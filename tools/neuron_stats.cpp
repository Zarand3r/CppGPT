// One corpus pass over fch_gelu: for every MLP neuron, the contexts that
// activate it most (M7-4).
//
// This is the first producer for the D11 artifact channel, and it exists in the
// same change as the channel deliberately -- a loader with nothing to load is a
// speculative abstraction, which this repo's doctrine forbids.
//
// WHAT IT DOES NOT DO. It reports what a neuron fires on, which is a
// correlation. Nothing here establishes that the neuron's activation CAUSES the
// downstream behaviour, and docs/INTERPRETING.md is emphatic that naming a
// component from correlation alone is the standard error in this field. The
// causal half is M7-6.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/dataloader.hpp"
#include "cppgpt/interp/artifact.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "tools/cli.hpp"
#include "tools/io.hpp"

using namespace cppgpt;

namespace {

struct Entry {
    float act;
    std::int32_t focus;
};

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv,
                         {"checkpoint", "data", "out", "windows", "seq", "top-k", "ctx", "seed"});
    const std::string ckpt(args.str("checkpoint", ""));
    const std::string data(args.str("data", ""));
    const std::string out(args.str("out", ""));
    if (ckpt.empty() || data.empty() || out.empty()) {
        std::fprintf(stderr,
                     "usage: neuron_stats --checkpoint X.ckpt --data corpus.bin --out N.art\n"
                     "                    [--windows 128] [--seq 32] [--top-k 4] [--ctx 24]\n"
                     "                    [--seed 1337]\n");
        return 2;
    }
    const int n_windows = args.integer("windows", 128);
    const int top_k = args.integer("top-k", 4);
    const int ctx_len = args.integer("ctx", 24);
    if (n_windows < 1 || top_k < 1 || ctx_len < 1) {
        std::fprintf(stderr, "neuron_stats: --windows/--top-k/--ctx must all be >= 1\n");
        return 2;
    }

    auto peek = CheckpointFile::open(ckpt.c_str());
    if (!peek) {
        std::fprintf(stderr, "neuron_stats: cannot read '%s': %s\n", ckpt.c_str(),
                     describe(peek.error()));
        return 1;
    }
    const CheckpointHeader& hdr = peek->header();
    Config cfg{};
    cfg.max_seq_len = hdr.max_seq_len;
    cfg.vocab_size = hdr.vocab_size;
    cfg.n_layer = hdr.n_layer;
    cfg.n_head = hdr.n_head;
    cfg.n_embd = hdr.n_embd;

    const int T = std::min(args.integer("seq", 32), cfg.max_seq_len);
    if (T < 2) {
        std::fprintf(stderr, "neuron_stats: --seq must be >= 2\n");
        return 2;
    }
    GPT2 model(cfg, /*B=*/1, T);
    if (const auto r = model.load_checkpoint(ckpt.c_str()); !r) {
        std::fprintf(stderr, "neuron_stats: cannot load '%s': %s\n", ckpt.c_str(),
                     describe(r.error()));
        return 1;
    }

    const auto toks = toolio::read_tokens(data.c_str());
    if (!toks) {
        std::fprintf(stderr, "neuron_stats: cannot read '%s': %s\n", data.c_str(),
                     describe(toks.error()));
        return 1;
    }
    if (toks->size() < static_cast<std::size_t>(T) + 1) {
        std::fprintf(stderr, "neuron_stats: corpus has %zu tokens; need at least %d\n",
                     toks->size(), T + 1);
        return 1;
    }

    const int L = cfg.n_layer, C = cfg.n_embd;
    const auto n_neurons = static_cast<std::size_t>(L) * static_cast<std::size_t>(4 * C);
    const auto kz = static_cast<std::size_t>(top_k);

    // A min-heap per neuron would be tidier; a sorted top-k array of 4 is
    // smaller and faster at this size, and the whole table is n_neurons * top_k.
    std::vector<Entry> best(n_neurons * kz, Entry{-1.0f, -1});
    std::vector<std::int32_t> best_ctx(n_neurons * kz * static_cast<std::size_t>(ctx_len), 0);
    std::vector<int> window(static_cast<std::size_t>(T));

    Generator gen(static_cast<std::uint64_t>(args.integer("seed", 1337)));
    const std::size_t hi = toks->size() - static_cast<std::size_t>(T) - 1;
    std::vector<std::size_t> starts(static_cast<std::size_t>(n_windows));
    for (auto& s : starts)
        s = static_cast<std::size_t>(gen.uniform_int(0, static_cast<std::int64_t>(hi)));

    std::printf("neuron_stats: %s over %s\n", ckpt.c_str(), data.c_str());
    std::printf("  L%d C%d | %zu neurons | %d windows x %d tokens | top-%d, ctx %d\n", L, C,
                n_neurons, n_windows, T, top_k, ctx_len);

    for (int w = 0; w < n_windows; ++w) {
        const std::size_t start = starts[static_cast<std::size_t>(w)];
        for (int t = 0; t < T; ++t)
            window[static_cast<std::size_t>(t)] =
                static_cast<int>((*toks)[start + static_cast<std::size_t>(t)]);
        model.forward(window.data(), nullptr);

        const float* g = model.acts().fch_gelu;
        for (int l = 0; l < L; ++l) {
            const float* layer = g + static_cast<std::size_t>(l) * static_cast<std::size_t>(T) *
                                         static_cast<std::size_t>(4 * C);
            for (int t = 0; t < T; ++t) {
                const float* row = layer + static_cast<std::size_t>(t) * static_cast<std::size_t>(4 * C);
                for (int u = 0; u < 4 * C; ++u) {
                    const auto ni = static_cast<std::size_t>(l) * static_cast<std::size_t>(4 * C) +
                                    static_cast<std::size_t>(u);
                    Entry* slot = best.data() + ni * kz;
                    if (row[u] <= slot[kz - 1].act) continue;  // not in this neuron's top-k

                    // Shift down from the tail; top_k is 4, so this is cheaper
                    // than any structure with a pointer in it.
                    std::size_t pos = kz - 1;
                    while (pos > 0 && slot[pos - 1].act < row[u]) {
                        slot[pos] = slot[pos - 1];
                        std::memcpy(best_ctx.data() + (ni * kz + pos) * static_cast<std::size_t>(ctx_len),
                                    best_ctx.data() + (ni * kz + pos - 1) * static_cast<std::size_t>(ctx_len),
                                    static_cast<std::size_t>(ctx_len) * sizeof(std::int32_t));
                        --pos;
                    }
                    slot[pos] = Entry{row[u], 0};
                    // Right-aligned context ending at the firing position, so the
                    // last real token is always the one that fired.
                    std::int32_t* dst =
                        best_ctx.data() + (ni * kz + pos) * static_cast<std::size_t>(ctx_len);
                    for (int c = 0; c < ctx_len; ++c) {
                        const int src = t - (ctx_len - 1 - c);
                        dst[c] = src >= 0 ? static_cast<std::int32_t>(window[static_cast<std::size_t>(src)])
                                          : -1;
                    }
                    slot[pos].focus = static_cast<std::int32_t>(ctx_len - 1);
                }
            }
        }
    }

    NeuronTopKHeader nh{};
    nh.n_neurons = static_cast<std::uint32_t>(n_neurons);
    nh.top_k = static_cast<std::uint32_t>(top_k);
    nh.ctx_len = static_cast<std::uint32_t>(ctx_len);
    nh.reserved = 0;

    std::string payload;
    payload.resize(sizeof(nh) + n_neurons * kz * neuron_entry_bytes(nh.ctx_len));
    std::memcpy(payload.data(), &nh, sizeof(nh));
    std::size_t off = sizeof(nh);
    int dead = 0;
    for (std::size_t ni = 0; ni < n_neurons; ++ni) {
        if (best[ni * kz].focus < 0) ++dead;
        for (std::size_t j = 0; j < kz; ++j) {
            const Entry& e = best[ni * kz + j];
            const float act = e.focus < 0 ? 0.0f : e.act;
            std::memcpy(payload.data() + off, &act, sizeof(act));
            off += sizeof(act);
            std::memcpy(payload.data() + off, &e.focus, sizeof(e.focus));
            off += sizeof(e.focus);
            std::memcpy(payload.data() + off,
                        best_ctx.data() + (ni * kz + j) * static_cast<std::size_t>(ctx_len),
                        static_cast<std::size_t>(ctx_len) * sizeof(std::int32_t));
            off += static_cast<std::size_t>(ctx_len) * sizeof(std::int32_t);
        }
    }
    ASSERT_MSG(off == payload.size(), "neuron_stats: payload size does not match what was written");

    if (const auto r = write_artifact(out.c_str(), ArtifactKind::NeuronTopK, hdr.checksum, payload);
        !r) {
        std::fprintf(stderr, "neuron_stats: writing '%s' failed: %s\n", out.c_str(),
                     describe(r.error()));
        return 1;
    }
    std::printf("  wrote %s (%zu bytes, %d neurons never activated)\n", out.c_str(),
                payload.size(), dead);
    return 0;
}
