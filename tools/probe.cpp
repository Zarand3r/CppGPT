// Linear probes over the residual stream, with causal validation (M7-5, M7-6).
//
// For each (layer, property): fit a direction that decodes the property, report
// held-out accuracy against its base rate and a shuffled-label control, then
// STEER along the direction and compare the output shift against a random
// direction of the same norm.
//
// The two halves answer different questions and neither substitutes for the
// other. Decodability says the property is present in the representation.
// Steering says the model's output depends on that direction. A property can be
// perfectly decodable from a direction the model never reads.
//
// Properties are chosen for what a 4-layer character model could plausibly
// encode -- capitalisation, line structure, punctuation context -- and each
// needs CONTEXT, not just the current character. A probe that recovers "this
// character is a vowel" from layer 0 has rediscovered the embedding.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/interp/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/tokenizer.hpp"
#include "tools/cli.hpp"
#include "tools/io.hpp"

using namespace cppgpt;

namespace {

struct Property {
    const char* name;
    const char* what;
};

// Labels depend on the character AND its predecessor, so none is recoverable
// from the current token embedding alone.
constexpr Property kProps[] = {
    {"after_newline", "the previous character was a line break"},
    {"after_space", "the previous character was a space"},
    {"after_punct", "the previous character was punctuation"},
    {"in_caps_run", "this and the previous character are both uppercase"},
    {"after_vowel", "the previous character was a vowel"},
};

bool label_for(const Property& p, char prev, char cur) {
    const std::string_view name(p.name);
    if (name == "after_newline") return prev == '\n';
    if (name == "after_space") return prev == ' ';
    if (name == "after_punct")
        return prev == '.' || prev == ',' || prev == ';' || prev == ':' || prev == '!' ||
               prev == '?';
    if (name == "in_caps_run")
        return prev >= 'A' && prev <= 'Z' && cur >= 'A' && cur <= 'Z';
    return prev == 'a' || prev == 'e' || prev == 'i' || prev == 'o' || prev == 'u' ||
           prev == 'A' || prev == 'E' || prev == 'I' || prev == 'O' || prev == 'U';
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv,
                         {"checkpoint", "vocab", "data", "windows", "seq", "seed", "scale"});
    const std::string ckpt(args.str("checkpoint", ""));
    const std::string vocab(args.str("vocab", ""));
    const std::string data(args.str("data", ""));
    if (ckpt.empty() || vocab.empty() || data.empty()) {
        std::fprintf(stderr,
                     "usage: probe --checkpoint X.ckpt --vocab X.vocab --data corpus.bin\n"
                     "             [--windows 64] [--seq 32] [--seed 1337] [--scale 2.0]\n");
        return 2;
    }
    const int n_windows = args.integer("windows", 64);
    const float scale = static_cast<float>(args.real("scale", 2.0f));

    auto peek = CheckpointFile::open(ckpt.c_str());
    if (!peek) {
        std::fprintf(stderr, "probe: cannot read '%s': %s\n", ckpt.c_str(), describe(peek.error()));
        return 1;
    }
    const CheckpointHeader& h = peek->header();
    Config cfg{};
    cfg.max_seq_len = h.max_seq_len;
    cfg.vocab_size = h.vocab_size;
    cfg.n_layer = h.n_layer;
    cfg.n_head = h.n_head;
    cfg.n_embd = h.n_embd;

    const int T = std::min(args.integer("seq", 32), cfg.max_seq_len);
    GPT2 model(cfg, 1, T);
    if (const auto r = model.load_checkpoint(ckpt.c_str()); !r) {
        std::fprintf(stderr, "probe: cannot load '%s': %s\n", ckpt.c_str(), describe(r.error()));
        return 1;
    }
    const auto vocab_bytes = toolio::read_file(vocab.c_str());
    if (!vocab_bytes) {
        std::fprintf(stderr, "probe: cannot read '%s': %s\n", vocab.c_str(),
                     describe(vocab_bytes.error()));
        return 1;
    }
    const CharTokenizer tok(*vocab_bytes);

    const auto toks = toolio::read_tokens(data.c_str());
    if (!toks) {
        std::fprintf(stderr, "probe: cannot read '%s': %s\n", data.c_str(), describe(toks.error()));
        return 1;
    }

    const int L = cfg.n_layer, C = cfg.n_embd;
    Generator gen(static_cast<std::uint64_t>(args.integer("seed", 1337)));
    const std::size_t hi = toks->size() - static_cast<std::size_t>(T) - 1;

    // Windows are drawn up front and used in order, so the train/test split is
    // by CORPUS POSITION. A random row split would put a character's neighbours
    // on both sides, and adjacent positions are not independent samples.
    std::vector<std::size_t> starts(static_cast<std::size_t>(n_windows));
    for (auto& s : starts)
        s = static_cast<std::size_t>(gen.uniform_int(0, static_cast<std::int64_t>(hi)));

    const int rows = n_windows * (T - 1);  // position 0 has no predecessor
    const int n_train = (rows * 3) / 4;
    std::vector<float> feats(static_cast<std::size_t>(rows) * C);
    std::vector<std::vector<std::uint8_t>> labels(std::size(kProps),
                                                  std::vector<std::uint8_t>(rows));
    std::vector<int> window(static_cast<std::size_t>(T));

    std::printf("probe: %s over %s\n", ckpt.c_str(), data.c_str());
    std::printf("  %d windows x %d tokens = %d rows, %d train / %d held out, scale %.1f\n",
                n_windows, T, rows, n_train, rows - n_train, static_cast<double>(scale));
    std::printf("\n  %-14s %-6s %8s %8s %9s %10s %10s %8s\n", "property", "layer", "acc", "base",
                "shuffled", "steer KL", "null mean", "beats");

    for (int l = 0; l < L; ++l) {
        int row = 0;
        for (int w = 0; w < n_windows; ++w) {
            const std::size_t st = starts[static_cast<std::size_t>(w)];
            for (int t = 0; t < T; ++t)
                window[static_cast<std::size_t>(t)] =
                    static_cast<int>((*toks)[st + static_cast<std::size_t>(t)]);
            model.forward(window.data(), nullptr);
            const float* resid = layer_slice(model.acts().residual3, l, 1, T, C);
            for (int t = 1; t < T; ++t) {
                std::memcpy(feats.data() + static_cast<std::size_t>(row) * C,
                            resid + static_cast<std::size_t>(t) * C,
                            static_cast<std::size_t>(C) * sizeof(float));
                const int one_prev[1] = {window[static_cast<std::size_t>(t - 1)]};
                const int one_cur[1] = {window[static_cast<std::size_t>(t)]};
                const char prev = tok.decode(std::span<const int>(one_prev, 1))[0];
                const char cur = tok.decode(std::span<const int>(one_cur, 1))[0];
                for (std::size_t pi = 0; pi < std::size(kProps); ++pi)
                    labels[pi][static_cast<std::size_t>(row)] =
                        label_for(kProps[pi], prev, cur) ? 1 : 0;
                ++row;
            }
        }

        std::vector<float> dir(static_cast<std::size_t>(C));
        for (std::size_t pi = 0; pi < std::size(kProps); ++pi) {
            const ProbeResult pr = fit_probe(feats.data(), labels[pi].data(), rows, C, n_train,
                                             dir.data(), gen);
            // Steer on one window; the direction is what is under test, and the
            // random control is drawn from the same generator.
            for (int t = 0; t < T; ++t)
                window[static_cast<std::size_t>(t)] = static_cast<int>((*toks)[starts[0] + static_cast<std::size_t>(t)]);
            const SteerResult sr =
                steer_effect(model, window.data(), l, T - 1, dir.data(), scale, 20, gen);
            std::printf("  %-14s L%-5d %8.3f %8.3f %9.3f %10.4f %10.4f %7.0f%%\n", kProps[pi].name, l,
                        static_cast<double>(pr.accuracy), static_cast<double>(pr.base_rate),
                        static_cast<double>(pr.shuffled), static_cast<double>(sr.kl_direction),
                        static_cast<double>(sr.kl_random_mean),
                        100.0 * static_cast<double>(sr.beats_random));
        }
    }
    std::printf(
        "\n  A direction is a finding only if acc clears base AND shuffled sits at base\n"
        "  AND it beats most of a 20-draw random null. Decodable-but-not-causal is the common case.\n");
    return 0;
}
