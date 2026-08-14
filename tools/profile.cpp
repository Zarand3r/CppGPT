// profile — how long does one forward pass take?
//
// This is the budget for everything interactive. Ablation, patching, and every
// "what if" question costs one forward pass each, so the sweep in tools/inspect
// is exactly L*(NH+2) of whatever this prints.
//
// Two measurements, deliberately separate:
//
//   1. End-to-end forward latency, warmup then best-of-N. Best rather than mean
//      because it is the least-noisy estimate of what the machine can do; the
//      median is printed alongside so a large gap between them exposes a noisy
//      box rather than hiding it.
//   2. A per-op breakdown, timed standalone at the model's own shapes. These are
//      INDEPENDENT measurements and will not sum to the end-to-end number —
//      cache state differs, and the model reuses arenas the microbenchmarks do
//      not. Read the ratios, not the sum. The sum is printed with its error
//      against the end-to-end time so the gap is visible instead of implied.
//
// Architecture comes from the checkpoint, never from flags — same rule as
// tools/inspect, for the same reason: a flag that disagrees with the weights
// profiles a model that does not exist.
//
// Usage: profile --checkpoint <f.ckpt> [--batch 1] [--seq 64] [--reps 50]
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/core.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/ops.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/storage.hpp"
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Best and median of `v`, in ms. Median needs a copy since we sort.
struct Stat {
    double best, median;
};
Stat summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size() / 2]};
}

// Time `fn` reps times after `warm` warmup iterations.
template <class F>
Stat time_it(int warm, int reps, F&& fn) {
    for (int i = 0; i < warm; ++i) fn();
    std::vector<double> t;
    t.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        const auto t0 = Clock::now();
        fn();
        t.push_back(ms_since(t0));
    }
    return summarize(std::move(t));
}

// Analytic forward FLOPs (multiply and add counted separately, matching how the
// bench tool reports, so the two numbers are comparable).
double forward_gflop(const Config& c, int B, int T) {
    const double b = B, t = T, C = c.n_embd, V = c.vocab_size, L = c.n_layer;
    const double per_layer = 2.0 * b * t * C * (3.0 * C)      // qkv
                             + 2.0 * b * t * t * C           // attention scores + weighted sum
                             + 2.0 * b * t * C * C           // attn out projection
                             + 2.0 * b * t * C * (4.0 * C)   // mlp fc
                             + 2.0 * b * t * (4.0 * C) * C;  // mlp proj
    return (L * per_layer + 2.0 * b * t * C * V) / 1e9;       // + tied classifier
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv, {"checkpoint", "batch", "seq", "reps"});
    const std::string ckpt(args.str("checkpoint", ""));
    if (ckpt.empty()) {
        std::fprintf(stderr,
                     "usage: profile --checkpoint <f.ckpt> [--batch 1] [--seq 64] [--reps 50]\n");
        return 2;
    }
    const int reps = args.integer("reps", 50);
    if (reps < 1) {
        std::fprintf(stderr, "profile: --reps must be >= 1\n");
        return 2;
    }

    auto peek = CheckpointFile::open(ckpt.c_str());
    if (!peek) {
        std::fprintf(stderr, "profile: cannot read '%s': %s\n", ckpt.c_str(),
                     describe(peek.error()));
        return 1;
    }
    const CheckpointHeader& h = peek->header();
    const Config cfg{h.max_seq_len, h.vocab_size, h.n_layer, h.n_head, h.n_embd};

    const int B = args.integer("batch", 1);
    const int T = args.integer("seq", cfg.max_seq_len);
    if (B < 1 || T < 1 || T > cfg.max_seq_len) {
        std::fprintf(stderr, "profile: --batch must be >= 1 and --seq in 1..%d\n", cfg.max_seq_len);
        return 1;
    }

    GPT2 model(cfg, B, T);
    if (const auto r = model.load_checkpoint(ckpt.c_str()); !r) {
        std::fprintf(stderr, "profile: cannot load '%s': %s\n", ckpt.c_str(), describe(r.error()));
        return 1;
    }

    Generator gen(1234ULL);
    std::vector<int> tokens(static_cast<std::size_t>(B) * static_cast<std::size_t>(T));
    for (auto& x : tokens) x = static_cast<int>(gen.uniform_int(0, cfg.vocab_size - 1));

    std::printf("profile: %s\n", ckpt.c_str());
    std::printf("  config   L%d H%d C%d V%d ctx%d   |   B=%d T=%d (%d tokens/pass)\n", cfg.n_layer,
                cfg.n_head, cfg.n_embd, cfg.vocab_size, cfg.max_seq_len, B, T, B * T);

    // ---- 1. end-to-end forward ------------------------------------------------
    const Stat fwd = time_it(3, reps, [&] { model.forward(tokens.data(), nullptr); });
    const double gf = forward_gflop(cfg, B, T);
    std::printf("\n  forward pass (inference, no targets), best of %d:\n", reps);
    std::printf("    %.3f ms   (median %.3f ms)\n", fwd.best, fwd.median);
    std::printf("    %.0f tokens/s   %.2f GFLOP/pass   %.1f GFLOP/s\n",
                (B * T) / (fwd.best / 1000.0), gf, gf / (fwd.best / 1000.0));

    // A forward WITH targets additionally runs softmax + cross-entropy over the
    // whole [B,T,V] logit block, which is what training pays and inference does
    // not — worth separating, since at small C the softmax is not negligible.
    std::vector<int> targets(tokens.size());
    for (std::size_t i = 0; i < targets.size(); ++i)
        targets[i] = tokens[(i + 1) % tokens.size()];
    const Stat fwd_loss = time_it(3, reps, [&] { model.forward(tokens.data(), targets.data()); });
    std::printf("    with loss (softmax + cross-entropy): %.3f ms  (+%.1f%%)\n", fwd_loss.best,
                100.0 * (fwd_loss.best - fwd.best) / fwd.best);

    // ---- 2. per-op breakdown --------------------------------------------------
    // Standalone buffers at the model's shapes. These are independent timings:
    // see the header note on why they do not sum to the end-to-end number.
    const int C = cfg.n_embd, NH = cfg.n_head, V = cfg.vocab_size;
    const auto BTC = static_cast<std::size_t>(B) * T * C;
    const auto BT = static_cast<std::size_t>(B) * T;
    Storage buf(6 * BTC + 4 * BTC + BT * 2 + static_cast<std::size_t>(B) * NH * T * T * 2 +
                BT * static_cast<std::size_t>(V) + static_cast<std::size_t>(C) * 4 * C + 4 * C);
    float* x = buf.alloc(BTC);
    float* y = buf.alloc(BTC);
    float* qkv = buf.alloc(3 * BTC);
    float* big = buf.alloc(4 * BTC);
    float* mean = buf.alloc(BT);
    float* rstd = buf.alloc(BT);
    float* preatt = buf.alloc(static_cast<std::size_t>(B) * NH * T * T);
    float* att = buf.alloc(static_cast<std::size_t>(B) * NH * T * T);
    float* logits = buf.alloc(BT * static_cast<std::size_t>(V));
    for (std::size_t i = 0; i < BTC; ++i) x[i] = gen.normal();

    const ParamTensors& p = model.params();
    struct Row {
        const char* name;
        double ms;
        int per_pass;  // how many times a full forward runs this
    };
    std::vector<Row> rows;
    const int L = cfg.n_layer;
    rows.push_back({"layernorm",
                    time_it(2, reps, [&] {
                        layernorm_forward(y, mean, rstd, x, p.ln1w, p.ln1b, B, T, C);
                    }).best,
                    2 * L});
    rows.push_back({"matmul qkv    [C->3C]",
                    time_it(2, reps, [&] { matmul_forward(qkv, x, p.qkvw, p.qkvb, B, T, C, 3 * C); })
                        .best,
                    L});
    rows.push_back({"attention",
                    time_it(2, reps, [&] {
                        attention_forward(y, preatt, att, qkv, B, T, C, NH);
                    }).best,
                    L});
    rows.push_back({"matmul attproj[C->C]",
                    time_it(2, reps, [&] {
                        matmul_forward(y, x, p.attprojw, p.attprojb, B, T, C, C);
                    }).best,
                    L});
    rows.push_back({"matmul fc     [C->4C]",
                    time_it(2, reps, [&] { matmul_forward(big, x, p.fcw, p.fcb, B, T, C, 4 * C); })
                        .best,
                    L});
    rows.push_back({"gelu",
                    time_it(2, reps, [&] {
                        gelu_forward(big, big, static_cast<int>(BTC) * 4);
                    }).best,
                    L});
    rows.push_back({"matmul fcproj [4C->C]",
                    time_it(2, reps, [&] {
                        matmul_forward(y, big, p.fcprojw, p.fcprojb, B, T, 4 * C, C);
                    }).best,
                    L});
    rows.push_back({"residual add",
                    time_it(2, reps, [&] { residual_forward(y, x, y, static_cast<int>(BTC)); }).best,
                    2 * L});
    rows.push_back({"matmul lm_head[C->V]",
                    time_it(2, reps, [&] { matmul_forward(logits, x, p.wte, nullptr, B, T, C, V); })
                        .best,
                    1});

    double sum = 0.0;
    for (const Row& r : rows) sum += r.ms * r.per_pass;
    std::printf("\n  per-op, timed standalone at these shapes (x = calls per forward):\n");
    std::printf("    %-24s %10s %5s %10s %7s\n", "op", "ms/call", "x", "ms/pass", "share");
    for (const Row& r : rows) {
        const double total = r.ms * r.per_pass;
        std::printf("    %-24s %10.4f %5d %10.4f %6.1f%%\n", r.name, r.ms, r.per_pass, total,
                    100.0 * total / sum);
    }
    std::printf("    %-24s %10s %5s %10.4f\n", "sum of parts", "", "", sum);
    std::printf("    end-to-end was %.4f ms — parts differ by %+.1f%% (independent timings, "
                "different cache state)\n",
                fwd.best, 100.0 * (sum - fwd.best) / fwd.best);
    return 0;
}
