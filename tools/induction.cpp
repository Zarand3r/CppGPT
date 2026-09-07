// Does this model have induction heads? (B4)
//
// The canonical operational test (Olsson et al. 2022): feed a random block
// repeated twice and see whether any head attends from the second copy back to
// what followed the matching token in the first.
//
// This exists to CHECK M-22, which concluded there are no copying heads and
// therefore no induction heads using a weight-space statistic of this repo's own
// construction. This measurement shares none of that machinery -- it reads
// attention from a real forward -- so agreement makes the negative rest on two
// independent methods rather than one.
//
// Averaged over many random blocks, because one block is one sample and this
// repo has already published two single-sample claims it had to retract.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/interp/artifact.hpp"
#include "cppgpt/interp/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "tools/cli.hpp"

using namespace cppgpt;

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv, {"checkpoint", "half", "trials", "seed", "out"});
    const std::string ckpt(args.str("checkpoint", ""));
    if (ckpt.empty()) {
        std::fprintf(stderr,
                     "usage: induction --checkpoint X.ckpt [--half 16] [--trials 64] [--seed 7]\n");
        return 2;
    }
    const int trials = args.integer("trials", 64);
    const std::string out(args.str("out", ""));

    auto peek = CheckpointFile::open(ckpt.c_str());
    if (!peek) {
        std::fprintf(stderr, "induction: cannot read '%s': %s\n", ckpt.c_str(),
                     describe(peek.error()));
        return 1;
    }
    const CheckpointHeader& h = peek->header();
    Config cfg{h.max_seq_len, h.vocab_size, h.n_layer, h.n_head, h.n_embd};

    const int half = std::min(args.integer("half", 16), cfg.max_seq_len / 2);
    if (half < 2) {
        std::fprintf(stderr, "induction: --half must be >= 2 and at most ctx/2\n");
        return 2;
    }
    const int T = 2 * half, L = cfg.n_layer, NH = cfg.n_head;

    GPT2 model(cfg, 1, T);
    if (const auto r = model.load_checkpoint(ckpt.c_str()); !r) {
        std::fprintf(stderr, "induction: cannot load '%s': %s\n", ckpt.c_str(), describe(r.error()));
        return 1;
    }

    Generator gen(static_cast<std::uint64_t>(args.integer("seed", 7)));
    const auto n_head_total = static_cast<std::size_t>(L) * static_cast<std::size_t>(NH);
    std::vector<double> sum(n_head_total, 0.0), sumsq(n_head_total, 0.0);
    std::vector<float> per_head(n_head_total);
    std::vector<int> tok(static_cast<std::size_t>(T));
    float uniform = 0.0f;

    // A CONTROL run on non-repeated sequences. Without it a raw score is
    // unreadable: some attention lands on those positions by geometry alone.
    std::vector<double> ctrl(n_head_total, 0.0);

    for (int t = 0; t < trials; ++t) {
        for (int i = 0; i < half; ++i) {
            tok[static_cast<std::size_t>(i)] =
                static_cast<int>(gen.uniform_int(0, cfg.vocab_size - 1));
            tok[static_cast<std::size_t>(half + i)] = tok[static_cast<std::size_t>(i)];
        }
        model.forward(tok.data(), nullptr);
        induction_scores(model, half, per_head.data(), &uniform);
        for (std::size_t i = 0; i < n_head_total; ++i) {
            sum[i] += per_head[i];
            sumsq[i] += static_cast<double>(per_head[i]) * per_head[i];
        }

        // Same length, no repetition.
        for (int i = 0; i < T; ++i)
            tok[static_cast<std::size_t>(i)] =
                static_cast<int>(gen.uniform_int(0, cfg.vocab_size - 1));
        model.forward(tok.data(), nullptr);
        induction_scores(model, half, per_head.data(), &uniform);
        for (std::size_t i = 0; i < n_head_total; ++i) ctrl[i] += per_head[i];
    }

    std::printf("induction: %s\n", ckpt.c_str());
    std::printf("  %d trials, block of %d repeated twice (T=%d), vocab %d\n", trials, half, T,
                cfg.vocab_size);
    std::printf("  uniform-attention baseline on these positions: %.4f\n\n", static_cast<double>(uniform));
    std::printf("  %-8s %10s %8s %10s %10s\n", "head", "repeated", "sd", "control", "x uniform");

    const double n = static_cast<double>(trials);
    std::vector<std::size_t> order(n_head_total);
    for (std::size_t i = 0; i < n_head_total; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return sum[a] > sum[b];
    });
    double best = 0.0;
    for (const std::size_t i : order) {
        const double mean = sum[i] / n;
        const double sd = std::sqrt(std::max(sumsq[i] / n - mean * mean, 0.0));
        std::printf("  L%zuH%-5zu %10.4f %8.4f %10.4f %10.2f\n", i / static_cast<std::size_t>(NH),
                    i % static_cast<std::size_t>(NH), mean, sd, ctrl[i] / n,
                    mean / static_cast<double>(uniform));
        best = std::max(best, mean / static_cast<double>(uniform));
    }
    std::printf(
        "\n  An induction head attends far above uniform on repeated blocks and not on the\n"
        "  control. Best here is %.2fx uniform.\n", best);

    if (!out.empty()) {
        InductionHeader ih{};
        ih.n_layer = static_cast<std::uint32_t>(L);
        ih.n_head = static_cast<std::uint32_t>(NH);
        ih.trials = static_cast<std::uint32_t>(trials);
        ih.half = static_cast<std::uint32_t>(half);
        ih.uniform = uniform;
        ih.reserved = 0;
        std::vector<InductionRecord> recs(n_head_total);
        for (std::size_t i = 0; i < n_head_total; ++i)
            recs[i] = InductionRecord{static_cast<float>(sum[i] / n),
                                      static_cast<float>(ctrl[i] / n)};
        std::string payload;
        payload.resize(sizeof(ih) + recs.size() * sizeof(InductionRecord));
        std::memcpy(payload.data(), &ih, sizeof(ih));
        std::memcpy(payload.data() + sizeof(ih), recs.data(), recs.size() * sizeof(InductionRecord));
        if (const auto r = write_artifact(out.c_str(), ArtifactKind::InductionScores, h.checksum,
                                          payload);
            !r) {
            std::fprintf(stderr, "induction: writing '%s' failed: %s\n", out.c_str(),
                         describe(r.error()));
            return 1;
        }
        std::printf("  wrote %s (%zu heads)\n", out.c_str(), recs.size());
    }
    return 0;
}
