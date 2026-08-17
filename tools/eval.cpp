// eval — how good is this model, measured against baselines that make the number
// mean something.
//
// A cross-entropy of 1.81 nats is uninterpretable alone. It is only meaningful
// next to what you would get for free, so this reports the model and three
// reference points on the SAME validation tokens:
//
//   uniform  — ln(V). Knowing nothing at all.
//   unigram  — character frequencies from the training split. Knowing English
//              letter statistics and nothing else.
//   bigram   — P(next | previous char) from the training split. One character of
//              context, the cheapest thing that could be called a model.
//
// A transformer that does not beat bigram has learned nothing a lookup table
// could not do. That comparison is `ROADMAP.md` M2's convergence gate ("val loss
// ... below a same-corpus bigram baseline"), which until now had no
// implementation and therefore could not be checked.
//
// The baselines are counted on the TRAINING split and scored on the VALIDATION
// split. Counting them on the validation split would let them memorise the very
// text they are scored on, which would flatter the baselines and understate the
// model — the comparison has to be honest in the direction that hurts us.
//
// Differences from the eval inside tools/train:
//   * Deterministic and complete. Sequential non-overlapping windows over the
//     whole file, no shuffle, so two runs on one checkpoint give the same number
//     and two checkpoints are compared on identical tokens. train's eval pulls a
//     drifting subset from a shuffled cursor.
//   * Reports perplexity and bits/character alongside nats.
//   * States how many tokens it dropped rather than truncating silently.
//
// Usage: eval --checkpoint <f.ckpt> --data <f.val.bin> [--train <f.train.bin>]
//             [--batch 32] [--max-batches N]
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/core.hpp"
#include "cppgpt/model.hpp"
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;

// A bigram table is V*V counts. Fine for a character model (65 -> 34 KB) and
// absurd for a BPE one (50257 -> 20 GB), so refuse loudly rather than attempt it.
constexpr int kMaxBigramVocab = 2048;

// Read a flat little-endian uint16 token file. The validation split is small by
// construction (it is a held-out fraction, not a corpus), so this reads it whole;
// the training split is only ever streamed, never held.
std::vector<std::uint16_t> read_tokens(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "eval: cannot open '%s'\n", path.c_str());
        std::exit(1);
    }
    const std::streamoff bytes = f.tellg();
    if (bytes < 0 || bytes % 2 != 0) {
        std::fprintf(stderr, "eval: '%s' is %lld bytes, not a whole number of uint16 tokens\n",
                     path.c_str(), static_cast<long long>(bytes));
        std::exit(1);
    }
    std::vector<std::uint16_t> t(static_cast<std::size_t>(bytes) / 2);
    f.seekg(0);
    if (!t.empty() && !f.read(reinterpret_cast<char*>(t.data()), bytes)) {
        std::fprintf(stderr, "eval: short read on '%s'\n", path.c_str());
        std::exit(1);
    }
    return t;
}

struct Baselines {
    bool have = false;
    double uniform = 0.0, unigram = 0.0, bigram = 0.0;
};

// Count unigrams and bigrams over the training split, then score the same
// (previous, target) pairs the model is scored on. Laplace (add-one) smoothing,
// so an unseen pair costs a large but finite number of nats instead of infinity.
Baselines baselines_on(const std::string& train_path, const std::vector<std::uint16_t>& val,
                       const std::vector<std::pair<std::size_t, std::size_t>>& pairs, int V) {
    Baselines b;
    if (V > kMaxBigramVocab) {
        std::fprintf(stderr,
                     "eval: vocab %d exceeds the %d-symbol bigram limit (the table would be %.1f GB);"
                     " skipping baselines\n",
                     V, kMaxBigramVocab,
                     static_cast<double>(V) * V * 8.0 / 1e9);
        return b;
    }
    const std::vector<std::uint16_t> train = read_tokens(train_path);
    if (train.size() < 2) {
        std::fprintf(stderr, "eval: training split has %zu tokens; too few to count\n", train.size());
        return b;
    }
    const auto Vz = static_cast<std::size_t>(V);
    std::vector<std::uint64_t> uni(Vz, 0), bi(Vz * Vz, 0), row(Vz, 0);
    for (std::size_t i = 0; i < train.size(); ++i) {
        const std::size_t c = train[i];
        if (c >= Vz) {
            std::fprintf(stderr, "eval: training token %zu is out of vocab range\n", c);
            std::exit(1);
        }
        ++uni[c];
        if (i + 1 < train.size()) {
            const std::size_t n = train[i + 1];
            if (n < Vz) {
                ++bi[c * Vz + n];
                ++row[c];
            }
        }
    }
    const double total = static_cast<double>(train.size());
    double s_uni = 0.0, s_bi = 0.0;
    for (const auto& [prev, tgt] : pairs) {
        const std::size_t p = val[prev], t = val[tgt];
        s_uni += -std::log((static_cast<double>(uni[t]) + 1.0) / (total + static_cast<double>(V)));
        s_bi += -std::log((static_cast<double>(bi[p * Vz + t]) + 1.0) /
                          (static_cast<double>(row[p]) + static_cast<double>(V)));
    }
    const double n = static_cast<double>(pairs.size());
    b.have = true;
    b.uniform = std::log(static_cast<double>(V));
    b.unigram = s_uni / n;
    b.bigram = s_bi / n;
    return b;
}

void report(const char* name, double nats, double ref) {
    // Perplexity is exp(nats); bits/char is nats/ln2. For a character-level model
    // "per token" and "per character" are the same thing.
    std::printf("  %-22s %8.4f %10.2f %10.3f", name, nats, std::exp(nats), nats / std::log(2.0));
    if (ref > 0.0) std::printf("   %+7.1f%%", 100.0 * (nats - ref) / ref);
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv, {"checkpoint", "data", "train", "batch", "max-batches"});
    const std::string ckpt(args.str("checkpoint", ""));
    const std::string data(args.str("data", ""));
    if (ckpt.empty() || data.empty()) {
        std::fprintf(stderr,
                     "usage: eval --checkpoint <f.ckpt> --data <f.val.bin>\n"
                     "            [--train <f.train.bin>] [--batch 32] [--max-batches N]\n");
        return 2;
    }
    const int B = args.integer("batch", 32);
    const int max_batches = args.integer("max-batches", 0);  // 0 = the whole file
    if (B < 1) {
        std::fprintf(stderr, "eval: --batch must be >= 1\n");
        return 2;
    }

    auto peek = CheckpointFile::open(ckpt.c_str());
    if (!peek) {
        std::fprintf(stderr, "eval: cannot read '%s': %s\n", ckpt.c_str(), describe(peek.error()));
        return 1;
    }
    const CheckpointHeader& h = peek->header();
    const Config cfg{h.max_seq_len, h.vocab_size, h.n_layer, h.n_head, h.n_embd};
    const int T = cfg.max_seq_len, V = cfg.vocab_size;

    const std::vector<std::uint16_t> val = read_tokens(data);
    // Example k spans [k*T, k*T+T]: T inputs and the T targets one step ahead, so
    // it needs T+1 tokens.
    const std::size_t n_examples = val.size() > static_cast<std::size_t>(T)
                                       ? (val.size() - 1) / static_cast<std::size_t>(T)
                                       : 0;
    if (n_examples == 0) {
        std::fprintf(stderr, "eval: '%s' holds %zu tokens, fewer than one %d-token window\n",
                     data.c_str(), val.size(), T + 1);
        return 1;
    }
    std::size_t n_batches = n_examples / static_cast<std::size_t>(B);
    if (n_batches == 0) {
        std::fprintf(stderr, "eval: %zu examples is fewer than one batch of %d; lower --batch\n",
                     n_examples, B);
        return 1;
    }
    if (max_batches > 0) n_batches = std::min(n_batches, static_cast<std::size_t>(max_batches));

    GPT2 model(cfg, B, T);
    if (const auto r = model.load_checkpoint(ckpt.c_str(), GPT2::LoadMode::WeightsOnly); !r) {
        std::fprintf(stderr, "eval: cannot load '%s': %s\n", ckpt.c_str(), describe(r.error()));
        return 1;
    }

    const auto per_batch = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    std::vector<int> inputs(per_batch), targets(per_batch);
    // The exact (previous, target) index pairs the model is scored on, so the
    // baselines are scored on precisely the same predictions and not merely the
    // same file.
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    pairs.reserve(n_batches * per_batch);

    double sum_nats = 0.0;
    for (std::size_t b = 0; b < n_batches; ++b) {
        for (int i = 0; i < B; ++i) {
            const std::size_t base =
                (b * static_cast<std::size_t>(B) + static_cast<std::size_t>(i)) *
                static_cast<std::size_t>(T);
            for (int t = 0; t < T; ++t) {
                const auto k = static_cast<std::size_t>(i) * static_cast<std::size_t>(T) +
                               static_cast<std::size_t>(t);
                inputs[k] = static_cast<int>(val[base + static_cast<std::size_t>(t)]);
                targets[k] = static_cast<int>(val[base + static_cast<std::size_t>(t) + 1]);
                pairs.emplace_back(base + static_cast<std::size_t>(t),
                                   base + static_cast<std::size_t>(t) + 1);
            }
        }
        // Next-token evaluation means the targets ARE the inputs shifted by one.
        // Stated as an invariant rather than trusted to the index arithmetic
        // above: an off-by-one here scores the model on a different prediction
        // entirely, and on an undertrained model the resulting loss looks
        // entirely plausible — close enough to the right answer that no bound on
        // the reported number can separate them. Cheap, and it turns a silent
        // wrong answer into a crash.
        for (int i = 0; i < B; ++i) {
            const auto row = static_cast<std::size_t>(i) * static_cast<std::size_t>(T);
            for (int t = 0; t + 1 < T; ++t) {
                const auto k = row + static_cast<std::size_t>(t);
                ASSERT_MSG(targets[k] == inputs[k + 1],
                           "eval: targets are not the inputs shifted by one");
            }
        }
        model.forward(inputs.data(), targets.data());
        // mean_loss is the mean over this batch's B*T predictions; weight it back
        // up so the global figure is a true per-token mean.
        sum_nats += static_cast<double>(model.mean_loss()) * static_cast<double>(per_batch);
    }
    const double n_tokens = static_cast<double>(n_batches * per_batch);
    const double model_nats = sum_nats / n_tokens;

    std::printf("eval: %s on %s\n", ckpt.c_str(), data.c_str());
    std::printf("  L%d H%d C%d V%d ctx%d | %zu of %zu tokens scored (%zu batches of %d x %d)\n",
                cfg.n_layer, cfg.n_head, cfg.n_embd, V, T, static_cast<std::size_t>(n_tokens),
                val.size(), n_batches, B, T);
    // Never let a partial tail vanish silently: a number computed on 90% of the
    // file must say so.
    const std::size_t dropped = val.size() - static_cast<std::size_t>(n_tokens);
    if (dropped > 0)
        std::printf("  %zu tokens not scored (partial trailing window/batch)\n", dropped);

    const std::string train_path(args.str("train", ""));
    const Baselines base = train_path.empty() ? Baselines{} : baselines_on(train_path, val, pairs, V);

    std::printf("\n  %-22s %8s %10s %10s   %8s\n", "", "nats/tok", "perplexity", "bits/char",
                "vs model");
    report("model", model_nats, 0.0);
    if (base.have) {
        report("bigram baseline", base.bigram, model_nats);
        report("unigram baseline", base.unigram, model_nats);
        report("uniform baseline", base.uniform, model_nats);
        std::printf("\n  %s the bigram baseline by %.3f nats (%.1f%%).\n",
                    model_nats < base.bigram ? "Beats" : "LOSES TO",
                    std::fabs(base.bigram - model_nats),
                    100.0 * std::fabs(base.bigram - model_nats) / base.bigram);
        if (model_nats >= base.bigram)
            std::printf("  A transformer that does not beat one character of context has learned"
                        " nothing a lookup table could not.\n");
    } else if (train_path.empty()) {
        std::printf("\n  Pass --train <f.train.bin> for the baselines that make this number"
                    " interpretable.\n");
    }
    return 0;
}
