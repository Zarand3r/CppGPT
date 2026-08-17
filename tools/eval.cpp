// eval — how good is this model, measured against baselines strong enough for
// the answer to mean something.
//
// A cross-entropy of 1.82 nats is uninterpretable alone, and beating a bigram is
// a low bar that any model which learns anything at all will clear. The question
// worth asking is whether a transformer beats a *well-smoothed high-order n-gram*
// — because if it does not, it has learned nothing a counting table could not,
// and all the depth is decoration.
//
// So this reports, on identical validation tokens:
//
//   uniform            ln(V). Knowing nothing.
//   unigram .. K-gram  Witten-Bell interpolated n-grams counted on the TRAINING
//                      split. The ladder matters: where the model sits on it says
//                      how many characters of context it is really worth.
//   model              the checkpoint.
//
// Four things beyond the scalar loss, because one number hides too much:
//
//   * Loss by context position. The single most diagnostic view here. If loss
//     stops falling after t characters, the model is using t characters —
//     whatever its depth suggests.
//   * Top-1 accuracy, which is what "predicts the next letter" actually means.
//   * Calibration (expected calibration error). A model at 90% confidence should
//     be right 90% of the time; systematic overconfidence is invisible in the
//     loss but obvious to anyone reading the predictions.
//   * Optional train-split score, so the train/val gap shows memorisation.
//
// Baselines are counted on the TRAINING split and scored on the VALIDATION split.
// Counting them on the validation split would let them memorise the text they are
// scored on — flattering the baselines and understating the model. The comparison
// has to be honest in the direction that hurts us.
//
// The n-gram sees exactly the context the model sees: within-window only, and
// truncated at the window start. Giving the baseline context the model does not
// have would be the same dishonesty pointing the other way.
//
// Smoothing is Witten-Bell interpolation: P_k = λ·P_ML + (1−λ)·P_{k−1} with
// λ = N(ctx)/(N(ctx)+D(ctx)), recursing down to uniform. Parameter-free (nothing
// to tune in the baseline's favour or against it) and, unlike stupid-backoff,
// normalised — which cross-entropy requires to mean anything.
//
// Usage: eval --checkpoint <f.ckpt> --data <f.val.bin> [--train <f.train.bin>]
//             [--batch 32] [--max-batches N] [--order 6]
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/core.hpp"
#include "cppgpt/model.hpp"
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;

// Read a flat little-endian uint16 token file. A validation split is a held-out
// fraction, not a corpus, so it is read whole; the training split is only ever
// streamed into counts and never held alongside it.
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

// Highest context length whose packed base-V key still fits a uint64. Contexts
// are packed as a base-V fold, so this is a hard representational limit, not a
// tuning choice — exceeding it would alias two different contexts onto one key
// and silently corrupt every count.
int max_order_for(int V) {
    int k = 0;
    long double cap = 1.0L;
    while (k < 16) {
        cap *= static_cast<long double>(V);
        if (cap > 9.0e18L) break;
        ++k;
    }
    return k;
}

// Interpolated n-gram counted over a training split. Level k is indexed by the k
// preceding tokens; level 0 is the context-free unigram.
class NGram {
public:
    NGram(const std::vector<std::uint16_t>& train, int V, int order) : V_(V), order_(order) {
        cnt_.resize(static_cast<std::size_t>(order) + 1);
        ctx_.resize(static_cast<std::size_t>(order) + 1);
        const auto Vz = static_cast<std::uint64_t>(V);
        for (std::size_t i = 0; i < train.size(); ++i) {
            const auto tok = static_cast<std::uint64_t>(train[i]);
            if (tok >= Vz) {
                std::fprintf(stderr, "eval: training token %llu is outside the vocabulary\n",
                             static_cast<unsigned long long>(tok));
                std::exit(1);
            }
            std::uint64_t key = 0;
            for (int k = 0; k <= order && static_cast<std::size_t>(k) <= i; ++k) {
                if (k > 0) {
                    // Extend the context leftwards one token at a time; the key
                    // for level k is the base-V fold of train[i-k .. i-1].
                    key = 0;
                    for (int j = k; j >= 1; --j)
                        key = key * Vz + static_cast<std::uint64_t>(train[i - static_cast<std::size_t>(j)]);
                }
                auto& c = cnt_[static_cast<std::size_t>(k)][key * Vz + tok];
                auto& s = ctx_[static_cast<std::size_t>(k)][key];
                if (c == 0) ++s.second;  // a continuation seen here for the first time
                ++c;
                ++s.first;
            }
        }
    }

    // P(token | the last `n` entries of ctx), interpolated down to uniform.
    // `ctx` points at the tokens immediately preceding the target, oldest first.
    [[nodiscard]] double prob(const std::uint16_t* ctx, int n, int token) const {
        const auto Vz = static_cast<std::uint64_t>(V_);
        double p = 1.0 / static_cast<double>(V_);  // the uniform floor
        const int top = std::min(n, order_);
        for (int k = 0; k <= top; ++k) {
            std::uint64_t key = 0;
            for (int j = k; j >= 1; --j) key = key * Vz + static_cast<std::uint64_t>(ctx[n - j]);
            const auto s = ctx_[static_cast<std::size_t>(k)].find(key);
            if (s == ctx_[static_cast<std::size_t>(k)].end()) break;  // unseen: keep the lower order
            const double total = s->second.first, distinct = s->second.second;
            const auto c = cnt_[static_cast<std::size_t>(k)].find(key * Vz +
                                                                 static_cast<std::uint64_t>(token));
            const double hits = (c == cnt_[static_cast<std::size_t>(k)].end())
                                    ? 0.0
                                    : static_cast<double>(c->second);
            // Witten-Bell: trust this order in proportion to how much evidence it
            // has, relative to how many distinct continuations it has seen.
            const double lambda = total / (total + distinct);
            p = lambda * (hits / total) + (1.0 - lambda) * p;
        }
        return p;
    }

    [[nodiscard]] std::size_t entries() const {
        std::size_t n = 0;
        for (const auto& m : cnt_) n += m.size();
        return n;
    }

private:
    int V_, order_;
    std::vector<std::unordered_map<std::uint64_t, std::uint32_t>> cnt_;
    // context -> (total occurrences, distinct continuations)
    std::vector<std::unordered_map<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>>> ctx_;
};

const char* order_name(int k) {
    switch (k) {
        case 0: return "unigram baseline";
        case 1: return "bigram baseline";
        case 2: return "trigram baseline";
        default: return nullptr;
    }
}

void report(const char* name, double nats, double ref) {
    std::printf("  %-22s %8.4f %10.2f %10.3f", name, nats, std::exp(nats), nats / std::log(2.0));
    if (ref > 0.0) std::printf("   %+7.1f%%", 100.0 * (nats - ref) / ref);
    std::printf("\n");
}

// One pass of the model over sequential windows. Returns mean nats/token and
// fills the per-position, accuracy and calibration accumulators when asked.
struct Pass {
    double nats = 0.0;
    double accuracy = 0.0;
    double ece = 0.0;
    std::size_t tokens = 0;
    std::vector<double> by_pos;  // mean loss at each within-window position
};

Pass run_model(GPT2& model, const std::vector<std::uint16_t>& toks, int B, int T,
               std::size_t n_batches,
               std::vector<std::pair<std::size_t, std::size_t>>* pairs_out) {
    const int V = model.config().vocab_size;
    const auto per_batch = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    std::vector<int> inputs(per_batch), targets(per_batch);
    Pass p;
    p.by_pos.assign(static_cast<std::size_t>(T), 0.0);
    std::size_t correct = 0;
    // 10 equal-width confidence bins for the calibration error.
    std::vector<double> bin_conf(10, 0.0), bin_hit(10, 0.0), bin_n(10, 0.0);

    for (std::size_t b = 0; b < n_batches; ++b) {
        for (int i = 0; i < B; ++i) {
            const std::size_t base =
                (b * static_cast<std::size_t>(B) + static_cast<std::size_t>(i)) *
                static_cast<std::size_t>(T);
            for (int t = 0; t < T; ++t) {
                const auto k = static_cast<std::size_t>(i) * static_cast<std::size_t>(T) +
                               static_cast<std::size_t>(t);
                inputs[k] = static_cast<int>(toks[base + static_cast<std::size_t>(t)]);
                targets[k] = static_cast<int>(toks[base + static_cast<std::size_t>(t) + 1]);
                if (pairs_out) pairs_out->emplace_back(base, base + static_cast<std::size_t>(t) + 1);
            }
        }
        // Next-token evaluation means the targets ARE the inputs shifted by one.
        // Stated as an invariant rather than trusted to the index arithmetic: an
        // off-by-one scores the model on a different prediction entirely, and on
        // an undertrained model the resulting loss looks entirely plausible — no
        // bound on the reported number can separate them.
        for (int i = 0; i < B; ++i) {
            const auto row = static_cast<std::size_t>(i) * static_cast<std::size_t>(T);
            for (int t = 0; t + 1 < T; ++t)
                ASSERT_MSG(targets[row + static_cast<std::size_t>(t)] ==
                               inputs[row + static_cast<std::size_t>(t) + 1],
                           "eval: targets are not the inputs shifted by one");
        }
        model.forward(inputs.data(), targets.data());

        const ActTensors& a = model.acts();
        for (std::size_t k = 0; k < per_batch; ++k) {
            const double l = a.losses[k];
            p.nats += l;
            p.by_pos[k % static_cast<std::size_t>(T)] += l;
            const float* pr = a.probs + k * static_cast<std::size_t>(V);
            int arg = 0;
            for (int v = 1; v < V; ++v)
                if (pr[v] > pr[arg]) arg = v;
            const double conf = pr[arg];
            const bool hit = (arg == targets[k]);
            correct += hit ? 1 : 0;
            const auto bin = std::min<std::size_t>(9, static_cast<std::size_t>(conf * 10.0));
            bin_conf[bin] += conf;
            bin_hit[bin] += hit ? 1.0 : 0.0;
            bin_n[bin] += 1.0;
        }
        p.tokens += per_batch;
    }

    const double n = static_cast<double>(p.tokens);
    p.nats /= n;
    p.accuracy = static_cast<double>(correct) / n;
    const double rows = static_cast<double>(n_batches) * static_cast<double>(B);
    for (auto& v : p.by_pos) v /= rows;
    for (std::size_t i = 0; i < bin_n.size(); ++i)
        if (bin_n[i] > 0)
            p.ece += (bin_n[i] / n) * std::fabs(bin_hit[i] / bin_n[i] - bin_conf[i] / bin_n[i]);
    return p;
}

std::size_t batches_in(const std::vector<std::uint16_t>& toks, int B, int T, int cap) {
    if (toks.size() <= static_cast<std::size_t>(T)) return 0;
    const std::size_t ex = (toks.size() - 1) / static_cast<std::size_t>(T);
    std::size_t nb = ex / static_cast<std::size_t>(B);
    if (cap > 0) nb = std::min(nb, static_cast<std::size_t>(cap));
    return nb;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv,
                         {"checkpoint", "data", "train", "batch", "max-batches", "order"});
    const std::string ckpt(args.str("checkpoint", ""));
    const std::string data(args.str("data", ""));
    if (ckpt.empty() || data.empty()) {
        std::fprintf(stderr,
                     "usage: eval --checkpoint <f.ckpt> --data <f.val.bin>\n"
                     "            [--train <f.train.bin>] [--batch 32] [--max-batches N]\n"
                     "            [--order 6]\n");
        return 2;
    }
    const int B = args.integer("batch", 32);
    const int max_batches = args.integer("max-batches", 0);
    const int want_order = args.integer("order", 6);
    if (B < 1) {
        std::fprintf(stderr, "eval: --batch must be >= 1\n");
        return 2;
    }
    if (want_order < 0) {
        std::fprintf(stderr, "eval: --order must be >= 0\n");
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
    const std::size_t n_batches = batches_in(val, B, T, max_batches);
    if (n_batches == 0) {
        std::fprintf(stderr,
                     "eval: '%s' holds %zu tokens — fewer than one batch of %d windows of %d\n",
                     data.c_str(), val.size(), B, T + 1);
        return 1;
    }

    GPT2 model(cfg, B, T);
    if (const auto r = model.load_checkpoint(ckpt.c_str(), GPT2::LoadMode::WeightsOnly); !r) {
        std::fprintf(stderr, "eval: cannot load '%s': %s\n", ckpt.c_str(), describe(r.error()));
        return 1;
    }

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    pairs.reserve(n_batches * static_cast<std::size_t>(B) * static_cast<std::size_t>(T));
    const Pass mp = run_model(model, val, B, T, n_batches, &pairs);

    std::printf("eval: %s on %s\n", ckpt.c_str(), data.c_str());
    std::printf("  L%d H%d C%d V%d ctx%d | %zu of %zu tokens scored (%zu batches of %d x %d)\n",
                cfg.n_layer, cfg.n_head, cfg.n_embd, V, T, mp.tokens, val.size(), n_batches, B, T);
    const std::size_t dropped = val.size() - mp.tokens;
    if (dropped > 0)
        std::printf("  %zu tokens not scored (partial trailing window/batch)\n", dropped);

    std::printf("\n  %-22s %8s %10s %10s   %8s\n", "", "nats/tok", "perplexity", "bits/char",
                "vs model");
    report("model", mp.nats, 0.0);

    const std::string train_path(args.str("train", ""));
    int best_order = -1;
    double best_nats = 0.0;
    if (!train_path.empty()) {
        const int order = std::min(want_order, max_order_for(V));
        if (order < want_order)
            std::printf("  (n-gram order capped at %d: vocab %d makes a longer packed context key"
                        " overflow 64 bits)\n",
                        order, V);
        const std::vector<std::uint16_t> train = read_tokens(train_path);
        const NGram ng(train, V, order);

        // Score every order on exactly the predictions the model was scored on,
        // with exactly the context the model had: within-window, truncated at the
        // window start.
        std::vector<double> sums(static_cast<std::size_t>(order) + 1, 0.0);
        for (const auto& [base, tgt] : pairs) {
            const auto have = static_cast<int>(tgt - base);  // context tokens available
            for (int k = 0; k <= order; ++k) {
                // The context is the `use` tokens ENDING at tgt-1, not starting at
                // the window base. Anchoring at the base reads the beginning of the
                // window as if it were the text just before the target: the counts
                // are then looked up for a context that never precedes this token,
                // so every order misses and higher orders miss harder. It showed up
                // as bigram scoring worse than uniform.
                const int use = std::min(k, have);
                sums[static_cast<std::size_t>(k)] -= std::log(ng.prob(
                    val.data() + tgt - static_cast<std::size_t>(use), use,
                    static_cast<int>(val[tgt])));
            }
        }
        const double n = static_cast<double>(pairs.size());
        for (int k = order; k >= 0; --k) {
            char buf[32];
            const char* nm = order_name(k);
            if (nm == nullptr) {
                std::snprintf(buf, sizeof(buf), "%d-gram baseline", k + 1);
                nm = buf;
            }
            const double nats = sums[static_cast<std::size_t>(k)] / n;
            report(nm, nats, mp.nats);
            if (best_order < 0 || nats < best_nats) {
                best_nats = nats;
                best_order = k;
            }
        }
        report("uniform baseline", std::log(static_cast<double>(V)), mp.nats);
        std::printf("  (%zu n-gram entries counted over %zu training tokens)\n", ng.entries(),
                    train.size());
    }

    // ---- what the scalar hides -------------------------------------------------
    std::printf("\n  top-1 accuracy %.1f%%   calibration error %.3f (0 = confidence matches"
                " reality)\n",
                100.0 * mp.accuracy, mp.ece);

    // Loss by position is the most diagnostic view: where it stops falling is how
    // much context the model is actually using, regardless of how much it is given.
    std::printf("\n  loss by context length (nats; position t predicts with t+1 chars of"
                " context):\n   ");
    const int show = std::min(T, 16);
    for (int t = 0; t < show; ++t) std::printf(" %5d", t + 1);
    std::printf("\n   ");
    for (int t = 0; t < show; ++t) std::printf(" %5.2f", mp.by_pos[static_cast<std::size_t>(t)]);
    std::printf("\n");
    if (T > show) {
        std::printf("    ... last position (%d chars of context): %.2f\n", T,
                    mp.by_pos[static_cast<std::size_t>(T - 1)]);
    }
    // Quantify "stops improving" rather than leaving it to the eye: the first
    // context length within 2% of the best any length achieves.
    const double floor_ = *std::min_element(mp.by_pos.begin(), mp.by_pos.end());
    int saturate = T;
    for (int t = 0; t < T; ++t)
        if (mp.by_pos[static_cast<std::size_t>(t)] <= floor_ * 1.02) {
            saturate = t + 1;
            break;
        }
    std::printf("    within 2%% of its best by %d characters of context (of %d available)\n",
                saturate, T);

    if (best_order >= 0) {
        std::printf("\n  %s the best n-gram (%d-gram) by %.3f nats (%.1f%%).\n",
                    mp.nats < best_nats ? "Beats" : "LOSES TO", best_order + 1,
                    std::fabs(best_nats - mp.nats), 100.0 * std::fabs(best_nats - mp.nats) / best_nats);
        if (mp.nats >= best_nats)
            std::printf("  A transformer that loses to a counting table has learned nothing the"
                        " table could not.\n");
    } else if (train_path.empty()) {
        std::printf("\n  Pass --train <f.train.bin> for the baselines that make this"
                    " interpretable.\n");
    }
    return 0;
}
