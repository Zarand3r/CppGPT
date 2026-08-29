// inspect — dump one forward pass as JSON for the interpretability viewer.
//
// Runs a prompt through a trained checkpoint and writes everything the viewer
// needs: tokens, per-layer attention for selected heads, residual-stream norms,
// the logit-lens top-k at every layer, direct logit attribution per head, and an
// exhaustive ablation sweep.
//
// The two causal views are deliberately both present, because they disagree in
// an informative way. Attribution is what a component wrote into the output
// direction, with the final layernorm's scale frozen, and costs no extra forward
// pass. Ablation is the total downstream effect of silencing it, and costs one
// forward each. A head with small attribution but large ablation KL is acting
// through later layers rather than on the logit directly.
//
// Size is a first-class constraint, not an afterthought. Attention is
// O(L * NH * T^2): ~0.5 MB of JSON at toy scale, but ~1.2 GB at GPT-2 scale
// (L12 NH12 T1024). So --layers/--heads selection and a hard size ceiling exist
// from the first commit; exceeding the ceiling is an error naming the flags that
// would shrink it, never a file no browser can open.
//
// Usage:
//   inspect --checkpoint <f.ckpt> --vocab <f.vocab> --prompt "text" --out run.json
//           [--layers 0,2] [--heads 0,1] [--top-k K] [--max-mb N]
#include <cstddef>
#include <cstdio>
#include <charconv>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/core.hpp"
#include "cppgpt/dataloader.hpp"
#include "cppgpt/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/tokenizer.hpp"
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;

// 5 adds "residual_mid"; 4 "run_url"; 3 "pos_embed"; 2 "attribution"/"ablation".
constexpr int kSchemaVersion = 5;

std::string read_file(const std::string& path, const char* what) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "inspect: cannot open %s '%s'\n", what, path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// "0,2,5" -> {0,2,5}; empty selects everything in [0, n).
std::vector<int> parse_selection(std::string_view spec, int n, const char* what) {
    std::vector<int> out;
    if (spec.empty()) {
        out.resize(static_cast<std::size_t>(n));
        std::iota(out.begin(), out.end(), 0);
        return out;
    }
    std::string s(spec);
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        int v = 0;
        const auto* first = item.data();
        const auto res = std::from_chars(first, first + item.size(), v);
        if (res.ec != std::errc{} || res.ptr != first + item.size()) {
            std::fprintf(stderr, "inspect: --%s expects comma-separated integers, got '%s'\n",
                         what, item.c_str());
            std::exit(2);
        }
        if (v < 0 || v >= n) {
            std::fprintf(stderr, "inspect: --%s index %d out of range [0,%d)\n", what, v, n);
            std::exit(2);
        }
        out.push_back(v);
    }
    if (out.empty()) {
        std::fprintf(stderr, "inspect: --%s selected nothing\n", what);
        std::exit(2);
    }
    // A repeat is a typo, not a request. Honouring it silently dumps the same
    // matrix twice, doubles the size estimate the --max-mb ceiling is computed
    // from, and renders duplicate panels that look like two different heads.
    std::vector<int> sorted = out;
    std::sort(sorted.begin(), sorted.end());
    if (const auto dup = std::adjacent_find(sorted.begin(), sorted.end()); dup != sorted.end()) {
        std::fprintf(stderr, "inspect: --%s lists %d more than once\n", what, *dup);
        std::exit(2);
    }
    return out;
}

// JSON string escaping. The vocabulary is arbitrary bytes — control characters,
// quotes and backslashes all occur in ordinary text — so anything not printable
// ASCII goes out as \u00XX. Emitting a raw byte here would produce a file the
// viewer cannot parse, from a corpus the user considers perfectly normal.
void json_escape(std::string& out, std::string_view s) {
    static const char* hex = "0123456789abcdef";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20 || c > 0x7E) {
                    out += "\\u00";
                    out += hex[(c >> 4) & 0xF];
                    out += hex[c & 0xF];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
}

void append_float(std::string& out, float v) {
    char buf[32];
    // %.5g keeps attention weights and norms readable without bloating the file;
    // this is a visualization artifact, not a numerical one.
    // A non-finite activation means the numerics broke upstream. Emitting 0 would
    // draw a confident-looking zero in the viewer -- the silent zero-fill the
    // constitution names as a deal-breaker. Fail fast instead.
    ASSERT_MSG(std::isfinite(v), "inspect: activation is not finite");
    std::snprintf(buf, sizeof(buf), "%.5g", static_cast<double>(v));
    out += buf;
}

struct TopK {
    std::vector<int> ids;
    std::vector<float> probs;
};

// Softmax over `logits` then the k largest, descending.
TopK top_k_of(const float* logits, int V, int k) {
    std::vector<int> idx(static_cast<std::size_t>(V));
    std::iota(idx.begin(), idx.end(), 0);
    // Clamp, not min: a negative k put partial_sort's middle iterator BEFORE
    // begin, which ASan reports as a heap-buffer-overflow and the release build
    // silently reads out of bounds and exits 0.
    const int kk = std::clamp(k, 0, V);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    const float mx = *std::max_element(logits, logits + V);
    double sum = 0.0;
    for (int i = 0; i < V; ++i) sum += std::exp(static_cast<double>(logits[i] - mx));
    TopK t;
    for (int i = 0; i < kk; ++i) {
        t.ids.push_back(idx[static_cast<std::size_t>(i)]);
        t.probs.push_back(
            static_cast<float>(std::exp(static_cast<double>(logits[idx[static_cast<std::size_t>(i)]] - mx)) / sum));
    }
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv,
                         {"checkpoint", "vocab", "prompt", "out", "layers", "heads", "top-k",
                          "max-mb", "ablate", "run-url"});

    const std::string ckpt(args.str("checkpoint", ""));
    const std::string vocab_path(args.str("vocab", ""));
    const std::string prompt(args.str("prompt", ""));
    const std::string out_path(args.str("out", ""));
    if (ckpt.empty() || vocab_path.empty() || prompt.empty() || out_path.empty()) {
        std::fprintf(stderr,
                     "usage: inspect --checkpoint <f.ckpt> --vocab <f.vocab> --prompt \"text\"\n"
                     "               --out run.json [--layers 0,2] [--heads 0,1] [--top-k K]\n"
                     "               [--max-mb N] [--ablate 0|1] [--run-url URL]\n");
        return 2;
    }
    const int top_k = args.integer("top-k", 8);
    // top_k_of clamps to [0,V] so a negative value cannot walk off the buffer
    // (it once did, ASan-confirmed). But clamping a nonsensical request and
    // reporting success is the silent no-op cli.hpp exists to prevent: --top-k 0
    // produced an empty prediction list in every panel, which reads as "the model
    // predicted nothing" rather than "you asked for nothing".
    if (top_k < 1) {
        std::fprintf(stderr, "inspect: --top-k must be >= 1 (got %d)\n", top_k);
        return 2;
    }
    const double max_mb = static_cast<double>(args.real("max-mb", 64.0f));
    // The sweep costs L·(NH+2) forward passes: milliseconds at toy scale, and the
    // whole point of the tool. --ablate 0 turns it off for a model where it isn't.
    const bool do_ablate = args.integer("ablate", 1) != 0;
    // Provenance travels WITH the dump rather than being hardcoded in the page:
    // a URL baked into viewer.html silently goes stale the moment a different
    // checkpoint is served, and a wrong provenance link is worse than none.
    const std::string run_url(args.str("run-url", ""));

    // Architecture comes from the checkpoint, never from flags.
    auto peek = CheckpointFile::open(ckpt.c_str());
    if (!peek) {
        std::fprintf(stderr, "inspect: cannot read checkpoint '%s': %s\n", ckpt.c_str(),
                     describe(peek.error()));
        return 1;
    }
    const CheckpointHeader& h = peek->header();
    Config cfg{h.max_seq_len, h.vocab_size, h.n_layer, h.n_head, h.n_embd};

    const std::string vocab = read_file(vocab_path, "vocab");
    const CharTokenizer tok(vocab);
    if (tok.vocab_size() != cfg.vocab_size) {
        std::fprintf(stderr, "inspect: vocab has %d symbols, checkpoint expects %d\n",
                     tok.vocab_size(), cfg.vocab_size);
        return 1;
    }

    const std::vector<int> ids = tok.encode(prompt);
    if (ids.empty() || static_cast<int>(ids.size()) > cfg.max_seq_len) {
        std::fprintf(stderr, "inspect: prompt is %zu tokens; must be 1..%d\n", ids.size(),
                     cfg.max_seq_len);
        return 1;
    }
    const auto n_pos = static_cast<int>(ids.size());

    const std::vector<int> layers = parse_selection(args.str("layers", ""), cfg.n_layer, "layers");
    const std::vector<int> heads = parse_selection(args.str("heads", ""), cfg.n_head, "heads");

    // Refuse an unopenable file rather than writing one. ~8 chars per float of
    // JSON is the empirical rate for %.5g values with separators.
    const double att_floats = static_cast<double>(layers.size()) * static_cast<double>(heads.size()) *
                              static_cast<double>(n_pos) * static_cast<double>(n_pos);
    const double est_mb = att_floats * 8.0 / (1024.0 * 1024.0);
    if (est_mb > max_mb) {
        std::fprintf(stderr,
                     "inspect: attention alone would be ~%.0f MB (%zu layers x %zu heads x %d^2),\n"
                     "  over the --max-mb %.0f ceiling. Narrow it with --layers / --heads, use a\n"
                     "  shorter --prompt, or raise --max-mb if you really want the whole thing.\n",
                     est_mb, layers.size(), heads.size(), n_pos, max_mb);
        return 1;
    }

    // Run at exactly the prompt length rather than the padded context. Attention
    // is O(T²) and the lens is O(T·C·V), so on a short prompt against a 64-token
    // context this is the largest saving available here. Causality makes the
    // padded tail inert, so the short run reproduces the padded one EXACTLY at
    // every position both contain — pinned by a unit test (interpret_test), not
    // merely asserted in a comment.
    const int T = n_pos, C = cfg.n_embd, V = cfg.vocab_size;
    GPT2 model(cfg, /*B=*/1, T);
    if (const auto r = model.load_checkpoint(ckpt.c_str()); !r) {
        std::fprintf(stderr, "inspect: cannot load '%s': %s\n", ckpt.c_str(), describe(r.error()));
        return 1;
    }

    const std::vector<int> buf(ids.begin(), ids.end());
    model.forward(buf.data(), nullptr);

    const ActTensors& a = model.acts();
    const int B_dim = model.batch();
    std::string js;
    js.reserve(1 << 20);
    js += "{\n  \"schema\": ";
    js += std::to_string(kSchemaVersion);
    js += ",\n  \"config\": {\"n_layer\": " + std::to_string(cfg.n_layer) +
          ", \"n_head\": " + std::to_string(cfg.n_head) +
          ", \"n_embd\": " + std::to_string(cfg.n_embd) +
          ", \"vocab_size\": " + std::to_string(V) +
          ", \"max_seq_len\": " + std::to_string(cfg.max_seq_len) + "},\n";
    js += "  \"run_url\": \"";
    json_escape(js, run_url);
    js += "\",\n";
    js += "  \"prompt\": \"";
    json_escape(js, prompt);
    js += "\",\n  \"n_positions\": " + std::to_string(n_pos) + ",\n";

    // tokens ------------------------------------------------------------------
    js += "  \"tokens\": [";
    for (int i = 0; i < n_pos; ++i) {
        if (i) js += ", ";
        const std::vector<int> one{ids[static_cast<std::size_t>(i)]};
        js += "{\"id\": " + std::to_string(ids[static_cast<std::size_t>(i)]) + ", \"text\": \"";
        json_escape(js, tok.decode(one));
        js += "\"}";
    }
    js += "],\n";

    // residual-stream norm per layer per position ------------------------------
    js += "  \"residual_norms\": [";
    for (int l = 0; l < cfg.n_layer; ++l) {
        if (l) js += ", ";
        js += "[";
        // [L, B, T, C] — the B factor is required. It was missing here, correct
        // only because this tool happens to build with B == 1; a third
        // re-derivation of a stride the model already knows.
        const float* res = layer_slice(a.residual3, l, B_dim, T, C);
        for (int t = 0; t < n_pos; ++t) {
            if (t) js += ", ";
            double ss = 0.0;
            for (int c = 0; c < C; ++c) {
                const double x = res[static_cast<std::size_t>(t) * C + c];
                ss += x * x;
            }
            append_float(js, static_cast<float>(std::sqrt(ss)));
        }
        js += "]";
    }
    js += "],\n";

    // A transformer block has TWO residual adds, not one: attention writes back,
    // then the MLP writes back. residual3 is only the second, so a diagram drawn
    // from it alone silently halves the structure. residual2 is the stream after
    // the attention sublayer.
    js += "  \"residual_mid\": [";
    for (int l = 0; l < cfg.n_layer; ++l) {
        if (l) js += ", ";
        js += "[";
        // [L, B, T, C] — the B factor is required. It was missing here, correct
        // only because this tool happens to build with B == 1; a third
        // re-derivation of a stride the model already knows.
        const float* res = layer_slice(a.residual2, l, B_dim, T, C);
        for (int t = 0; t < n_pos; ++t) {
            if (t) js += ", ";
            double ss = 0.0;
            for (int c = 0; c < C; ++c) {
                const double x = res[static_cast<std::size_t>(t) * C + c];
                ss += x * x;
            }
            append_float(js, static_cast<float>(std::sqrt(ss)));
        }
        js += "]";
    }
    js += "],\n";

    // Softmax / KL over a distribution at one position. Shared by the lens series
    // and the ablation sweep, which must agree on both or their numbers are not
    // comparable.
    const auto softmax_of = [V](const float* logits) {
        std::vector<double> p(static_cast<std::size_t>(V));
        const float mx = *std::max_element(logits, logits + V);
        double sum = 0.0;
        for (int i = 0; i < V; ++i) {
            p[static_cast<std::size_t>(i)] = std::exp(static_cast<double>(logits[i] - mx));
            sum += p[static_cast<std::size_t>(i)];
        }
        for (auto& x : p) x /= sum;
        return p;
    };
    // KL(a || b), guarded: b_i is never 0 after softmax, but clamp anyway so a
    // denormal cannot produce inf and silently poison the whole series.
    const auto kl = [V](const std::vector<double>& a_, const std::vector<double>& b_) {
        double d = 0.0;
        for (int i = 0; i < V; ++i) {
            const double ai = a_[static_cast<std::size_t>(i)];
            if (ai > 1e-12)
                d += ai * std::log(ai / std::max(b_[static_cast<std::size_t>(i)], 1e-30));
        }
        return d;
    };
    const auto last_off = static_cast<std::size_t>(n_pos - 1) * static_cast<std::size_t>(V);

    // Three views of the logit lens — the per-position top-1 grid, the last
    // position's top-k, and the per-layer KL series — from ONE lens pass per
    // layer. Each view used to recompute the same [T,V] projection for itself,
    // so a 4-layer model ran 12 lens passes to produce 4 distinct results.
    {
        std::vector<float> lens(static_cast<std::size_t>(T) * V);
        std::vector<float> scratch(static_cast<std::size_t>(T) * C + 2 * static_cast<std::size_t>(T));
        std::string js_grid, js_lens;
        std::vector<std::vector<double>> per_layer;
        per_layer.reserve(static_cast<std::size_t>(cfg.n_layer));

        for (int l = 0; l < cfg.n_layer; ++l) {
            logit_lens(model, l, lens.data(), scratch.data());

            // Per-position top-1: the "watch the sequence transform" view,
            // bounded to top-1 so it stays small even at large T.
            if (l) js_grid += ", ";
            js_grid += "[";
            for (int t = 0; t < n_pos; ++t) {
                if (t) js_grid += ", ";
                const TopK k1 = top_k_of(lens.data() + static_cast<std::size_t>(t) * V, V, 1);
                const std::vector<int> one{k1.ids[0]};
                js_grid += "{\"id\": " + std::to_string(k1.ids[0]) + ", \"text\": \"";
                json_escape(js_grid, tok.decode(one));
                js_grid += "\", \"p\": ";
                append_float(js_grid, k1.probs[0]);
                js_grid += "}";
            }
            js_grid += "]";

            if (l) js_lens += ", ";
            const TopK tk = top_k_of(lens.data() + last_off, V, top_k);
            js_lens += "{\"layer\": " + std::to_string(l) + ", \"top\": [";
            for (std::size_t i = 0; i < tk.ids.size(); ++i) {
                if (i) js_lens += ", ";
                const std::vector<int> one{tk.ids[i]};
                js_lens += "{\"id\": " + std::to_string(tk.ids[i]) + ", \"text\": \"";
                json_escape(js_lens, tok.decode(one));
                js_lens += "\", \"p\": ";
                append_float(js_lens, tk.probs[i]);
                js_lens += "}";
            }
            js_lens += "]}";

            per_layer.push_back(softmax_of(lens.data() + last_off));
        }

        js += "  \"lens_grid\": [" + js_grid + "],\n";
        js += "  \"logit_lens\": [" + js_lens + "],\n";

        // Per-layer KL: how much each layer moved the prediction, in nats.
        //   step[l]     = KL(P_l || P_{l-1})  -- divergence this layer introduced
        //   to_final[l] = KL(P_out || P_l)    -- how far this layer still is from the output
        // The residual norm plotted elsewhere is only a proxy for "did this layer
        // do work"; this is the quantity itself, and a near-zero step[l]
        // identifies a layer that is near-identity for this input.
        const std::vector<double> p_out = softmax_of(a.logits + last_off);
        js += "  \"layer_kl\": [";
        for (int l = 0; l < cfg.n_layer; ++l) {
            if (l) js += ", ";
            const double step = (l == 0) ? 0.0
                                         : kl(per_layer[static_cast<std::size_t>(l)],
                                              per_layer[static_cast<std::size_t>(l - 1)]);
            js += "{\"layer\": " + std::to_string(l) + ", \"step\": ";
            append_float(js, static_cast<float>(step));
            js += ", \"to_final\": ";
            append_float(js, static_cast<float>(kl(p_out, per_layer[static_cast<std::size_t>(l)])));
            js += "}";
        }
        js += "],\n";
    }

    // final distribution at the last position ----------------------------------
    {
        const TopK t = top_k_of(a.logits + static_cast<std::size_t>(n_pos - 1) * V, V, top_k);
        js += "  \"final_top\": [";
        for (std::size_t i = 0; i < t.ids.size(); ++i) {
            if (i) js += ", ";
            const std::vector<int> one{t.ids[i]};
            js += "{\"id\": " + std::to_string(t.ids[i]) + ", \"text\": \"";
            json_escape(js, tok.decode(one));
            js += "\", \"p\": ";
            append_float(js, t.probs[i]);
            js += "}";
        }
        js += "],\n";
    }

    // Per-head summary: mean entropy (how focused) and mean attention distance
    // (q-k weighted by attention; ~1 marks a previous-token head, ~q marks a head
    // that stares at position 0 — the "attention sink"). Cheap, and it turns a
    // wall of heatmaps into something you can rank.
    js += "  \"head_stats\": [";
    {
        bool first = true;
        for (int l = 0; l < cfg.n_layer; ++l) {
            for (int h = 0; h < cfg.n_head; ++h) {
                const float* ah = head_slice(a.att, l, h, B_dim, cfg.n_head, T);
                double ent = 0.0, dist = 0.0, sink = 0.0;
                for (int q = 0; q < n_pos; ++q) {
                    const float* row = ah + static_cast<std::size_t>(q) * T;
                    for (int k = 0; k <= q; ++k) {
                        const double w = row[k];
                        if (w > 1e-9) ent += -w * std::log(w);
                        dist += w * static_cast<double>(q - k);
                    }
                    sink += row[0];
                }
                const double n = n_pos;
                if (!first) js += ", ";
                first = false;
                js += "{\"layer\": " + std::to_string(l) + ", \"head\": " + std::to_string(h) +
                      ", \"entropy\": ";
                append_float(js, static_cast<float>(ent / n));
                js += ", \"distance\": ";
                append_float(js, static_cast<float>(dist / n));
                js += ", \"to_first\": ";
                append_float(js, static_cast<float>(sink / n));
                js += "}";
            }
        }
    }
    js += "],\n";

    // attention [selected layers][selected heads][n_pos][n_pos] ------------------
    js += "  \"attention\": {\"layers\": [";
    for (std::size_t i = 0; i < layers.size(); ++i) js += (i ? ", " : "") + std::to_string(layers[i]);
    js += "], \"heads\": [";
    for (std::size_t i = 0; i < heads.size(); ++i) js += (i ? ", " : "") + std::to_string(heads[i]);
    js += "], \"data\": [";
    for (std::size_t li = 0; li < layers.size(); ++li) {
        if (li) js += ", ";
        js += "[";
        const float* att_l = head_slice(a.att, layers[li], 0, B_dim, cfg.n_head, T);
        for (std::size_t hi = 0; hi < heads.size(); ++hi) {
            if (hi) js += ", ";
            js += "[";
            const float* att_h = att_l + static_cast<std::size_t>(heads[hi]) *
                                            static_cast<std::size_t>(T) * static_cast<std::size_t>(T);
            for (int q = 0; q < n_pos; ++q) {
                if (q) js += ", ";
                js += "[";
                for (int k = 0; k < n_pos; ++k) {
                    if (k) js += ", ";
                    // Emit 0 above the diagonal rather than reading it:
                    // attention_forward only writes t2 <= t, and the activation
                    // arena is never zeroed, so those cells are uninitialised
                    // memory -- which we were drawing as attention weights.
                    append_float(js, k <= q ? att_h[static_cast<std::size_t>(q) * T + k] : 0.0f);
                }
                js += "]";
            }
            js += "]";
        }
        js += "]";
    }
    js += "]},\n";

    // Learned positional embeddings. Canonical GPT-2 learns `wpe` rather than
    // using fixed sinusoids, so unlike the sinusoidal case there is something to
    // look at: these values were shaped by training and can be inspected for what
    // the model discovered about position.
    //
    // Cosine similarity between position rows is the informative view. A model
    // that learned locality shows a bright band along the diagonal (nearby
    // positions encoded similarly); an undertrained one still looks like the
    // N(0, 0.02) it was initialised to, where distinct rows are near-orthogonal
    // and the matrix is dark everywhere off the diagonal. The norms say how far
    // each position has moved from initialisation at all.
    {
        const float* wpe = model.params().wpe;
        std::vector<double> nrm(static_cast<std::size_t>(n_pos));
        js += "  \"pos_embed\": {\"norms\": [";
        for (int t = 0; t < n_pos; ++t) {
            const float* r = wpe + static_cast<std::size_t>(t) * C;
            double ss = 0.0;
            for (int c = 0; c < C; ++c) ss += static_cast<double>(r[c]) * static_cast<double>(r[c]);
            nrm[static_cast<std::size_t>(t)] = std::sqrt(ss);
            if (t) js += ", ";
            append_float(js, static_cast<float>(nrm[static_cast<std::size_t>(t)]));
        }
        js += "], \"similarity\": [";
        for (int i = 0; i < n_pos; ++i) {
            if (i) js += ", ";
            js += "[";
            const float* ri = wpe + static_cast<std::size_t>(i) * C;
            for (int j = 0; j < n_pos; ++j) {
                if (j) js += ", ";
                const float* rj = wpe + static_cast<std::size_t>(j) * C;
                double d = 0.0;
                for (int c = 0; c < C; ++c)
                    d += static_cast<double>(ri[c]) * static_cast<double>(rj[c]);
                // A zero row would be a division by zero; it also means that
                // position never trained, which is worth showing as 0 similarity
                // rather than as a NaN that append_float would abort on.
                const double den =
                    nrm[static_cast<std::size_t>(i)] * nrm[static_cast<std::size_t>(j)];
                append_float(js, static_cast<float>(den > 1e-30 ? d / den : 0.0));
            }
            js += "]";
        }
        js += "]},\n";
    }

    // Direct logit attribution for the model's own top-1 prediction: how much
    // each head, each MLP, the embedding and the biases pushed that token's
    // logit. This costs NO extra forward pass — it reads activations the forward
    // already produced — and the parts sum to the logit exactly.
    const int dla_token = static_cast<int>(
        std::max_element(a.logits + last_off, a.logits + last_off + V) - (a.logits + last_off));
    {
        const auto Lz = static_cast<std::size_t>(cfg.n_layer);
        const auto NHz = static_cast<std::size_t>(cfg.n_head);
        std::vector<float> heads(Lz * NHz), mlps(Lz);
        std::vector<float> dscratch(4 * static_cast<std::size_t>(C));
        float embed = 0.0f, bias = 0.0f;
        direct_logit_attribution(model, n_pos - 1, dla_token, heads.data(), mlps.data(), &embed,
                                 &bias, dscratch.data());

        const std::vector<int> one{dla_token};
        js += "  \"attribution\": {\"token\": {\"id\": " + std::to_string(dla_token) +
              ", \"text\": \"";
        json_escape(js, tok.decode(one));
        js += "\"}, \"embed\": ";
        append_float(js, embed);
        js += ", \"bias\": ";
        append_float(js, bias);
        js += ", \"heads\": [";
        for (std::size_t l = 0; l < Lz; ++l) {
            if (l) js += ", ";
            js += "[";
            for (std::size_t h = 0; h < NHz; ++h) {
                if (h) js += ", ";
                append_float(js, heads[l * NHz + h]);
            }
            js += "]";
        }
        js += "], \"mlps\": [";
        for (std::size_t l = 0; l < Lz; ++l) {
            if (l) js += ", ";
            append_float(js, mlps[l]);
        }
        js += "]},\n";
    }

    // Ablation sweep: silence one component, re-run, and measure how far the
    // output distribution moved (KL from baseline, in nats) plus what happened to
    // the baseline's top-1 probability. Every head, every MLP and every attention
    // block — an exhaustive causal map, infeasible at production scale and only
    // L·(NH+2) forward passes here.
    //
    // Attribution and ablation answer different questions and will disagree:
    // attribution is the direct write into the output direction with the
    // layernorm scale frozen, ablation is the total downstream effect including
    // everything the later layers do differently. A head with small attribution
    // and large ablation KL acts through later layers, not on the logit.
    //
    // This CLOBBERS the activation arena, so it must follow every baseline read.
    js += "  \"ablation\": [";
    if (do_ablate) {
        const std::vector<double> p_base = softmax_of(a.logits + last_off);
        const auto top_base = static_cast<std::size_t>(dla_token);
        std::vector<float> saved(ablation_scratch(cfg));
        bool first = true;
        const auto sweep = [&](Ablation kind, const char* name, int layer, int head) {
            save_and_ablate(model, kind, layer, head, saved.data());
            model.forward(buf.data(), nullptr);
            const std::vector<double> p = softmax_of(a.logits + last_off);
            restore_ablation(model, kind, layer, head, saved.data());

            if (!first) js += ", ";
            first = false;
            js += "{\"kind\": \"";
            js += name;
            js += "\", \"layer\": " + std::to_string(layer) +
                  ", \"head\": " + std::to_string(head) + ", \"kl\": ";
            append_float(js, static_cast<float>(kl(p_base, p)));
            js += ", \"dtop\": ";
            append_float(js, static_cast<float>(p[top_base] - p_base[top_base]));
            js += "}";
        };
        for (int l = 0; l < cfg.n_layer; ++l) {
            for (int h = 0; h < cfg.n_head; ++h) sweep(Ablation::Head, "head", l, h);
            sweep(Ablation::Mlp, "mlp", l, -1);
            sweep(Ablation::AttnBlock, "attn", l, -1);
        }
    }
    js += "]\n}\n";

    // run.json is consumed by serve_viewer.py and make-site.sh, so it gets the
    // same atomic write as every other cross-tool file (L5/L15). ofstream(trunc)
    // here was a recurrence of the signature those lessons already swept.
    if (const auto r = write_file_atomic(out_path.c_str(), js); !r) {
        std::fprintf(stderr, "inspect: writing '%s' failed: %s\n", out_path.c_str(),
                     describe(r.error()));
        return 1;
    }
    std::printf("inspect: %s — %d tokens, %zu layers x %zu heads of attention, %.2f MB\n",
                out_path.c_str(), n_pos, layers.size(), heads.size(),
                static_cast<double>(js.size()) / (1024.0 * 1024.0));
    return 0;
}
