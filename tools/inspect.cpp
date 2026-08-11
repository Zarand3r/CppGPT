// inspect — dump one forward pass as JSON for the interpretability viewer.
//
// Runs a prompt through a trained checkpoint and writes everything the viewer
// needs: tokens, per-layer attention for selected heads, residual-stream norms,
// and the logit-lens top-k at every layer.
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
#include "cppgpt/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/tokenizer.hpp"
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;

constexpr int kSchemaVersion = 1;

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
        const int v = std::atoi(item.c_str());
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
    std::snprintf(buf, sizeof(buf), "%.5g", static_cast<double>(std::isfinite(v) ? v : 0.0f));
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
    const int kk = std::min(k, V);
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
                          "max-mb"});

    const std::string ckpt(args.str("checkpoint", ""));
    const std::string vocab_path(args.str("vocab", ""));
    const std::string prompt(args.str("prompt", ""));
    const std::string out_path(args.str("out", ""));
    if (ckpt.empty() || vocab_path.empty() || prompt.empty() || out_path.empty()) {
        std::fprintf(stderr,
                     "usage: inspect --checkpoint <f.ckpt> --vocab <f.vocab> --prompt \"text\"\n"
                     "               --out run.json [--layers 0,2] [--heads 0,1] [--top-k K]\n"
                     "               [--max-mb N]\n");
        return 2;
    }
    const int top_k = args.integer("top-k", 8);
    const double max_mb = static_cast<double>(args.real("max-mb", 64.0f));

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

    const int T = cfg.max_seq_len, C = cfg.n_embd, V = cfg.vocab_size;
    GPT2 model(cfg, /*B=*/1, T);
    if (const auto r = model.load_checkpoint(ckpt.c_str()); !r) {
        std::fprintf(stderr, "inspect: cannot load '%s': %s\n", ckpt.c_str(), describe(r.error()));
        return 1;
    }

    // Right-pad to the fixed context; causality makes the tail inert, so every
    // value we read at positions [0, n_pos) is what an unpadded forward produces.
    std::vector<int> buf(static_cast<std::size_t>(T), 0);
    std::copy(ids.begin(), ids.end(), buf.begin());
    model.forward(buf.data(), nullptr);

    const ActTensors& a = model.acts();
    std::string js;
    js.reserve(1 << 20);
    js += "{\n  \"schema\": ";
    js += std::to_string(kSchemaVersion);
    js += ",\n  \"config\": {\"n_layer\": " + std::to_string(cfg.n_layer) +
          ", \"n_head\": " + std::to_string(cfg.n_head) +
          ", \"n_embd\": " + std::to_string(cfg.n_embd) +
          ", \"vocab_size\": " + std::to_string(V) +
          ", \"max_seq_len\": " + std::to_string(T) + "},\n";
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
        const float* res = a.residual3 + static_cast<std::size_t>(l) * T * C;
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

    // logit lens: top-k at the LAST position, per layer -------------------------
    {
        std::vector<float> lens(static_cast<std::size_t>(T) * V);
        std::vector<float> scratch(static_cast<std::size_t>(T) * C + 2 * static_cast<std::size_t>(T));
        js += "  \"logit_lens\": [";
        for (int l = 0; l < cfg.n_layer; ++l) {
            if (l) js += ", ";
            logit_lens(model, l, lens.data(), scratch.data());
            const TopK t = top_k_of(lens.data() + static_cast<std::size_t>(n_pos - 1) * V, V, top_k);
            js += "{\"layer\": " + std::to_string(l) + ", \"top\": [";
            for (std::size_t i = 0; i < t.ids.size(); ++i) {
                if (i) js += ", ";
                const std::vector<int> one{t.ids[i]};
                js += "{\"id\": " + std::to_string(t.ids[i]) + ", \"text\": \"";
                json_escape(js, tok.decode(one));
                js += "\", \"p\": ";
                append_float(js, t.probs[i]);
                js += "}";
            }
            js += "]}";
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

    // attention [selected layers][selected heads][n_pos][n_pos] ------------------
    js += "  \"attention\": {\"layers\": [";
    for (std::size_t i = 0; i < layers.size(); ++i) js += (i ? ", " : "") + std::to_string(layers[i]);
    js += "], \"heads\": [";
    for (std::size_t i = 0; i < heads.size(); ++i) js += (i ? ", " : "") + std::to_string(heads[i]);
    js += "], \"data\": [";
    const auto NH = static_cast<std::size_t>(cfg.n_head);
    for (std::size_t li = 0; li < layers.size(); ++li) {
        if (li) js += ", ";
        js += "[";
        const float* att_l =
            a.att + static_cast<std::size_t>(layers[li]) * NH * static_cast<std::size_t>(T) * T;
        for (std::size_t hi = 0; hi < heads.size(); ++hi) {
            if (hi) js += ", ";
            js += "[";
            const float* att_h =
                att_l + static_cast<std::size_t>(heads[hi]) * static_cast<std::size_t>(T) * T;
            for (int q = 0; q < n_pos; ++q) {
                if (q) js += ", ";
                js += "[";
                for (int k = 0; k < n_pos; ++k) {
                    if (k) js += ", ";
                    append_float(js, att_h[static_cast<std::size_t>(q) * T + k]);
                }
                js += "]";
            }
            js += "]";
        }
        js += "]";
    }
    js += "]}\n}\n";

    std::ofstream of(out_path, std::ios::binary | std::ios::trunc);
    of << js;
    of.close();
    if (!of) {
        std::fprintf(stderr, "inspect: writing '%s' failed\n", out_path.c_str());
        return 1;
    }
    std::printf("inspect: %s — %d tokens, %zu layers x %zu heads of attention, %.2f MB\n",
                out_path.c_str(), n_pos, layers.size(), heads.size(),
                static_cast<double>(js.size()) / (1024.0 * 1024.0));
    return 0;
}
