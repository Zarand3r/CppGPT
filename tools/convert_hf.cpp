// convert_hf — HuggingFace GPT-2 safetensors -> a cppgpt checkpoint.
//
// Reads `model.safetensors` directly. No Python, no pickle: safetensors is an
// 8-byte little-endian header length, then a FLAT JSON header (name -> {dtype,
// shape, data_offsets}), then raw tensor bytes. `pytorch_model.bin` is a Python
// pickle -- a stack VM whose opcodes call arbitrary callables -- and is never
// touched.
//
// Two things this must get right, both of which produce a model that RUNS and is
// WRONG if missed:
//
//   1. HF's GPT-2 uses Conv1D, which stores weights [in, out]. matmul_forward
//      needs [out, in]. Every c_attn / c_proj / c_fc is transposed here.
//   2. The file holds 160 tensors; we have 148 parameters. The 12 extra are
//      `h.{i}.attn.bias`, shape [1,1,1024,1024] -- causal MASKS, not parameters
//      (our attention masks in code). They are skipped, and the 148/160
//      reconciliation is asserted: it is the cheapest possible check that
//      everything else was mapped.
//
// Usage: convert_hf --model model.safetensors --config config.json --out gpt2.ckpt
//                   [--vocab vocab.json --merges merges.txt]
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cppgpt/bpe.hpp"
#include "cppgpt/checkpoint.hpp"
#include "cppgpt/core.hpp"
#include "cppgpt/model.hpp"
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;

struct Entry {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t begin = 0, end = 0;
};

// Targeted parser for the safetensors header: depth 2, three known fields, no
// nesting beyond an integer array. A general JSON parser would be larger and a
// wider surface on a downloaded file.
bool parse_header(std::string_view s, std::vector<std::pair<std::string, Entry>>& out) {
    std::size_t i = 0;
    const auto ws = [&] { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i; };
    const auto str = [&](std::string& d) {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) ++i;
            d += s[i++];
        }
        if (i >= s.size()) return false;
        ++i;
        return true;
    };
    const auto num = [&](std::int64_t& v) {
        ws();
        bool neg = false;
        if (i < s.size() && (s[i] == '-')) { neg = true; ++i; }
        if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
        v = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
        if (neg) v = -v;
        return true;
    };
    ws();
    if (i >= s.size() || s[i] != '{') return false;
    ++i;
    ws();
    if (i < s.size() && s[i] == '}') return true;
    while (i < s.size()) {
        ws();
        std::string name;
        if (!str(name)) return false;
        ws();
        if (i >= s.size() || s[i] != ':') return false;
        ++i;
        ws();
        if (i >= s.size()) return false;
        if (name == "__metadata__") {  // an object of strings; skip it by depth
            int depth = 0;
            do {
                if (s[i] == '{') ++depth;
                else if (s[i] == '}') --depth;
                else if (s[i] == '"') { std::string t; if (!str(t)) return false; continue; }
                ++i;
            } while (i < s.size() && depth > 0);
        } else {
            if (s[i] != '{') return false;
            ++i;
            Entry e;
            while (i < s.size()) {
                ws();
                std::string k;
                if (!str(k)) return false;
                ws();
                if (i >= s.size() || s[i] != ':') return false;
                ++i;
                ws();
                if (k == "dtype") {
                    if (!str(e.dtype)) return false;
                } else if (k == "shape" || k == "data_offsets") {
                    if (i >= s.size() || s[i] != '[') return false;
                    ++i;
                    std::vector<std::int64_t> v;
                    ws();
                    if (i < s.size() && s[i] == ']') { ++i; }
                    else {
                        while (i < s.size()) {
                            std::int64_t x = 0;
                            if (!num(x)) return false;
                            v.push_back(x);
                            ws();
                            if (i < s.size() && s[i] == ',') { ++i; continue; }
                            if (i < s.size() && s[i] == ']') { ++i; break; }
                            return false;
                        }
                    }
                    if (k == "shape") e.shape = v;
                    else if (v.size() == 2) { e.begin = static_cast<std::uint64_t>(v[0]); e.end = static_cast<std::uint64_t>(v[1]); }
                    else return false;
                } else return false;
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; break; }
                return false;
            }
            out.emplace_back(std::move(name), std::move(e));
        }
        ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') return true;
        return false;
    }
    return false;
}

// Pull one integer field out of config.json. n_head cannot be derived from any
// tensor shape, so it must be read; everything else is cross-checked against it.
bool config_int(std::string_view s, const char* key, int* out) {
    const std::string pat = std::string("\"") + key + "\"";
    const std::size_t k = s.find(pat);
    if (k == std::string_view::npos) return false;
    std::size_t i = s.find(':', k + pat.size());
    if (i == std::string_view::npos) return false;
    ++i;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n')) ++i;
    if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
    long v = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
    *out = static_cast<int>(v);
    return true;
}

std::string slurp(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv, {"model", "config", "out", "vocab", "merges"});
    const std::string model(args.str("model", ""));
    const std::string cfg_path(args.str("config", ""));
    const std::string out(args.str("out", ""));
    if (model.empty() || cfg_path.empty() || out.empty()) {
        std::fprintf(stderr,
                     "usage: convert_hf --model model.safetensors --config config.json\n"
                     "                  --out gpt2.ckpt [--vocab vocab.json --merges merges.txt]\n");
        return 2;
    }

    std::ifstream f(model, std::ios::binary);
    if (!f) { std::fprintf(stderr, "convert_hf: cannot open '%s'\n", model.c_str()); return 1; }
    std::uint64_t hlen = 0;
    if (!f.read(reinterpret_cast<char*>(&hlen), 8)) { std::fprintf(stderr, "convert_hf: truncated\n"); return 1; }
    if (hlen == 0 || hlen > (1ULL << 28)) { std::fprintf(stderr, "convert_hf: implausible header length %llu\n", static_cast<unsigned long long>(hlen)); return 1; }
    std::string hdr(static_cast<std::size_t>(hlen), '\0');
    if (!f.read(hdr.data(), static_cast<std::streamsize>(hlen))) { std::fprintf(stderr, "convert_hf: truncated header\n"); return 1; }

    std::vector<std::pair<std::string, Entry>> ents;
    if (!parse_header(hdr, ents)) { std::fprintf(stderr, "convert_hf: malformed safetensors header\n"); return 1; }
    std::printf("convert_hf: %zu tensors in header\n", ents.size());

    const std::string cfg_json = slurp(cfg_path.c_str());
    int n_head = 0, n_layer = 0, n_embd = 0, n_pos = 0, n_vocab = 0;
    if (!config_int(cfg_json, "n_head", &n_head) || !config_int(cfg_json, "n_layer", &n_layer) ||
        !config_int(cfg_json, "n_embd", &n_embd) || !config_int(cfg_json, "n_positions", &n_pos) ||
        !config_int(cfg_json, "vocab_size", &n_vocab)) {
        std::fprintf(stderr, "convert_hf: config.json missing a required field\n");
        return 1;
    }
    const Config cfg{n_pos, n_vocab, n_layer, n_head, n_embd};
    std::printf("  config: L%d H%d C%d V%d ctx%d\n", n_layer, n_head, n_embd, n_vocab, n_pos);

    const auto find = [&](const std::string& n) -> const Entry* {
        for (const auto& [k, v] : ents) if (k == n) return &v;
        return nullptr;
    };
    // Read a tensor into `dst`. `transpose` converts HF's Conv1D [in, out] to the
    // [out, in] matmul_forward requires -- the bug that yields a model which runs
    // and is wrong.
    std::vector<char> raw;
    const auto load = [&](const std::string& name, float* dst, std::size_t want, bool transpose) -> bool {
        const Entry* e = find(name);
        if (e == nullptr) { std::fprintf(stderr, "  missing tensor: %s\n", name.c_str()); return false; }
        if (e->dtype != "F32") { std::fprintf(stderr, "  %s has dtype %s, expected F32\n", name.c_str(), e->dtype.c_str()); return false; }
        const std::uint64_t bytes = e->end - e->begin;
        if (bytes != want * sizeof(float)) {
            std::fprintf(stderr, "  %s is %llu bytes, expected %zu\n", name.c_str(),
                         static_cast<unsigned long long>(bytes), want * sizeof(float));
            return false;
        }
        raw.resize(static_cast<std::size_t>(bytes));
        f.clear();
        f.seekg(static_cast<std::streamoff>(8 + hlen + e->begin));
        if (!f.read(raw.data(), static_cast<std::streamsize>(bytes))) { std::fprintf(stderr, "  short read on %s\n", name.c_str()); return false; }
        const float* src = reinterpret_cast<const float*>(raw.data());
        if (!transpose) { std::memcpy(dst, src, static_cast<std::size_t>(bytes)); return true; }
        if (e->shape.size() != 2) { std::fprintf(stderr, "  %s: transpose needs a 2-D shape\n", name.c_str()); return false; }
        const auto r = static_cast<std::size_t>(e->shape[0]), c = static_cast<std::size_t>(e->shape[1]);
        for (std::size_t a = 0; a < r; ++a)
            for (std::size_t b = 0; b < c; ++b) dst[b * r + a] = src[a * c + b];
        return true;
    };

    GPT2 m(cfg, 1, 1);  // T=1: the arenas are irrelevant here, only the params
    ParamTensors& p = m.params();
    const auto C = static_cast<std::size_t>(n_embd);
    const auto V = static_cast<std::size_t>(n_vocab);
    const auto P = static_cast<std::size_t>(n_pos);
    int mapped = 0;
    bool ok = true;
    const auto take = [&](const std::string& n, float* d, std::size_t w, bool t) { if (ok && !(ok = load(n, d, w, t))) return; ++mapped; };

    take("wte.weight", p.wte, V * C, false);
    take("wpe.weight", p.wpe, P * C, false);
    for (int l = 0; l < n_layer && ok; ++l) {
        const std::string h = "h." + std::to_string(l) + ".";
        const auto lz = static_cast<std::size_t>(l);
        take(h + "ln_1.weight", p.ln1w + lz * C, C, false);
        take(h + "ln_1.bias", p.ln1b + lz * C, C, false);
        take(h + "attn.c_attn.weight", p.qkvw + lz * 3 * C * C, 3 * C * C, true);
        take(h + "attn.c_attn.bias", p.qkvb + lz * 3 * C, 3 * C, false);
        take(h + "attn.c_proj.weight", p.attprojw + lz * C * C, C * C, true);
        take(h + "attn.c_proj.bias", p.attprojb + lz * C, C, false);
        take(h + "ln_2.weight", p.ln2w + lz * C, C, false);
        take(h + "ln_2.bias", p.ln2b + lz * C, C, false);
        take(h + "mlp.c_fc.weight", p.fcw + lz * 4 * C * C, 4 * C * C, true);
        take(h + "mlp.c_fc.bias", p.fcb + lz * 4 * C, 4 * C, false);
        take(h + "mlp.c_proj.weight", p.fcprojw + lz * C * 4 * C, 4 * C * C, true);
        take(h + "mlp.c_proj.bias", p.fcprojb + lz * C, C, false);
    }
    take("ln_f.weight", p.lnfw, C, false);
    take("ln_f.bias", p.lnfb, C, false);
    if (!ok) return 1;

    // The reconciliation: 148 mapped + one causal mask per layer == what the file
    // holds. If this fails, a tensor was missed or double-counted, and that is
    // caught here rather than by wrong numbers later.
    int masks = 0;
    for (const auto& [k, v] : ents)
        if (k.size() > 10 && k.rfind(".attn.bias") == k.size() - 10) ++masks;
    std::printf("  mapped %d parameters + %d causal masks skipped = %d (header has %zu)\n",
                mapped, masks, mapped + masks, ents.size());
    if (static_cast<std::size_t>(mapped + masks) != ents.size()) {
        std::fprintf(stderr, "convert_hf: %zu tensors unaccounted for — refusing to write\n",
                     ents.size() - static_cast<std::size_t>(mapped + masks));
        return 1;
    }

    std::uint64_t fp = 0;
    const std::string vocab(args.str("vocab", "")), merges(args.str("merges", ""));
    if (!vocab.empty() && !merges.empty()) {
        auto t = BpeTokenizer::open(vocab.c_str(), merges.c_str());
        if (!t) { std::fprintf(stderr, "convert_hf: cannot load tokenizer\n"); return 1; }
        if (t->vocab_size() != n_vocab) {
            std::fprintf(stderr, "convert_hf: tokenizer has %d tokens, model expects %d\n", t->vocab_size(), n_vocab);
            return 1;
        }
        fp = t->fingerprint();
    }
    if (const auto r = m.save_checkpoint(out.c_str(), kTokenizerGpt2Bpe, fp); !r) {
        std::fprintf(stderr, "convert_hf: save failed: %s\n", describe(r.error()));
        return 1;
    }
    std::printf("  wrote %s (%zu params, tokenizer fingerprint %016llx)\n", out.c_str(),
                m.param_count(), static_cast<unsigned long long>(fp));
    return 0;
}
