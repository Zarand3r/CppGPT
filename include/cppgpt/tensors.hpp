// cppgpt tensor tables: the single source of truth for both arenas' layout.
//
// Every parameter and activation is one row here. The row carries the tensor's
// name, its per-layer element count, whether it is stacked as [L, ...], whether
// AdamW decays it, and how init_weights fills it.
//
// This replaces a five-way correspondence that was maintained BY HAND with no
// compile-time enforcement: struct field order == param_sizes index ==
// point_params index == kDecay index == the .bin order that
// scripts/gen_fixtures.py reproduces as a separate Python list. Six of the
// sixteen parameter tensors have the identical size L*C, so transposing two of
// them changed no size, no offset and no allocation — only meaning — and the
// only thing standing in the way was one whole-model numeric comparison that
// reports "wrong" without saying where.
//
// Deliberately NOT a Tensor class: no runtime shapes, no strides, no dtype, no
// dispatch. The ops keep raw float* + int dims, and src/ops.cpp is untouched, so
// the measured matmul path is byte-identical (docs/measurements.md M-1).
//
// Order is the .bin order and is load-bearing: checkpoints, the PyTorch parity
// fixture and any future weight converter all depend on it. Reordering rows
// changes the on-disk format.
#pragma once

#include <cstddef>

#include "cppgpt/model_config.hpp"

namespace cppgpt {

// How init_weights fills a parameter.
enum class Init {
    Normal,      // N(0, 0.02) — plain weights and embeddings
    NormalProj,  // N(0, 0.02/sqrt(2L)) — residual projections (GPT-2's depth scaling)
    Zero,        // biases
    One,         // LayerNorm gains
};

struct ParamSpec {
    const char* name;
    std::size_t (*elems)(const Config&);  // elements PER LAYER when per_layer
    bool per_layer;                       // stored as [L, ...]
    bool decay;                           // AdamW 2-group split: >=2D weights decay
    Init init;
};

namespace detail {
constexpr std::size_t sz(int v) { return static_cast<std::size_t>(v); }
}  // namespace detail

// THE .bin ORDER. Changing it changes the on-disk checkpoint format.
inline constexpr ParamSpec kParamSpecs[] = {
    {"wte",      [](const Config& c) { return detail::sz(c.vocab_size) * detail::sz(c.n_embd); },      false, true,  Init::Normal},
    {"wpe",      [](const Config& c) { return detail::sz(c.max_seq_len) * detail::sz(c.n_embd); },     false, true,  Init::Normal},
    {"ln1w",     [](const Config& c) { return detail::sz(c.n_embd); },                                 true,  false, Init::One},
    {"ln1b",     [](const Config& c) { return detail::sz(c.n_embd); },                                 true,  false, Init::Zero},
    {"qkvw",     [](const Config& c) { return 3 * detail::sz(c.n_embd) * detail::sz(c.n_embd); },      true,  true,  Init::Normal},
    {"qkvb",     [](const Config& c) { return 3 * detail::sz(c.n_embd); },                             true,  false, Init::Zero},
    {"attprojw", [](const Config& c) { return detail::sz(c.n_embd) * detail::sz(c.n_embd); },          true,  true,  Init::NormalProj},
    {"attprojb", [](const Config& c) { return detail::sz(c.n_embd); },                                 true,  false, Init::Zero},
    {"ln2w",     [](const Config& c) { return detail::sz(c.n_embd); },                                 true,  false, Init::One},
    {"ln2b",     [](const Config& c) { return detail::sz(c.n_embd); },                                 true,  false, Init::Zero},
    {"fcw",      [](const Config& c) { return 4 * detail::sz(c.n_embd) * detail::sz(c.n_embd); },      true,  true,  Init::Normal},
    {"fcb",      [](const Config& c) { return 4 * detail::sz(c.n_embd); },                             true,  false, Init::Zero},
    {"fcprojw",  [](const Config& c) { return detail::sz(c.n_embd) * 4 * detail::sz(c.n_embd); },      true,  true,  Init::NormalProj},
    {"fcprojb",  [](const Config& c) { return detail::sz(c.n_embd); },                                 true,  false, Init::Zero},
    {"lnfw",     [](const Config& c) { return detail::sz(c.n_embd); },                                 false, false, Init::One},
    {"lnfb",     [](const Config& c) { return detail::sz(c.n_embd); },                                 false, false, Init::Zero},
};
inline constexpr int kNumParams = static_cast<int>(sizeof(kParamSpecs) / sizeof(kParamSpecs[0]));

// Total elements for one parameter tensor, including the [L, ...] stacking.
[[nodiscard]] constexpr std::size_t param_total(const ParamSpec& s, const Config& c) {
    return s.elems(c) * (s.per_layer ? detail::sz(c.n_layer) : 1);
}

// Sum over the whole parameter arena, in .bin order.
[[nodiscard]] constexpr std::size_t params_total(const Config& c) {
    std::size_t n = 0;
    for (const ParamSpec& s : kParamSpecs) n += param_total(s, c);
    return n;
}

// --- activations -------------------------------------------------------------
// Same idea for the activation arena, which is sized for a fixed B and T. The
// per-layer flag means [L, ...] stacking exactly as for parameters.
struct ActSpec {
    const char* name;
    std::size_t (*elems)(const Config&, int B, int T);  // PER LAYER when per_layer
    bool per_layer;
};

// THE activation order. Must match ActTensors field order (static_assert'd where
// they are bound), and act_sizes' original hand-written list.
inline constexpr ActSpec kActSpecs[] = {
    {"encoded",   [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, false},
    {"ln1",       [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"ln1_mean",  [](const Config&,   int B, int T) { return detail::sz(B) * detail::sz(T); },                        true},
    {"ln1_rstd",  [](const Config&,   int B, int T) { return detail::sz(B) * detail::sz(T); },                        true},
    {"qkv",       [](const Config& c, int B, int T) { return 3 * detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"atty",      [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"preatt",    [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(c.n_head) * detail::sz(T) * detail::sz(T); }, true},
    {"att",       [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(c.n_head) * detail::sz(T) * detail::sz(T); }, true},
    {"attproj",   [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"residual2", [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"ln2",       [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"ln2_mean",  [](const Config&,   int B, int T) { return detail::sz(B) * detail::sz(T); },                        true},
    {"ln2_rstd",  [](const Config&,   int B, int T) { return detail::sz(B) * detail::sz(T); },                        true},
    {"fch",       [](const Config& c, int B, int T) { return 4 * detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"fch_gelu",  [](const Config& c, int B, int T) { return 4 * detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"fcproj",    [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"residual3", [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, true},
    {"lnf",       [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.n_embd); }, false},
    {"lnf_mean",  [](const Config&,   int B, int T) { return detail::sz(B) * detail::sz(T); },                        false},
    {"lnf_rstd",  [](const Config&,   int B, int T) { return detail::sz(B) * detail::sz(T); },                        false},
    {"logits",    [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.vocab_size); }, false},
    {"probs",     [](const Config& c, int B, int T) { return detail::sz(B) * detail::sz(T) * detail::sz(c.vocab_size); }, false},
    {"losses",    [](const Config&,   int B, int T) { return detail::sz(B) * detail::sz(T); },                        false},
};
inline constexpr int kNumActs = static_cast<int>(sizeof(kActSpecs) / sizeof(kActSpecs[0]));

[[nodiscard]] constexpr std::size_t act_total(const ActSpec& s, const Config& c, int B, int T) {
    return s.elems(c, B, T) * (s.per_layer ? detail::sz(c.n_layer) : 1);
}

[[nodiscard]] constexpr std::size_t acts_total(const Config& c, int B, int T) {
    std::size_t n = 0;
    for (const ActSpec& s : kActSpecs) n += act_total(s, c, B, T);
    return n;
}

}  // namespace cppgpt
