// cppgpt interpretability: read the model's intermediate state.
//
// Everything here is READ-ONLY over activations a forward pass already produced.
// No hooks are needed: the arena retains every intermediate because backward
// consumes them, so GPT2::acts() already exposes the whole forward pass.
#pragma once

#include <cstddef>

#include "cppgpt/model.hpp"

namespace cppgpt {

// Logit lens (nostalgebraist, 2020): project the residual stream at `layer`
// through the model's FINAL layernorm and its tied unembedding, answering "what
// would the model predict if it stopped after this layer?" Watching the top-1
// sharpen with depth shows where the prediction is actually made.
//
// `out_logits` receives [B*T, V] for the model's fixed B and T; the caller owns it.
// `scratch` must be at least [B*T*(C+2)] floats: the normalised residual, then
// layernorm's per-position mean and rstd. The mean/rstd live in caller scratch
// rather than the model's own lnf_mean/lnf_rstd so this really is read-only with
// respect to the activation arena — otherwise a caller inspecting acts() after a
// lens call would find those two buffers silently overwritten.
//
// Caveat worth stating where it will be read: the lens borrows the FINAL
// layernorm for every layer, but each layer's residual has its own scale, so
// early-layer readings are systematically distorted. Belrose et al.'s "tuned
// lens" fixes this with a learned per-layer probe; we do not implement it, so
// treat early layers as indicative rather than quantitative.
//
// At `layer == n_layer - 1` the lens IS the model's own final computation, so its
// output must equal acts().logits bit-identically — which is how it is tested.
void logit_lens(const GPT2& model, int layer, float* out_logits, float* scratch) noexcept;

// ---------------------------------------------------------------------------
// Direct logit attribution
// ---------------------------------------------------------------------------
//
// Every component that writes to the residual stream — the embedding, each
// attention head, each MLP — contributes additively to the final residual. The
// final layernorm is AFFINE once its mean and rstd are fixed, so freezing those
// at the values the real forward produced makes the decomposition exact:
//
//   logit[token] = embed + Σ_heads + Σ_mlps + bias
//
// That identity is the test (tests/unit/interpret_test.cpp). A threshold check
// ("contributions look plausible") would pass for a transposed unembedding, a
// mis-strided head slice, or a dropped centering term; the sum rule passes for
// none of them.
//
// Freezing the layernorm scale is the standard convention (TransformerLens), and
// it is a modelling choice rather than a free lunch: ablating a component also
// changes rstd, so attribution and ablation answer related but distinct
// questions. Attribution says "what did this component write into the output
// direction"; ablation says "what happens downstream if it stops writing".
//
// Head h of layer l reaches the residual only through columns [h·hs, (h+1)·hs)
// of attprojw[l] (out = inp @ weightᵀ, so the input channels of a head are a
// column block) — the same slice `save_and_ablate` zeroes, which is why the two
// views stay consistent.
//
// out_heads receives [n_layer · n_head] (layer-major), out_mlps [n_layer]; both
// are caller-owned, as are out_embed and out_bias. `scratch` needs [4 · n_embd]
// floats. Reads batch 0. Read-only with respect to the model.
void direct_logit_attribution(const GPT2& model, int pos, int token, float* out_heads,
                              float* out_mlps, float* out_embed, float* out_bias,
                              float* scratch) noexcept;

// ---------------------------------------------------------------------------
// Ablation
// ---------------------------------------------------------------------------
//
// What to silence. Each is implemented by zeroing the weights that carry the
// component's output into the residual stream, so no forward-pass hook is
// needed — the model runs unmodified code.
//
//   Head      — one head's column block of attprojw. Equivalent to zeroing that
//               head's channels of `atty`, an identity the unit test proves
//               directly rather than assuming.
//   Mlp       — a whole MLP block's output (fcprojw + fcprojb).
//   AttnBlock — a whole attention block's output (attprojw + attprojb).
//
// Ablating the weights rather than the activations matters: it is exact, it
// survives into every position at once, and it costs one forward pass.
enum class Ablation { Head, Mlp, AttnBlock };

// Floats `save_and_ablate` may write for ANY component of `cfg` — size one
// buffer with this and reuse it across a sweep.
[[nodiscard]] std::size_t ablation_scratch(const Config& cfg) noexcept;

// Copy the affected weights into `saved`, then zero them. `head` is ignored
// unless kind == Ablation::Head. MUTATES the model's parameters; the caller must
// pair every call with restore_ablation to put the model back. Run a forward
// pass after this to measure the effect.
void save_and_ablate(GPT2& model, Ablation kind, int layer, int head, float* saved) noexcept;

// Exact inverse of the matching save_and_ablate call: restores the parameters
// bit-for-bit. Pass the same kind/layer/head and the buffer it filled.
void restore_ablation(GPT2& model, Ablation kind, int layer, int head, const float* saved) noexcept;

}  // namespace cppgpt
