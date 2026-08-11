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

}  // namespace cppgpt
