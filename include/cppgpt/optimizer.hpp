// cppgpt AdamW optimizer.
//
// `AdamW` bundles the training hyperparameters (canonical GPT-2 defaults) so call
// sites pass one config, not a soup of positional floats. `adamw_update` is the
// per-tensor kernel — raw pointers + dims like the other ops (the device seam) —
// so it takes the scalars directly; the model unpacks an AdamW into it and applies
// the 2-group weight-decay split.
#pragma once

#include "cppgpt/device.hpp"

namespace cppgpt {

// AdamW hyperparameters. betas/eps default to canonical GPT-2 (0.9/0.95, 1e-8);
// lr and weight_decay are set per run.
struct AdamW {
    float lr = 3e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 1e-8f;
    float weight_decay = 0.0f;
};

// One AdamW step over an n-element tensor (decoupled weight decay): updates
// `param` in place and advances the `m`/`v` moment buffers. `t` is the 1-based
// step index (bias correction). Pass weight_decay = 0 for tensors that must not
// decay (biases, LayerNorm gains).
void adamw_update(float* param, const float* grad, float* m, float* v, int n, float lr,
                  float beta1, float beta2, float eps, float weight_decay, int t,
                  Device dev = Device::CPU) noexcept;

}  // namespace cppgpt
