// cppgpt forward/backward ops. Free functions over dense fp32 buffers, no
// allocation inside, caller passes output buffers. Each op carries a Device and
// dispatches on it (CPU only in v1; the CUDA phase adds kernels behind the same
// call site). Signatures mirror llm.c so the eventual HF-weight loader and
// PyTorch fixtures stay interchangeable.
//
// Layout convention (PyTorch nn.Linear / llm.c): activations are row-major
// [B, T, C]; the weight is [OC, C]; so out = inp @ weightᵀ + bias.
#pragma once

#include "cppgpt/device.hpp"

namespace cppgpt {

// out[B,T,OC] = inp[B,T,C] @ weight[OC,C]ᵀ + bias[OC].
// `bias` may be null (no bias added). Overwrites `out`. Pointers must not alias.
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC, Device dev) noexcept;

// Backward of matmul_forward. ACCUMULATES (+=) into dinp, dweight, dbias so
// gradient accumulation composes — the caller zeroes them first.
//   dinp[B,T,C]  += dout @ weight
//   dweight[OC,C] += doutᵀ @ inp
//   dbias[OC]     += sum over (B,T) of dout
// `dbias` may be null (skip the bias gradient). Other pointers are required.
void matmul_backward(float* dinp, float* dweight, float* dbias, const float* dout,
                     const float* inp, const float* weight, int B, int T, int C, int OC,
                     Device dev) noexcept;

// GELU, tanh approximation (canonical GPT-2 `gelu_new`), element-wise over N:
//   out = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
// Overwrites `out`; in-place (out == inp) is allowed.
void gelu_forward(float* out, const float* inp, int N, Device dev) noexcept;

// Backward of gelu_forward. ACCUMULATES (+=) into dinp (caller zeroes first).
void gelu_backward(float* dinp, const float* inp, const float* dout, int N, Device dev) noexcept;

// Element-wise residual add: out = a + b over N. Overwrites `out`; in-place
// (out == a or out == b) is allowed.
void residual_forward(float* out, const float* a, const float* b, int N, Device dev) noexcept;

// Backward of residual_forward. ACCUMULATES (+=) into da and db (caller zeroes):
// both receive the upstream gradient, since d(a+b)/da = d(a+b)/db = 1.
void residual_backward(float* da, float* db, const float* dout, int N, Device dev) noexcept;

// LayerNorm over the last dim C of an [B,T,C] activation (canonical GPT-2:
// affine, eps=1e-5). Per row: out = (x − mean)/√(var + eps) · weight + bias.
// Writes `out`, and `mean`/`rstd` ([B*T] each) — the per-row mean and
// 1/√(var+eps) saved for the backward pass. `out` must not alias `inp`: the
// backward reads the original `inp`, so an in-place forward would corrupt it.
void layernorm_forward(float* out, float* mean, float* rstd, const float* inp,
                       const float* weight, const float* bias, int B, int T, int C,
                       Device dev) noexcept;

// Backward of layernorm_forward. ACCUMULATES (+=) into dinp, dweight, dbias
// (caller zeroes first); `mean`/`rstd` are the buffers saved by the forward.
void layernorm_backward(float* dinp, float* dweight, float* dbias, const float* dout,
                        const float* inp, const float* weight, const float* mean,
                        const float* rstd, int B, int T, int C, Device dev) noexcept;

// Softmax over a vector of length N (N > 0): out[i] = exp(inp[i]) / Σ exp(inp),
// using the max-subtraction trick for numerical stability. Overwrites `out`.
void softmax_forward(float* out, const float* inp, int N, Device dev) noexcept;

// Backward of softmax_forward from the upstream gradient `dout` and the softmax
// output `out`. ACCUMULATES (+=) into dinp (caller zeroes):
//   dinp[i] += out[i] · (dout[i] − Σ_j dout[j]·out[j]).
void softmax_backward(float* dinp, const float* dout, const float* out, int N,
                      Device dev) noexcept;

// Causal multi-head self-attention core. `inp` is [B,T,3C] (concatenated q,k,v,
// each [B,T,C]; head h occupies columns [h·hs,(h+1)·hs), hs = C/NH). Writes `out`
// [B,T,C] and the scores `preatt`/`att` ([B,NH,T,T]; only the causal triangle
// t2≤t is written/meaningful). C must be divisible by NH.
void attention_forward(float* out, float* preatt, float* att, const float* inp, int B, int T,
                       int C, int NH, Device dev) noexcept;

// Backward of attention_forward. ACCUMULATES (+=) into dinp [B,T,3C] (caller
// zeroes). `att` is the buffer saved by the forward; `datt`/`dpreatt` ([B,NH,T,T])
// are scratch fully overwritten by this call (need not be zeroed).
void attention_backward(float* dinp, float* datt, float* dpreatt, const float* dout,
                        const float* inp, const float* att, int B, int T, int C, int NH,
                        Device dev) noexcept;

// Token + learned-position embedding: out[b,t,:] = wte[tokens[b,t],:] + wpe[t,:].
// `tokens` is [B*T] ids in [0,V); `wte` is [V,C]; `wpe` is [T,C] (or larger).
// Writes `out` [B,T,C]. An out-of-range token id aborts (fail fast).
void encoder_forward(float* out, const int* tokens, const float* wte, const float* wpe, int B,
                     int T, int C, int V, Device dev) noexcept;

// Backward of encoder_forward. ACCUMULATES (+=) into dwte [V,C] and dwpe [T,C]
// (caller zeroes), scattering dout to the looked-up token row and position row.
void encoder_backward(float* dwte, float* dwpe, const int* tokens, const float* dout, int B,
                      int T, int C, int V, Device dev) noexcept;

// Per-position cross-entropy from softmax probabilities: losses[b,t] =
// −log(probs[b,t,targets[b,t]]). `probs` is [B,T,V] (a softmax over V); `targets`
// is [B*T] in [0,V). Writes `losses` [B*T]; the mean over positions is the scalar
// training loss.
void cross_entropy_forward(float* losses, const float* probs, const int* targets, int B, int T,
                           int V, Device dev) noexcept;

// Gradient of the MEAN cross-entropy w.r.t. the logits, fusing the softmax
// backward: ACCUMULATES (+=) dlogits[b,t,v] += (probs[b,t,v] − [v==target]) / (B·T).
void cross_entropy_backward(float* dlogits, const float* probs, const int* targets, int B, int T,
                            int V, Device dev) noexcept;

// AdamW step for one parameter tensor of `n` floats, with decoupled weight decay
// (Loshchilov & Hutter; matches torch.optim.AdamW). Updates `param` in place and
// the running first/second moments `m`, `v` (both start at zero); `grad` is read,
// never modified. `t` is the 1-based step index, used for bias correction. With
// bc1 = 1/(1−β1ᵗ), bc2 = 1/(1−β2ᵗ):
//     m ← β1·m + (1−β1)·g;   v ← β2·v + (1−β2)·g²
//     param −= lr · ( (m·bc1)/(√(v·bc2) + eps) + weight_decay·param )
// Pass weight_decay = 0 for tensors that must not decay (biases, LayerNorm gains).
void adamw_update(float* param, const float* grad, float* m, float* v, int n, float lr,
                  float beta1, float beta2, float eps, float weight_decay, int t,
                  Device dev) noexcept;

}  // namespace cppgpt
