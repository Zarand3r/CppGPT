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

}  // namespace cppgpt
