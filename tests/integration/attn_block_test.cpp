// Integration: the GPT-2 attention sub-block, forward + backward end to end.
//
//   out = x + c_proj(attention(c_attn(layernorm(x))))
//
// Composes layernorm + matmul (c_attn, C->3C) + causal attention + matmul
// (c_proj, C->C) + residual. Like the MLP-block test, it verifies the ops wire
// together and the backward chain composes in reverse — including dx accumulating
// from both the residual and the layernorm paths — via a finite-difference
// gradient check of dx and an inner weight (dW_attn). Buffers come from one
// Storage arena, the model's pattern.
#include <cstddef>

#include "cppgpt/ops.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/storage.hpp"
#include "tests/check.hpp"
#include "tests/numeric.hpp"

using namespace cppgpt;
using cppgpt::test::dot;
using cppgpt::test::grad_check;

int main() {
    const int B = 2, T = 3, C = 4, NH = 2;
    const int bt = B * T;
    const std::size_t n_x = static_cast<std::size_t>(bt * C);
    const std::size_t n_qkv = static_cast<std::size_t>(bt * 3 * C);
    const std::size_t n_att = static_cast<std::size_t>(B * NH * T * T);
    const std::size_t n_wattn = static_cast<std::size_t>(3 * C * C);
    const std::size_t n_wproj = static_cast<std::size_t>(C * C);
    const std::size_t cz = static_cast<std::size_t>(C);
    const std::size_t btz = static_cast<std::size_t>(bt);

    Storage arena(4096);
    Generator gen(0xA77B10CULL);
    auto rnd = [&](std::size_t n) {
        float* p = arena.alloc(n);
        for (std::size_t i = 0; i < n; ++i) p[i] = gen.normal();
        return p;
    };
    auto zeros = [&](std::size_t n) { return arena.alloc_zeroed(n); };

    // Parameters + input.
    float* x = rnd(n_x);
    float* w_ln = rnd(cz);
    float* b_ln = rnd(cz);
    float* w_attn = rnd(n_wattn);
    float* b_attn = rnd(static_cast<std::size_t>(3 * C));
    float* w_proj = rnd(n_wproj);
    float* b_proj = rnd(cz);

    // Forward activations (scratch).
    float* ln = arena.alloc(n_x);
    float* ln_mean = arena.alloc(btz);
    float* ln_rstd = arena.alloc(btz);
    float* qkv = arena.alloc(n_qkv);
    float* preatt = arena.alloc(n_att);
    float* att = arena.alloc(n_att);
    float* att_out = arena.alloc(n_x);
    float* proj = arena.alloc(n_x);
    float* out = arena.alloc(n_x);

    auto forward = [&]() {
        layernorm_forward(ln, ln_mean, ln_rstd, x, w_ln, b_ln, B, T, C, Device::CPU);
        matmul_forward(qkv, ln, w_attn, b_attn, B, T, C, 3 * C, Device::CPU);
        attention_forward(att_out, preatt, att, qkv, B, T, C, NH, Device::CPU);
        matmul_forward(proj, att_out, w_proj, b_proj, B, T, C, C, Device::CPU);
        residual_forward(out, x, proj, bt * C, Device::CPU);
    };

    forward();
    float* dout = rnd(n_x);
    auto loss = [&]() {
        forward();
        return dot(dout, out, n_x);
    };

    // Backward chain (reverse) into zeroed gradient buffers; datt/dpreatt scratch.
    float* dx = zeros(n_x);
    float* dproj = zeros(n_x);
    float* datt_out = zeros(n_x);
    float* dqkv = zeros(n_qkv);
    float* dln = zeros(n_x);
    float* dw_ln = zeros(cz);
    float* db_ln = zeros(cz);
    float* dw_attn = zeros(n_wattn);
    float* db_attn = zeros(static_cast<std::size_t>(3 * C));
    float* dw_proj = zeros(n_wproj);
    float* db_proj = zeros(cz);
    float* datt = arena.alloc(n_att);
    float* dpreatt = arena.alloc(n_att);

    residual_backward(dx, dproj, dout, bt * C, Device::CPU);
    matmul_backward(datt_out, dw_proj, db_proj, dproj, att_out, w_proj, B, T, C, C, Device::CPU);
    attention_backward(dqkv, datt, dpreatt, datt_out, qkv, att, B, T, C, NH, Device::CPU);
    matmul_backward(dln, dw_attn, db_attn, dqkv, ln, w_attn, B, T, C, 3 * C, Device::CPU);
    layernorm_backward(dx, dw_ln, db_ln, dln, x, w_ln, ln_mean, ln_rstd, B, T, C, Device::CPU);

    // dx flows through the whole block (incl. residual + layernorm accumulation).
    CHECK(grad_check(loss, x, dx, n_x) < 2e-2);
    // dW_attn exercises the inner-weight gradient path through attention + proj.
    CHECK(grad_check(loss, w_attn, dw_attn, n_wattn) < 2e-2);

    return cppgpt::test::summary();
}
