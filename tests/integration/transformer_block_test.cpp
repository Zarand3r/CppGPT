// Integration: a full GPT-2 transformer block, forward + backward end to end.
//
//   a   = x + attn_proj(attention(c_attn(ln_1(x))))     // attention sub-block
//   out = a + mlp_proj(gelu(c_fc(ln_2(a))))             // MLP sub-block
//
// This is the unit the model repeats n_layer times. It composes every op
// (layernorm ×2, matmul ×4, attention, gelu, residual ×2) and exercises the part
// unit tests can't: that the two sub-blocks chain (the attention output `a` feeds
// the MLP sub-block, and the backward flows back through both), with `da`
// accumulating from the MLP residual + ln_2 and `dx` from the attention residual
// + ln_1. Pinned by a finite-difference gradient check of dx and of one weight in
// each sub-block. Buffers come from one Storage arena, the model's pattern.
//
// Supersedes the per-sub-block mlp_block / attn_block tests (which it contains).
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
    const int B = 2, T = 3, C = 4, NH = 2, H = 4 * C;
    const int bt = B * T;
    const std::size_t n_x = static_cast<std::size_t>(bt * C);
    const std::size_t n_qkv = static_cast<std::size_t>(bt * 3 * C);
    const std::size_t n_h = static_cast<std::size_t>(bt * H);
    const std::size_t n_att = static_cast<std::size_t>(B * NH * T * T);
    const std::size_t n_attnw = static_cast<std::size_t>(3 * C * C);
    const std::size_t n_aprojw = static_cast<std::size_t>(C * C);
    const std::size_t n_fcw = static_cast<std::size_t>(H * C);
    const std::size_t n_mprojw = static_cast<std::size_t>(C * H);
    const std::size_t cz = static_cast<std::size_t>(C);
    const std::size_t hz = static_cast<std::size_t>(H);
    const std::size_t c3 = static_cast<std::size_t>(3 * C);
    const std::size_t btz = static_cast<std::size_t>(bt);

    Storage arena(16384);
    Generator gen(0x7B10C7ULL);
    auto rnd = [&](std::size_t n) {
        float* p = arena.alloc(n);
        for (std::size_t i = 0; i < n; ++i) p[i] = gen.normal();
        return p;
    };
    auto zeros = [&](std::size_t n) { return arena.alloc_zeroed(n); };

    // Parameters + input.
    float* x = rnd(n_x);
    float* ln1_w = rnd(cz);
    float* ln1_b = rnd(cz);
    float* attn_w = rnd(n_attnw);
    float* attn_b = rnd(c3);
    float* aproj_w = rnd(n_aprojw);
    float* aproj_b = rnd(cz);
    float* ln2_w = rnd(cz);
    float* ln2_b = rnd(cz);
    float* fc_w = rnd(n_fcw);
    float* fc_b = rnd(hz);
    float* mproj_w = rnd(n_mprojw);
    float* mproj_b = rnd(cz);

    // Forward activations (scratch).
    float* ln1 = arena.alloc(n_x);
    float* ln1_mean = arena.alloc(btz);
    float* ln1_rstd = arena.alloc(btz);
    float* qkv = arena.alloc(n_qkv);
    float* preatt = arena.alloc(n_att);
    float* att = arena.alloc(n_att);
    float* attn_out = arena.alloc(n_x);
    float* aproj = arena.alloc(n_x);
    float* a = arena.alloc(n_x);
    float* ln2 = arena.alloc(n_x);
    float* ln2_mean = arena.alloc(btz);
    float* ln2_rstd = arena.alloc(btz);
    float* fc = arena.alloc(n_h);
    float* g = arena.alloc(n_h);
    float* mproj = arena.alloc(n_x);
    float* out = arena.alloc(n_x);

    auto forward = [&]() {
        layernorm_forward(ln1, ln1_mean, ln1_rstd, x, ln1_w, ln1_b, B, T, C);
        matmul_forward(qkv, ln1, attn_w, attn_b, B, T, C, 3 * C);
        attention_forward(attn_out, preatt, att, qkv, B, T, C, NH);
        matmul_forward(aproj, attn_out, aproj_w, aproj_b, B, T, C, C);
        residual_forward(a, x, aproj, bt * C);
        layernorm_forward(ln2, ln2_mean, ln2_rstd, a, ln2_w, ln2_b, B, T, C);
        matmul_forward(fc, ln2, fc_w, fc_b, B, T, C, H);
        gelu_forward(g, fc, bt * H);
        matmul_forward(mproj, g, mproj_w, mproj_b, B, T, H, C);
        residual_forward(out, a, mproj, bt * C);
    };

    forward();
    float* dout = rnd(n_x);
    auto loss = [&]() {
        forward();
        return dot(dout, out, n_x);
    };

    // Gradient buffers (zeroed); attention scratch (overwritten).
    float* dx = zeros(n_x);
    float* da = zeros(n_x);
    float* dmproj = zeros(n_x);
    float* dg = zeros(n_h);
    float* dfc = zeros(n_h);
    float* dln2 = zeros(n_x);
    float* dln2_w = zeros(cz);
    float* dln2_b = zeros(cz);
    float* dfc_w = zeros(n_fcw);
    float* dfc_b = zeros(hz);
    float* dmproj_w = zeros(n_mprojw);
    float* dmproj_b = zeros(cz);
    float* daproj = zeros(n_x);
    float* dattn_out = zeros(n_x);
    float* dqkv = zeros(n_qkv);
    float* dln1 = zeros(n_x);
    float* dln1_w = zeros(cz);
    float* dln1_b = zeros(cz);
    float* dattn_w = zeros(n_attnw);
    float* dattn_b = zeros(c3);
    float* daproj_w = zeros(n_aprojw);
    float* daproj_b = zeros(cz);
    float* datt = arena.alloc(n_att);
    float* dpreatt = arena.alloc(n_att);

    // MLP sub-block backward (fills da from the residual + ln_2 paths).
    residual_backward(da, dmproj, dout, bt * C);
    matmul_backward(dg, dmproj_w, dmproj_b, dmproj, g, mproj_w, B, T, H, C);
    gelu_backward(dfc, fc, dg, bt * H);
    matmul_backward(dln2, dfc_w, dfc_b, dfc, ln2, fc_w, B, T, C, H);
    layernorm_backward(da, dln2_w, dln2_b, dln2, a, ln2_w, ln2_mean, ln2_rstd, B, T, C);

    // Attention sub-block backward (da is now the full gradient w.r.t. a).
    residual_backward(dx, daproj, da, bt * C);
    matmul_backward(dattn_out, daproj_w, daproj_b, daproj, attn_out, aproj_w, B, T, C, C);
    attention_backward(dqkv, datt, dpreatt, dattn_out, qkv, att, B, T, C, NH);
    matmul_backward(dln1, dattn_w, dattn_b, dqkv, ln1, attn_w, B, T, C, 3 * C);
    layernorm_backward(dx, dln1_w, dln1_b, dln1, x, ln1_w, ln1_mean, ln1_rstd, B, T, C);

    // dx through the whole block; one weight gradient in each sub-block.
    CHECK(grad_check(loss, x, dx, n_x) < 3e-2);
    CHECK(grad_check(loss, attn_w, dattn_w, n_attnw) < 3e-2);
    CHECK(grad_check(loss, fc_w, dfc_w, n_fcw) < 3e-2);

    return cppgpt::test::summary();
}
