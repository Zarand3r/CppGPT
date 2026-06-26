// Integration: the GPT-2 MLP sub-block, forward + backward end to end.
//
//   out = x + c_proj(gelu(c_fc(layernorm(x))))
//
// This composes matmul (M0) with gelu, layernorm, and residual, and exercises
// the part unit tests can't: that the ops wire together and the backward chain
// composes in reverse — in particular that dx ACCUMULATES from both the residual
// path and the layernorm path. Correctness is pinned by a finite-difference
// gradient check of dx and an inner weight gradient (dW_fc) against the analytic
// backward. Buffers come from a single Storage arena (alloc / alloc_zeroed),
// the same pattern the model will use.
#include <cstddef>
#include <vector>

#include "cppgpt/ops.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/storage.hpp"
#include "tests/check.hpp"
#include "tests/numeric.hpp"

using namespace cppgpt;
using cppgpt::test::dot;
using cppgpt::test::grad_check;

int main() {
    const int B = 2, T = 3, C = 4, H = 4 * C;  // H = MLP hidden = 4C
    const int dev_bt = B * T;
    const std::size_t n_x = static_cast<std::size_t>(dev_bt * C);
    const std::size_t n_h = static_cast<std::size_t>(dev_bt * H);
    const std::size_t n_wfc = static_cast<std::size_t>(H * C);
    const std::size_t n_wproj = static_cast<std::size_t>(C * H);
    const std::size_t cz = static_cast<std::size_t>(C);
    const std::size_t hz = static_cast<std::size_t>(H);
    const std::size_t bt = static_cast<std::size_t>(dev_bt);

    Storage arena(8192);
    Generator gen(0x6217A11ULL);
    auto rnd = [&](std::size_t n) {
        float* p = arena.alloc(n);
        for (std::size_t i = 0; i < n; ++i) p[i] = gen.normal();
        return p;
    };
    auto zeros = [&](std::size_t n) { return arena.alloc_zeroed(n); };

    // Parameters + input (persistent across the gradient check).
    float* x = rnd(n_x);
    float* w_ln = rnd(cz);
    float* b_ln = rnd(cz);
    float* w_fc = rnd(n_wfc);
    float* b_fc = rnd(hz);
    float* w_proj = rnd(n_wproj);
    float* b_proj = rnd(cz);

    // Forward activations (scratch — recomputed each loss() call).
    float* ln = arena.alloc(n_x);
    float* ln_mean = arena.alloc(bt);
    float* ln_rstd = arena.alloc(bt);
    float* fc = arena.alloc(n_h);
    float* g = arena.alloc(n_h);
    float* proj = arena.alloc(n_x);
    float* out = arena.alloc(n_x);

    auto forward = [&]() {
        layernorm_forward(ln, ln_mean, ln_rstd, x, w_ln, b_ln, B, T, C, Device::CPU);
        matmul_forward(fc, ln, w_fc, b_fc, B, T, C, H, Device::CPU);
        gelu_forward(g, fc, dev_bt * H, Device::CPU);
        matmul_forward(proj, g, w_proj, b_proj, B, T, H, C, Device::CPU);
        residual_forward(out, x, proj, dev_bt * C, Device::CPU);
    };

    // Base forward, then a fixed upstream gradient and the scalar loss <dout, out>.
    forward();
    float* dout = rnd(n_x);
    auto loss = [&]() {
        forward();
        return dot(dout, out, n_x);
    };

    // Backward chain (reverse order) into zeroed gradient buffers.
    float* dx = zeros(n_x);
    float* dproj = zeros(n_x);
    float* dg = zeros(n_h);
    float* dfc = zeros(n_h);
    float* dln = zeros(n_x);
    float* dw_ln = zeros(cz);
    float* db_ln = zeros(cz);
    float* dw_fc = zeros(n_wfc);
    float* db_fc = zeros(hz);
    float* dw_proj = zeros(n_wproj);
    float* db_proj = zeros(cz);

    residual_backward(dx, dproj, dout, dev_bt * C, Device::CPU);
    matmul_backward(dg, dw_proj, db_proj, dproj, g, w_proj, B, T, H, C, Device::CPU);
    gelu_backward(dfc, fc, dg, dev_bt * H, Device::CPU);
    matmul_backward(dln, dw_fc, db_fc, dfc, ln, w_fc, B, T, C, H, Device::CPU);
    // dx accumulates here on top of the residual contribution above:
    layernorm_backward(dx, dw_ln, db_ln, dln, x, w_ln, ln_mean, ln_rstd, B, T, C, Device::CPU);

    // dx flows through the whole block (incl. the residual+layernorm accumulation).
    CHECK(grad_check(loss, x, dx, n_x) < 2e-2);
    // dW_fc exercises the inner-weight gradient path through gelu + proj.
    CHECK(grad_check(loss, w_fc, dw_fc, n_wfc) < 2e-2);

    return cppgpt::test::summary();
}
