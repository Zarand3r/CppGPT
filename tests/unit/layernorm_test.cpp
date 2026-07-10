// layernorm_forward / layernorm_backward (canonical GPT-2, eps=1e-5).
//
// Forward pinned by: the normalized row has mean 0 / var 1 (weight=1,bias=0), and
// a double-precision recompute of the full affine formula. Backward (the
// error-prone part) pinned by finite-difference gradient checks on dinp, dweight,
// and dbias — independent of the hand-derived backward algebra.
#include "cppgpt/ops.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "cppgpt/random.hpp"
#include "tests/check.hpp"
#include "tests/numeric.hpp"

using namespace cppgpt;
using cppgpt::test::dot;
using cppgpt::test::grad_check;
using cppgpt::test::rel_close;

int main() {
    const int B = 2, T = 3, C = 8;
    const std::size_t BT = static_cast<std::size_t>(B * T);
    const std::size_t n = static_cast<std::size_t>(B * T * C);
    const std::size_t cz = static_cast<std::size_t>(C);

    Generator gen(0x1A4E12ULL);
    std::vector<float> inp(n), weight(cz), bias(cz), out(n), mean(BT), rstd(BT);
    for (auto& x : inp) x = gen.normal();

    // ---- forward property: normalized rows are mean 0, var 1 (weight=1, bias=0) ----
    {
        std::vector<float> w1(cz, 1.0f), b0(cz, 0.0f);
        layernorm_forward(out.data(), mean.data(), rstd.data(), inp.data(), w1.data(), b0.data(),
                          B, T, C);
        for (std::size_t r = 0; r < BT; ++r) {
            const float* o = out.data() + r * cz;
            double mu = 0.0;
            for (std::size_t c = 0; c < cz; ++c) mu += o[c];
            mu /= static_cast<double>(C);
            double var = 0.0;
            for (std::size_t c = 0; c < cz; ++c) var += (o[c] - mu) * (o[c] - mu);
            var /= static_cast<double>(C);
            CHECK(std::fabs(mu) < 1e-5);
            CHECK(rel_close(var, 1.0, 1e-3));
        }
    }

    // ---- forward formula: independent double-precision recompute ----
    for (auto& x : weight) x = gen.normal();
    for (auto& x : bias) x = gen.normal();
    layernorm_forward(out.data(), mean.data(), rstd.data(), inp.data(), weight.data(), bias.data(),
                      B, T, C);
    {
        bool fwd_ok = true;
        for (std::size_t r = 0; r < BT; ++r) {
            const float* x = inp.data() + r * cz;
            double mu = 0.0;
            for (std::size_t c = 0; c < cz; ++c) mu += x[c];
            mu /= static_cast<double>(C);
            double var = 0.0;
            for (std::size_t c = 0; c < cz; ++c) var += (x[c] - mu) * (x[c] - mu);
            var /= static_cast<double>(C);
            const double s = 1.0 / std::sqrt(var + 1e-5);
            for (std::size_t c = 0; c < cz; ++c) {
                const double ref = (x[c] - mu) * s * weight[c] + bias[c];
                fwd_ok = fwd_ok && rel_close(out[r * cz + c], ref, 1e-4);
            }
        }
        CHECK(fwd_ok);
    }

    // ---- backward: finite-difference gradient checks on dinp, dweight, dbias ----
    {
        std::vector<float> dout(n), dinp(n, 0.0f), dweight(cz, 0.0f), dbias(cz, 0.0f);
        for (auto& x : dout) x = gen.normal();

        // analytic grads use mean/rstd from the base forward above.
        layernorm_backward(dinp.data(), dweight.data(), dbias.data(), dout.data(), inp.data(),
                           weight.data(), mean.data(), rstd.data(), B, T, C);

        // loss = <dout, forward(inp, weight, bias)>; recomputes mean/rstd each call.
        auto loss = [&]() {
            layernorm_forward(out.data(), mean.data(), rstd.data(), inp.data(), weight.data(),
                              bias.data(), B, T, C);
            return dot(dout.data(), out.data(), n);
        };
        CHECK(grad_check(loss, inp.data(), dinp.data(), n) < 5e-3);
        CHECK(grad_check(loss, weight.data(), dweight.data(), cz) < 5e-3);
        CHECK(grad_check(loss, bias.data(), dbias.data(), cz) < 5e-3);
    }

    return cppgpt::test::summary();
}
