// matmul_forward / matmul_backward correctness.
//
// No external (PyTorch) oracle is available in this environment, so correctness
// is established by two self-contained, independent means:
//
//   * Forward — exact hand-computed fixtures pin the out = inp @ Wᵀ + b
//     convention (catches transpose / indexing / bias bugs).
//   * Backward — the adjoint identity. matmul is linear in each of inp, weight,
//     bias, so the backward must be the mathematical transpose of the forward:
//         F := <dout, inp @ Wᵀ>  ==  <dinp, inp>  ==  <dweight, weight>
//         <dout, broadcast(bias)>  ==  <dbias, bias>
//     These hold to floating-point rounding, independent of how the backward is
//     coded — for a linear op this is the exact, complete form of a gradient
//     check, so no separate finite-difference pass is needed here. (The MLP-block
//     integration test exercises matmul inside a composed finite-difference check.)
//
// (When a PyTorch environment is available, scripts/gen_fixtures.py will add a
// cross-framework .bin fixture comparison; the adjoint law already pins matmul.)
#include "cppgpt/ops.hpp"

#include <cstddef>
#include <vector>

#include "cppgpt/random.hpp"
#include "tests/check.hpp"
#include "tests/numeric.hpp"

using namespace cppgpt;
using cppgpt::test::dot;
using cppgpt::test::rel_close;

int main() {
    // ---- Forward: exact fixtures (integer-valued, so == is safe in fp32) ----
    {
        // B=1,T=1,C=3,OC=2; selector weight + bias -> [inp0+10, inp1+20].
        const float inp[3] = {1.0f, 2.0f, 3.0f};
        const float w[6] = {1, 0, 0, 0, 1, 0};
        const float b[2] = {10.0f, 20.0f};
        float out[2] = {0, 0};
        matmul_forward(out, inp, w, b, 1, 1, 3, 2, Device::CPU);
        CHECK(out[0] == 11.0f);
        CHECK(out[1] == 22.0f);

        float out_nb[2] = {0, 0};  // bias = null path
        matmul_forward(out_nb, inp, w, nullptr, 1, 1, 3, 2, Device::CPU);
        CHECK(out_nb[0] == 1.0f && out_nb[1] == 2.0f);
    }
    {
        // Negatives + accumulation: OC=1, C=2 -> 1.5*2 + (-2)*3 = -3.
        const float inp[2] = {1.5f, -2.0f};
        const float w[2] = {2.0f, 3.0f};
        float out[1] = {0};
        matmul_forward(out, inp, w, nullptr, 1, 1, 2, 1, Device::CPU);
        CHECK(out[0] == -3.0f);
    }
    {
        // Batched: B=2 distinct rows exercise inp/out row strides.
        const float inp[4] = {1, 2, 3, 4};  // two rows of C=2
        const float w[2] = {1, 1};          // OC=1 -> row sum
        float out[2] = {0, 0};
        matmul_forward(out, inp, w, nullptr, 2, 1, 2, 1, Device::CPU);
        CHECK(out[0] == 3.0f && out[1] == 7.0f);
    }

    // ---- Backward: adjoint identities (exact to fp rounding) ----
    {
        const int B = 2, T = 3, C = 4, OC = 5;
        const std::size_t n_in = static_cast<std::size_t>(B * T * C);
        const std::size_t n_w = static_cast<std::size_t>(OC * C);
        const std::size_t n_out = static_cast<std::size_t>(B * T * OC);

        Generator gen(0xC0FFEEULL);
        std::vector<float> inp(n_in), w(n_w), bias(static_cast<std::size_t>(OC)), dout(n_out);
        for (auto& x : inp) x = gen.normal();
        for (auto& x : w) x = gen.normal();
        for (auto& x : bias) x = gen.normal();
        for (auto& x : dout) x = gen.normal();

        // Pure bilinear output (no bias) for the adjoint form.
        std::vector<float> out_nb(n_out);
        matmul_forward(out_nb.data(), inp.data(), w.data(), nullptr, B, T, C, OC, Device::CPU);

        std::vector<float> dinp(n_in, 0.0f), dweight(n_w, 0.0f),
            dbias(static_cast<std::size_t>(OC), 0.0f);
        matmul_backward(dinp.data(), dweight.data(), dbias.data(), dout.data(), inp.data(),
                        w.data(), B, T, C, OC, Device::CPU);

        const double form = dot(dout.data(), out_nb.data(), n_out);  // <dout, inp@Wᵀ>
        CHECK(rel_close(form, dot(dinp.data(), inp.data(), n_in), 1e-4));    // == <dinp, inp>
        CHECK(rel_close(form, dot(dweight.data(), w.data(), n_w), 1e-4));    // == <dweight, weight>

        // bias adjoint: <dout, broadcast(bias)> == <dbias, bias>.
        double lhs_bias = 0.0;
        for (int bt = 0; bt < B * T; ++bt)
            for (int oc = 0; oc < OC; ++oc)
                lhs_bias += static_cast<double>(dout[static_cast<std::size_t>(bt * OC + oc)]) *
                            static_cast<double>(bias[static_cast<std::size_t>(oc)]);
        CHECK(rel_close(lhs_bias, dot(dbias.data(), bias.data(), static_cast<std::size_t>(OC)),
                        1e-4));

        // dbias is exactly the column sum of dout.
        for (int oc = 0; oc < OC; ++oc) {
            double col = 0.0;
            for (int bt = 0; bt < B * T; ++bt)
                col += static_cast<double>(dout[static_cast<std::size_t>(bt * OC + oc)]);
            CHECK(rel_close(col, static_cast<double>(dbias[static_cast<std::size_t>(oc)]), 1e-5));
        }

        // dbias = null path: skipping the bias gradient must not change dinp/dweight.
        std::vector<float> dinp2(n_in, 0.0f), dweight2(n_w, 0.0f);
        matmul_backward(dinp2.data(), dweight2.data(), nullptr, dout.data(), inp.data(), w.data(),
                        B, T, C, OC, Device::CPU);
        CHECK(dinp2 == dinp);
        CHECK(dweight2 == dweight);
    }

    return cppgpt::test::summary();
}
