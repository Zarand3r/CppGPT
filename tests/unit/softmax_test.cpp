// softmax_forward / softmax_backward over a vector of length N.
//
// Forward pinned by: sums to 1, strictly positive, order-preserving,
// shift-invariant (softmax(x+c)=softmax(x)), uniform on equal inputs. Backward
// pinned by a finite-difference gradient check of the softmax Jacobian.
#include "cppgpt/ops.hpp"

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
    // ---- forward ----
    {
        const float in[3] = {0.0f, 1.0f, 2.0f};
        float out[3] = {0, 0, 0};
        softmax_forward(out, in, 3);
        CHECK(rel_close(static_cast<double>(out[0]) + out[1] + out[2], 1.0, 1e-6));  // sums to 1
        CHECK(out[0] > 0.0f && out[1] > 0.0f && out[2] > 0.0f);                       // positive
        CHECK(out[0] < out[1] && out[1] < out[2]);                                    // order-preserving

        // shift invariance: softmax(x + 10) == softmax(x).
        const float in2[3] = {10.0f, 11.0f, 12.0f};
        float out2[3] = {0, 0, 0};
        softmax_forward(out2, in2, 3);
        for (int i = 0; i < 3; ++i) CHECK(rel_close(out[i], out2[i], 1e-5));

        // uniform on equal inputs.
        const float eq[4] = {5, 5, 5, 5};
        float ou[4] = {0, 0, 0, 0};
        softmax_forward(ou, eq, 4);
        for (int i = 0; i < 4; ++i) CHECK(rel_close(ou[i], 0.25, 1e-6));
    }

    // ---- backward: finite-difference gradient check ----
    {
        const int N = 12;
        Generator gen(0x50F7ULL);
        std::vector<float> inp(N), dout(N), out(N), dinp(N, 0.0f);
        for (auto& x : inp) x = gen.normal();
        for (auto& x : dout) x = gen.normal();

        softmax_forward(out.data(), inp.data(), N);
        softmax_backward(dinp.data(), dout.data(), out.data(), N);

        auto loss = [&]() {
            softmax_forward(out.data(), inp.data(), N);
            return dot(dout.data(), out.data(), static_cast<std::size_t>(N));
        };
        CHECK(grad_check(loss, inp.data(), dinp.data(), static_cast<std::size_t>(N)) < 2e-3);
    }

    return cppgpt::test::summary();
}
