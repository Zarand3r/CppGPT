// gelu_forward / gelu_backward (canonical GPT-2 tanh approximation).
//
// Forward pinned by: gelu(0)=0; the known value gelu_new(1)≈0.8411920; the exact
// structural identity gelu(x)−gelu(−x)=x; and the x→±∞ asymptotics. Backward
// pinned by a finite-difference gradient check (independent of the derivative
// formula).
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
    // ---- forward ----
    {
        float in0 = 0.0f, o0 = -1.0f;
        gelu_forward(&o0, &in0, 1, Device::CPU);
        CHECK(o0 == 0.0f);  // gelu(0) = 0 exactly

        float in1 = 1.0f, o1 = 0.0f;
        gelu_forward(&o1, &in1, 1, Device::CPU);
        CHECK(rel_close(o1, 0.8411920, 1e-4));  // known canonical value

        // gelu(x) - gelu(-x) = x, exact to fp rounding.
        const float xs[6] = {0.5f, 1.0f, 2.0f, -0.7f, 3.3f, -2.1f};
        for (float x : xs) {
            float in[2] = {x, -x};
            float out[2] = {0, 0};
            gelu_forward(out, in, 2, Device::CPU);
            CHECK(rel_close(static_cast<double>(out[0]) - out[1], x, 1e-5));
        }

        // Asymptotics: gelu(20) ≈ 20, gelu(-20) ≈ 0.
        float big[2] = {20.0f, -20.0f}, ob[2] = {0, 0};
        gelu_forward(ob, big, 2, Device::CPU);
        CHECK(rel_close(ob[0], 20.0, 1e-4));
        CHECK(std::fabs(ob[1]) < 1e-4f);

        // In-place (out == inp) is allowed.
        float v[3] = {-1.0f, 0.5f, 2.0f};
        float ref[3];
        gelu_forward(ref, v, 3, Device::CPU);
        gelu_forward(v, v, 3, Device::CPU);
        CHECK(v[0] == ref[0] && v[1] == ref[1] && v[2] == ref[2]);
    }

    // ---- backward: finite-difference gradient check ----
    {
        const int N = 64;
        Generator gen(0xABCDEF01ULL);
        std::vector<float> inp(N), dout(N), out(N), dinp(N, 0.0f);
        for (auto& x : inp) x = gen.normal();
        for (auto& x : dout) x = gen.normal();

        gelu_backward(dinp.data(), inp.data(), dout.data(), N, Device::CPU);  // analytic

        auto loss = [&]() {
            gelu_forward(out.data(), inp.data(), N, Device::CPU);
            return dot(dout.data(), out.data(), static_cast<std::size_t>(N));
        };
        const double err = grad_check(loss, inp.data(), dinp.data(), static_cast<std::size_t>(N));
        CHECK(err < 3e-3);

        // Accumulation: a second backward doubles the gradient.
        std::vector<float> dinp2 = dinp;
        gelu_backward(dinp2.data(), inp.data(), dout.data(), N, Device::CPU);
        bool doubled = true;
        for (int i = 0; i < N; ++i)
            doubled = doubled && rel_close(dinp2[i], 2.0 * dinp[i], 1e-5);
        CHECK(doubled);
    }

    return cppgpt::test::summary();
}
