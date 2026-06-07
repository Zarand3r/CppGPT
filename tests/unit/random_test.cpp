#include "cppgpt/random.hpp"

#include <cmath>

#include "tests/check.hpp"

int main() {
    using cppgpt::Generator;

    // determinism: same seed -> identical sequence
    {
        Generator a(123);
        Generator b(123);
        for (int i = 0; i < 100; ++i) {
            CHECK(a.next_u64() == b.next_u64());
        }
    }

    // different seeds diverge
    {
        Generator a(1);
        Generator b(2);
        bool differ = false;
        for (int i = 0; i < 8; ++i) {
            if (a.next_u64() != b.next_u64()) differ = true;
        }
        CHECK(differ);
    }

    // uniform() in [0, 1)
    {
        Generator g(7);
        for (int i = 0; i < 1000; ++i) {
            const float u = g.uniform();
            CHECK(u >= 0.0f && u < 1.0f);
        }
    }

    // uniform_int() in [lo, hi]
    {
        Generator g(7);
        for (int i = 0; i < 1000; ++i) {
            const std::int64_t v = g.uniform_int(-3, 5);
            CHECK(v >= -3 && v <= 5);
        }
    }

    // normal(): finite, sample mean ~ 0 (deterministic seed, loose bound)
    {
        Generator g(7);
        const int n = 20000;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            const float x = g.normal();
            CHECK(std::isfinite(x));
            sum += x;
        }
        CHECK(std::fabs(sum / n) < 0.05);
    }

    return cppgpt::test::summary();
}
