// argmax / sample: pick a token from a logits vector.
//
// argmax is exact. `sample` is pinned by: determinism (same seed ⇒ same token),
// the greedy limits (tiny temperature or top_k=1 ⇒ argmax), top-k restriction
// (only the k highest-logit tokens ever appear), and a frequency check (over many
// draws the empirical distribution matches softmax).
#include "cppgpt/sample.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

int main() {
    // ---- argmax: index of the max logit, ties → lowest index ----
    {
        const float a[4] = {1.0f, 5.0f, 2.0f, 5.0f};
        CHECK(argmax(a, 4) == 1);
        const float b[3] = {-2.0f, -9.0f, -1.0f};
        CHECK(argmax(b, 3) == 2);
    }

    // ---- determinism: same seed ⇒ same sequence of samples ----
    {
        const float logits[5] = {0.3f, 1.1f, -0.5f, 2.0f, 0.7f};
        Generator g1(42ULL), g2(42ULL);
        bool same = true;
        for (int i = 0; i < 20; ++i)
            same = same && (sample(logits, 5, 1.0f, 0, g1) == sample(logits, 5, 1.0f, 0, g2));
        CHECK(same);
    }

    // ---- greedy limits: tiny temperature and top_k=1 both collapse to argmax ----
    {
        const float logits[5] = {0.3f, 1.1f, -0.5f, 2.0f, 0.7f};  // argmax = 3
        Generator g(7ULL);
        bool tiny_temp_ok = true, topk1_ok = true;
        for (int i = 0; i < 50; ++i) {
            tiny_temp_ok = tiny_temp_ok && (sample(logits, 5, 1e-4f, 0, g) == 3);
            topk1_ok = topk1_ok && (sample(logits, 5, 1.0f, 1, g) == 3);
        }
        CHECK(tiny_temp_ok);
        CHECK(topk1_ok);
    }

    // ---- top-k restriction: only the k highest-logit tokens ever appear ----
    {
        const float logits[6] = {0.0f, 3.0f, 1.0f, 2.5f, -1.0f, 0.5f};  // top-2 = {1, 3}
        Generator g(99ULL);
        bool restricted = true;
        for (int i = 0; i < 200; ++i) {
            const int t = sample(logits, 6, 1.0f, 2, g);
            restricted = restricted && (t == 1 || t == 3);
        }
        CHECK(restricted);
    }

    // ---- frequency: over many draws, P(token 1) ≈ softmax over {0,1} ----
    {
        const float logits[2] = {0.0f, 1.0f};  // p1 = e/(1+e) ≈ 0.7311
        Generator g(2024ULL);
        int count1 = 0;
        const int N = 20000;
        for (int i = 0; i < N; ++i)
            if (sample(logits, 2, 1.0f, 0, g) == 1) ++count1;
        const double freq = static_cast<double>(count1) / N;
        CHECK(freq > 0.72 && freq < 0.745);  // ≈ 0.7311
    }

    // ---- top_k == 1 is exactly argmax, even when the maximum is TIED ----
    // The top-k threshold is `>= kth_largest`, so ties would all stay eligible
    // and softmax would spread over them — breaking the documented "top_k = 1
    // gives greedy decoding" and greedy determinism vs the PyTorch reference.
    {
        const float logits[6] = {3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f};  // three-way tie
        Generator g(31337ULL);
        bool always_argmax = true;
        for (int i = 0; i < 200; ++i)
            always_argmax = always_argmax && (sample(logits, 6, 1.0f, 1, g) == argmax(logits, 6));
        CHECK(always_argmax);
    }

    // ---- non-finite logits fail fast instead of silently returning a token ----
    // A NaN/inf distribution means upstream numerics already broke (diverged lr,
    // corrupt weights). Returning token 0 forever would hide it.
    {
        const float nan_logits[3] = {std::nanf(""), std::nanf(""), std::nanf("")};
        Generator g(5ULL);
        CHECK_DIES(IGNORE(sample(nan_logits, 3, 1.0f, 0, g)));

        const float inf_logits[3] = {-std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
        CHECK_DIES(IGNORE(sample(inf_logits, 3, 1.0f, 0, g)));
    }

    return cppgpt::test::summary();
}
