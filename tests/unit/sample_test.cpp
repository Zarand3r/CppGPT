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
        // The excluded tokens sit just BELOW the cut (2.80 against a 2.90 threshold),
        // not far below it. The earlier version used a 1.5 gap, which meant a
        // threshold wrong by up to 1.5 admitted no new token and was invisible --
        // a mutation lowering the threshold by 1.0 survived it. Margin in test
        // data hides exactly the errors the test exists to find.
        const float logits[6] = {0.0f, 3.00f, 2.90f, 2.80f, 2.70f, 0.5f};  // top-2 = {1, 2}
        Generator g(99ULL);
        bool restricted = true;
        for (int i = 0; i < 400; ++i) {
            const int t = sample(logits, 6, 1.0f, 2, g);
            restricted = restricted && (t == 1 || t == 2);
        }
        CHECK(restricted);

        // ...and both eligible tokens must actually appear, or "restricted" would
        // also pass for a threshold so high that only one token survives.
        bool saw1 = false, saw2 = false;
        for (int i = 0; i < 400; ++i) {
            const int t = sample(logits, 6, 1.0f, 2, g);
            saw1 = saw1 || (t == 1);
            saw2 = saw2 || (t == 2);
        }
        CHECK(saw1 && saw2);

        // The cut must land at exactly k tokens for every k, with near-tied logits
        // either side. This is what pins the k-th-LARGEST index; picking from the
        // wrong end of nth_element leaves the count wrong.
        bool k_exact = true;
        for (int k = 1; k <= 5; ++k) {
            bool seen[6] = {false, false, false, false, false, false};
            for (int i = 0; i < 600; ++i) seen[sample(logits, 6, 1.0f, k, g)] = true;
            int n = 0;
            for (const bool b : seen) n += b ? 1 : 0;
            k_exact = k_exact && (n == k);
        }
        CHECK(k_exact);

        // The same property over MANY random vectors, not one. Whether a given
        // off-by-one partition index happens to leave the correct value at the
        // position we read is a property of libstdc++'s partitioning for that
        // exact input -- so one vector can pass by luck. Twenty cannot.
        {
            Generator rg(20260821ULL);
            bool wide_exact = true;
            for (int trial = 0; trial < 20 && wide_exact; ++trial) {
                float wide[24];
                // NEAR-TIED on purpose. With a wide spread the k-th eligible
                // token can have negligible probability and never be drawn, so
                // `n == k` fails on correct code -- which it did. The property
                // under test is which tokens are ELIGIBLE, not how often each is
                // chosen, so the logits must keep them all reachable.
                for (int i2 = 0; i2 < 24; ++i2) wide[i2] = 0.2f * rg.normal();
                for (int k = 1; k <= 6 && wide_exact; ++k) {
                    bool seen[24] = {};
                    for (int i2 = 0; i2 < 2000; ++i2) seen[sample(wide, 24, 1.0f, k, rg)] = true;
                    int n = 0;
                    for (const bool b : seen) n += b ? 1 : 0;
                    wide_exact = wide_exact && (n == k);
                }
            }
            CHECK(wide_exact);
        }
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

    // Greedy decoding must hit the finite guard too. top_k == 1 returns via
    // argmax, whose `>` comparison is false for NaN, so it returned token 0
    // forever on a broken model — the L2 incident, reopened on this branch by
    // the L4 fix. The two lessons' remedies collided and nothing tested the seam.
    {
        const float nan3[3] = {std::nanf(""), std::nanf(""), std::nanf("")};
        Generator g(9ULL);
        CHECK_DIES(IGNORE(sample(nan3, 3, 1.0f, 1, g)));   // greedy path
        CHECK_DIES(IGNORE(sample(nan3, 3, 1.0f, 0, g)));   // sampling path
    }

    return cppgpt::test::summary();
}
