// clip_grad_norm + cosine_lr: the pure optimizer utilities. Exact checks on
// known-norm gradient vectors and closed-form schedule values (warmup ramp,
// peak, cosine midpoint, floor, and monotonicity).
#include <cmath>
#include <vector>

#include "cppgpt/optimizer.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

namespace {
bool close(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }
}  // namespace

int main() {
    // ---- clip_grad_norm: no clip when under the threshold ----
    {
        std::vector<float> g{3.0f, 4.0f};  // L2 norm == 5
        const float norm = clip_grad_norm(g.data(), 2, 10.0f);
        CHECK(close(norm, 5.0f));
        CHECK(close(g[0], 3.0f) && close(g[1], 4.0f));  // untouched
    }

    // ---- clip_grad_norm: scales down to max_norm when over ----
    {
        std::vector<float> g{3.0f, 4.0f};  // norm 5
        const float norm = clip_grad_norm(g.data(), 2, 1.0f);
        CHECK(close(norm, 5.0f));  // returns PRE-clip norm
        const float scale = 1.0f / (5.0f + 1e-6f);
        CHECK(close(g[0], 3.0f * scale) && close(g[1], 4.0f * scale));
        const float new_norm = std::sqrt(g[0] * g[0] + g[1] * g[1]);
        CHECK(close(new_norm, 1.0f, 1e-3f));  // clipped to ~max_norm
    }

    // ---- clip_grad_norm: zero gradients, no clip, no div-by-zero ----
    {
        std::vector<float> g{0.0f, 0.0f, 0.0f};
        const float norm = clip_grad_norm(g.data(), 3, 1.0f);
        CHECK(close(norm, 0.0f));
        CHECK(close(g[0], 0.0f));
    }

    // ---- clip_grad_norm: max_norm <= 0 disables clipping (norm still reported) ----
    {
        std::vector<float> g{6.0f, 8.0f};  // norm 10
        const float norm = clip_grad_norm(g.data(), 2, 0.0f);
        CHECK(close(norm, 10.0f));
        CHECK(close(g[0], 6.0f) && close(g[1], 8.0f));  // unchanged
    }

    // ---- cosine_lr: warmup ramp reaches max at the end of warmup ----
    {
        const float mx = 1.0f, mn = 0.1f;
        const int W = 10, S = 100;
        CHECK(close(cosine_lr(0, mx, mn, W, S), mx * 1.0f / 10.0f));
        CHECK(close(cosine_lr(W - 1, mx, mn, W, S), mx));  // step 9 -> max_lr
        // Strictly increasing across warmup.
        bool up = true;
        for (int s = 1; s < W; ++s) up = up && (cosine_lr(s, mx, mn, W, S) > cosine_lr(s - 1, mx, mn, W, S));
        CHECK(up);
    }

    // ---- cosine_lr: cosine start == max, midpoint == mean, end == min ----
    {
        const float mx = 1.0f, mn = 0.1f;
        const int W = 10, S = 110;  // decay window length 100, midpoint at step 60
        CHECK(close(cosine_lr(W, mx, mn, W, S), mx));           // decay_ratio 0
        CHECK(close(cosine_lr(60, mx, mn, W, S), (mx + mn) / 2.0f, 1e-4f));  // ratio 0.5
        CHECK(close(cosine_lr(S, mx, mn, W, S), mn));           // ratio 1
        CHECK(close(cosine_lr(S + 50, mx, mn, W, S), mn));      // held after the end
        // Strictly decreasing across the cosine phase.
        bool down = true;
        for (int s = W + 1; s <= S; ++s)
            down = down && (cosine_lr(s, mx, mn, W, S) < cosine_lr(s - 1, mx, mn, W, S));
        CHECK(down);
    }

    // ---- cosine_lr: warmup == 0 goes straight to the cosine phase ----
    {
        const float mx = 2.0f, mn = 0.0f;
        const int S = 50;
        CHECK(close(cosine_lr(0, mx, mn, 0, S), mx));   // no ramp; starts at max
        CHECK(close(cosine_lr(S, mx, mn, 0, S), mn));
    }

    return cppgpt::test::summary();
}
