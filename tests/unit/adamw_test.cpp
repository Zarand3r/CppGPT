// adamw_update: single-tensor AdamW step with decoupled weight decay.
//
// Correctness is established four independent ways:
//   * Cross-framework fixture — 5 steps of a fixed-gradient two-group scenario
//     match torch.optim.AdamW (fp64 reference values, generated in .venv) to fp32
//     rounding. This pins the exact update algorithm, bias correction, and the
//     decay/no-decay split against the canonical oracle.
//   * Analytic first step — from m=v=0, the first update is −lr·(sign(g) + wd·p)
//     (bias correction cancels to m̂=g, v̂=g², so m̂/√v̂ = sign(g)).
//   * Weight-decay isolation — with g=0, a decaying tensor moves by exactly
//     −lr·wd·p and a non-decaying tensor does not move at all.
//   * Convergence — on a convex quadratic (grad = x−a) the iterate approaches the
//     minimum a, exercising the moments and bias correction over many steps.
#include "cppgpt/optimizer.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "tests/check.hpp"
#include "tests/numeric.hpp"

using namespace cppgpt;
using cppgpt::test::rel_close;

int main() {
    // ---- Cross-framework fixture: torch.optim.AdamW, lr=0.1, betas=(0.9,0.95), eps=1e-8 ----
    // 5 steps, same gradient each step. Group w decays (wd=0.1); group b does not.
    {
        const float lr = 0.1f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
        std::vector<float> w = {0.5f, -0.3f, 1.2f, -0.75f};
        const std::vector<float> gw = {0.1f, -0.2f, 0.05f, 0.3f};
        std::vector<float> b = {1.0f, -2.0f};
        const std::vector<float> gb = {0.05f, -0.1f};
        std::vector<float> mw(4, 0.0f), vw(4, 0.0f), mb(2, 0.0f), vb(2, 0.0f);

        for (int t = 1; t <= 5; ++t) {
            adamw_update(w.data(), gw.data(), mw.data(), vw.data(), 4, lr, b1, b2, eps, 0.1f, t,
                         Device::CPU);
            adamw_update(b.data(), gb.data(), mb.data(), vb.data(), 2, lr, b1, b2, eps, 0.0f, t,
                         Device::CPU);
        }
        // torch fp64 references (scripts: two-group AdamW, 5 fixed-gradient steps).
        const double w_ref[4] = {-0.014604427040054865, 0.20480246152502624, 0.6510886568998806,
                                 -1.2033420220883504};
        const double b_ref[2] = {0.5000000999999801, -1.5000000499999948};
        bool ok = true;
        for (int i = 0; i < 4; ++i) ok = ok && (std::fabs(w[i] - w_ref[i]) < 1e-4);
        for (int i = 0; i < 2; ++i) ok = ok && (std::fabs(b[i] - b_ref[i]) < 1e-4);
        CHECK(ok);
    }

    // ---- Analytic first step: Δp = −lr·(sign(g) + wd·p) from m=v=0 ----
    {
        const float lr = 0.01f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.1f;
        std::vector<float> p = {2.0f, -3.0f, 0.5f};
        const std::vector<float> g = {0.7f, -0.2f, 4.0f};  // magnitudes irrelevant to direction
        const std::vector<float> p0 = p;
        std::vector<float> m(3, 0.0f), v(3, 0.0f);
        adamw_update(p.data(), g.data(), m.data(), v.data(), 3, lr, b1, b2, eps, wd, 1, Device::CPU);
        bool ok = true;
        for (int i = 0; i < 3; ++i) {
            const float sign = (g[i] > 0.0f) ? 1.0f : -1.0f;
            const float expected = p0[i] - lr * (sign + wd * p0[i]);
            ok = ok && rel_close(p[i], expected, 1e-5);
        }
        CHECK(ok);
    }

    // ---- Weight-decay isolation: with g=0, decay ⇒ Δp=−lr·wd·p; no-decay ⇒ no move ----
    {
        const float lr = 0.1f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
        std::vector<float> pd = {1.5f, -2.5f};   // decaying
        std::vector<float> pn = {1.5f, -2.5f};   // non-decaying
        const std::vector<float> g = {0.0f, 0.0f};
        std::vector<float> md(2, 0.0f), vd(2, 0.0f), mn(2, 0.0f), vn(2, 0.0f);
        const std::vector<float> p0 = pd;
        adamw_update(pd.data(), g.data(), md.data(), vd.data(), 2, lr, b1, b2, eps, 0.2f, 1,
                     Device::CPU);
        adamw_update(pn.data(), g.data(), mn.data(), vn.data(), 2, lr, b1, b2, eps, 0.0f, 1,
                     Device::CPU);
        bool ok = true;
        for (int i = 0; i < 2; ++i) {
            ok = ok && rel_close(pd[i], p0[i] - lr * 0.2f * p0[i], 1e-6);  // pure decay
            ok = ok && (pn[i] == p0[i]);                                   // exactly unchanged
        }
        CHECK(ok);
    }

    // ---- Convergence: grad = x − a (∇ of ½‖x−a‖²) drives x → a ----
    {
        const float lr = 0.05f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        const std::vector<float> a = {1.0f, -2.0f, 3.5f, 0.25f};
        const std::size_t n = a.size();
        std::vector<float> x(n, 0.0f), m(n, 0.0f), v(n, 0.0f), g(n);
        for (int t = 1; t <= 3000; ++t) {
            for (std::size_t i = 0; i < n; ++i) g[i] = x[i] - a[i];
            adamw_update(x.data(), g.data(), m.data(), v.data(), static_cast<int>(n), lr, b1, b2,
                         eps, 0.0f, t, Device::CPU);
        }
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) ok = ok && (std::fabs(x[i] - a[i]) < 1e-3);
        CHECK(ok);
    }

    return cppgpt::test::summary();
}
