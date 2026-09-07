// Linear probes over the residual stream (M7-5, plan property P4).
//
// P4: every reported number comes from data the probe was not fitted on, and a
// probe fitted on SHUFFLED labels must land at the base rate. That control is
// the whole test. Without it a probe reports high accuracy on a property that is
// 95% one class and looks like a discovery.
//
// One control the mutations could NOT distinguish, stated rather than implied:
// replacing the shuffled-label refit with "score the real weights on shuffled
// labels" passes every check here. On a clean split the two estimate the same
// quantity; they diverge only when the split leaks, which is what the control
// exists to detect and what these constructed splits deliberately do not do.
//
// The cases below are constructed so the right answer is known in advance:
// a linearly separable property must be learned, an unlearnable one must not,
// and a lopsided one must not beat its own base rate.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "cppgpt/interp/interpret.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

int main() {
    Generator g(4242ULL);
    const int n = 800, dim = 16, n_train = 600;

    // ---- a linearly separable property IS learned ----
    {
        std::vector<float> x(static_cast<std::size_t>(n) * dim);
        std::vector<std::uint8_t> y(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const bool cls = (i % 2) == 0;
            y[static_cast<std::size_t>(i)] = cls ? 1 : 0;
            for (int d = 0; d < dim; ++d)
                x[static_cast<std::size_t>(i) * dim + d] = static_cast<float>(g.normal());
            // Signal on one axis only; everything else is noise.
            x[static_cast<std::size_t>(i) * dim] += cls ? 2.5f : -2.5f;
        }
        std::vector<float> dir(static_cast<std::size_t>(dim));
        const ProbeResult r = fit_probe(x.data(), y.data(), n, dim, n_train, dir.data(), g);
        CHECK(r.accuracy > 0.9f);
        // The control must NOT learn it. If shuffling still scores high, the
        // split leaks and the accuracy above means nothing.
        CHECK(r.shuffled < r.accuracy);
        CHECK(std::fabs(r.shuffled - r.base_rate) < 0.15f);
        // The signal axis must dominate the direction, or "we found the
        // direction" is unearned even when the accuracy is high.
        double best = 0.0;
        int arg = -1;
        for (int d = 0; d < dim; ++d)
            if (std::fabs(static_cast<double>(dir[static_cast<std::size_t>(d)])) > best) {
                best = std::fabs(static_cast<double>(dir[static_cast<std::size_t>(d)]));
                arg = d;
            }
        CHECK(arg == 0);
    }

    // ---- an UNLEARNABLE property is not learned ----
    // Labels independent of the features: accuracy must sit at the base rate.
    {
        std::vector<float> x(static_cast<std::size_t>(n) * dim);
        std::vector<std::uint8_t> y(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(g.uniform_int(0, 1));
            for (int d = 0; d < dim; ++d)
                x[static_cast<std::size_t>(i) * dim + d] = static_cast<float>(g.normal());
        }
        std::vector<float> dir(static_cast<std::size_t>(dim));
        const ProbeResult r = fit_probe(x.data(), y.data(), n, dim, n_train, dir.data(), g);
        CHECK(r.accuracy < r.base_rate + 0.15f);
    }

    // ---- a LOPSIDED property does not look like a discovery ----
    // 95% one class, and the features carry nothing. A probe reporting 0.95 here
    // has learned the prior, which is why base_rate is returned beside it.
    {
        std::vector<float> x(static_cast<std::size_t>(n) * dim);
        std::vector<std::uint8_t> y(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] = (i % 20 == 0) ? 0 : 1;
            for (int d = 0; d < dim; ++d)
                x[static_cast<std::size_t>(i) * dim + d] = static_cast<float>(g.normal());
        }
        std::vector<float> dir(static_cast<std::size_t>(dim));
        const ProbeResult r = fit_probe(x.data(), y.data(), n, dim, n_train, dir.data(), g);
        CHECK(r.base_rate > 0.9f);
        CHECK(r.accuracy <= r.base_rate + 0.05f);
    }

    // ---- accuracy is HELD-OUT, and a memorising probe proves it ----
    //
    // Found by mutation: with dim=16 and 600 training rows the probe cannot
    // memorise, so training and held-out accuracy agree and "score the training
    // rows instead" passed every check above. The distinction only shows up when
    // the probe HAS the capacity to fit noise.
    //
    // Here dim > n_train with random labels: the training rows are perfectly
    // separable, so a probe scored on them reports ~1.0, while the honest
    // held-out answer is chance.
    {
        const int nm = 60, dm = 80, tm = 40;
        std::vector<float> x(static_cast<std::size_t>(nm) * dm);
        std::vector<std::uint8_t> y(static_cast<std::size_t>(nm));
        for (int i = 0; i < nm; ++i) {
            y[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(g.uniform_int(0, 1));
            for (int d = 0; d < dm; ++d)
                x[static_cast<std::size_t>(i) * dm + d] = static_cast<float>(g.normal());
        }
        std::vector<float> dir(static_cast<std::size_t>(dm));
        const ProbeResult r = fit_probe(x.data(), y.data(), nm, dm, tm, dir.data(), g);
        // Labels are noise, so the held-out number must stay near the base rate
        // however well the training rows were fitted.
        CHECK(r.accuracy < r.base_rate + 0.2f);
        CHECK(r.shuffled < r.base_rate + 0.2f);
    }

    // ---- the split is held-out, not the training rows ----
    // Fitting on rows the probe also scores is the standard way probe numbers
    // become meaningless. With n_train == n there is no held-out data at all,
    // which must fail loudly rather than report a training-set number.
    {
        std::vector<float> x(static_cast<std::size_t>(n) * dim, 0.0f);
        std::vector<std::uint8_t> y(static_cast<std::size_t>(n), 0);
        std::vector<float> dir(static_cast<std::size_t>(dim));
        CHECK_DIES_WITH(IGNORE(fit_probe(x.data(), y.data(), n, dim, n, dir.data(), g)),
                        "no held-out rows");
        CHECK_DIES_WITH(IGNORE(fit_probe(x.data(), y.data(), n, dim, 0, dir.data(), g)),
                        "no training rows");
    }

    return cppgpt::test::summary();
}
