// Shared numeric test helpers: double-accumulated dot, relative closeness, and a
// central-difference gradient check. The gradient check is the independent
// backward oracle for the op tests (no PyTorch in this environment): it compares
// each analytic gradient against a finite difference of the forward pass, so a
// derivation error in a hand-written backward is caught regardless of how the
// backward is coded.
#pragma once

#include <cmath>
#include <cstddef>

namespace cppgpt::test {

inline double dot(const float* a, const float* b, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return s;
}

inline bool rel_close(double x, double y, double tol) {
    const double denom = std::fmax(1.0, std::fmax(std::fabs(x), std::fabs(y)));
    return std::fabs(x - y) / denom <= tol;
}

// Central-difference gradient check. `loss()` returns a scalar computed from the
// current contents of x[0..n); `analytic[i]` is the claimed dL/dx[i]. Perturbs
// each x[i] by ±h, restoring it, and returns the max relative error between the
// finite difference and the analytic gradient.
template <class Loss>
double grad_check(Loss loss, float* x, const float* analytic, std::size_t n, float h = 1e-2f) {
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float x0 = x[i];
        x[i] = x0 + h;
        const double lp = loss();
        x[i] = x0 - h;
        const double lm = loss();
        x[i] = x0;  // restore exactly
        const double fd = (lp - lm) / (2.0 * static_cast<double>(h));
        const double a = static_cast<double>(analytic[i]);
        const double denom = std::fmax(1.0, std::fmax(std::fabs(fd), std::fabs(a)));
        worst = std::fmax(worst, std::fabs(fd - a) / denom);
    }
    return worst;
}

}  // namespace cppgpt::test
