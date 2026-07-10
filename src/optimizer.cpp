#include "cppgpt/optimizer.hpp"

#include <cmath>

#include "cppgpt/core.hpp"

namespace cppgpt {
namespace {

void adamw_update_cpu(float* param, const float* grad, float* m, float* v, int n, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) noexcept {
    // Bias-correction scales, computed once per tensor per step.
    const float bc1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(t)));
    const float bc2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(t)));
    for (int i = 0; i < n; ++i) {
        const float g = grad[i];
        const float mi = beta1 * m[i] + (1.0f - beta1) * g;
        const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        const float mhat = mi * bc1;
        const float vhat = vi * bc2;
        param[i] -= lr * (mhat / (std::sqrt(vhat) + eps) + weight_decay * param[i]);
    }
}

}  // namespace

void adamw_update(float* param, const float* grad, float* m, float* v, int n, float lr,
                  float beta1, float beta2, float eps, float weight_decay, int t,
                  Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(param != nullptr && grad != nullptr && m != nullptr && v != nullptr);
    ASSERT(n >= 0 && t >= 1);
    ASSERT(beta1 >= 0.0f && beta1 < 1.0f && beta2 >= 0.0f && beta2 < 1.0f);
    adamw_update_cpu(param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, t);
}

}  // namespace cppgpt
