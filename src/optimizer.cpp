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

float clip_grad_norm(float* grad, int n, float max_norm) noexcept {
    ASSERT(grad != nullptr && n >= 0);
    double sumsq = 0.0;  // double so the reduction is stable across millions of terms
    for (int i = 0; i < n; ++i) {
        const double g = static_cast<double>(grad[i]);
        sumsq += g * g;
    }
    const float norm = static_cast<float>(std::sqrt(sumsq));
    if (max_norm > 0.0f && norm > max_norm) {
        const float scale = max_norm / (norm + 1e-6f);
        for (int i = 0; i < n; ++i) grad[i] *= scale;
    }
    return norm;
}

float cosine_lr(int step, float max_lr, float min_lr, int warmup, int max_steps) noexcept {
    ASSERT(warmup >= 0 && warmup < max_steps && min_lr <= max_lr);
    if (step < warmup) {  // linear warmup (warmup > 0 here, so no division by zero)
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup);
    }
    if (step >= max_steps) return min_lr;  // hold the floor after the schedule ends
    const float decay = static_cast<float>(step - warmup) / static_cast<float>(max_steps - warmup);
    const float coeff = 0.5f * (1.0f + std::cos(3.14159265358979323846f * decay));  // 1 -> 0
    return min_lr + coeff * (max_lr - min_lr);
}

}  // namespace cppgpt
