#include "cppgpt/ops.hpp"

#include <cstddef>

#include "cppgpt/core.hpp"

namespace cppgpt {
namespace {

void matmul_forward_cpu(float* out, const float* inp, const float* weight, const float* bias,
                        int B, int T, int C, int OC) noexcept {
    const std::size_t BT = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    const std::size_t Cz = static_cast<std::size_t>(C);
    const std::size_t OCz = static_cast<std::size_t>(OC);
    for (std::size_t bt = 0; bt < BT; ++bt) {
        const float* inp_bt = inp + bt * Cz;
        float* out_bt = out + bt * OCz;
        for (std::size_t oc = 0; oc < OCz; ++oc) {
            const float* w_oc = weight + oc * Cz;
            float acc = bias != nullptr ? bias[oc] : 0.0f;
            for (std::size_t c = 0; c < Cz; ++c) acc += inp_bt[c] * w_oc[c];
            out_bt[oc] = acc;
        }
    }
}

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias, const float* dout,
                         const float* inp, const float* weight, int B, int T, int C,
                         int OC) noexcept {
    const std::size_t BT = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    const std::size_t Cz = static_cast<std::size_t>(C);
    const std::size_t OCz = static_cast<std::size_t>(OC);
    for (std::size_t bt = 0; bt < BT; ++bt) {
        const float* dout_bt = dout + bt * OCz;
        float* dinp_bt = dinp + bt * Cz;
        const float* inp_bt = inp + bt * Cz;
        for (std::size_t oc = 0; oc < OCz; ++oc) {
            const float d = dout_bt[oc];
            const float* w_oc = weight + oc * Cz;
            float* dw_oc = dweight + oc * Cz;
            for (std::size_t c = 0; c < Cz; ++c) {
                dinp_bt[c] += d * w_oc[c];
                dw_oc[c] += d * inp_bt[c];
            }
            if (dbias != nullptr) dbias[oc] += d;
        }
    }
}

}  // namespace

void matmul_forward(float* out, const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC, Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(out != nullptr && inp != nullptr && weight != nullptr);
    ASSERT(B >= 0 && T >= 0 && C >= 0 && OC >= 0);
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
}

void matmul_backward(float* dinp, float* dweight, float* dbias, const float* dout,
                     const float* inp, const float* weight, int B, int T, int C, int OC,
                     Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(dinp != nullptr && dweight != nullptr && dout != nullptr && inp != nullptr &&
           weight != nullptr);
    ASSERT(B >= 0 && T >= 0 && C >= 0 && OC >= 0);
    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
}

}  // namespace cppgpt
