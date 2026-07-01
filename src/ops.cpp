#include "cppgpt/ops.hpp"

#include <cmath>
#include <cstddef>

#include "cppgpt/core.hpp"

namespace cppgpt {
namespace {

// Canonical GPT-2 gelu_new constants: √(2/π) and the cubic coefficient.
constexpr float kGeluCoef = 0.7978845608028654f;
constexpr float kGeluCubic = 0.044715f;

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

void gelu_forward_cpu(float* out, const float* inp, int N) noexcept {
    const std::size_t n = static_cast<std::size_t>(N);
    for (std::size_t i = 0; i < n; ++i) {
        const float x = inp[i];
        const float inner = kGeluCoef * (x + kGeluCubic * x * x * x);
        out[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void gelu_backward_cpu(float* dinp, const float* inp, const float* dout, int N) noexcept {
    const std::size_t n = static_cast<std::size_t>(N);
    for (std::size_t i = 0; i < n; ++i) {
        const float x = inp[i];
        const float inner = kGeluCoef * (x + kGeluCubic * x * x * x);
        const float th = std::tanh(inner);
        const float dinner = kGeluCoef * (1.0f + 3.0f * kGeluCubic * x * x);  // d(inner)/dx
        // d/dx [0.5·x·(1+tanh(inner))] = 0.5·(1+tanh) + 0.5·x·(1−tanh²)·inner'
        const float local = 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * dinner;
        dinp[i] += local * dout[i];
    }
}

void residual_forward_cpu(float* out, const float* a, const float* b, int N) noexcept {
    const std::size_t n = static_cast<std::size_t>(N);
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void residual_backward_cpu(float* da, float* db, const float* dout, int N) noexcept {
    const std::size_t n = static_cast<std::size_t>(N);
    for (std::size_t i = 0; i < n; ++i) {
        da[i] += dout[i];
        db[i] += dout[i];
    }
}

// Canonical GPT-2 LayerNorm epsilon (protocol-fixed, not a tunable).
constexpr float kLayerNormEps = 1e-5f;

void layernorm_forward_cpu(float* out, float* mean, float* rstd, const float* inp,
                           const float* weight, const float* bias, int B, int T, int C) noexcept {
    const std::size_t BT = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    const std::size_t Cz = static_cast<std::size_t>(C);
    const float inv_c = 1.0f / static_cast<float>(C);
    for (std::size_t bt = 0; bt < BT; ++bt) {
        const float* x = inp + bt * Cz;
        float m = 0.0f;
        for (std::size_t c = 0; c < Cz; ++c) m += x[c];
        m *= inv_c;
        float v = 0.0f;
        for (std::size_t c = 0; c < Cz; ++c) {
            const float d = x[c] - m;
            v += d * d;
        }
        v *= inv_c;
        const float s = 1.0f / std::sqrt(v + kLayerNormEps);
        float* out_bt = out + bt * Cz;
        for (std::size_t c = 0; c < Cz; ++c) out_bt[c] = (x[c] - m) * s * weight[c] + bias[c];
        mean[bt] = m;
        rstd[bt] = s;
    }
}

void layernorm_backward_cpu(float* dinp, float* dweight, float* dbias, const float* dout,
                            const float* inp, const float* weight, const float* mean,
                            const float* rstd, int B, int T, int C) noexcept {
    const std::size_t BT = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    const std::size_t Cz = static_cast<std::size_t>(C);
    const float inv_c = 1.0f / static_cast<float>(C);
    for (std::size_t bt = 0; bt < BT; ++bt) {
        const float* dout_bt = dout + bt * Cz;
        const float* x = inp + bt * Cz;
        float* dinp_bt = dinp + bt * Cz;
        const float m = mean[bt];
        const float s = rstd[bt];
        // Two reductions over the row: mean of dnorm and of dnorm·norm.
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (std::size_t c = 0; c < Cz; ++c) {
            const float norm = (x[c] - m) * s;
            const float dnorm = weight[c] * dout_bt[c];
            dnorm_mean += dnorm;
            dnorm_norm_mean += dnorm * norm;
        }
        dnorm_mean *= inv_c;
        dnorm_norm_mean *= inv_c;
        for (std::size_t c = 0; c < Cz; ++c) {
            const float norm = (x[c] - m) * s;
            const float dnorm = weight[c] * dout_bt[c];
            dbias[c] += dout_bt[c];
            dweight[c] += norm * dout_bt[c];
            dinp_bt[c] += (dnorm - dnorm_mean - norm * dnorm_norm_mean) * s;
        }
    }
}

void softmax_forward_cpu(float* out, const float* inp, int N) noexcept {
    const std::size_t n = static_cast<std::size_t>(N);
    float maxv = inp[0];
    for (std::size_t i = 1; i < n; ++i) maxv = std::fmax(maxv, inp[i]);
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float e = std::exp(inp[i] - maxv);
        out[i] = e;
        sum += e;
    }
    const float inv = 1.0f / sum;
    for (std::size_t i = 0; i < n; ++i) out[i] *= inv;
}

void softmax_backward_cpu(float* dinp, const float* dout, const float* out, int N) noexcept {
    const std::size_t n = static_cast<std::size_t>(N);
    // dinp_i = out_i·(dout_i − Σ_j dout_j·out_j); the subtracted term is the
    // projection of dout onto the softmax output.
    float d = 0.0f;
    for (std::size_t i = 0; i < n; ++i) d += dout[i] * out[i];
    for (std::size_t i = 0; i < n; ++i) dinp[i] += out[i] * (dout[i] - d);
}

void attention_forward_cpu(float* out, float* preatt, float* att, const float* inp, int B, int T,
                           int C, int NH) noexcept {
    const int hs = C / NH;  // head size
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
    const std::size_t C3 = static_cast<std::size_t>(3 * C);
    const std::size_t Cz = static_cast<std::size_t>(C);
    const std::size_t Tz = static_cast<std::size_t>(T);
    const std::size_t hz = static_cast<std::size_t>(hs);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                const float* q = inp + (static_cast<std::size_t>(b) * Tz + t) * C3 + h * hz;
                float* preatt_row = preatt + ((static_cast<std::size_t>(b) * NH + h) * Tz + t) * Tz;
                float* att_row = att + ((static_cast<std::size_t>(b) * NH + h) * Tz + t) * Tz;
                for (int t2 = 0; t2 <= t; ++t2) {  // causal: only keys at t2 <= t
                    const float* k = inp + (static_cast<std::size_t>(b) * Tz + t2) * C3 + Cz + h * hz;
                    float s = 0.0f;
                    for (std::size_t i = 0; i < hz; ++i) s += q[i] * k[i];
                    preatt_row[t2] = s * scale;
                }
                softmax_forward_cpu(att_row, preatt_row, t + 1);
                float* out_row = out + (static_cast<std::size_t>(b) * Tz + t) * Cz + h * hz;
                for (std::size_t i = 0; i < hz; ++i) out_row[i] = 0.0f;
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* v =
                        inp + (static_cast<std::size_t>(b) * Tz + t2) * C3 + 2 * Cz + h * hz;
                    const float a = att_row[t2];
                    for (std::size_t i = 0; i < hz; ++i) out_row[i] += a * v[i];
                }
            }
        }
    }
}

void attention_backward_cpu(float* dinp, float* datt, float* dpreatt, const float* dout,
                            const float* inp, const float* att, int B, int T, int C,
                            int NH) noexcept {
    const int hs = C / NH;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));
    const std::size_t C3 = static_cast<std::size_t>(3 * C);
    const std::size_t Cz = static_cast<std::size_t>(C);
    const std::size_t Tz = static_cast<std::size_t>(T);
    const std::size_t hz = static_cast<std::size_t>(hs);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < NH; ++h) {
                const std::size_t row = ((static_cast<std::size_t>(b) * NH + h) * Tz + t) * Tz;
                const float* att_row = att + row;
                float* datt_row = datt + row;
                float* dpreatt_row = dpreatt + row;
                const float* dout_row = dout + (static_cast<std::size_t>(b) * Tz + t) * Cz + h * hz;
                const float* q = inp + (static_cast<std::size_t>(b) * Tz + t) * C3 + h * hz;
                float* dq = dinp + (static_cast<std::size_t>(b) * Tz + t) * C3 + h * hz;
                // 1. datt[t2] = Σ_i v_{t2}[i]·dout[i];  dv_{t2} += att·dout
                for (int t2 = 0; t2 <= t; ++t2) {
                    const std::size_t base = (static_cast<std::size_t>(b) * Tz + t2) * C3 + h * hz;
                    const float* v = inp + base + 2 * Cz;
                    float* dv = dinp + base + 2 * Cz;
                    const float a = att_row[t2];
                    float d = 0.0f;
                    for (std::size_t i = 0; i < hz; ++i) {
                        d += v[i] * dout_row[i];
                        dv[i] += a * dout_row[i];
                    }
                    datt_row[t2] = d;
                }
                // 2. softmax backward (zero the causal row, then accumulate the Jacobian).
                for (int t2 = 0; t2 <= t; ++t2) dpreatt_row[t2] = 0.0f;
                softmax_backward_cpu(dpreatt_row, datt_row, att_row, t + 1);
                // 3. scores backward: preatt[t2] = (q·k_{t2})·scale.
                for (int t2 = 0; t2 <= t; ++t2) {
                    const std::size_t base = (static_cast<std::size_t>(b) * Tz + t2) * C3 + h * hz;
                    const float* k = inp + base + Cz;
                    float* dk = dinp + base + Cz;
                    const float g = dpreatt_row[t2] * scale;
                    for (std::size_t i = 0; i < hz; ++i) {
                        dq[i] += g * k[i];
                        dk[i] += g * q[i];
                    }
                }
            }
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

void softmax_forward(float* out, const float* inp, int N, Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(out != nullptr && inp != nullptr);
    ASSERT(N > 0);
    softmax_forward_cpu(out, inp, N);
}

void softmax_backward(float* dinp, const float* dout, const float* out, int N,
                      Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(dinp != nullptr && dout != nullptr && out != nullptr);
    ASSERT(N > 0);
    softmax_backward_cpu(dinp, dout, out, N);
}

void attention_forward(float* out, float* preatt, float* att, const float* inp, int B, int T,
                       int C, int NH, Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(out != nullptr && preatt != nullptr && att != nullptr && inp != nullptr);
    ASSERT(B >= 0 && T >= 0 && C > 0 && NH > 0 && C % NH == 0);
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
}

void attention_backward(float* dinp, float* datt, float* dpreatt, const float* dout,
                        const float* inp, const float* att, int B, int T, int C, int NH,
                        Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(dinp != nullptr && datt != nullptr && dpreatt != nullptr && dout != nullptr &&
           inp != nullptr && att != nullptr);
    ASSERT(B >= 0 && T >= 0 && C > 0 && NH > 0 && C % NH == 0);
    attention_backward_cpu(dinp, datt, dpreatt, dout, inp, att, B, T, C, NH);
}

void gelu_forward(float* out, const float* inp, int N, Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(out != nullptr && inp != nullptr);
    ASSERT(N >= 0);
    gelu_forward_cpu(out, inp, N);
}

void gelu_backward(float* dinp, const float* inp, const float* dout, int N, Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(dinp != nullptr && inp != nullptr && dout != nullptr);
    ASSERT(N >= 0);
    gelu_backward_cpu(dinp, inp, dout, N);
}

void residual_forward(float* out, const float* a, const float* b, int N, Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(out != nullptr && a != nullptr && b != nullptr);
    ASSERT(N >= 0);
    residual_forward_cpu(out, a, b, N);
}

void residual_backward(float* da, float* db, const float* dout, int N, Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(da != nullptr && db != nullptr && dout != nullptr);
    ASSERT(N >= 0);
    residual_backward_cpu(da, db, dout, N);
}

void layernorm_forward(float* out, float* mean, float* rstd, const float* inp,
                       const float* weight, const float* bias, int B, int T, int C,
                       Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(out != nullptr && mean != nullptr && rstd != nullptr && inp != nullptr &&
           weight != nullptr && bias != nullptr);
    ASSERT(B >= 0 && T >= 0 && C > 0);
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);
}

void layernorm_backward(float* dinp, float* dweight, float* dbias, const float* dout,
                        const float* inp, const float* weight, const float* mean,
                        const float* rstd, int B, int T, int C, Device dev) noexcept {
    ASSERT(dev == Device::CPU);
    ASSERT(dinp != nullptr && dweight != nullptr && dbias != nullptr && dout != nullptr &&
           inp != nullptr && weight != nullptr && mean != nullptr && rstd != nullptr);
    ASSERT(B >= 0 && T >= 0 && C > 0);
    layernorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
}

}  // namespace cppgpt
