#include "cppgpt/model.hpp"

#include <cmath>
#include <cstddef>

#include "cppgpt/core.hpp"
#include "cppgpt/ops.hpp"

namespace cppgpt {
namespace {

// Parameter tensor sizes in declaration / .bin order; returns the total.
std::size_t param_sizes(const Config& c, std::size_t s[kNumParamTensors]) {
    const std::size_t V = static_cast<std::size_t>(c.vocab_size);
    const std::size_t C = static_cast<std::size_t>(c.n_embd);
    const std::size_t L = static_cast<std::size_t>(c.n_layer);
    const std::size_t maxT = static_cast<std::size_t>(c.max_seq_len);
    s[0] = V * C;          // wte
    s[1] = maxT * C;       // wpe
    s[2] = L * C;          // ln1w
    s[3] = L * C;          // ln1b
    s[4] = L * 3 * C * C;  // qkvw
    s[5] = L * 3 * C;      // qkvb
    s[6] = L * C * C;      // attprojw
    s[7] = L * C;          // attprojb
    s[8] = L * C;          // ln2w
    s[9] = L * C;          // ln2b
    s[10] = L * 4 * C * C;  // fcw
    s[11] = L * 4 * C;      // fcb
    s[12] = L * C * 4 * C;  // fcprojw
    s[13] = L * C;          // fcprojb
    s[14] = C;              // lnfw
    s[15] = C;              // lnfb
    std::size_t tot = 0;
    for (int i = 0; i < kNumParamTensors; ++i) tot += s[i];
    return tot;
}

void point_params(ParamTensors& p, float* base, const std::size_t s[kNumParamTensors]) {
    float* ptr[kNumParamTensors];
    std::size_t off = 0;
    for (int i = 0; i < kNumParamTensors; ++i) {
        ptr[i] = base + off;
        off += s[i];
    }
    p.wte = ptr[0];   p.wpe = ptr[1];   p.ln1w = ptr[2];     p.ln1b = ptr[3];
    p.qkvw = ptr[4];  p.qkvb = ptr[5];  p.attprojw = ptr[6]; p.attprojb = ptr[7];
    p.ln2w = ptr[8];  p.ln2b = ptr[9];  p.fcw = ptr[10];     p.fcb = ptr[11];
    p.fcprojw = ptr[12]; p.fcprojb = ptr[13]; p.lnfw = ptr[14]; p.lnfb = ptr[15];
}

// Activation tensor sizes in declaration order (for fixed B, T); returns total.
std::size_t act_sizes(const Config& c, int B, int T, std::size_t s[kNumActTensors]) {
    const std::size_t Bz = static_cast<std::size_t>(B);
    const std::size_t Tz = static_cast<std::size_t>(T);
    const std::size_t C = static_cast<std::size_t>(c.n_embd);
    const std::size_t L = static_cast<std::size_t>(c.n_layer);
    const std::size_t NH = static_cast<std::size_t>(c.n_head);
    const std::size_t V = static_cast<std::size_t>(c.vocab_size);
    const std::size_t BTC = Bz * Tz * C;
    const std::size_t BT = Bz * Tz;
    const std::size_t ATT = Bz * NH * Tz * Tz;
    s[0] = BTC;          // encoded
    s[1] = L * BTC;      // ln1
    s[2] = L * BT;       // ln1_mean
    s[3] = L * BT;       // ln1_rstd
    s[4] = L * BTC * 3;  // qkv
    s[5] = L * BTC;      // atty
    s[6] = L * ATT;      // preatt
    s[7] = L * ATT;      // att
    s[8] = L * BTC;      // attproj
    s[9] = L * BTC;      // residual2
    s[10] = L * BTC;     // ln2
    s[11] = L * BT;      // ln2_mean
    s[12] = L * BT;      // ln2_rstd
    s[13] = L * BTC * 4;  // fch
    s[14] = L * BTC * 4;  // fch_gelu
    s[15] = L * BTC;     // fcproj
    s[16] = L * BTC;     // residual3
    s[17] = BTC;         // lnf
    s[18] = BT;          // lnf_mean
    s[19] = BT;          // lnf_rstd
    s[20] = BT * V;      // logits
    s[21] = BT * V;      // probs
    s[22] = BT;          // losses
    std::size_t tot = 0;
    for (int i = 0; i < kNumActTensors; ++i) tot += s[i];
    return tot;
}

void point_acts(ActTensors& a, float* base, const std::size_t s[kNumActTensors]) {
    float* ptr[kNumActTensors];
    std::size_t off = 0;
    for (int i = 0; i < kNumActTensors; ++i) {
        ptr[i] = base + off;
        off += s[i];
    }
    a.encoded = ptr[0];   a.ln1 = ptr[1];      a.ln1_mean = ptr[2];  a.ln1_rstd = ptr[3];
    a.qkv = ptr[4];       a.atty = ptr[5];     a.preatt = ptr[6];    a.att = ptr[7];
    a.attproj = ptr[8];   a.residual2 = ptr[9]; a.ln2 = ptr[10];     a.ln2_mean = ptr[11];
    a.ln2_rstd = ptr[12]; a.fch = ptr[13];     a.fch_gelu = ptr[14]; a.fcproj = ptr[15];
    a.residual3 = ptr[16]; a.lnf = ptr[17];    a.lnf_mean = ptr[18]; a.lnf_rstd = ptr[19];
    a.logits = ptr[20];   a.probs = ptr[21];   a.losses = ptr[22];
}

void fill_normal(float* p, std::size_t n, float std, Generator& gen) {
    for (std::size_t i = 0; i < n; ++i) p[i] = std * gen.normal();
}
void fill_const(float* p, std::size_t n, float v) {
    for (std::size_t i = 0; i < n; ++i) p[i] = v;
}

}  // namespace

GPT2::GPT2(const Config& cfg, int B, int T)
    : cfg_(cfg),
      B_(B),
      T_(T),
      param_store_([&] {
          std::size_t s[kNumParamTensors];
          return Storage(param_sizes(cfg, s));
      }()),
      act_store_([&] {
          std::size_t s[kNumActTensors];
          return Storage(act_sizes(cfg, B, T, s));
      }()) {
    ASSERT(cfg.n_embd > 0 && cfg.n_head > 0 && cfg.n_embd % cfg.n_head == 0);
    ASSERT(cfg.n_layer >= 0 && cfg.vocab_size > 0 && T <= cfg.max_seq_len);
    std::size_t ps[kNumParamTensors];
    const std::size_t ptot = param_sizes(cfg, ps);
    point_params(params_, param_store_.alloc(ptot), ps);
    std::size_t as[kNumActTensors];
    const std::size_t atot = act_sizes(cfg, B, T, as);
    point_acts(acts_, act_store_.alloc(atot), as);
}

void GPT2::init_weights(Generator& gen) {
    const int C = cfg_.n_embd, L = cfg_.n_layer, V = cfg_.vocab_size, maxT = cfg_.max_seq_len;
    const auto Cz = static_cast<std::size_t>(C);
    const auto Lz = static_cast<std::size_t>(L);
    const float w_std = 0.02f;
    const float proj_std = 0.02f / std::sqrt(2.0f * static_cast<float>(L));
    const ParamTensors& p = params_;
    fill_normal(p.wte, static_cast<std::size_t>(V) * Cz, w_std, gen);
    fill_normal(p.wpe, static_cast<std::size_t>(maxT) * Cz, w_std, gen);
    fill_const(p.ln1w, Lz * Cz, 1.0f);
    fill_const(p.ln1b, Lz * Cz, 0.0f);
    fill_normal(p.qkvw, Lz * 3 * Cz * Cz, w_std, gen);
    fill_const(p.qkvb, Lz * 3 * Cz, 0.0f);
    fill_normal(p.attprojw, Lz * Cz * Cz, proj_std, gen);
    fill_const(p.attprojb, Lz * Cz, 0.0f);
    fill_const(p.ln2w, Lz * Cz, 1.0f);
    fill_const(p.ln2b, Lz * Cz, 0.0f);
    fill_normal(p.fcw, Lz * 4 * Cz * Cz, w_std, gen);
    fill_const(p.fcb, Lz * 4 * Cz, 0.0f);
    fill_normal(p.fcprojw, Lz * Cz * 4 * Cz, proj_std, gen);
    fill_const(p.fcprojb, Lz * Cz, 0.0f);
    fill_const(p.lnfw, Cz, 1.0f);
    fill_const(p.lnfb, Cz, 0.0f);
}

void GPT2::forward(const int* tokens, const int* targets, int B, int T) {
    ASSERT(B == B_ && T == T_);
    const int C = cfg_.n_embd, L = cfg_.n_layer, NH = cfg_.n_head, V = cfg_.vocab_size;
    const auto BTC = static_cast<std::size_t>(B) * T * C;
    const auto BT = static_cast<std::size_t>(B) * T;
    const auto ATT = static_cast<std::size_t>(B) * NH * T * T;
    const auto Cz = static_cast<std::size_t>(C);
    const ParamTensors& p = params_;
    const ActTensors& a = acts_;

    encoder_forward(a.encoded, tokens, p.wte, p.wpe, B, T, C, V, Device::CPU);

    for (int l = 0; l < L; ++l) {
        const auto ll = static_cast<std::size_t>(l);
        // Input residual: encoded for layer 0, else the previous block output.
        const float* res = (l == 0) ? a.encoded : a.residual3 + (ll - 1) * BTC;
        // Per-layer parameter slices.
        const float* ln1w = p.ln1w + ll * Cz;
        const float* ln1b = p.ln1b + ll * Cz;
        const float* qkvw = p.qkvw + ll * 3 * Cz * Cz;
        const float* qkvb = p.qkvb + ll * 3 * Cz;
        const float* aprojw = p.attprojw + ll * Cz * Cz;
        const float* aprojb = p.attprojb + ll * Cz;
        const float* ln2w = p.ln2w + ll * Cz;
        const float* ln2b = p.ln2b + ll * Cz;
        const float* fcw = p.fcw + ll * 4 * Cz * Cz;
        const float* fcb = p.fcb + ll * 4 * Cz;
        const float* fcprojw = p.fcprojw + ll * Cz * 4 * Cz;
        const float* fcprojb = p.fcprojb + ll * Cz;
        // Per-layer activation slices.
        float* ln1 = a.ln1 + ll * BTC;
        float* ln1_mean = a.ln1_mean + ll * BT;
        float* ln1_rstd = a.ln1_rstd + ll * BT;
        float* qkv = a.qkv + ll * BTC * 3;
        float* atty = a.atty + ll * BTC;
        float* preatt = a.preatt + ll * ATT;
        float* att = a.att + ll * ATT;
        float* attproj = a.attproj + ll * BTC;
        float* residual2 = a.residual2 + ll * BTC;
        float* ln2 = a.ln2 + ll * BTC;
        float* ln2_mean = a.ln2_mean + ll * BT;
        float* ln2_rstd = a.ln2_rstd + ll * BT;
        float* fch = a.fch + ll * BTC * 4;
        float* fch_gelu = a.fch_gelu + ll * BTC * 4;
        float* fcproj = a.fcproj + ll * BTC;
        float* residual3 = a.residual3 + ll * BTC;

        layernorm_forward(ln1, ln1_mean, ln1_rstd, res, ln1w, ln1b, B, T, C, Device::CPU);
        matmul_forward(qkv, ln1, qkvw, qkvb, B, T, C, 3 * C, Device::CPU);
        attention_forward(atty, preatt, att, qkv, B, T, C, NH, Device::CPU);
        matmul_forward(attproj, atty, aprojw, aprojb, B, T, C, C, Device::CPU);
        residual_forward(residual2, res, attproj, static_cast<int>(BTC), Device::CPU);
        layernorm_forward(ln2, ln2_mean, ln2_rstd, residual2, ln2w, ln2b, B, T, C, Device::CPU);
        matmul_forward(fch, ln2, fcw, fcb, B, T, C, 4 * C, Device::CPU);
        gelu_forward(fch_gelu, fch, static_cast<int>(BTC) * 4, Device::CPU);
        matmul_forward(fcproj, fch_gelu, fcprojw, fcprojb, B, T, 4 * C, C, Device::CPU);
        residual_forward(residual3, residual2, fcproj, static_cast<int>(BTC), Device::CPU);
    }

    const float* last = (L == 0) ? a.encoded : a.residual3 + static_cast<std::size_t>(L - 1) * BTC;
    layernorm_forward(a.lnf, a.lnf_mean, a.lnf_rstd, last, p.lnfw, p.lnfb, B, T, C, Device::CPU);
    matmul_forward(a.logits, a.lnf, p.wte, nullptr, B, T, C, V, Device::CPU);  // tied classifier
    for (std::size_t bt = 0; bt < BT; ++bt)
        softmax_forward(a.probs + bt * static_cast<std::size_t>(V),
                        a.logits + bt * static_cast<std::size_t>(V), V, Device::CPU);
    cross_entropy_forward(a.losses, a.probs, targets, B, T, V, Device::CPU);

    double sum = 0.0;
    for (std::size_t bt = 0; bt < BT; ++bt) sum += a.losses[bt];
    mean_loss_ = static_cast<float>(sum / static_cast<double>(BT));
}

}  // namespace cppgpt
