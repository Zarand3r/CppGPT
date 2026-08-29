#include "cppgpt/interpret.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>

#include "cppgpt/core.hpp"
#include "cppgpt/ops.hpp"

namespace cppgpt {

void logit_lens(const GPT2& model, int layer, float* out_logits, float* scratch) noexcept {
    const Config& cfg = model.config();
    ASSERT(layer >= 0 && layer < cfg.n_layer);
    ASSERT(out_logits != nullptr && scratch != nullptr);

    const int B = model.batch(), T = model.seq_len();
    const int C = cfg.n_embd, V = cfg.vocab_size;
    const ActTensors& a = model.acts();
    const ParamTensors& p = model.params();

    // residual3 is [L, B, T, C]; take this layer's block.
    const float* residual = layer_slice(a.residual3, layer, B, T, C);

    // mean/rstd go in CALLER scratch, not the model's lnf_mean/lnf_rstd: writing
    // those would silently corrupt state a caller may still be inspecting, which
    // would make the "read-only" contract above a lie.
    const auto bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    float* norm = scratch;
    float* mean = scratch + bt * static_cast<std::size_t>(C);
    float* rstd = mean + bt;
    layernorm_forward(norm, mean, rstd, residual, p.lnfw, p.lnfb, B, T, C);
    matmul_forward(out_logits, norm, p.wte, nullptr, B, T, C, V);  // tied classifier
}

void direct_logit_attribution(const GPT2& model, int pos, int token, float* out_heads,
                              float* out_mlps, float* out_embed, float* out_bias,
                              float* scratch) noexcept {
    const Config& cfg = model.config();
    const int B = model.batch(), T = model.seq_len();
    const int C = cfg.n_embd, L = cfg.n_layer, NH = cfg.n_head;
    ASSERT(pos >= 0 && pos < T);
    ASSERT(token >= 0 && token < cfg.vocab_size);
    ASSERT(out_heads != nullptr && out_mlps != nullptr && out_embed != nullptr);
    ASSERT(out_bias != nullptr && scratch != nullptr);

    const ActTensors& a = model.acts();
    const ParamTensors& p = model.params();
    const auto Cz = static_cast<std::size_t>(C);
    const auto off = static_cast<std::size_t>(pos) * Cz;  // batch 0, position `pos`

    // The layernorm scale is FROZEN at what the real forward produced. That is
    // what makes the decomposition exact rather than approximate: with rstd
    // fixed, the final layernorm is affine, and an affine map of a sum is the
    // sum of the maps.
    const auto rstd = static_cast<double>(a.lnf_rstd[static_cast<std::size_t>(pos)]);
    const float* wte_row = p.wte + static_cast<std::size_t>(token) * Cz;

    // u = lnfw ⊙ wte[token]: the direction in residual space that this token's
    // logit reads. Every component is scored by its projection onto u, centred
    // exactly as the layernorm centres.
    float* u = scratch;
    float* wu = scratch + Cz;       // per input channel: Σ_o attprojw[o,i]·u[o]
    float* ws = scratch + 2 * Cz;   // per input channel: Σ_o attprojw[o,i]
    float* bsum = scratch + 3 * Cz; // Σ_l attprojb[l]
    double u_sum = 0.0;
    for (int c = 0; c < C; ++c) {
        u[c] = p.lnfw[c] * wte_row[c];
        u_sum += static_cast<double>(u[c]);
        bsum[c] = 0.0f;
    }

    // attrib(x) = rstd · Σ_c (x[c] − mean(x))·u[c]. Linear in x, which is exactly
    // why the parts sum to the whole.
    const auto attrib = [&](const float* x) {
        double dot = 0.0, sum = 0.0;
        for (int c = 0; c < C; ++c) {
            dot += static_cast<double>(x[c]) * static_cast<double>(u[c]);
            sum += static_cast<double>(x[c]);
        }
        return static_cast<float>(rstd * (dot - (sum / C) * u_sum));
    };

    *out_embed = attrib(a.encoded + off);

    const auto hs = static_cast<std::size_t>(C / NH);
    for (int l = 0; l < L; ++l) {
        const auto lz = static_cast<std::size_t>(l);
        const float* atty = layer_slice(a.atty, l, B, T, C) + off;
        const float* w = p.attprojw + lz * Cz * Cz;

        // Fold u through the projection once per layer (C² work), so each head
        // costs only hs. Head h reaches the residual through input channels
        // [h·hs, (h+1)·hs) — the column block save_and_ablate zeroes.
        for (std::size_t i = 0; i < Cz; ++i) {
            double du = 0.0, ds = 0.0;
            for (std::size_t o = 0; o < Cz; ++o) {
                const double wo = static_cast<double>(w[o * Cz + i]);
                du += wo * static_cast<double>(u[o]);
                ds += wo;
            }
            wu[i] = static_cast<float>(du);
            ws[i] = static_cast<float>(ds);
        }
        for (int h = 0; h < NH; ++h) {
            const auto lo = static_cast<std::size_t>(h) * hs;
            double dot = 0.0, sum = 0.0;
            for (std::size_t i = 0; i < hs; ++i) {
                const double av = static_cast<double>(atty[lo + i]);
                dot += av * static_cast<double>(wu[lo + i]);
                sum += av * static_cast<double>(ws[lo + i]);
            }
            out_heads[lz * static_cast<std::size_t>(NH) + static_cast<std::size_t>(h)] =
                static_cast<float>(rstd * (dot - (sum / C) * u_sum));
        }

        // The MLP's residual write is fcproj, bias included (matmul_forward adds
        // it), so this row needs no separate bias accounting.
        out_mlps[lz] = attrib(layer_slice(a.fcproj, l, B, T, C) + off);

        // attprojb is added to attproj AFTER the heads, so it belongs to no head.
        const float* ab = p.attprojb + lz * Cz;
        for (int c = 0; c < C; ++c) bsum[c] += ab[c];
    }

    // Two bias terms with different treatment, because they enter at different
    // points: the attention projection biases are part of the residual (centred
    // and scaled), while lnfb is added AFTER normalisation (neither).
    double lnfb_term = 0.0;
    for (int c = 0; c < C; ++c)
        lnfb_term += static_cast<double>(p.lnfb[c]) * static_cast<double>(wte_row[c]);
    *out_bias = attrib(bsum) + static_cast<float>(lnfb_term);
}

void softmax_into(float* out, const float* logits, int V) noexcept {
    ASSERT(out != nullptr && logits != nullptr && V > 0);
    float mx = logits[0];
    for (int i = 1; i < V; ++i) mx = std::fmax(mx, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < V; ++i) {
        const double e = std::exp(static_cast<double>(logits[i] - mx));
        out[i] = static_cast<float>(e);
        sum += e;
    }
    const double inv = 1.0 / sum;
    for (int i = 0; i < V; ++i) out[i] = static_cast<float>(static_cast<double>(out[i]) * inv);
}

double kl_divergence(const float* p, const float* q, int V) noexcept {
    ASSERT(p != nullptr && q != nullptr && V > 0);
    double d = 0.0;
    for (int i = 0; i < V; ++i) {
        const double pi = p[i];
        if (pi > 1e-12) d += pi * std::log(pi / std::max(static_cast<double>(q[i]), 1e-30));
    }
    return d;
}

std::size_t ablation_scratch(const Config& cfg) noexcept {
    // The MLP slice is the largest: fcprojw [C, 4C] plus fcprojb [C].
    const auto C = static_cast<std::size_t>(cfg.n_embd);
    return 4 * C * C + C;
}

namespace {

// The contiguous weight+bias pair an Mlp/AttnBlock ablation clears. Head is
// strided and handled separately.
struct Slice {
    float* w;
    std::size_t w_n;
    float* b;
    std::size_t b_n;
};

Slice block_slice(ParamTensors& p, Ablation kind, int layer, std::size_t C) noexcept {
    const auto lz = static_cast<std::size_t>(layer);
    if (kind == Ablation::Mlp) return {p.fcprojw + lz * C * 4 * C, 4 * C * C, p.fcprojb + lz * C, C};
    return {p.attprojw + lz * C * C, C * C, p.attprojb + lz * C, C};
}

}  // namespace

void save_and_ablate(GPT2& model, Ablation kind, int layer, int head, float* saved) noexcept {
    const Config& cfg = model.config();
    ASSERT(saved != nullptr);
    ASSERT(layer >= 0 && layer < cfg.n_layer);
    const auto C = static_cast<std::size_t>(cfg.n_embd);
    ParamTensors& p = model.params();

    if (kind == Ablation::Head) {
        ASSERT(head >= 0 && head < cfg.n_head);
        const auto hs = C / static_cast<std::size_t>(cfg.n_head);
        const auto lo = static_cast<std::size_t>(head) * hs;
        float* w = p.attprojw + static_cast<std::size_t>(layer) * C * C;
        // out = inp @ weightᵀ with weight [C_out, C_in]: a head owns a COLUMN
        // block of every row, not a contiguous span.
        for (std::size_t o = 0; o < C; ++o) {
            for (std::size_t i = 0; i < hs; ++i) {
                saved[o * hs + i] = w[o * C + lo + i];
                w[o * C + lo + i] = 0.0f;
            }
        }
        return;
    }
    const Slice s = block_slice(p, kind, layer, C);
    for (std::size_t i = 0; i < s.w_n; ++i) {
        saved[i] = s.w[i];
        s.w[i] = 0.0f;
    }
    for (std::size_t i = 0; i < s.b_n; ++i) {
        saved[s.w_n + i] = s.b[i];
        s.b[i] = 0.0f;
    }
}

void restore_ablation(GPT2& model, Ablation kind, int layer, int head,
                      const float* saved) noexcept {
    const Config& cfg = model.config();
    ASSERT(saved != nullptr);
    ASSERT(layer >= 0 && layer < cfg.n_layer);
    const auto C = static_cast<std::size_t>(cfg.n_embd);
    ParamTensors& p = model.params();

    if (kind == Ablation::Head) {
        ASSERT(head >= 0 && head < cfg.n_head);
        const auto hs = C / static_cast<std::size_t>(cfg.n_head);
        const auto lo = static_cast<std::size_t>(head) * hs;
        float* w = p.attprojw + static_cast<std::size_t>(layer) * C * C;
        for (std::size_t o = 0; o < C; ++o)
            for (std::size_t i = 0; i < hs; ++i) w[o * C + lo + i] = saved[o * hs + i];
        return;
    }
    const Slice s = block_slice(p, kind, layer, C);
    for (std::size_t i = 0; i < s.w_n; ++i) s.w[i] = saved[i];
    for (std::size_t i = 0; i < s.b_n; ++i) s.b[i] = saved[s.w_n + i];
}

void capture_site(const GPT2& model, PatchSite site, int layer, int head, float* out) noexcept {
    const Config& cfg = model.config();
    const int B = model.batch(), T = model.seq_len(), C = cfg.n_embd, NH = cfg.n_head;
    ASSERT_MSG(layer >= 0 && layer < cfg.n_layer, "capture_site: layer out of range");
    ASSERT(out != nullptr);

    const ActTensors& a = model.acts();
    // Each site is the layer's [B,T,C] slice of the activation the patch would
    // overwrite. Same tensors, same strides as apply_patch's destinations --
    // derived through layer_slice rather than re-computed, which is the rule
    // that exists because this stride was mis-derived three times.
    const float* base = (site == PatchSite::MlpOut)        ? a.fcproj
                        : (site == PatchSite::AttnBlockOut) ? a.attproj
                                                            : a.atty;
    detail::capture_from(site, head, layer_slice(base, layer, B, T, C), out, B, T, C, NH);
}

Component component_at(const Config& cfg, int index) noexcept {
    ASSERT_MSG(index >= 0 && index < n_components(cfg), "component_at: index out of range");
    const int per_layer = cfg.n_head + 2;
    const int layer = index / per_layer;
    const int slot = index % per_layer;
    if (slot < cfg.n_head) return {PatchSite::HeadOut, layer, slot};
    if (slot == cfg.n_head) return {PatchSite::MlpOut, layer, -1};
    return {PatchSite::AttnBlockOut, layer, -1};
}

void component_label(const Config& cfg, int index, char* out, std::size_t cap) noexcept {
    const Component c = component_at(cfg, index);
    ASSERT(out != nullptr && cap >= 16);
    if (c.site == PatchSite::HeadOut)
        (void)std::snprintf(out, cap, "L%dH%d", c.layer, c.head);
    else if (c.site == PatchSite::MlpOut)
        (void)std::snprintf(out, cap, "L%dmlp", c.layer);
    else
        (void)std::snprintf(out, cap, "L%dattn", c.layer);
}

void coax_sweep(GPT2& model, const int* tokens, int pos, const float* replacements,
                std::size_t stride, double* out_marginal, double* out_growth) noexcept {
    const Config& cfg = model.config();
    const int n = n_components(cfg), V = cfg.vocab_size;
    ASSERT(replacements != nullptr && out_marginal != nullptr && out_growth != nullptr);
    ASSERT_MSG(pos >= 0 && pos < model.seq_len(), "coax_sweep: pos out of range");

    const auto off = static_cast<std::size_t>(pos) * static_cast<std::size_t>(V);
    std::vector<float> p_clean(static_cast<std::size_t>(V)), p_i(static_cast<std::size_t>(V)),
        p_j(static_cast<std::size_t>(V)), p_ij(static_cast<std::size_t>(V));

    const auto patch_for = [&](int idx) {
        const Component c = component_at(cfg, idx);
        return Patch{c.site, c.layer, c.head,
                     replacements + static_cast<std::size_t>(idx) * stride};
    };

    model.forward(tokens, nullptr);
    softmax_into(p_clean.data(), model.acts().logits + off, V);

    // Marginal effects first: E(j) = KL(clean || ablate j). These are also the
    // |S| = 0 case of the conditional score, which is what makes the reduction
    // testable rather than asserted.
    for (int j = 0; j < n; ++j) {
        const Patch pj = patch_for(j);
        model.forward(tokens, nullptr, -1, &pj, 1);
        softmax_into(p_j.data(), model.acts().logits + off, V);
        out_marginal[j] = kl_divergence(p_clean.data(), p_j.data(), V);
    }

    for (int i = 0; i < n; ++i) {
        // The reference for every pair in this row is the i-ablated output, NOT
        // the clean one: the question is how much j moves the model that has
        // already lost i.
        const Patch pi = patch_for(i);
        model.forward(tokens, nullptr, -1, &pi, 1);
        softmax_into(p_i.data(), model.acts().logits + off, V);

        for (int j = 0; j < n; ++j) {
            if (i == j) {
                // Silencing the same component twice is not a pair. Recording 0
                // would read as "no interaction", which is a claim; NaN is not a
                // number this can be confused with, and the writer skips it.
                out_growth[static_cast<std::size_t>(i) * n + j] =
                    std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            const Patch pair[2] = {patch_for(i), patch_for(j)};
            model.forward(tokens, nullptr, -1, pair, 2);
            softmax_into(p_ij.data(), model.acts().logits + off, V);
            const double cond = kl_divergence(p_i.data(), p_ij.data(), V);
            out_growth[static_cast<std::size_t>(i) * n + j] = cond - out_marginal[j];
        }
    }
}

}  // namespace cppgpt
