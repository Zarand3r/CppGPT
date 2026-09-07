#include "cppgpt/interp/interpret.hpp"

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

namespace {

// ln1 applied to one token embedding, into `out` [C]. The layernorm is exact
// here: the input really is that one row of wte, so mean and variance are known
// rather than frozen from some forward pass.
void ln1_of_token(const GPT2& model, int layer, int token, float* out) noexcept {
    const Config& cfg = model.config();
    const auto C = static_cast<std::size_t>(cfg.n_embd);
    const ParamTensors& p = model.params();
    const float* x = p.wte + static_cast<std::size_t>(token) * C;
    const float* w = p.ln1w + static_cast<std::size_t>(layer) * C;
    const float* b = p.ln1b + static_cast<std::size_t>(layer) * C;

    double mean = 0.0;
    for (std::size_t i = 0; i < C; ++i) mean += static_cast<double>(x[i]);
    mean /= static_cast<double>(C);
    double var = 0.0;
    for (std::size_t i = 0; i < C; ++i) {
        const double d = static_cast<double>(x[i]) - mean;
        var += d * d;
    }
    var /= static_cast<double>(C);
    const double rstd = 1.0 / std::sqrt(var + static_cast<double>(kLayerNormEps));
    for (std::size_t i = 0; i < C; ++i)
        out[i] = static_cast<float>((static_cast<double>(x[i]) - mean) * rstd *
                                        static_cast<double>(w[i]) +
                                    static_cast<double>(b[i]));
}

// One head's slice of qkvw. `third` selects Q (0), K (1) or V (2): the qkv
// vector is [Q | K | V] each of width C, and head h owns [h*hs, (h+1)*hs)
// within each -- the same convention attention_forward reads.
const float* qkv_head(const GPT2& model, int layer, int head, int third) noexcept {
    const Config& cfg = model.config();
    const auto C = static_cast<std::size_t>(cfg.n_embd);
    const auto hs = C / static_cast<std::size_t>(cfg.n_head);
    // qkvw is [3C, C] in [out, in] order, so the row index is the out-feature.
    const std::size_t row = static_cast<std::size_t>(third) * C +
                            static_cast<std::size_t>(head) * hs;
    return model.params().qkvw + static_cast<std::size_t>(layer) * 3 * C * C + row * C;
}

}  // namespace

std::size_t circuit_floats(const Config& cfg) noexcept {
    const auto V = static_cast<std::size_t>(cfg.vocab_size);
    return V * V;
}

void ov_circuit(const GPT2& model, int layer, int head, float* out) noexcept {
    const Config& cfg = model.config();
    const int V = cfg.vocab_size, C = cfg.n_embd, NH = cfg.n_head;
    ASSERT_MSG(layer >= 0 && layer < cfg.n_layer, "ov_circuit: layer out of range");
    ASSERT_MSG(head >= 0 && head < NH, "ov_circuit: head out of range");
    ASSERT(out != nullptr);

    const auto Cz = static_cast<std::size_t>(C);
    const auto hs = Cz / static_cast<std::size_t>(NH);
    const ParamTensors& p = model.params();
    const float* wv = qkv_head(model, layer, head, 2);
    const float* wo = p.attprojw + static_cast<std::size_t>(layer) * Cz * Cz;
    const std::size_t lo = static_cast<std::size_t>(head) * hs;

    std::vector<float> x(Cz), v(hs), y(Cz);
    for (int t = 0; t < V; ++t) {
        ln1_of_token(model, layer, t, x.data());

        // v = x . W_V^T, one head's rows only.
        for (std::size_t i = 0; i < hs; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < Cz; ++j)
                s += static_cast<double>(x[j]) * static_cast<double>(wv[i * Cz + j]);
            v[i] = static_cast<float>(s);
        }
        // y = v . W_O^T. Head h reaches the residual only through attprojw's
        // input columns [lo, lo+hs) -- the block save_and_ablate zeroes, so the
        // weight view here and the ablation view stay consistent by definition.
        for (std::size_t o = 0; o < Cz; ++o) {
            double s = 0.0;
            for (std::size_t i = 0; i < hs; ++i)
                s += static_cast<double>(v[i]) * static_cast<double>(wo[o * Cz + lo + i]);
            y[o] = static_cast<float>(s);
        }
        // logits through the tied unembedding. No final layernorm: its scale is
        // input-dependent, so magnitudes are relative and the row ranking is not.
        for (int k = 0; k < V; ++k) {
            double s = 0.0;
            const float* u = p.wte + static_cast<std::size_t>(k) * Cz;
            for (std::size_t o = 0; o < Cz; ++o)
                s += static_cast<double>(y[o]) * static_cast<double>(u[o]);
            out[static_cast<std::size_t>(t) * static_cast<std::size_t>(V) +
                static_cast<std::size_t>(k)] = static_cast<float>(s);
        }
    }

    // Centre each COLUMN: subtract, for every target token, its mean over all
    // sources.
    //
    // WHY. Without it the table is dominated by the unembedding rather than by
    // the head. Measured on the Shakespeare checkpoint, L0H1's top promotion was
    // the character '&' in 21 of 65 rows and 'Q' in a further 16 -- over half the
    // table was two rare characters, which have large embedding norms and so win
    // a dot product against almost any direction.
    //
    // The correction is principled rather than cosmetic: a constant offset per
    // target is shared by every source, so by construction it carries no
    // information about what THIS head does with THIS source. Removing it leaves
    // exactly the source-dependent part, which is the question the panel asks.
    const auto Vz = static_cast<std::size_t>(V);
    for (int k = 0; k < V; ++k) {
        double mean = 0.0;
        for (int t = 0; t < V; ++t)
            mean += static_cast<double>(out[static_cast<std::size_t>(t) * Vz +
                                            static_cast<std::size_t>(k)]);
        mean /= static_cast<double>(V);
        for (int t = 0; t < V; ++t) {
            float& e = out[static_cast<std::size_t>(t) * Vz + static_cast<std::size_t>(k)];
            e = static_cast<float>(static_cast<double>(e) - mean);
        }
    }
}

void qk_circuit(const GPT2& model, int layer, int head, float* out) noexcept {
    const Config& cfg = model.config();
    const int V = cfg.vocab_size, C = cfg.n_embd, NH = cfg.n_head;
    ASSERT_MSG(layer >= 0 && layer < cfg.n_layer, "qk_circuit: layer out of range");
    ASSERT_MSG(head >= 0 && head < NH, "qk_circuit: head out of range");
    ASSERT(out != nullptr);

    const auto Cz = static_cast<std::size_t>(C);
    const auto hs = Cz / static_cast<std::size_t>(NH);
    const float* wq = qkv_head(model, layer, head, 0);
    const float* wk = qkv_head(model, layer, head, 1);
    // The BIASES matter here. attention_forward scores with
    // (W_Q x + b_Q) . (W_K x + b_K), and the cross term b_Q . (W_K x_s) varies
    // with the SOURCE, so dropping it changes the within-row ranking -- which is
    // the only thing this table is read for. On the trained checkpoint it moved
    // the most-preferred source on 35% of rows, and on 96.9% for L0H0.
    //
    // ov_circuit needs no equivalent: b_V and b_O contribute a per-TARGET
    // constant, which its column-centring removes exactly.
    const ParamTensors& pp = model.params();
    const auto bofs = static_cast<std::size_t>(layer) * 3 * Cz + static_cast<std::size_t>(head) * hs;
    const float* bq = pp.qkvb + bofs;
    const float* bk = pp.qkvb + bofs + Cz;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hs));

    // Project every token once into q- and k-space, then take all V^2 dot
    // products. Doing it per pair would repeat the projection V times.
    std::vector<float> x(Cz), q(static_cast<std::size_t>(V) * hs),
        k(static_cast<std::size_t>(V) * hs);
    for (int tk = 0; tk < V; ++tk) {
        ln1_of_token(model, layer, tk, x.data());
        for (std::size_t i = 0; i < hs; ++i) {
            double sq = 0.0, sk = 0.0;
            for (std::size_t j = 0; j < Cz; ++j) {
                sq += static_cast<double>(x[j]) * static_cast<double>(wq[i * Cz + j]);
                sk += static_cast<double>(x[j]) * static_cast<double>(wk[i * Cz + j]);
            }
            q[static_cast<std::size_t>(tk) * hs + i] =
                static_cast<float>(sq + static_cast<double>(bq[i]));
            k[static_cast<std::size_t>(tk) * hs + i] =
                static_cast<float>(sk + static_cast<double>(bk[i]));
        }
    }
    for (int d = 0; d < V; ++d)
        for (int s = 0; s < V; ++s) {
            double acc = 0.0;
            for (std::size_t i = 0; i < hs; ++i)
                acc += static_cast<double>(q[static_cast<std::size_t>(d) * hs + i]) *
                       static_cast<double>(k[static_cast<std::size_t>(s) * hs + i]);
            out[static_cast<std::size_t>(d) * static_cast<std::size_t>(V) +
                static_cast<std::size_t>(s)] = static_cast<float>(acc) * scale;
        }
}

float copying_score(const float* ov, int V) noexcept {
    ASSERT(ov != nullptr && V > 0);
    int hits = 0;
    for (int t = 0; t < V; ++t) {
        const float* row = ov + static_cast<std::size_t>(t) * static_cast<std::size_t>(V);
        int best = 0;
        for (int k = 1; k < V; ++k)
            if (row[k] > row[best]) best = k;
        hits += (best == t) ? 1 : 0;
    }
    return static_cast<float>(hits) / static_cast<float>(V);
}

void svd_square(const float* a, int n, float* u, float* s, float* vt) noexcept {
    ASSERT(a != nullptr && u != nullptr && s != nullptr && vt != nullptr);
    ASSERT_MSG(n > 0, "svd_square: n must be positive");
    const auto nz = static_cast<std::size_t>(n);

    // Work on a copy: the input is const and callers rely on that (the circuit
    // tables are recomputed rarely and read often).
    //
    // A non-finite entry is refused rather than propagated. Audit finding: one
    // +Inf made this return NORMALLY with NaN in u and a non-finite singular
    // value -- output indistinguishable from a result. A decomposition of a
    // matrix containing infinity is not defined, so this is an invalid input,
    // and invalid input fails fast here as it does everywhere else in the repo.
    std::vector<double> w(nz * nz);
    for (std::size_t i = 0; i < nz * nz; ++i) {
        ASSERT_MSG(std::isfinite(a[i]), "svd_square: input is not finite");
        w[i] = static_cast<double>(a[i]);
    }

    // V accumulates the rotations; it starts as the identity, which is also the
    // right answer for the zero matrix -- so a degenerate input still yields a
    // valid orthonormal basis rather than uninitialised memory.
    std::vector<double> v(nz * nz, 0.0);
    for (std::size_t i = 0; i < nz; ++i) v[i * nz + i] = 1.0;

    // One-sided Jacobi. Sweep over column pairs, rotating each to make them
    // orthogonal. Converged when no pair in a whole sweep needed a rotation.
    //
    // The cap is an assert rather than a silent return: not converging means the
    // output is not a decomposition, and returning it quietly would put a
    // meaningless basis on the screen labelled as a direction.
    // Classical Jacobi converges in ~10 sweeps for well-scaled input; the cap is
    // a runaway backstop, not a tuning knob, so it is set well clear of any
    // legitimate case rather than snugly above the observed one. An audit noted
    // 60 was tight enough to be reachable by ordinary matrices.
    constexpr int kMaxSweeps = 200;
    constexpr double kTol = 1e-14;

    // The scale every convergence test is measured against.
    //
    // A purely PER-PAIR relative test livelocks. Adversarial review found this:
    // `|apq| <= kTol*sqrt(app*aqq)` compares a pair against itself, so the
    // threshold shrinks in lockstep with the column it is judging. A column that
    // is nothing but rounding residue never converges -- each rotation halves it
    // and halves the threshold, forever. On an 8x8 matrix of ones the loop
    // reached a bit-exact fixed point and spun until the sweep cap aborted the
    // PROCESS. It reproduced from n=6 upward, and the unit test missed it only
    // because that block was pinned to n=5.
    //
    // The fix is an absolute floor: a column negligible against the LARGEST
    // column in the matrix is already converged, whatever its own norm is.
    double scale = 0.0;
    for (std::size_t k = 0; k < nz; ++k) {
        double nk = 0.0;
        for (std::size_t i = 0; i < nz; ++i) nk += w[i * nz + k] * w[i * nz + k];
        scale = std::fmax(scale, std::sqrt(nk));
    }
    const double floor2 = (scale * 1e-15) * (scale * 1e-15);  // squared, to compare with app/aqq

    int sweep = 0;
    for (; sweep < kMaxSweeps; ++sweep) {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < nz; ++p)
            for (std::size_t q = p + 1; q < nz; ++q) {
                double app = 0.0, aqq = 0.0, apq = 0.0;
                for (std::size_t i = 0; i < nz; ++i) {
                    const double xp = w[i * nz + p], xq = w[i * nz + q];
                    app += xp * xp;
                    aqq += xq * xq;
                    apq += xp * xq;
                }
                // Already orthogonal, both columns vanish, or one is negligible
                // against the matrix as a whole.
                if (apq == 0.0) continue;
                if (app <= floor2 || aqq <= floor2) continue;
                if (std::fabs(apq) <= kTol * std::sqrt(app * aqq)) continue;

                // The rotation that zeroes apq. Written via tau/t rather than
                // atan2 for the usual numerical reason: it stays accurate when
                // the two column norms are nearly equal.
                const double tau = (aqq - app) / (2.0 * apq);
                // tau*tau overflows to +inf above sqrt(DBL_MAX) ~ 1.34e154, which
                // gives t = 0 and an IDENTITY rotation -- a no-op that still
                // counted as progress, so the loop could never terminate at any
                // sweep cap. Above the threshold the exact formula degenerates to
                // its asymptote t -> 1/(2|tau|), which is what is used.
                //
                // DEFENCE IN DEPTH, and deliberately not separately gated: with
                // the two guards above in place this branch appears unreachable.
                // |tau| > 1e150 needs |apq| < |aqq-app|/2e150, while the relative
                // guard needs |apq| > 1e-14*sqrt(app*aqq); together those require
                // aqq/app < 2.5e-273, and the absolute floor already requires
                // aqq/app > 1e-30. Removing it does not fail any test.
                //
                // It stays anyway. That reachability argument is arithmetic about
                // floating-point edge cases, which is exactly the kind of
                // reasoning that turns out to be wrong, and the cost of being
                // wrong here is an infinite loop that kills the process.
                constexpr double kTauMax = 1e150;
                const double sign = (tau >= 0.0 ? 1.0 : -1.0);
                const double tt = (std::fabs(tau) > kTauMax)
                                      ? sign / (2.0 * std::fabs(tau))
                                      : sign / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + tt * tt);
                const double sn = c * tt;

                // A rotation that is numerically the identity changes nothing, so
                // reporting it as progress is what turns a fixed point into an
                // infinite loop. Belt and braces alongside the two guards above.
                if (sn == 0.0) continue;

                for (std::size_t i = 0; i < nz; ++i) {
                    const double xp = w[i * nz + p], xq = w[i * nz + q];
                    w[i * nz + p] = c * xp - sn * xq;
                    w[i * nz + q] = sn * xp + c * xq;
                    const double vp = v[i * nz + p], vq = v[i * nz + q];
                    v[i * nz + p] = c * vp - sn * vq;
                    v[i * nz + q] = sn * vp + c * vq;
                }
                rotated = true;
            }
        if (!rotated) break;
    }
    ASSERT_MSG(sweep < kMaxSweeps, "svd_square: Jacobi did not converge");

    // Column norms of the rotated matrix are the singular values; normalising
    // those columns gives U. A zero column means a zero singular value, and its
    // U column is filled from the identity so the basis stays well-formed.
    std::vector<std::size_t> order(nz);
    std::vector<double> sig(nz);
    for (std::size_t k = 0; k < nz; ++k) {
        double norm = 0.0;
        for (std::size_t i = 0; i < nz; ++i) norm += w[i * nz + k] * w[i * nz + k];
        sig[k] = std::sqrt(norm);
        order[k] = k;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
        return sig[x] > sig[y];
    });

    // A zero-singular-value direction still needs *some* orthonormal vector.
    // Gram-Schmidt against what is already placed is the cheap way to get one,
    // and at these sizes the cost is irrelevant.
    for (std::size_t out = 0; out < nz; ++out) {
        const std::size_t k = order[out];
        s[out] = static_cast<float>(sig[k]);
        for (std::size_t j = 0; j < nz; ++j) vt[out * nz + j] = static_cast<float>(v[j * nz + k]);

        std::vector<double> col(nz);
        if (sig[k] > 0.0) {
            for (std::size_t i = 0; i < nz; ++i) col[i] = w[i * nz + k] / sig[k];
        } else {
            // Pick the basis vector with the LARGEST residual after projecting
            // out what is already placed, not the first one over a threshold.
            // Accepting a small residual and normalising it multiplies the
            // round-off in that column by 1/residual, which is how a fallback
            // meant to keep the basis well-formed ends up degrading it.
            std::vector<double> best(nz, 0.0);
            double best_norm = -1.0;
            for (std::size_t cand = 0; cand < nz; ++cand) {
                for (std::size_t i = 0; i < nz; ++i) col[i] = (i == cand) ? 1.0 : 0.0;
                // Twice, for stability: one pass leaves components of the earlier
                // vectors behind when they are nearly parallel to the candidate.
                for (int pass = 0; pass < 2; ++pass)
                    for (std::size_t prev = 0; prev < out; ++prev) {
                        double dot = 0.0;
                        for (std::size_t i = 0; i < nz; ++i)
                            dot += col[i] * static_cast<double>(u[i * nz + prev]);
                        for (std::size_t i = 0; i < nz; ++i)
                            col[i] -= dot * static_cast<double>(u[i * nz + prev]);
                    }
                double norm = 0.0;
                for (std::size_t i = 0; i < nz; ++i) norm += col[i] * col[i];
                if (norm > best_norm) {
                    best_norm = norm;
                    best = col;
                }
            }
            ASSERT_MSG(best_norm > 1e-12, "svd_square: no independent direction left for the basis");
            const double bn = std::sqrt(best_norm);
            for (std::size_t i = 0; i < nz; ++i) col[i] = best[i] / bn;
        }
        for (std::size_t i = 0; i < nz; ++i) u[i * nz + out] = static_cast<float>(col[i]);
    }
}

void svd_circuit(const GPT2& model, int layer, int head, CircuitKind kind, float* u, float* s,
                 float* vt) noexcept {
    const Config& cfg = model.config();
    std::vector<float> tbl(circuit_floats(cfg));
    if (kind == CircuitKind::Ov)
        ov_circuit(model, layer, head, tbl.data());
    else
        qk_circuit(model, layer, head, tbl.data());
    svd_square(tbl.data(), cfg.vocab_size, u, s, vt);
}

namespace {

// Accuracy of w.x + b on rows [lo, hi), against `labels`.
double probe_accuracy(const float* x, const std::uint8_t* labels, int lo, int hi, int dim,
                      const std::vector<double>& w, double b) noexcept {
    int right = 0;
    for (int i = lo; i < hi; ++i) {
        double z = b;
        for (int d = 0; d < dim; ++d)
            z += w[static_cast<std::size_t>(d)] *
                 static_cast<double>(x[static_cast<std::size_t>(i) * dim + d]);
        right += ((z > 0.0) == (labels[static_cast<std::size_t>(i)] != 0)) ? 1 : 0;
    }
    return static_cast<double>(right) / static_cast<double>(hi - lo);
}

// Logistic regression by gradient descent on rows [0, n_train).
void probe_fit_weights(const float* x, const std::uint8_t* labels, int n_train, int dim,
                       std::vector<double>& w, double& b) noexcept {
    std::fill(w.begin(), w.end(), 0.0);
    b = 0.0;
    constexpr int kEpochs = 200;
    constexpr double kLr = 0.5;
    std::vector<double> gw(static_cast<std::size_t>(dim));
    for (int e = 0; e < kEpochs; ++e) {
        std::fill(gw.begin(), gw.end(), 0.0);
        double gb = 0.0;
        for (int i = 0; i < n_train; ++i) {
            double z = b;
            for (int d = 0; d < dim; ++d)
                z += w[static_cast<std::size_t>(d)] *
                     static_cast<double>(x[static_cast<std::size_t>(i) * dim + d]);
            const double p = 1.0 / (1.0 + std::exp(-z));
            const double e_ = p - (labels[static_cast<std::size_t>(i)] != 0 ? 1.0 : 0.0);
            for (int d = 0; d < dim; ++d)
                gw[static_cast<std::size_t>(d)] +=
                    e_ * static_cast<double>(x[static_cast<std::size_t>(i) * dim + d]);
            gb += e_;
        }
        const double s = kLr / static_cast<double>(n_train);
        for (int d = 0; d < dim; ++d) w[static_cast<std::size_t>(d)] -= s * gw[static_cast<std::size_t>(d)];
        b -= s * gb;
    }
}

}  // namespace

ProbeResult fit_probe(const float* x, const std::uint8_t* y, int n, int dim, int n_train,
                      float* out_direction, Generator& gen) noexcept {
    ASSERT(x != nullptr && y != nullptr && out_direction != nullptr);
    ASSERT_MSG(n > 1 && dim > 0, "fit_probe: needs at least two rows and one dimension");
    // Both halves must be non-empty. A probe scored on its training rows is the
    // standard way these numbers become meaningless, so it aborts rather than
    // reporting one.
    ASSERT_MSG(n_train > 0, "fit_probe: no training rows");
    ASSERT_MSG(n_train < n, "fit_probe: no held-out rows");

    std::vector<double> w(static_cast<std::size_t>(dim));
    double b = 0.0;
    probe_fit_weights(x, y, n_train, dim, w, b);
    for (int d = 0; d < dim; ++d)
        out_direction[static_cast<std::size_t>(d)] = static_cast<float>(w[static_cast<std::size_t>(d)]);

    ProbeResult r{};
    r.accuracy = static_cast<float>(probe_accuracy(x, y, n_train, n, dim, w, b));

    // Base rate on the held-out split: the accuracy of always guessing the
    // majority class. A property that is 95% one class yields a 95% probe that
    // learned the prior, so this is reported beside the accuracy, not derived
    // from it by a reader.
    int ones = 0;
    for (int i = n_train; i < n; ++i) ones += (y[static_cast<std::size_t>(i)] != 0) ? 1 : 0;
    const int held = n - n_train;
    r.base_rate = static_cast<float>(std::max(ones, held - ones)) / static_cast<float>(held);

    // The control: refit on permuted labels. This must land at the base rate.
    // If it does not, the split is leaking -- which for character data is the
    // default outcome, since adjacent positions are not independent.
    std::vector<std::uint8_t> shuffled(y, y + static_cast<std::size_t>(n));
    for (std::size_t i = shuffled.size(); i > 1; --i) {
        const auto j = static_cast<std::size_t>(gen.uniform_int(0, static_cast<std::int64_t>(i - 1)));
        std::swap(shuffled[i - 1], shuffled[j]);
    }
    std::vector<double> w2(static_cast<std::size_t>(dim));
    double b2 = 0.0;
    probe_fit_weights(x, shuffled.data(), n_train, dim, w2, b2);
    r.shuffled = static_cast<float>(probe_accuracy(x, shuffled.data(), n_train, n, dim, w2, b2));
    return r;
}

SteerResult steer_effect(GPT2& model, const int* tokens, int layer, int pos,
                         const float* direction, float scale, int n_random,
                         Generator& gen) noexcept {
    const Config& cfg = model.config();
    const int V = cfg.vocab_size, C = cfg.n_embd;
    ASSERT_MSG(layer >= 0 && layer < cfg.n_layer, "steer_effect: layer out of range");
    ASSERT_MSG(pos >= 0 && pos < model.seq_len(), "steer_effect: pos out of range");
    ASSERT(tokens != nullptr && direction != nullptr);
    ASSERT_MSG(n_random > 0, "steer_effect: the null needs at least one draw");

    const auto Cz = static_cast<std::size_t>(C);
    double norm = 0.0;
    for (std::size_t i = 0; i < Cz; ++i)
        norm += static_cast<double>(direction[i]) * static_cast<double>(direction[i]);
    // Normalising is what makes `scale` mean the same thing for every direction;
    // a zero-norm one has no direction to steer along and is a caller error.
    ASSERT_MSG(norm > 0.0, "steer_effect: zero-norm direction");
    norm = std::sqrt(norm);

    const auto off = static_cast<std::size_t>(pos) * static_cast<std::size_t>(V);
    const std::size_t n = patch_floats(cfg, PatchSite::MlpOut, model.batch(), model.seq_len());
    std::vector<float> base(n), buf(n);
    std::vector<float> p_clean(static_cast<std::size_t>(V)), p_out(static_cast<std::size_t>(V));

    model.forward(tokens, nullptr);
    softmax_into(p_clean.data(), model.acts().logits + off, V);
    // Capture BEFORE any patched forward overwrites the arena.
    capture_site(model, PatchSite::MlpOut, layer, -1, base.data());

    // Steering is additive: the layer's MLP write plus the scaled direction,
    // applied at every position. Patching replaces, so the sum is formed here.
    const auto run = [&](const float* unit) {
        for (std::size_t i = 0; i < n; ++i)
            buf[i] = base[i] + scale * unit[i % Cz];
        const Patch p{PatchSite::MlpOut, layer, -1, buf.data()};
        model.forward(tokens, nullptr, -1, &p, 1);
        softmax_into(p_out.data(), model.acts().logits + off, V);
        return static_cast<float>(kl_divergence(p_clean.data(), p_out.data(), V));
    };

    std::vector<float> unit(Cz);
    for (std::size_t i = 0; i < Cz; ++i)
        unit[i] = static_cast<float>(static_cast<double>(direction[i]) / norm);

    SteerResult r{};
    r.kl_direction = run(unit.data());

    // The null: many random directions of the SAME norm. If the probe direction
    // does not stand out against them, it is decodable but not one the model
    // reads -- which is the common case and the thing this measurement exists to
    // detect.
    std::vector<float> rnd(Cz);
    double sum = 0.0, sumsq = 0.0;
    int beat = 0;
    for (int k = 0; k < n_random; ++k) {
        double rn = 0.0;
        for (std::size_t i = 0; i < Cz; ++i) {
            rnd[i] = static_cast<float>(gen.normal());
            rn += static_cast<double>(rnd[i]) * static_cast<double>(rnd[i]);
        }
        rn = std::sqrt(std::max(rn, 1e-30));
        for (std::size_t i = 0; i < Cz; ++i)
            rnd[i] = static_cast<float>(static_cast<double>(rnd[i]) / rn);
        const double k_r = run(rnd.data());
        sum += k_r;
        sumsq += k_r * k_r;
        beat += (static_cast<double>(r.kl_direction) > k_r) ? 1 : 0;
    }
    const double nd = static_cast<double>(n_random);
    const double mean = sum / nd;
    r.kl_random_mean = static_cast<float>(mean);
    r.kl_random_sd = static_cast<float>(std::sqrt(std::max(sumsq / nd - mean * mean, 0.0)));
    r.beats_random = static_cast<float>(beat) / static_cast<float>(n_random);
    return r;
}

}  // namespace cppgpt
