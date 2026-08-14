// Interpretability: the logit lens, direct logit attribution, and ablation.
//
// Each of the three has one load-bearing check, chosen so that it cannot pass
// for a broken implementation:
//   logit_lens  — bit-identity with the model's own logits at the last layer.
//   attribution — the parts sum to the true logit (exact by construction).
//   ablation    — zeroing attprojw's column block equals zeroing atty's channels.
// The weaker "output looks like a valid distribution" versions of all three pass
// for a transposed unembedding.
//
// The load-bearing check is bit-identity at the last layer. There, the lens IS
// the model's own final computation (final layernorm over residual3[L-1], then
// the tied classifier), so anything less than an exact match means the lens is
// wired wrong — a mis-strided residual, the wrong layernorm weights, or a
// transposed unembedding all fail it. Asserting only "the lens produces a valid
// distribution" would pass for all three.
#include "cppgpt/interpret.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "cppgpt/model.hpp"
#include "cppgpt/ops.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

int main() {
    Config cfg{};
    cfg.max_seq_len = 8;
    cfg.vocab_size = 11;
    cfg.n_layer = 3;
    cfg.n_head = 2;
    cfg.n_embd = 16;
    const int B = 1, T = 6, V = cfg.vocab_size, C = cfg.n_embd;

    Generator g(31337ULL);
    GPT2 m(cfg, B, T);
    m.init_weights(g);
    std::vector<int> tok(static_cast<std::size_t>(B * T));
    for (auto& x : tok) x = static_cast<int>(g.uniform_int(0, V - 1));
    m.forward(tok.data(), nullptr);

    const auto n = static_cast<std::size_t>(B) * T;
    std::vector<float> lens(n * static_cast<std::size_t>(V));
    std::vector<float> scratch(n * static_cast<std::size_t>(C) + 2 * n);  // norm + mean + rstd

    // ---- last layer: the lens must reproduce the model's own logits EXACTLY ----
    {
        logit_lens(m, cfg.n_layer - 1, lens.data(), scratch.data());
        bool identical = true;
        for (std::size_t i = 0; i < lens.size(); ++i)
            identical = identical && (lens[i] == m.acts().logits[i]);
        CHECK(identical);
    }

    // ---- earlier layers must differ, or the lens is reading the wrong block ----
    {
        std::vector<float> early(n * static_cast<std::size_t>(V));
        logit_lens(m, 0, early.data(), scratch.data());
        bool any_diff = false;
        for (std::size_t i = 0; i < early.size(); ++i)
            any_diff = any_diff || (early[i] != m.acts().logits[i]);
        CHECK(any_diff);
    }

    // ---- every layer yields finite logits (a NaN here means a bad layernorm) ----
    {
        bool finite = true;
        for (int l = 0; l < cfg.n_layer; ++l) {
            logit_lens(m, l, lens.data(), scratch.data());
            for (float x : lens) finite = finite && std::isfinite(x);
        }
        CHECK(finite);
    }

    // ---- read-only for real: lensing every layer must leave the ENTIRE activation
    // arena untouched, including lnf_mean/lnf_rstd. Re-running forward would hide a
    // violation by recomputing, so snapshot and compare without one. ----
    {
        const std::vector<float> logits_before(m.acts().logits,
                                               m.acts().logits + n * static_cast<std::size_t>(V));
        const std::vector<float> mean_before(m.acts().lnf_mean, m.acts().lnf_mean + n);
        const std::vector<float> rstd_before(m.acts().lnf_rstd, m.acts().lnf_rstd + n);
        // Lens layer 0 LAST, deliberately. Ending on the final layer would hide a
        // violation: there the lens recomputes the model's own final layernorm, so
        // an implementation that scribbles into lnf_mean/lnf_rstd writes back
        // identical values and the snapshot still matches. (Found by mutation
        // testing this very check — the first version of it passed the mutant.)
        for (int l = cfg.n_layer - 1; l >= 0; --l) logit_lens(m, l, lens.data(), scratch.data());
        bool untouched = true;
        for (std::size_t i = 0; i < logits_before.size(); ++i)
            untouched = untouched && (m.acts().logits[i] == logits_before[i]);
        for (std::size_t i = 0; i < n; ++i)
            untouched = untouched && (m.acts().lnf_mean[i] == mean_before[i]) &&
                        (m.acts().lnf_rstd[i] == rstd_before[i]);
        CHECK(untouched);
    }

    // ---- direct logit attribution: the parts must sum to the whole ------------
    //
    // This is an identity, not a tolerance dial. Freezing the final layernorm's
    // mean/rstd makes it affine, and an affine map of a sum is the sum of the
    // maps — so the decomposition is exact by construction. A transposed
    // unembedding, a dropped centering term, a mis-strided head slice or a
    // forgotten bias each break it. "Contributions look plausible" would not.
    {
        const auto NHz = static_cast<std::size_t>(cfg.n_head);
        const auto Lz = static_cast<std::size_t>(cfg.n_layer);
        std::vector<float> heads(Lz * NHz), mlps(Lz);
        std::vector<float> dscratch(4 * static_cast<std::size_t>(C));
        float embed = 0.0f, bias = 0.0f;

        double worst = 0.0;
        bool any_head_nonzero = false, heads_differ = false;
        for (int pos = 0; pos < T; ++pos) {
            for (int v = 0; v < V; ++v) {
                direct_logit_attribution(m, pos, v, heads.data(), mlps.data(), &embed, &bias,
                                         dscratch.data());
                double total = static_cast<double>(embed) + static_cast<double>(bias);
                for (float x : heads) total += static_cast<double>(x);
                for (float x : mlps) total += static_cast<double>(x);
                const double truth = static_cast<double>(
                    m.acts().logits[static_cast<std::size_t>(pos) * V + static_cast<std::size_t>(v)]);
                worst = std::max(worst, std::fabs(total - truth));
                for (float x : heads) {
                    any_head_nonzero = any_head_nonzero || (x != 0.0f);
                    heads_differ = heads_differ || (x != heads[0]);
                }
            }
        }
        std::printf("  DLA sum rule: worst error %.3e over %d positions x %d tokens\n", worst, T, V);
        // Measured 2.14e-08 — pure fp32 roundoff. The bound is ~47x that, not the
        // 1e-4 a "looks close enough" instinct suggests: any real defect here
        // (dropped centering, transposed unembed, wrong stride) misses by O(1),
        // so a loose bound buys nothing and hides drift.
        CHECK(worst < 1e-6);
        // An all-zero or constant decomposition would satisfy the sum rule and
        // carry no information whatsoever; rule both out.
        CHECK(any_head_nonzero);
        CHECK(heads_differ);
    }

    // ---- the identity that ablation rests on ---------------------------------
    //
    // save_and_ablate silences a head by zeroing its COLUMN block of attprojw,
    // because that needs no forward-pass hook. The claim is that this equals
    // zeroing the head's channels of `atty`. Prove it rather than assume it.
    {
        const int NH = cfg.n_head, hs = C / NH;
        Generator g2(99ULL);
        std::vector<float> atty(static_cast<std::size_t>(C));
        std::vector<float> w(static_cast<std::size_t>(C) * static_cast<std::size_t>(C));
        std::vector<float> bias(static_cast<std::size_t>(C));
        for (auto& x : atty) x = g2.normal();
        for (auto& x : w) x = g2.normal();
        for (auto& x : bias) x = g2.normal();

        const int h_abl = NH / 2;
        std::vector<float> atty_z = atty;
        for (int i = 0; i < hs; ++i) atty_z[static_cast<std::size_t>(h_abl * hs + i)] = 0.0f;
        std::vector<float> w_z = w;
        for (int o = 0; o < C; ++o)
            for (int i = 0; i < hs; ++i)
                w_z[static_cast<std::size_t>(o) * C + static_cast<std::size_t>(h_abl * hs + i)] = 0.0f;

        std::vector<float> out_a(static_cast<std::size_t>(C)), out_b(static_cast<std::size_t>(C));
        matmul_forward(out_a.data(), atty_z.data(), w.data(), bias.data(), 1, 1, C, C);
        matmul_forward(out_b.data(), atty.data(), w_z.data(), bias.data(), 1, 1, C, C);
        bool same = true;
        for (int c = 0; c < C; ++c) same = same && (out_a[c] == out_b[c]);
        CHECK(same);
    }

    // ---- every ablation changes the output, and restore is bit-exact ---------
    {
        const auto n_log = static_cast<std::size_t>(B) * T * V;
        const std::vector<float> logits_before(m.acts().logits, m.acts().logits + n_log);
        // wte is the first tensor in .bin order, so it is the arena base: this
        // spans the WHOLE parameter block, not just the embedding.
        const std::vector<float> params_before(m.params().wte, m.params().wte + m.param_count());
        std::vector<float> saved(ablation_scratch(cfg));

        const Ablation kinds[] = {Ablation::Head, Ablation::Mlp, Ablation::AttnBlock};
        for (Ablation k : kinds) {
            save_and_ablate(m, k, /*layer=*/1, /*head=*/0, saved.data());
            m.forward(tok.data(), nullptr);
            bool changed = false;
            for (std::size_t i = 0; i < n_log; ++i)
                changed = changed || (m.acts().logits[i] != logits_before[i]);
            CHECK(changed);  // a silent no-op would look like "this component does not matter"

            restore_ablation(m, k, 1, 0, saved.data());
            bool params_restored = true;
            for (std::size_t i = 0; i < params_before.size(); ++i)
                params_restored = params_restored && (m.params().wte[i] == params_before[i]);
            CHECK(params_restored);

            m.forward(tok.data(), nullptr);
            bool logits_restored = true;
            for (std::size_t i = 0; i < n_log; ++i)
                logits_restored = logits_restored && (m.acts().logits[i] == logits_before[i]);
            CHECK(logits_restored);
        }
    }

    // ---- a single head ablation must zero the COLUMN block, not the row block -
    //
    // Both tile the matrix, so "ablating every head leaves zeros" passes for a
    // transposed implementation, and the algebraic identity above is proved on
    // local buffers rather than through save_and_ablate. This is the check that
    // ties the implementation to the math: everything inside the head's input
    // channels goes to zero, everything outside is untouched.
    {
        std::vector<float> saved(ablation_scratch(cfg));
        const auto Cz = static_cast<std::size_t>(C);
        const auto hs = Cz / static_cast<std::size_t>(cfg.n_head);
        const int h = cfg.n_head - 1;
        const auto lo = static_cast<std::size_t>(h) * hs;
        const std::vector<float> before(m.params().attprojw, m.params().attprojw + Cz * Cz);

        save_and_ablate(m, Ablation::Head, /*layer=*/0, h, saved.data());
        bool shape_ok = true;
        for (std::size_t o = 0; o < Cz; ++o)
            for (std::size_t i = 0; i < Cz; ++i) {
                const float v = m.params().attprojw[o * Cz + i];
                const bool in_block = (i >= lo && i < lo + hs);
                shape_ok = shape_ok && (in_block ? (v == 0.0f) : (v == before[o * Cz + i]));
            }
        CHECK(shape_ok);
        restore_ablation(m, Ablation::Head, 0, h, saved.data());
    }

    // ---- the head column blocks tile attprojw exactly -------------------------
    // Ablating every head of a layer must leave the entire projection zero. If
    // the blocks overlapped or left a gap, some head would be double-counted or
    // unreachable — and a sweep would silently under-report.
    {
        std::vector<std::vector<float>> saves;
        for (int h = 0; h < cfg.n_head; ++h) {
            saves.emplace_back(ablation_scratch(cfg));
            save_and_ablate(m, Ablation::Head, /*layer=*/0, h, saves.back().data());
        }
        const auto Cz = static_cast<std::size_t>(C);
        bool all_zero = true;
        for (std::size_t i = 0; i < Cz * Cz; ++i)
            all_zero = all_zero && (m.params().attprojw[i] == 0.0f);
        CHECK(all_zero);
        for (int h = cfg.n_head - 1; h >= 0; --h)
            restore_ablation(m, Ablation::Head, 0, h, saves[static_cast<std::size_t>(h)].data());
    }

    // ---- right-sizing T is safe: causality makes the padded tail inert --------
    // inspect runs the model at T = prompt length instead of the padded
    // max_seq_len, and attention is O(T²), so this is the single biggest win
    // available there. It is only legitimate if the short run reproduces the
    // padded one EXACTLY at the positions both contain.
    {
        Generator g3(31337ULL);  // same seed as `m`, so identical weights
        GPT2 padded(cfg, B, cfg.max_seq_len);
        padded.init_weights(g3);
        std::vector<int> long_tok(static_cast<std::size_t>(B) * cfg.max_seq_len, 0);
        std::copy(tok.begin(), tok.end(), long_tok.begin());
        padded.forward(long_tok.data(), nullptr);

        bool identical = true;
        for (int t = 0; t < T; ++t)
            for (int v = 0; v < V; ++v)
                identical = identical &&
                            (padded.acts().logits[static_cast<std::size_t>(t) * V + v] ==
                             m.acts().logits[static_cast<std::size_t>(t) * V + v]);
        CHECK(identical);
    }

    return cppgpt::test::summary();
}
