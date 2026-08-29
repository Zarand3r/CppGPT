// Conditional co-ablation (IMPLEMENTATION_PLAN.md Step 3, property P5).
//
// The load-bearing check is P5: with an EMPTY primary set the conditional score
// must reproduce the marginal sweep exactly. A conditional score that silently
// measures something else would still produce a full matrix of plausible
// numbers, and nothing downstream could tell. Comparing against a separately
// computed marginal sweep is the only check that cannot pass for such a bug.
//
// The second check is that the matrix is NOT symmetric by construction. If
// growth[i][j] were computed once and mirrored, every backup relationship would
// read as mutual, which is the opposite of what the measurement is for -- a
// backup takes over for its primary, not the other way round.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <vector>

#include "cppgpt/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/patch.hpp"
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
    const int B = 1, T = 6, V = cfg.vocab_size;

    Generator g(31337ULL);
    GPT2 m(cfg, B, T);
    m.init_weights(g);
    std::vector<int> tok(static_cast<std::size_t>(B) * T);
    for (auto& x : tok) x = static_cast<int>(g.uniform_int(0, V - 1));

    const int n = n_components(cfg);
    CHECK(n == cfg.n_layer * (cfg.n_head + 2));

    // ---- the component enumeration is the contract every sweep shares ----
    // Walk it and assert it is a bijection onto the (site, layer, head) space.
    // A duplicated or skipped component silently mis-indexes the donor cache.
    {
        int heads = 0, mlps = 0, attns = 0;
        bool in_range = true;
        for (int i = 0; i < n; ++i) {
            const Component c = component_at(cfg, i);
            in_range = in_range && c.layer >= 0 && c.layer < cfg.n_layer;
            if (c.site == PatchSite::HeadOut) {
                ++heads;
                in_range = in_range && c.head >= 0 && c.head < cfg.n_head;
            } else if (c.site == PatchSite::MlpOut) {
                ++mlps;
                in_range = in_range && c.head == -1;
            } else {
                ++attns;
                in_range = in_range && c.head == -1;
            }
        }
        CHECK(in_range);
        CHECK(heads == cfg.n_layer * cfg.n_head);
        CHECK(mlps == cfg.n_layer);
        CHECK(attns == cfg.n_layer);
        CHECK_DIES_WITH((void)component_at(cfg, n), "index out of range");
    }

    const std::size_t stride = patch_floats(cfg, PatchSite::AttnBlockOut, B, T);
    const std::vector<float> zeros(static_cast<std::size_t>(n) * stride, 0.0f);

    std::vector<double> marginal(static_cast<std::size_t>(n));
    std::vector<double> growth(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    coax_sweep(m, tok.data(), T - 1, zeros.data(), stride, marginal.data(), growth.data());

    // ---- P5: the marginal sweep, recomputed independently ----
    // Deliberately NOT reusing coax_sweep's own numbers: this loop runs the
    // ablation the way tools/inspect does, so agreement means two paths agree
    // rather than one path agreeing with itself.
    {
        std::vector<float> p_clean(static_cast<std::size_t>(V)), p(static_cast<std::size_t>(V));
        const auto off = static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(V);
        m.forward(tok.data(), nullptr);
        softmax_into(p_clean.data(), m.acts().logits + off, V);

        bool identical = true;
        bool any_nonzero = false;
        for (int j = 0; j < n; ++j) {
            const Component c = component_at(cfg, j);
            const Patch pj{c.site, c.layer, c.head, zeros.data()};
            m.forward(tok.data(), nullptr, -1, &pj, 1);
            softmax_into(p.data(), m.acts().logits + off, V);
            const double want = kl_divergence(p_clean.data(), p.data(), V);
            identical = identical && (want == marginal[static_cast<std::size_t>(j)]);
            any_nonzero = any_nonzero || want > 0.0;
        }
        CHECK(identical);
        // Without this, P5 passes if every ablation is a no-op and every KL is 0.
        CHECK(any_nonzero);
    }

    // ---- the DEFINITION of growth, recomputed independently ----
    // Found in review: the structural checks below (shape, enumeration,
    // asymmetry, baseline sensitivity) all pass for a coax_sweep that computes
    // the WRONG QUANTITY. Two mutations survived them -- referencing the
    // conditional against the clean output instead of the i-ablated one, and
    // subtracting E(i) instead of E(j). Both change what the matrix means while
    // leaving its shape intact.
    //
    // So this recomputes one cell from the definition, with code written out
    // here rather than borrowed from the implementation:
    //
    //   growth[i][j] = KL(ablate i || ablate i and j) - KL(clean || ablate j)
    {
        std::vector<float> pc(static_cast<std::size_t>(V)), pi(static_cast<std::size_t>(V)),
            pj(static_cast<std::size_t>(V)), pij(static_cast<std::size_t>(V));
        const auto off = static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(V);
        const auto pat = [&](int idx) {
            const Component c = component_at(cfg, idx);
            return Patch{c.site, c.layer, c.head,
                         zeros.data() + static_cast<std::size_t>(idx) * stride};
        };

        bool all_match = true;
        bool any_nonzero = false;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                m.forward(tok.data(), nullptr);
                softmax_into(pc.data(), m.acts().logits + off, V);

                const Patch pI = pat(i);
                m.forward(tok.data(), nullptr, -1, &pI, 1);
                softmax_into(pi.data(), m.acts().logits + off, V);

                const Patch pJ = pat(j);
                m.forward(tok.data(), nullptr, -1, &pJ, 1);
                softmax_into(pj.data(), m.acts().logits + off, V);

                const Patch both[2] = {pat(i), pat(j)};
                m.forward(tok.data(), nullptr, -1, both, 2);
                softmax_into(pij.data(), m.acts().logits + off, V);

                const double want = kl_divergence(pi.data(), pij.data(), V) -
                                    kl_divergence(pc.data(), pj.data(), V);
                const double got = growth[static_cast<std::size_t>(i) * n + j];
                all_match = all_match && (want == got);
                any_nonzero = any_nonzero || want != 0.0;
            }
        CHECK(all_match);
        CHECK(any_nonzero);  // else every cell is 0 and the check above is vacuous
    }

    // ---- the diagonal is undefined, not zero ----
    // Zero would read as "this component does not interact with itself", which
    // is a claim about a pair that does not exist.
    {
        bool diag_nan = true;
        for (int i = 0; i < n; ++i)
            diag_nan = diag_nan &&
                       std::isnan(growth[static_cast<std::size_t>(i) * n + i]);
        CHECK(diag_nan);
    }

    // ---- the matrix is directional, not mirrored ----
    {
        bool any_asymmetric = false;
        bool all_finite = true;
        for (int i = 0; i < n && !any_asymmetric; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const double a = growth[static_cast<std::size_t>(i) * n + j];
                const double b = growth[static_cast<std::size_t>(j) * n + i];
                all_finite = all_finite && std::isfinite(a);
                if (a != b) any_asymmetric = true;
            }
        CHECK(all_finite);
        CHECK(any_asymmetric);
    }

    // ---- the baseline is a parameter, and it changes the answer ----
    // Replacements drawn from a different forward (a donor) must not reproduce
    // the zero-baseline matrix. If they did, `replacements` would be ignored and
    // every CoAx number would silently be a zero-ablation number.
    {
        std::vector<int> donor(tok.size());
        for (std::size_t i = 0; i < donor.size(); ++i)
            donor[i] = static_cast<int>((tok[i] + 3) % V);

        std::vector<float> cache(static_cast<std::size_t>(n) * stride, 0.0f);
        m.forward(donor.data(), nullptr);
        for (int i = 0; i < n; ++i) {
            const Component c = component_at(cfg, i);
            capture_site(m, c.site, c.layer, c.head,
                         cache.data() + static_cast<std::size_t>(i) * stride);
        }

        std::vector<double> m2(static_cast<std::size_t>(n));
        std::vector<double> g2(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
        coax_sweep(m, tok.data(), T - 1, cache.data(), stride, m2.data(), g2.data());

        bool differs = false;
        for (int j = 0; j < n && !differs; ++j)
            differs = m2[static_cast<std::size_t>(j)] != marginal[static_cast<std::size_t>(j)];
        CHECK(differs);
    }

    return cppgpt::test::summary();
}
