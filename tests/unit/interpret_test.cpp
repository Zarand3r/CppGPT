// logit_lens: the residual stream read through the unembedding at each layer.
//
// The load-bearing check is bit-identity at the last layer. There, the lens IS
// the model's own final computation (final layernorm over residual3[L-1], then
// the tied classifier), so anything less than an exact match means the lens is
// wired wrong — a mis-strided residual, the wrong layernorm weights, or a
// transposed unembedding all fail it. Asserting only "the lens produces a valid
// distribution" would pass for all three.
#include "cppgpt/interpret.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "cppgpt/model.hpp"
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

    return cppgpt::test::summary();
}
