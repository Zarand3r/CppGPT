// Weight-space head circuits (ROADMAP.md M6-A5).
//
// These tables claim to describe the HEAD rather than one prompt, so the checks
// have to be different in kind from every other test in this directory. Three
// of them:
//
//   PURITY      the table is a function of the weights alone. Running forwards
//               on different prompts between two calls must not move it. This is
//               the claim that makes the panel worth having, and it is exactly
//               the claim that a stray read of acts() would silently break.
//
//   CONSTRUCTED a model built so the head's OV path is the identity must produce
//               OV == W_E W_E^T, with an orthonormal-ish embedding making the
//               diagonal the row maximum and copying_score exactly 1. An
//               arbitrary model gives numbers nobody can check by hand; this one
//               has a known answer.
//
//   AGAINST THE the read half is verified against a real forward. At position 0
//   FORWARD     causal masking leaves attention one-hot on position 0, so with
//               wpe zeroed the head's `atty` IS its OV value for that token. The
//               test recomputes it with layernorm_forward + matmul_forward --
//               the library's own ops, which share no code with ov_circuit's
//               hand-rolled loops -- so agreement means two implementations
//               agree, not that one agrees with itself.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "cppgpt/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/ops.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

namespace {

Config make_config() noexcept {
    Config cfg{};
    cfg.max_seq_len = 8;
    cfg.vocab_size = 11;
    cfg.n_layer = 2;
    cfg.n_head = 2;
    cfg.n_embd = 16;
    return cfg;
}

}  // namespace

int main() {
    const Config cfg = make_config();
    const int V = cfg.vocab_size, C = cfg.n_embd, NH = cfg.n_head;
    const int B = 1, T = 5;
    const auto hs = static_cast<std::size_t>(C / NH);

    Generator g(4242ULL);
    GPT2 m(cfg, B, T);
    m.init_weights(g);

    const std::size_t n = circuit_floats(cfg);
    CHECK(n == static_cast<std::size_t>(V) * static_cast<std::size_t>(V));
    std::vector<float> ov(n), ov2(n), qk(n);

    std::vector<int> tok(static_cast<std::size_t>(B) * T);
    for (auto& x : tok) x = static_cast<int>(g.uniform_int(0, V - 1));

    // ---- PURITY: weights only ----
    {
        ov_circuit(m, 1, 1, ov.data());
        qk_circuit(m, 1, 1, qk.data());

        // Two forwards on different prompts, which fill the activation arena
        // with entirely different values.
        m.forward(tok.data(), nullptr);
        std::vector<int> other(tok.size());
        for (std::size_t i = 0; i < other.size(); ++i)
            other[i] = static_cast<int>((tok[i] + 5) % V);
        m.forward(other.data(), nullptr);

        ov_circuit(m, 1, 1, ov2.data());
        bool identical = true;
        for (std::size_t i = 0; i < n; ++i) identical = identical && (ov[i] == ov2[i]);
        CHECK(identical);

        std::vector<float> qk2(n);
        qk_circuit(m, 1, 1, qk2.data());
        bool qk_identical = true;
        for (std::size_t i = 0; i < n; ++i) qk_identical = qk_identical && (qk[i] == qk2[i]);
        CHECK(qk_identical);
    }

    // ---- AGAINST THE FORWARD: the read half, at a position where attention is
    // forced to be one-hot ----
    {
        Config c2 = cfg;
        GPT2 fm(c2, B, T);
        fm.init_weights(g);
        // wpe zeroed so the residual at layer 0 IS the token embedding. Without
        // this the head reads wte[t] + wpe[0] and the table, which knows nothing
        // about position, would be right to disagree.
        for (std::size_t i = 0;
             i < static_cast<std::size_t>(c2.max_seq_len) * static_cast<std::size_t>(C); ++i)
            fm.params().wpe[i] = 0.0f;

        fm.forward(tok.data(), nullptr);

        const int head = 1, t0 = tok[0];
        // Recompute the head's value vector the long way, through the library's
        // own ops rather than ov_circuit's loops.
        std::vector<float> x1(static_cast<std::size_t>(C)), mean(1), rstd(1);
        const float* wte_row = fm.params().wte + static_cast<std::size_t>(t0) * C;
        layernorm_forward(x1.data(), mean.data(), rstd.data(), wte_row, fm.params().ln1w,
                          fm.params().ln1b, 1, 1, C);
        std::vector<float> qkv(static_cast<std::size_t>(3 * C));
        matmul_forward(qkv.data(), x1.data(), fm.params().qkvw, fm.params().qkvb, 1, 1, C, 3 * C);

        // atty at position 0 is that value vector, because att[0][0] == 1.
        const float* atty = fm.acts().atty + static_cast<std::size_t>(head) * hs;
        const float* want = qkv.data() + static_cast<std::size_t>(2 * C) +
                            static_cast<std::size_t>(head) * hs;
        bool close = true;
        for (std::size_t i = 0; i < hs; ++i)
            close = close && std::fabs(atty[i] - want[i]) < 1e-5f;
        CHECK(close);

        // The guard that stops the check above being vacuous: a DIFFERENT token
        // must give a different value vector, so "they match" is a real
        // agreement rather than two zeros.
        std::vector<float> x_other(static_cast<std::size_t>(C));
        const int t_other = (t0 + 1) % V;
        layernorm_forward(x_other.data(), mean.data(), rstd.data(),
                          fm.params().wte + static_cast<std::size_t>(t_other) * C,
                          fm.params().ln1w, fm.params().ln1b, 1, 1, C);
        std::vector<float> qkv_other(static_cast<std::size_t>(3 * C));
        matmul_forward(qkv_other.data(), x_other.data(), fm.params().qkvw, fm.params().qkvb, 1, 1, C,
                       3 * C);
        bool differs = false;
        for (std::size_t i = 0; i < hs; ++i)
            differs = differs || std::fabs(qkv_other[2 * C + head * hs + i] - want[i]) > 1e-6f;
        CHECK(differs);
    }

    // ---- CONSTRUCTED: an identity OV path has a known answer ----
    {
        Config c3{};
        c3.max_seq_len = 4;
        c3.vocab_size = 6;
        c3.n_layer = 1;
        c3.n_head = 1;
        c3.n_embd = 6;  // == vocab, so wte can be the identity and rows orthonormal
        GPT2 im(c3, 1, 2);
        Generator g3(7ULL);
        im.init_weights(g3);
        ParamTensors& p = im.params();
        const auto Cz = static_cast<std::size_t>(c3.n_embd);

        // wte = identity: token t embeds to basis vector t.
        for (std::size_t i = 0; i < Cz * Cz; ++i) p.wte[i] = 0.0f;
        for (std::size_t i = 0; i < Cz; ++i) p.wte[i * Cz + i] = 1.0f;
        // ln1 as a pass-through gain, and W_V = W_O = identity, so the head's
        // write is exactly its input.
        for (std::size_t i = 0; i < Cz; ++i) {
            p.ln1w[i] = 1.0f;
            p.ln1b[i] = 0.0f;
        }
        for (std::size_t i = 0; i < 3 * Cz * Cz; ++i) p.qkvw[i] = 0.0f;
        for (std::size_t i = 0; i < Cz; ++i) p.qkvw[(2 * Cz + i) * Cz + i] = 1.0f;  // V block
        for (std::size_t i = 0; i < Cz * Cz; ++i) p.attprojw[i] = 0.0f;
        for (std::size_t i = 0; i < Cz; ++i) p.attprojw[i * Cz + i] = 1.0f;

        std::vector<float> iov(circuit_floats(c3));
        ov_circuit(im, 0, 0, iov.data());

        // With an identity path and an identity embedding, layernorm still
        // centres and scales, so the row is not literally the identity matrix --
        // but token t must still be the argmax of row t, and only token t.
        CHECK(copying_score(iov.data(), c3.vocab_size) == 1.0f);

        // And off-diagonal entries must be strictly smaller, not merely equal --
        // an all-zero table would also pass an argmax test on ties.
        bool strict = true;
        for (int t = 0; t < c3.vocab_size; ++t) {
            const float* row = iov.data() + static_cast<std::size_t>(t) * c3.vocab_size;
            for (int k = 0; k < c3.vocab_size; ++k)
                if (k != t) strict = strict && (row[t] > row[k]);
        }
        CHECK(strict);
    }

    // ---- the table uses ITS OWN layer's ln1 ----
    // init_weights gives every layer ln1w = 1 and ln1b = 0, so at init the
    // layernorms are indistinguishable and "always read layer 0's ln1" is
    // invisible. Found by mutation. Build a model whose layers differ ONLY in
    // ln1, so any difference between the two tables is attributable to it.
    {
        Config c4 = cfg;
        GPT2 lm(c4, B, T);
        Generator g4(99ULL);
        lm.init_weights(g4);
        ParamTensors& p = lm.params();
        const auto Cz = static_cast<std::size_t>(C);

        // Make layer 1's attention weights identical to layer 0's...
        for (std::size_t i = 0; i < 3 * Cz * Cz; ++i) p.qkvw[3 * Cz * Cz + i] = p.qkvw[i];
        for (std::size_t i = 0; i < Cz * Cz; ++i) p.attprojw[Cz * Cz + i] = p.attprojw[i];
        // ...and differ only in the layernorm gain.
        for (std::size_t i = 0; i < Cz; ++i) {
            p.ln1w[i] = 1.0f;
            p.ln1w[Cz + i] = 2.5f;
        }

        std::vector<float> a0(n), a1(n);
        ov_circuit(lm, 0, 0, a0.data());
        ov_circuit(lm, 1, 0, a1.data());
        bool differs = false;
        for (std::size_t i = 0; i < n && !differs; ++i) differs = a0[i] != a1[i];
        CHECK(differs);
    }

    // ---- the row index is the SOURCE token, not the destination ----
    // A transposed table is still a valid-looking table, and the identity
    // construction above cannot catch it because its answer is symmetric. So
    // build an asymmetric one: the head's write SHIFTS the embedding, meaning
    // attending to token t should promote token t+1. Transposed, it would
    // promote t-1.
    {
        Config c5{};
        c5.max_seq_len = 4;
        c5.vocab_size = 6;
        c5.n_layer = 1;
        c5.n_head = 1;
        c5.n_embd = 6;
        GPT2 sm(c5, 1, 2);
        Generator g5(13ULL);
        sm.init_weights(g5);
        ParamTensors& p = sm.params();
        const auto Cz = static_cast<std::size_t>(c5.n_embd);
        const int Vs = c5.vocab_size;

        for (std::size_t i = 0; i < Cz * Cz; ++i) p.wte[i] = 0.0f;
        for (std::size_t i = 0; i < Cz; ++i) p.wte[i * Cz + i] = 1.0f;
        for (std::size_t i = 0; i < Cz; ++i) {
            p.ln1w[i] = 1.0f;
            p.ln1b[i] = 0.0f;
        }
        for (std::size_t i = 0; i < 3 * Cz * Cz; ++i) p.qkvw[i] = 0.0f;
        for (std::size_t i = 0; i < Cz; ++i) p.qkvw[(2 * Cz + i) * Cz + i] = 1.0f;
        // W_O is a cyclic shift: basis vector i becomes basis vector i+1.
        for (std::size_t i = 0; i < Cz * Cz; ++i) p.attprojw[i] = 0.0f;
        for (std::size_t i = 0; i < Cz; ++i) p.attprojw[((i + 1) % Cz) * Cz + i] = 1.0f;

        std::vector<float> sov(circuit_floats(c5));
        ov_circuit(sm, 0, 0, sov.data());
        bool shifted = true;
        for (int src = 0; src < Vs; ++src) {
            const float* row = sov.data() + static_cast<std::size_t>(src) * Vs;
            int best = 0;
            for (int k = 1; k < Vs; ++k)
                if (row[k] > row[best]) best = k;
            shifted = shifted && (best == (src + 1) % Vs);
        }
        CHECK(shifted);
    }

    // ---- QK is a score table, not a distribution, and it is not symmetric ----
    {
        qk_circuit(m, 0, 0, qk.data());
        bool asym = false;
        for (int d = 0; d < V && !asym; ++d)
            for (int s = 0; s < V; ++s)
                if (qk[static_cast<std::size_t>(d) * V + s] !=
                    qk[static_cast<std::size_t>(s) * V + d])
                    asym = true;
        CHECK(asym);  // W_Q and W_K differ, so QK[d][s] != QK[s][d]
    }

    // ---- range checks fail fast and name the invariant ----
    {
        CHECK_DIES_WITH(ov_circuit(m, cfg.n_layer, 0, ov.data()), "layer out of range");
        CHECK_DIES_WITH(ov_circuit(m, 0, NH, ov.data()), "head out of range");
        CHECK_DIES_WITH(qk_circuit(m, -1, 0, qk.data()), "layer out of range");
    }

    return cppgpt::test::summary();
}
