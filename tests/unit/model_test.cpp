// GPT2 forward + backward: a baby model produces a finite loss ≈ ln(V) at init,
// the forward is deterministic (bit-for-bit), and gpt2_backward's gradients match
// finite differences of the loss end to end — including wte, whose gradient flows
// through both the tied classifier and the embedding (weight tying).
#include "cppgpt/model.hpp"

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstddef>
#include <vector>

#include "cppgpt/random.hpp"
#include "tests/check.hpp"
#include "tests/numeric.hpp"

using namespace cppgpt;
using cppgpt::test::grad_check;
using cppgpt::test::rel_close;

namespace {
// Sample standard deviation of n floats (mean is ~0 by construction here).
double stddev(const float* p, std::size_t n) {
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) m += p[i];
    m /= static_cast<double>(n);
    double v = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = p[i] - m;
        v += d * d;
    }
    return std::sqrt(v / static_cast<double>(n));
}
}  // namespace

// init_weights: the canonical GPT-2 scheme is a constitution-level claim, and it
// had no test at all — the parity gate overwrites the weights it would check.
// Dropping the residual-projection 1/sqrt(2L) scaling, a 10x weight std, and
// lnfw = 0.5 were all GREEN across the whole suite before this.
static void test_init_weights() {
    Config c{};
    c.max_seq_len = 16; c.vocab_size = 32; c.n_layer = 4; c.n_head = 2; c.n_embd = 64;
    Generator g(4242ULL);
    GPT2 m(c, 1, 8);
    m.init_weights(g);
    const auto C = static_cast<std::size_t>(c.n_embd);
    const auto L = static_cast<std::size_t>(c.n_layer);
    const ParamTensors& p = m.params();

    // LayerNorm gains are exactly 1, shifts and every bias exactly 0.
    bool gains = true, zeros = true;
    for (std::size_t i = 0; i < L * C; ++i) gains = gains && (p.ln1w[i] == 1.0f) && (p.ln2w[i] == 1.0f);
    for (std::size_t i = 0; i < C; ++i) gains = gains && (p.lnfw[i] == 1.0f);
    CHECK(gains);
    for (std::size_t i = 0; i < L * C; ++i)
        zeros = zeros && (p.ln1b[i] == 0.0f) && (p.ln2b[i] == 0.0f) && (p.attprojb[i] == 0.0f) &&
                (p.fcprojb[i] == 0.0f);
    for (std::size_t i = 0; i < C; ++i) zeros = zeros && (p.lnfb[i] == 0.0f);
    for (std::size_t i = 0; i < L * 3 * C; ++i) zeros = zeros && (p.qkvb[i] == 0.0f);
    CHECK(zeros);

    // Plain weights ~ N(0, 0.02); a 10x error is 900% off and cannot hide.
    const double s_wte = stddev(p.wte, static_cast<std::size_t>(c.vocab_size) * C);
    const double s_qkv = stddev(p.qkvw, L * 3 * C * C);
    CHECK(s_wte > 0.017 && s_wte < 0.023);
    CHECK(s_qkv > 0.018 && s_qkv < 0.022);

    // Residual projections carry the extra 1/sqrt(2L) — the whole point of the
    // GPT-2 init, and the term whose removal was previously invisible.
    const double expect = 0.02 / std::sqrt(2.0 * static_cast<double>(c.n_layer));
    const double s_proj = stddev(p.attprojw, L * C * C);
    const double s_fcp = stddev(p.fcprojw, L * C * 4 * C);
    CHECK(s_proj > 0.9 * expect && s_proj < 1.1 * expect);
    CHECK(s_fcp > 0.9 * expect && s_fcp < 1.1 * expect);
    CHECK(s_proj < 0.6 * s_qkv);  // strictly smaller than the unscaled weights
}

int main() {
    test_init_weights();
    Config cfg{};
    cfg.max_seq_len = 8;
    cfg.vocab_size = 8;
    cfg.n_layer = 2;
    cfg.n_head = 2;
    cfg.n_embd = 4;
    const int B = 2, T = 3, V = cfg.vocab_size;

    Generator gen(0x6072ULL);
    GPT2 model(cfg, B, T);
    model.init_weights(gen);

    std::vector<int> tokens(static_cast<std::size_t>(B * T));
    std::vector<int> targets(static_cast<std::size_t>(B * T));
    for (auto& t : tokens) t = static_cast<int>(gen.uniform_int(0, V - 1));
    for (auto& t : targets) t = static_cast<int>(gen.uniform_int(0, V - 1));

    model.forward(tokens.data(), targets.data());
    const float loss1 = model.mean_loss();
    CHECK(std::isfinite(loss1));
    CHECK(std::fabs(loss1 - std::log(static_cast<float>(V))) < 0.5f);  // ≈ ln(V) at init

    // Determinism: re-running yields the identical loss.
    model.forward(tokens.data(), targets.data());
    CHECK(model.mean_loss() == loss1);

    // A fresh model with the same seed reproduces it bit-for-bit.
    Generator gen2(0x6072ULL);
    GPT2 model2(cfg, B, T);
    model2.init_weights(gen2);
    model2.forward(tokens.data(), targets.data());
    CHECK(model2.mean_loss() == loss1);

    // ---- end-to-end gradient check ----
    model.forward(tokens.data(), targets.data());
    model.zero_grads();
    model.backward(tokens.data(), targets.data());

    auto loss = [&]() {
        model.forward(tokens.data(), targets.data());
        return static_cast<double>(model.mean_loss());
    };
    const int C = cfg.n_embd, L = cfg.n_layer, maxT = cfg.max_seq_len;
    const auto sz = [](int n) { return static_cast<std::size_t>(n); };

    // wte: the tied weight — its gradient must be correct through BOTH the
    // classifier head and the token embedding.
    CHECK(grad_check(loss, model.params().wte, model.grads().wte, sz(V * C)) < 3e-2);
    CHECK(grad_check(loss, model.params().wpe, model.grads().wpe, sz(maxT * C)) < 3e-2);
    // an attention weight and an MLP weight (per-layer backward chain)
    CHECK(grad_check(loss, model.params().qkvw, model.grads().qkvw, sz(L * 3 * C * C)) < 3e-2);
    CHECK(grad_check(loss, model.params().fcw, model.grads().fcw, sz(L * 4 * C * C)) < 3e-2);
    // final layernorm gain
    CHECK(grad_check(loss, model.params().lnfw, model.grads().lnfw, sz(C)) < 3e-2);

    // ---- training step (golden path): forward → backward → AdamW must overfit a
    // fixed batch, driving the loss far below its ≈ln(V) start. ----
    {
        Generator gen3(0x6072ULL);
        GPT2 trainee(cfg, B, T);
        trainee.init_weights(gen3);
        trainee.forward(tokens.data(), targets.data());
        const float start = trainee.mean_loss();
        float prev = start;
        bool monotonic = true;
        for (int step = 0; step < 200; ++step) {
            trainee.forward(tokens.data(), targets.data());
            const float cur = trainee.mean_loss();
            monotonic = monotonic && (cur <= prev + 1e-4f);  // non-increasing (small slack)
            prev = cur;
            trainee.zero_grads();
            trainee.backward(tokens.data(), targets.data());
            trainee.update(AdamW{.lr = 1e-2f});
        }
        trainee.forward(tokens.data(), targets.data());
        const float end = trainee.mean_loss();
        CHECK(std::isfinite(end));
        CHECK(monotonic);            // AdamW descends the fixed-batch loss
        CHECK(end < 0.1f * start);   // and overfits it far below the init loss
    }

    // ---- 2-group weight-decay routing: with zero grads and fresh moments (m=v=0),
    // one AdamW step is pure decay — decaying tensors scale by (1−lr·wd), non-decaying
    // tensors are bit-unchanged. This pins GPT2::update's kDecay mask to the right
    // tensors (the overfit smoke above runs wd=0 and can't catch a mis-mapped flag). ----
    {
        Generator gen4(0x6072ULL);
        GPT2 dm(cfg, B, T);
        dm.init_weights(gen4);
        const std::size_t nwte = sz(V * C);
        const std::vector<float> wte0(dm.params().wte, dm.params().wte + nwte);   // decays
        const std::vector<float> qkvw0(dm.params().qkvw,
                                       dm.params().qkvw + sz(L * 3 * C * C));      // decays
        const std::vector<float> lnfw0(dm.params().lnfw, dm.params().lnfw + sz(C));    // no decay
        const std::vector<float> qkvb0(dm.params().qkvb, dm.params().qkvb + sz(L * 3 * C));  // no

        dm.zero_grads();  // grads = 0 ⇒ moments stay 0 ⇒ step = −lr·wd·param on the decay group
        const float lr = 0.1f, wd = 0.5f, scale = 1.0f - lr * wd;
        dm.update(AdamW{.lr = lr, .weight_decay = wd});

        bool decay_ok = true, nodecay_ok = true;
        for (std::size_t i = 0; i < nwte; ++i)
            decay_ok = decay_ok && rel_close(dm.params().wte[i], wte0[i] * scale, 1e-5);
        for (std::size_t i = 0; i < sz(L * 3 * C * C); ++i)
            decay_ok = decay_ok && rel_close(dm.params().qkvw[i], qkvw0[i] * scale, 1e-5);
        for (std::size_t i = 0; i < sz(C); ++i)
            nodecay_ok = nodecay_ok && (dm.params().lnfw[i] == lnfw0[i]);  // exactly unchanged
        for (std::size_t i = 0; i < sz(L * 3 * C); ++i)
            nodecay_ok = nodecay_ok && (dm.params().qkvb[i] == qkvb0[i]);
        CHECK(decay_ok);
        CHECK(nodecay_ok);
    }

    // ---- a single-position forward is bit-identical at that position ---------
    //
    // This mode exists because the tied classifier is 79 GFLOP at GPT-2 scale and
    // generation reads one row. It is only legitimate if that row is EXACTLY what
    // the full path produces -- an optimisation that quietly changes the answer is
    // worse than the cost it saves. Positions before the last are deliberately
    // left stale, so they are not compared.
    {
        Config c{};
        c.max_seq_len = 16; c.vocab_size = 23; c.n_layer = 2; c.n_head = 2; c.n_embd = 16;
        const int B = 2, T = 6, V = c.vocab_size;
        Generator g(4242ULL);
        GPT2 m(c, B, T);
        m.init_weights(g);
        std::vector<int> tok(static_cast<std::size_t>(B * T));
        for (auto& x : tok) x = static_cast<int>(g.uniform_int(0, V - 1));

        m.forward(tok.data(), nullptr);
        std::vector<float> want(static_cast<std::size_t>(B) * V);
        for (int b = 0; b < B; ++b) {
            const auto row = static_cast<std::size_t>(b) * T + (T - 1);
            std::copy_n(m.acts().logits + row * V, V, want.begin() + static_cast<std::size_t>(b) * V);
        }

        m.forward(tok.data(), nullptr, /*logits_at=*/T - 1);
        bool identical = true;
        for (int b = 0; b < B; ++b) {
            const auto row = static_cast<std::size_t>(b) * T + (T - 1);
            for (int v = 0; v < V; ++v)
                identical = identical &&
                    (m.acts().logits[row * V + v] == want[static_cast<std::size_t>(b) * V + v]);
        }
        CHECK(identical);
        // And it must refuse a loss rather than compute one over stale logits.
        std::vector<int> tgt(tok.size(), 0);
        CHECK_DIES_WITH(m.forward(tok.data(), tgt.data(), /*logits_at=*/T - 1),
                        "cannot compute a loss");
    }

    return cppgpt::test::summary();
}
