// GPT2 forward + backward: a baby model produces a finite loss ≈ ln(V) at init,
// the forward is deterministic (bit-for-bit), and gpt2_backward's gradients match
// finite differences of the loss end to end — including wte, whose gradient flows
// through both the tied classifier and the embedding (weight tying).
#include "cppgpt/model.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "cppgpt/random.hpp"
#include "tests/check.hpp"
#include "tests/numeric.hpp"

using namespace cppgpt;
using cppgpt::test::grad_check;
using cppgpt::test::rel_close;

int main() {
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

    return cppgpt::test::summary();
}
