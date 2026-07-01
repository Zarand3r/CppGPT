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

    model.forward(tokens.data(), targets.data(), B, T);
    const float loss1 = model.mean_loss();
    CHECK(std::isfinite(loss1));
    CHECK(std::fabs(loss1 - std::log(static_cast<float>(V))) < 0.5f);  // ≈ ln(V) at init

    // Determinism: re-running yields the identical loss.
    model.forward(tokens.data(), targets.data(), B, T);
    CHECK(model.mean_loss() == loss1);

    // A fresh model with the same seed reproduces it bit-for-bit.
    Generator gen2(0x6072ULL);
    GPT2 model2(cfg, B, T);
    model2.init_weights(gen2);
    model2.forward(tokens.data(), targets.data(), B, T);
    CHECK(model2.mean_loss() == loss1);

    // ---- end-to-end gradient check ----
    model.forward(tokens.data(), targets.data(), B, T);
    model.zero_grads();
    model.backward(tokens.data(), targets.data(), B, T);

    auto loss = [&]() {
        model.forward(tokens.data(), targets.data(), B, T);
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
        trainee.forward(tokens.data(), targets.data(), B, T);
        const float start = trainee.mean_loss();
        float prev = start;
        bool monotonic = true;
        for (int step = 0; step < 200; ++step) {
            trainee.forward(tokens.data(), targets.data(), B, T);
            const float cur = trainee.mean_loss();
            monotonic = monotonic && (cur <= prev + 1e-4f);  // non-increasing (small slack)
            prev = cur;
            trainee.zero_grads();
            trainee.backward(tokens.data(), targets.data(), B, T);
            trainee.update(1e-2f, 0.9f, 0.95f, 1e-8f, 0.0f);
        }
        trainee.forward(tokens.data(), targets.data(), B, T);
        const float end = trainee.mean_loss();
        CHECK(std::isfinite(end));
        CHECK(monotonic);            // AdamW descends the fixed-batch loss
        CHECK(end < 0.1f * start);   // and overfits it far below the init loss
    }

    return cppgpt::test::summary();
}
