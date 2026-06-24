// GPT2 forward: a baby model produces a finite loss ≈ ln(V) at init, and the
// forward is deterministic (repeat and a fresh same-seed model reproduce it
// bit-for-bit). The end-to-end gradient check arrives with gpt2_backward.
#include "cppgpt/model.hpp"

#include <cmath>
#include <vector>

#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

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

    return cppgpt::test::summary();
}
