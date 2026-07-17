// Generation end to end: inference forward (no targets), determinism, and a
// golden test — a model overfit to a cyclic sequence must greedily regenerate it.
#include <cstddef>
#include <vector>

#include "cppgpt/generate.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/optimizer.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

int main() {
    // ---- inference forward (targets = nullptr) yields the same logits as training
    // forward, and does not require targets ----
    {
        Config cfg{};
        cfg.max_seq_len = 8;
        cfg.vocab_size = 5;
        cfg.n_layer = 2;
        cfg.n_head = 2;
        cfg.n_embd = 8;
        const int B = 1, T = 4, V = cfg.vocab_size;
        Generator g(11ULL);
        GPT2 m(cfg, B, T);
        m.init_weights(g);
        std::vector<int> tok(static_cast<std::size_t>(B * T)), tgt(static_cast<std::size_t>(B * T));
        for (auto& x : tok) x = static_cast<int>(g.uniform_int(0, V - 1));
        for (auto& x : tgt) x = static_cast<int>(g.uniform_int(0, V - 1));

        m.forward(tok.data(), tgt.data());
        std::vector<float> logits_with(m.acts().logits,
                                       m.acts().logits + static_cast<std::size_t>(B * T * V));
        m.forward(tok.data(), nullptr);  // inference — must not need targets
        bool same = true;
        for (std::size_t i = 0; i < logits_with.size(); ++i)
            same = same && (m.acts().logits[i] == logits_with[i]);
        CHECK(same);
    }

    // ---- generate determinism: same seed + same model ⇒ identical output ----
    {
        Config cfg{};
        cfg.max_seq_len = 8;
        cfg.vocab_size = 6;
        cfg.n_layer = 1;
        cfg.n_head = 2;
        cfg.n_embd = 8;
        const int T = 8;
        Generator init(3ULL);
        GPT2 m(cfg, 1, T);
        m.init_weights(init);
        std::vector<int> ctx(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) ctx[static_cast<std::size_t>(t)] = t % cfg.vocab_size;
        Generator a(77ULL), b(77ULL);
        const std::vector<int> ga = generate(m, ctx.data(), 16, 1.0f, 0, a);
        const std::vector<int> gb = generate(m, ctx.data(), 16, 1.0f, 0, b);
        CHECK(ga == gb);
    }

    // ---- golden: overfit a period-4 cycle, then greedy decoding reproduces it ----
    {
        Config cfg{};
        cfg.max_seq_len = 8;
        cfg.vocab_size = 4;
        cfg.n_layer = 1;
        cfg.n_head = 2;
        cfg.n_embd = 16;
        const int T = 8;
        Generator init(123ULL);
        GPT2 m(cfg, 1, T);
        m.init_weights(init);

        std::vector<int> seq(64);
        for (std::size_t i = 0; i < seq.size(); ++i) seq[i] = static_cast<int>(i % 4);  // 0,1,2,3,…

        const AdamW opt{.lr = 5e-3f};
        Generator batch(9ULL);
        std::vector<int> inp(static_cast<std::size_t>(T)), tgt(static_cast<std::size_t>(T));
        const int max_start = static_cast<int>(seq.size()) - T - 1;
        for (int step = 0; step < 600; ++step) {
            const int s = static_cast<int>(batch.uniform_int(0, max_start));
            for (int t = 0; t < T; ++t) {
                inp[static_cast<std::size_t>(t)] = seq[static_cast<std::size_t>(s + t)];
                tgt[static_cast<std::size_t>(t)] = seq[static_cast<std::size_t>(s + t + 1)];
            }
            m.forward(inp.data(), tgt.data());
            m.zero_grads();
            m.backward(inp.data(), tgt.data());
            m.update(opt);
        }

        // Greedy (top_k = 1) from the first window; the cycle should continue: i%4.
        Generator unused(0ULL);
        const std::vector<int> gen = generate(m, seq.data(), 8, 1.0f, 1, unused);
        bool cycle_ok = true;
        for (std::size_t i = 0; i < gen.size(); ++i)
            cycle_ok = cycle_ok && (gen[i] == static_cast<int>(i % 4));
        CHECK(cycle_ok);
    }

    return cppgpt::test::summary();
}
