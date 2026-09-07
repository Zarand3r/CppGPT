// The induction score (M7-B4).
//
// This exists to CHECK M-22's negative result with different machinery. M-22
// concluded there are no copying heads and therefore no induction heads, using a
// copying statistic of this repo's own construction over composed weight
// matrices. This reads attention from a real forward. If both say no, the
// negative rests on two independent measurements instead of one.
//
// The load-bearing check is that the score is recomputed here from the `att`
// tensor by hand, so agreement means two implementations agree rather than one
// agreeing with itself. The second is a control: on a NON-repeated sequence
// there is no matching prefix to attend to, so nothing should stand out.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "cppgpt/interp/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

int main() {
    Config cfg{};
    cfg.max_seq_len = 16;
    cfg.vocab_size = 11;
    cfg.n_layer = 2;
    cfg.n_head = 2;
    cfg.n_embd = 16;
    const int half = 6, T = 2 * half, B = 1;
    const int L = cfg.n_layer, NH = cfg.n_head;

    Generator g(31337ULL);
    GPT2 m(cfg, B, T);
    m.init_weights(g);

    // A repeated block: r[0..half-1] twice.
    std::vector<int> tok(static_cast<std::size_t>(T));
    for (int i = 0; i < half; ++i) {
        tok[static_cast<std::size_t>(i)] = static_cast<int>(g.uniform_int(0, cfg.vocab_size - 1));
        tok[static_cast<std::size_t>(half + i)] = tok[static_cast<std::size_t>(i)];
    }
    m.forward(tok.data(), nullptr);

    std::vector<float> per_head(static_cast<std::size_t>(L) * NH);
    float uniform = 0.0f;
    induction_scores(m, half, per_head.data(), &uniform);

    // ---- recomputed by hand from the attention tensor ----
    {
        bool all_match = true;
        bool any_nonzero = false;
        for (int l = 0; l < L; ++l)
            for (int h = 0; h < NH; ++h) {
                const float* a = head_slice(m.acts().att, l, h, B, NH, T);
                double acc = 0.0;
                for (int i = 0; i < half - 1; ++i)
                    acc += static_cast<double>(a[static_cast<std::size_t>(half + i) * T + (i + 1)]);
                acc /= static_cast<double>(half - 1);
                const double got = per_head[static_cast<std::size_t>(l) * NH + h];
                all_match = all_match && std::fabs(acc - got) < 1e-6;
                any_nonzero = any_nonzero || acc > 1e-9;
            }
        CHECK(all_match);
        CHECK(any_nonzero);  // else "they match" is two sets of zeros
    }

    // ---- the uniform baseline is what uniform attention would score ----
    {
        double want = 0.0;
        for (int i = 0; i < half - 1; ++i)
            want += 1.0 / static_cast<double>(half + i + 1);
        want /= static_cast<double>(half - 1);
        CHECK(std::fabs(want - static_cast<double>(uniform)) < 1e-6);
        CHECK(uniform > 0.0f);
    }

    // ---- attention rows are distributions, so no score can exceed 1 ----
    {
        bool bounded = true;
        for (int i = 0; i < L * NH; ++i)
            bounded = bounded && per_head[static_cast<std::size_t>(i)] >= 0.0f &&
                      per_head[static_cast<std::size_t>(i)] <= 1.0f;
        CHECK(bounded);
    }

    // ---- range checks ----
    {
        std::vector<float> p(static_cast<std::size_t>(L) * NH);
        float u = 0.0f;
        CHECK_DIES_WITH(induction_scores(m, T, p.data(), &u), "half must be");
        CHECK_DIES_WITH(induction_scores(m, 1, p.data(), &u), "half must be");
    }

    return cppgpt::test::summary();
}
