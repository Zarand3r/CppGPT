// Causal validation of a direction (M7-6, plan property P5).
//
// What a test CAN pin here is the mechanics: zero scale is exactly a no-op, the
// effect grows with the scale, the random control is a different direction of
// the same norm, and the whole thing is deterministic given a seed.
//
// What a test CANNOT pin is whether a particular direction beats its random
// control on a particular model. That is an empirical question about the model,
// and asserting it here would be asserting a result rather than a contract --
// which is exactly the reward-hacking shape this repo has already corrected
// twice. The comparison belongs in docs/measurements.md, with the control
// reported beside it.
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
    cfg.max_seq_len = 8;
    cfg.vocab_size = 11;
    cfg.n_layer = 3;
    cfg.n_head = 2;
    cfg.n_embd = 16;
    const int B = 1, T = 6, V = cfg.vocab_size, C = cfg.n_embd;

    Generator g(777ULL);
    GPT2 m(cfg, B, T);
    m.init_weights(g);
    std::vector<int> tok(static_cast<std::size_t>(B) * T);
    for (auto& x : tok) x = static_cast<int>(g.uniform_int(0, V - 1));

    std::vector<float> dir(static_cast<std::size_t>(C));
    for (auto& x : dir) x = static_cast<float>(g.normal());

    // ---- zero scale is EXACTLY a no-op ----
    // Not "small": the patch writes back what it captured, so this reduces to
    // the seam's own identity property and must be bit-exact. A tolerance here
    // would hide a steering path that perturbs the residual on its own.
    {
        Generator gg(1ULL);
        const SteerResult r = steer_effect(m, tok.data(), 1, T - 1, dir.data(), 0.0f, 4, gg);
        CHECK(r.kl_direction == 0.0f);
        CHECK(r.kl_random_mean == 0.0f);
    }

    // ---- the effect grows with the scale ----
    {
        Generator g1(2ULL), g2(2ULL);
        const SteerResult small = steer_effect(m, tok.data(), 1, T - 1, dir.data(), 0.5f, 8, g1);
        const SteerResult big = steer_effect(m, tok.data(), 1, T - 1, dir.data(), 4.0f, 8, g2);
        CHECK(big.kl_direction > small.kl_direction);
        CHECK(small.kl_direction > 0.0f);  // else "grows" is satisfied by two zeros
    }

    // ---- deterministic given a seed, including the random control ----
    {
        Generator g1(9ULL), g2(9ULL);
        const SteerResult a = steer_effect(m, tok.data(), 0, T - 1, dir.data(), 2.0f, 8, g1);
        const SteerResult b = steer_effect(m, tok.data(), 0, T - 1, dir.data(), 2.0f, 8, g2);
        CHECK(a.kl_direction == b.kl_direction);
        CHECK(a.kl_random_mean == b.kl_random_mean);
    }

    // ---- the control is a DIFFERENT direction, not the same one ----
    // If the "random" control reused the probe direction, the two numbers would
    // be identical and every direction would trivially pass its own test.
    {
        Generator g1(3ULL);
        const SteerResult r = steer_effect(m, tok.data(), 1, T - 1, dir.data(), 3.0f, 8, g1);
        CHECK(r.kl_random_mean != r.kl_direction);
        CHECK(r.kl_random_mean > 0.0f);   // the control must actually steer something
        CHECK(r.kl_random_sd > 0.0f);     // and the draws must differ from each other
    }

    // ---- scale is in units of residual norm, so a rescaled direction is the
    //      same direction ----
    // Without internal normalisation, "scale" would mean something different for
    // every probe and the numbers would not be comparable across properties.
    {
        std::vector<float> scaled(dir);
        for (auto& x : scaled) x *= 17.0f;
        Generator g1(5ULL), g2(5ULL);
        const SteerResult a = steer_effect(m, tok.data(), 2, T - 1, dir.data(), 1.5f, 8, g1);
        const SteerResult b = steer_effect(m, tok.data(), 2, T - 1, scaled.data(), 1.5f, 8, g2);
        CHECK(std::fabs(a.kl_direction - b.kl_direction) < 1e-5f);
    }

    // ---- invalid inputs fail fast ----
    {
        std::vector<float> zero(static_cast<std::size_t>(C), 0.0f);
        Generator g1(1ULL);
        CHECK_DIES_WITH(IGNORE(steer_effect(m, tok.data(), 1, T - 1, zero.data(), 1.0f, 8, g1)),
                        "zero-norm direction");
        CHECK_DIES_WITH(IGNORE(steer_effect(m, tok.data(), cfg.n_layer, T - 1, dir.data(), 1.0f, 8, g1)),
                        "layer out of range");
    }

    return cppgpt::test::summary();
}
