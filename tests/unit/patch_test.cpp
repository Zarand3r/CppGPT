// The intervention seam (IMPLEMENTATION_PLAN.md Step 1, properties P1-P4).
//
// Each check is chosen so it CANNOT pass for a broken seam. The weak versions —
// "the patched output differs from the unpatched one", "the logits are still a
// valid distribution" — pass for a patch written at the wrong stride, at the
// wrong layer, or into the wrong head, which are precisely the bugs available
// here. So every check below is bit equality against something computed a
// DIFFERENT way.
//
// P2 is the load-bearing one. `save_and_ablate` silences a head by zeroing
// attprojw's column block; a zero patch silences it by zeroing atty's channels.
// The two share no code and meet at exactly one point, so agreement to the last
// bit means both are wired right. (They agree only while the activations are
// finite: the weight path computes atty*0 and the patch path 0*attprojw, which
// differ if atty is inf or NaN. That is already a fail-fast condition.)
#include "cppgpt/patch.hpp"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

#include "cppgpt/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

namespace {

constexpr int kB = 1, kT = 6;

Config make_config() noexcept {
    Config cfg{};
    cfg.max_seq_len = 8;
    cfg.vocab_size = 11;
    cfg.n_layer = 3;
    cfg.n_head = 2;
    cfg.n_embd = 16;
    return cfg;
}

// FNV-1a over the parameter arena. Used for P4: the seam must not write a
// single byte of it.
std::uint64_t param_checksum(const GPT2& m) noexcept {
    const auto* p = reinterpret_cast<const unsigned char*>(m.params().wte);
    const std::size_t n = m.param_count() * sizeof(float);
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

bool logits_identical(const GPT2& m, const std::vector<float>& ref) noexcept {
    for (std::size_t i = 0; i < ref.size(); ++i)
        if (m.acts().logits[i] != ref[i]) return false;
    return true;
}

std::vector<float> snapshot_logits(const GPT2& m, int V) {
    const auto n = static_cast<std::size_t>(kB) * kT * static_cast<std::size_t>(V);
    return std::vector<float>(m.acts().logits, m.acts().logits + n);
}

}  // namespace

int main() {
    const Config cfg = make_config();
    const int L = cfg.n_layer, NH = cfg.n_head, V = cfg.vocab_size, C = cfg.n_embd;

    Generator g(31337ULL);
    GPT2 m(cfg, kB, kT);
    m.init_weights(g);

    std::vector<int> tok(static_cast<std::size_t>(kB) * kT);
    for (auto& x : tok) x = static_cast<int>(g.uniform_int(0, V - 1));

    m.forward(tok.data(), nullptr);
    const std::vector<float> clean = snapshot_logits(m, V);
    const std::uint64_t params_before = param_checksum(m);

    const std::size_t head_floats = patch_floats(cfg, PatchSite::HeadOut, kB, kT);
    const std::size_t block_floats = patch_floats(cfg, PatchSite::AttnBlockOut, kB, kT);

    // ---- P1: patching a site with the value it already holds changes nothing ----
    // Every site, every layer, every head. A wrong stride or a wrong offset
    // writes the right VALUES to the wrong PLACE, which this catches and a
    // "did the output move" check does not.
    {
        std::vector<float> buf(block_floats);
        bool all_identical = true;
        for (int l = 0; l < L; ++l) {
            for (int h = 0; h < NH; ++h) {
                m.forward(tok.data(), nullptr);  // refill the arena with clean values
                capture_site(m, PatchSite::HeadOut, l, h, buf.data());
                const Patch p{PatchSite::HeadOut, l, h, buf.data()};
                m.forward(tok.data(), nullptr, -1, &p, 1);
                all_identical = all_identical && logits_identical(m, clean);
            }
            for (const auto site : {PatchSite::AttnBlockOut, PatchSite::MlpOut}) {
                m.forward(tok.data(), nullptr);
                capture_site(m, site, l, -1, buf.data());
                const Patch p{site, l, -1, buf.data()};
                m.forward(tok.data(), nullptr, -1, &p, 1);
                all_identical = all_identical && logits_identical(m, clean);
            }
        }
        CHECK(all_identical);
    }

    // ---- P2: a zero patch equals weight ablation, bit for bit ----
    // The load-bearing check. Two implementations, no shared code.
    {
        const std::vector<float> zeros(head_floats, 0.0f);
        std::vector<float> saved(ablation_scratch(cfg));
        bool all_identical = true;
        bool any_moved = false;  // guard against both paths being no-ops

        for (int l = 0; l < L; ++l)
            for (int h = 0; h < NH; ++h) {
                save_and_ablate(m, Ablation::Head, l, h, saved.data());
                m.forward(tok.data(), nullptr);
                const std::vector<float> by_weights = snapshot_logits(m, V);
                restore_ablation(m, Ablation::Head, l, h, saved.data());

                const Patch p{PatchSite::HeadOut, l, h, zeros.data()};
                m.forward(tok.data(), nullptr, -1, &p, 1);

                all_identical = all_identical && logits_identical(m, by_weights);
                for (std::size_t i = 0; i < clean.size(); ++i)
                    any_moved = any_moved || (by_weights[i] != clean[i]);
            }
        CHECK(all_identical);
        // Without this, P2 would pass if ablation and patching were BOTH no-ops.
        CHECK(any_moved);
    }

    // ---- P4: the seam never writes to the parameter arena ----
    {
        std::vector<float> buf(block_floats, 0.5f);
        const Patch p{PatchSite::MlpOut, 1, -1, buf.data()};
        m.forward(tok.data(), nullptr, -1, &p, 1);
        CHECK(param_checksum(m) == params_before);
    }

    // ---- a patch at layer l leaves every earlier layer bit-identical ----
    // Catches a patch applied at the wrong point in the block, or to the wrong
    // layer's slice — both of which still produce a plausibly-changed output.
    {
        m.forward(tok.data(), nullptr);
        const auto btc = static_cast<std::size_t>(kB) * kT * static_cast<std::size_t>(C);
        std::vector<float> resid_before(m.acts().residual3, m.acts().residual3 + btc * L);

        std::vector<float> buf(block_floats, 0.25f);
        const int patched_layer = 2;
        const Patch p{PatchSite::MlpOut, patched_layer, -1, buf.data()};
        m.forward(tok.data(), nullptr, -1, &p, 1);

        bool prefix_identical = true;
        for (std::size_t i = 0; i < btc * static_cast<std::size_t>(patched_layer); ++i)
            prefix_identical = prefix_identical && (m.acts().residual3[i] == resid_before[i]);
        CHECK(prefix_identical);

        bool suffix_moved = false;
        for (std::size_t i = btc * static_cast<std::size_t>(patched_layer); i < btc * L; ++i)
            suffix_moved = suffix_moved || (m.acts().residual3[i] != resid_before[i]);
        CHECK(suffix_moved);  // else the "prefix identical" check is vacuous
    }

    // ---- capture_site round-trips through apply_patch ----
    // Independent of P1: compares the CAPTURED BYTES to the arena directly,
    // so a capture and an apply that were wrong in the same way would still fail.
    {
        m.forward(tok.data(), nullptr);
        std::vector<float> buf(head_floats);
        const int l = 1, h = 1;
        capture_site(m, PatchSite::HeadOut, l, h, buf.data());
        const float* atty = layer_slice(m.acts().atty, l, kB, kT, C);
        const auto hs = static_cast<std::size_t>(C / NH);
        bool matches = true;
        for (std::size_t i = 0; i < static_cast<std::size_t>(kB) * kT; ++i)
            for (std::size_t j = 0; j < hs; ++j)
                matches = matches &&
                          (buf[i * hs + j] ==
                           atty[i * static_cast<std::size_t>(C) + static_cast<std::size_t>(h) * hs + j]);
        CHECK(matches);
    }

    // ---- invalid inputs fail fast, and say which invariant broke ----
    {
        std::vector<float> buf(block_floats, 0.0f);
        CHECK_DIES_WITH(capture_site(m, PatchSite::HeadOut, L, 0, buf.data()), "layer out of range");

        const Patch bad_head{PatchSite::HeadOut, 0, NH, buf.data()};
        CHECK_DIES_WITH(m.forward(tok.data(), nullptr, -1, &bad_head, 1), "head out of range");

        const Patch null_repl{PatchSite::MlpOut, 0, -1, nullptr};
        CHECK_DIES_WITH(m.forward(tok.data(), nullptr, -1, &null_repl, 1), "replacement is null");

        // Found in the Step 1 review: a patch naming a layer that does not
        // exist used to match nothing, so the forward ran CLEAN and the caller
        // read an unpatched result as a patched one. A silent no-op is the
        // worst failure available here -- every downstream number is wrong and
        // nothing says so. This is the check that it now aborts instead.
        const Patch bad_layer{PatchSite::MlpOut, L, -1, buf.data()};
        CHECK_DIES_WITH(m.forward(tok.data(), nullptr, -1, &bad_layer, 1), "layer out of range");

        const Patch neg_layer{PatchSite::MlpOut, -1, -1, buf.data()};
        CHECK_DIES_WITH(m.forward(tok.data(), nullptr, -1, &neg_layer, 1), "layer out of range");

        // A patch pointer with a count of zero applies NOTHING and returns a
        // clean result the caller reads as patched -- the same silent class as
        // the out-of-range layer above, reachable by simply forgetting the count.
        CHECK_DIES_WITH(m.forward(tok.data(), nullptr, -1, &neg_layer, 0),
                        "pointer and count disagree");

        // Two patches writing the same region: whichever is applied last would
        // win silently, and the caller would report one intervention as two.
        const Patch dup_a{PatchSite::HeadOut, 0, 1, buf.data()};
        const Patch dup_b{PatchSite::HeadOut, 0, 1, buf.data()};
        const Patch dup_set[2] = {dup_a, dup_b};
        CHECK_DIES_WITH(m.forward(tok.data(), nullptr, -1, dup_set, 2), "same region");

        // ...but two patches on DIFFERENT regions are the whole point of a set.
        const Patch pair[2] = {{PatchSite::HeadOut, 0, 1, buf.data()},
                               {PatchSite::MlpOut, 2, -1, buf.data()}};
        m.forward(tok.data(), nullptr, -1, pair, 2);
        CHECK(std::isfinite(m.acts().logits[0]));
    }

    return cppgpt::test::summary();
}
