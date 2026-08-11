// The tensor table must reproduce the EXISTING layout exactly before anything
// downstream is allowed to depend on it. This is migration step 1: a pure
// equivalence proof against the hand-written sizes, across several configs,
// so the table can then replace them with nothing left to argue about.
#include "cppgpt/tensors.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "cppgpt/model.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

namespace {
// The hand-written .bin order, transcribed from the original param_sizes().
// If the table and this disagree, the table is wrong.
std::vector<std::size_t> reference_sizes(const Config& c) {
    const auto V = static_cast<std::size_t>(c.vocab_size);
    const auto C = static_cast<std::size_t>(c.n_embd);
    const auto L = static_cast<std::size_t>(c.n_layer);
    const auto maxT = static_cast<std::size_t>(c.max_seq_len);
    return {V * C,   maxT * C, L * C,         L * C, L * 3 * C * C, L * 3 * C,
            L * C * C, L * C,  L * C,         L * C, L * 4 * C * C, L * 4 * C,
            L * C * 4 * C, L * C, C,          C};
}
}  // namespace

int main() {
    const std::vector<Config> configs = {
        {8, 5, 2, 2, 8},          // the unit-test baby
        {16, 17, 2, 2, 8},        // the parity fixture
        {64, 65, 4, 4, 128},      // the Shakespeare toy
        {1024, 50257, 12, 12, 768},  // GPT-2 124M
        {32, 11, 0, 1, 16},       // L == 0: no layers at all
    };

    for (const Config& c : configs) {
        const std::vector<std::size_t> ref = reference_sizes(c);
        CHECK(ref.size() == static_cast<std::size_t>(kNumParams));

        // Per-tensor totals match, in order.
        bool sizes_match = true;
        std::size_t off = 0, ref_off = 0;
        bool offsets_match = true;
        for (int i = 0; i < kNumParams; ++i) {
            const std::size_t got = param_total(kParamSpecs[i], c);
            sizes_match = sizes_match && (got == ref[static_cast<std::size_t>(i)]);
            offsets_match = offsets_match && (off == ref_off);
            off += got;
            ref_off += ref[static_cast<std::size_t>(i)];
        }
        CHECK(sizes_match);
        CHECK(offsets_match);
        CHECK(params_total(c) == ref_off);

        // And the arena the model actually allocates agrees with the table.
        if (c.vocab_size > 0 && c.n_embd > 0) {
            GPT2 m(c, 1, c.max_seq_len < 4 ? c.max_seq_len : 4);
            CHECK(m.param_count() == params_total(c));
        }
    }

    // GPT-2 124M's canonical parameter count, as an external anchor.
    CHECK(params_total({1024, 50257, 12, 12, 768}) == 124439808u);

    // Every tensor has a distinct, non-empty name — the property that makes
    // ShapeMismatch and weight enumeration reportable at all.
    bool names_ok = true;
    for (int i = 0; i < kNumParams; ++i) {
        names_ok = names_ok && kParamSpecs[i].name != nullptr && kParamSpecs[i].name[0] != '\0';
        for (int j = i + 1; j < kNumParams; ++j)
            names_ok = names_ok && std::string(kParamSpecs[i].name) != kParamSpecs[j].name;
    }
    CHECK(names_ok);

    // The 2-group decay split: exactly the >=2D weights and embeddings decay.
    int decaying = 0;
    for (int i = 0; i < kNumParams; ++i) decaying += kParamSpecs[i].decay ? 1 : 0;
    CHECK(decaying == 6);  // wte, wpe, qkvw, attprojw, fcw, fcprojw

    return cppgpt::test::summary();
}
