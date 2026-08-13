// The tensor table must reproduce the EXISTING layout exactly before anything
// downstream is allowed to depend on it. This is migration step 1: a pure
// equivalence proof against the hand-written sizes, across several configs,
// so the table can then replace them with nothing left to argue about.
#include "cppgpt/tensors.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
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
// The hand-written activation order, transcribed from the ORIGINAL act_sizes().
// Independent of the table by construction — the previous version of this test
// summed the table and compared it to acts_total(), which IS that sum, so any row
// could be wrong upward and still pass.
std::vector<std::size_t> reference_act_sizes(const Config& c, int B, int T) {
    const auto C = static_cast<std::size_t>(c.n_embd);
    const auto L = static_cast<std::size_t>(c.n_layer);
    const auto NH = static_cast<std::size_t>(c.n_head);
    const auto V = static_cast<std::size_t>(c.vocab_size);
    const auto Bz = static_cast<std::size_t>(B);
    const auto Tz = static_cast<std::size_t>(T);
    const std::size_t BTC = Bz * Tz * C, BT = Bz * Tz, ATT = Bz * NH * Tz * Tz;
    return {BTC,     L * BTC, L * BT,  L * BT,  3 * L * BTC, L * BTC, L * ATT, L * ATT,
            L * BTC, L * BTC, L * BTC, L * BT,  L * BT,      4 * L * BTC, 4 * L * BTC,
            L * BTC, L * BTC, BTC,     BT,      BT,          BT * V,  BT * V,  BT};
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

    // Activation table vs the INDEPENDENT transcription, per tensor and by running
    // offset, at several (B, T) shapes.
    for (const Config& c : configs) {
        if (c.n_embd <= 0 || c.vocab_size <= 0) continue;
        for (const auto [B, T] : {std::pair{1, 4}, std::pair{2, 3}, std::pair{3, 1}}) {
            if (T > c.max_seq_len) continue;
            const std::vector<std::size_t> ref = reference_act_sizes(c, B, T);
            CHECK(ref.size() == static_cast<std::size_t>(kNumActs));
            bool sizes_match = true, offsets_match = true;
            std::size_t off = 0, ref_off = 0;
            for (int i = 0; i < kNumActs; ++i) {
                const std::size_t got = act_total(kActSpecs[i], c, B, T);
                sizes_match = sizes_match && (got == ref[static_cast<std::size_t>(i)]);
                offsets_match = offsets_match && (off == ref_off);
                off += got;
                ref_off += ref[static_cast<std::size_t>(i)];
            }
            CHECK(sizes_match);
            CHECK(offsets_match);
            CHECK(acts_total(c, B, T) == ref_off);
        }
    }

    // Activation names are distinct and non-empty too.
    {
        bool ok = true;
        for (int i = 0; i < kNumActs; ++i) {
            ok = ok && kActSpecs[i].name != nullptr && kActSpecs[i].name[0] != '\0';
            for (int j = i + 1; j < kNumActs; ++j)
                ok = ok && std::string(kActSpecs[i].name) != kActSpecs[j].name;
        }
        CHECK(ok);
        CHECK(kNumActs == 23);
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
    // By NAME, not by count: swapping ln1w->decay and attprojw->no-decay keeps the
    // count at 6 and used to pass, which is the drift the table was meant to stop.
    {
        const std::vector<std::string> should_decay = {"wte", "wpe", "qkvw", "attprojw", "fcw",
                                                       "fcprojw"};
        bool decay_ok = true;
        int decaying = 0;
        for (int i = 0; i < kNumParams; ++i) {
            const bool expect = std::find(should_decay.begin(), should_decay.end(),
                                          std::string(kParamSpecs[i].name)) != should_decay.end();
            decay_ok = decay_ok && (kParamSpecs[i].decay == expect);
            decaying += kParamSpecs[i].decay ? 1 : 0;
        }
        CHECK(decay_ok);
        CHECK(decaying == 6);
    }

    // Names must match ParamTensors field order. Nothing else ties them: the
    // static_asserts compare counts only, and point_params' index->field mapping is
    // hand-written, so a reordered table changes the on-disk format silently.
    {
        const std::vector<std::string> bin_order = {
            "wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb",
            "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb", "lnfw", "lnfb"};
        bool order_ok = bin_order.size() == static_cast<std::size_t>(kNumParams);
        for (int i = 0; order_ok && i < kNumParams; ++i)
            order_ok = order_ok && (bin_order[static_cast<std::size_t>(i)] == kParamSpecs[i].name);
        CHECK(order_ok);
    }

    return cppgpt::test::summary();
}
