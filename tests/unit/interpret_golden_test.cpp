// Golden baseline for the interpretability layer (IMPLEMENTATION_PLAN.md Step 0).
//
// WHY THIS EXISTS. The plan's next step adds a patch argument to GPT2::forward.
// Property P3 says the UNPATCHED forward must stay bit-identical to the
// pre-change build — and a claim about "the pre-change build" needs an artifact
// captured before the change lands, not a memory of it. This is that artifact.
//
// WHAT IT COVERS. Every number the interpretability layer produces: the logit
// lens at each layer and position, the full direct-logit-attribution
// decomposition, and the ablation KL of all L*(NH+2) components. Steps 1-4
// change exactly these, so a semantic change shows up here as a diff rather
// than as a number nobody re-read.
//
// WHY %a AND NOT %f. The whole plan is built on `==` rather than tolerances, so
// the golden is written in C99 hex-float form, which is exact and round-trips.
// A decimal golden would silently absorb the low-bit drift this test exists to
// catch.
//
// WHY A SYNTHETIC MODEL AND NOT data/shakespeare.ckpt. `data/` is gitignored --
// that checkpoint is 9.7 MB and is never committed. A test needing it could not
// be green on a clean checkout, which docs/engineering-lessons.md L11 forbids.
// The model here is built in-process from a fixed seed, so the fixture is a few
// hundred bytes of text and the test is hermetic. Real-model numbers live in
// docs/measurements.md with a reproduce command, which is this repo's existing
// convention for exactly that.
//
// DELIBERATE SCOPE NOTE. This covers the library, not tools/inspect.cpp's JSON
// emission, which lives inside that tool's main() and would need a refactor to
// reach from a unit test. The JSON schema is covered by
// //tests/integration:e2e_pipeline_test. The library is what Steps 1-4 change.
//
// TO REGENERATE after an INTENTIONAL semantic change: run the test, copy the
// "computed golden" block it prints on failure into
// tests/fixtures/interpret_golden.txt, and say in the PR why the numbers moved.
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cppgpt/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

namespace {

// The fixed synthetic model. Same shape as interpret_test's, so the two tests
// exercise the same configuration and a difference between them is a real
// difference rather than a shape artifact.
constexpr std::uint64_t kSeed = 31337ULL;
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

// One golden line. Hex float, so equality is bit equality.
void emit(std::vector<std::string>& out, const char* key, float v) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s %a", key, static_cast<double>(v));
    out.emplace_back(buf);
}

void emit_i(std::vector<std::string>& out, const char* key, int v) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s %d", key, v);
    out.emplace_back(buf);
}

// Argmax over one row of logits; ties broken by lowest index so the golden is
// deterministic even when two logits are exactly equal.
int argmax(const float* row, int n) noexcept {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (row[i] > row[best]) best = i;
    return best;
}

std::vector<std::string> compute_golden() {
    const Config cfg = make_config();
    const int L = cfg.n_layer, NH = cfg.n_head, V = cfg.vocab_size, C = cfg.n_embd;
    const auto n = static_cast<std::size_t>(kB) * kT;

    Generator g(kSeed);
    GPT2 m(cfg, kB, kT);
    m.init_weights(g);

    std::vector<int> tok(n);
    for (auto& x : tok) x = static_cast<int>(g.uniform_int(0, V - 1));
    m.forward(tok.data(), nullptr);

    std::vector<std::string> out;
    char key[128];

    // ---- tokens, so a change in the input is not mistaken for a change in the model ----
    for (std::size_t i = 0; i < n; ++i) {
        std::snprintf(key, sizeof(key), "token.%zu", i);
        emit_i(out, key, tok[i]);
    }

    // ---- logit lens: top-1 id and logit at every (layer, position) ----
    {
        std::vector<float> lens(n * static_cast<std::size_t>(V));
        std::vector<float> scratch(n * static_cast<std::size_t>(C) + 2 * n);
        for (int l = 0; l < L; ++l) {
            logit_lens(m, l, lens.data(), scratch.data());
            for (std::size_t p = 0; p < n; ++p) {
                const float* row = lens.data() + p * static_cast<std::size_t>(V);
                const int top = argmax(row, V);
                std::snprintf(key, sizeof(key), "lens.L%d.p%zu.top1", l, p);
                emit_i(out, key, top);
                std::snprintf(key, sizeof(key), "lens.L%d.p%zu.logit", l, p);
                emit(out, key, row[top]);
            }
        }
    }

    // ---- direct logit attribution at the last position, for the model's own top-1 ----
    {
        const int pos = kT - 1;
        const float* final_row =
            m.acts().logits + static_cast<std::size_t>(pos) * static_cast<std::size_t>(V);
        const int token = argmax(final_row, V);
        emit_i(out, "attr.token", token);

        std::vector<float> heads(static_cast<std::size_t>(L) * NH);
        std::vector<float> mlps(static_cast<std::size_t>(L));
        std::vector<float> scratch(static_cast<std::size_t>(4 * C));
        float embed = 0.0f, bias = 0.0f;
        direct_logit_attribution(m, pos, token, heads.data(), mlps.data(), &embed, &bias,
                                 scratch.data());
        for (int l = 0; l < L; ++l)
            for (int h = 0; h < NH; ++h) {
                std::snprintf(key, sizeof(key), "attr.L%d.H%d", l, h);
                emit(out, key, heads[static_cast<std::size_t>(l) * NH + h]);
            }
        for (int l = 0; l < L; ++l) {
            std::snprintf(key, sizeof(key), "attr.L%d.mlp", l);
            emit(out, key, mlps[static_cast<std::size_t>(l)]);
        }
        emit(out, "attr.embed", embed);
        emit(out, "attr.bias", bias);
    }

    // ---- ablation KL for every component, at the last position ----
    {
        const int pos = kT - 1;
        std::vector<float> base_probs(static_cast<std::size_t>(V));
        softmax_into(base_probs.data(),
                     m.acts().logits + static_cast<std::size_t>(pos) * static_cast<std::size_t>(V),
                     V);

        std::vector<float> saved(ablation_scratch(cfg));
        std::vector<float> probs(static_cast<std::size_t>(V));

        const auto sweep = [&](Ablation kind, const char* name, int l, int h) {
            save_and_ablate(m, kind, l, h, saved.data());
            m.forward(tok.data(), nullptr);
            softmax_into(
                probs.data(),
                m.acts().logits + static_cast<std::size_t>(pos) * static_cast<std::size_t>(V), V);
            restore_ablation(m, kind, l, h, saved.data());
            const double kl = kl_divergence(base_probs.data(), probs.data(), V);
            if (h >= 0)
                std::snprintf(key, sizeof(key), "abl.L%d.%s%d", l, name, h);
            else
                std::snprintf(key, sizeof(key), "abl.L%d.%s", l, name);
            emit(out, key, static_cast<float>(kl));
        };

        for (int l = 0; l < L; ++l) {
            for (int h = 0; h < NH; ++h) sweep(Ablation::Head, "H", l, h);
            sweep(Ablation::Mlp, "mlp", l, -1);
            sweep(Ablation::AttnBlock, "attn", l, -1);
        }

        // After a full sweep the model must be bit-identical to before it. A
        // missed restore would corrupt every later entry in the sweep while
        // still producing plausible numbers -- so check, rather than trust.
        m.forward(tok.data(), nullptr);
        for (std::size_t i = 0; i < static_cast<std::size_t>(V); ++i) {
            std::snprintf(key, sizeof(key), "restored.%zu", i);
            emit(out, key,
                 m.acts().logits[static_cast<std::size_t>(pos) * static_cast<std::size_t>(V) + i]);
        }
    }

    return out;
}

std::vector<std::string> read_fixture(const char* path) {
    std::vector<std::string> lines;
    std::FILE* f = std::fopen(path, "r");
    if (f == nullptr) return lines;  // caller reports the miss; empty is never a pass
    char buf[256];
    while (std::fgets(buf, sizeof(buf), f) != nullptr) {
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (!s.empty() && s[0] != '#') lines.push_back(std::move(s));
    }
    (void)std::fclose(f);
    return lines;
}

}  // namespace

int main() {
    const std::vector<std::string> got = compute_golden();
    const std::vector<std::string> want = read_fixture("tests/fixtures/interpret_golden.txt");

    // An empty or missing fixture is a FAILURE, not a vacuous pass -- the same
    // reason test::summary() fails a file that ran zero checks.
    CHECK(!want.empty());
    CHECK(got.size() == want.size());

    bool identical = got.size() == want.size();
    if (identical)
        for (std::size_t i = 0; i < got.size(); ++i)
            if (got[i] != want[i]) {
                identical = false;
                std::fprintf(stderr, "  first diff at line %zu:\n    want: %s\n    got:  %s\n", i + 1,
                             want[i].c_str(), got[i].c_str());
                break;
            }
    CHECK(identical);

    if (!identical) {
        std::fprintf(stderr, "\n--- computed golden (copy into tests/fixtures/interpret_golden.txt) ---\n");
        for (const auto& s : got) std::fprintf(stderr, "%s\n", s.c_str());
        std::fprintf(stderr, "--- end ---\n");
    }
    return cppgpt::test::summary();
}
