// ablation_stats — is a component important, or was it important on ONE prompt?
//
// Every ablation and attribution number in tools/inspect comes from a single
// forward pass. That is enough to say "silencing L2H0 moved THIS output by 0.63
// nats" and not enough to say anything about L2H0. Generalising from one input is
// the standard error in this field, and the viewer currently invites it.
//
// This runs the same exhaustive sweep over many windows sampled from a corpus and
// reports, per component, the mean effect and how much it VARIES. A head with a
// large mean and small spread is structural; one with a large spread was
// important for particular inputs and the single-prompt view overstated it.
//
// Data layout: components are a flat array indexed by layer * (n_head + 2) + slot,
// with Welford accumulators in parallel double arrays. One model, one saved-weight
// buffer and one probability pair are allocated up front and reused for every one
// of the N * L * (NH + 2) forward passes -- nothing is allocated inside the sweep.
//
// Usage: ablation_stats --checkpoint X.ckpt --data corpus.bin
//                       [--prompts 64] [--seq 32] [--seed 1337] [--top 12]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/core.hpp"
#include "cppgpt/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;

// All samples are kept, not summarised on the fly. The first version used
// Welford, which is the reflex for "accumulate a mean" -- but N is small and
// chosen by the caller (24 components x 1000 prompts is 192 KB), and streaming
// cost the MEDIAN. That turned out to matter: ablation effects are heavy-tailed
// (one component ranged 0.035 to 7.50 over 64 prompts), so the mean is set by a
// handful of prompts and says little about a typical one. Simpler and more
// informative.
struct Samples {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    [[nodiscard]] double mean() const {
        double s = 0.0;
        for (const double x : v) s += x;
        return v.empty() ? 0.0 : s / static_cast<double>(v.size());
    }
    [[nodiscard]] double quantile(double q) const {  // v is sorted by the caller first
        if (v.empty()) return 0.0;
        const auto i = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5);
        return v[std::min(i, v.size() - 1)];
    }
    // The fraction of prompts on which this component did anything at all. For a
    // heavy-tailed effect this separates "always somewhat involved" from
    // "irrelevant except on a few inputs" -- which a mean cannot.
    [[nodiscard]] double active(double thresh) const {
        std::size_t n = 0;
        for (const double x : v) n += (x >= thresh) ? 1 : 0;
        return v.empty() ? 0.0 : static_cast<double>(n) / static_cast<double>(v.size());
    }
};

std::vector<std::uint16_t> read_tokens(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "ablation_stats: cannot open '%s'\n", path.c_str());
        std::exit(1);
    }
    const std::streamoff bytes = f.tellg();
    if (bytes < 0 || bytes % 2 != 0) {
        std::fprintf(stderr, "ablation_stats: '%s' is not a whole number of uint16 tokens\n",
                     path.c_str());
        std::exit(1);
    }
    std::vector<std::uint16_t> t(static_cast<std::size_t>(bytes) / 2);
    f.seekg(0);
    if (!t.empty()) f.read(reinterpret_cast<char*>(t.data()), bytes);
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv,
                         {"checkpoint", "data", "prompts", "seq", "seed", "top", "baseline"});
    const std::string ckpt(args.str("checkpoint", ""));
    const std::string data(args.str("data", ""));
    if (ckpt.empty() || data.empty()) {
        std::fprintf(stderr,
                     "usage: ablation_stats --checkpoint X.ckpt --data corpus.bin\n"
                     "                      [--prompts 64] [--seq 32] [--seed 1337] [--top 12]\n"
                     "                      [--baseline zero|donor]\n");
        return 2;
    }
    // An unknown baseline is refused rather than defaulted: silently falling back
    // to zero would report an off-distribution measurement under whatever name
    // the caller asked for.
    const std::string baseline(args.str("baseline", "zero"));
    if (baseline != "zero" && baseline != "donor") {
        std::fprintf(stderr, "ablation_stats: --baseline must be 'zero' or 'donor', got '%s'\n",
                     baseline.c_str());
        return 2;
    }
    const bool use_donor = (baseline == "donor");
    const int n_prompts = args.integer("prompts", 64);
    const int show = args.integer("top", 12);
    if (n_prompts < 2) {
        std::fprintf(stderr, "ablation_stats: --prompts must be >= 2 (variance needs a sample)\n");
        return 2;
    }

    auto peek = CheckpointFile::open(ckpt.c_str());
    if (!peek) {
        std::fprintf(stderr, "ablation_stats: cannot read '%s': %s\n", ckpt.c_str(),
                     describe(peek.error()));
        return 1;
    }
    const CheckpointHeader& h = peek->header();
    const Config cfg{h.max_seq_len, h.vocab_size, h.n_layer, h.n_head, h.n_embd};
    const int T = std::min(args.integer("seq", 32), cfg.max_seq_len);
    if (T < 2) {
        std::fprintf(stderr, "ablation_stats: --seq must be >= 2\n");
        return 2;
    }

    const std::vector<std::uint16_t> toks = read_tokens(data);
    if (toks.size() < static_cast<std::size_t>(T) + 1) {
        std::fprintf(stderr, "ablation_stats: corpus has %zu tokens, need at least %d\n",
                     toks.size(), T + 1);
        return 1;
    }

    GPT2 model(cfg, 1, T);
    if (const auto r = model.load_checkpoint(ckpt.c_str(), GPT2::LoadMode::WeightsOnly); !r) {
        std::fprintf(stderr, "ablation_stats: cannot load '%s': %s\n", ckpt.c_str(),
                     describe(r.error()));
        return 1;
    }

    const int L = cfg.n_layer, NH = cfg.n_head, V = cfg.vocab_size;
    const int per_layer = NH + 2;              // NH heads, then the MLP, then the attn block
    const int n_comp = L * per_layer;
    std::vector<Samples> stat(static_cast<std::size_t>(n_comp));
    for (Samples& x : stat) x.v.reserve(static_cast<std::size_t>(n_prompts));

    // Everything the sweep touches is allocated once, here.
    std::vector<int> window(static_cast<std::size_t>(T));
    std::vector<float> saved(ablation_scratch(cfg));
    std::vector<float> p_base(static_cast<std::size_t>(V)), p_abl(static_cast<std::size_t>(V));
    Generator gen(static_cast<std::uint64_t>(args.integer("seed", 1337)));
    const std::size_t hi = toks.size() - static_cast<std::size_t>(T) - 1;
    const auto last = static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(V);

    // Every window start is drawn up front, so the donor for window p can be a
    // FIXED function of p -- the next window in the same list. No second
    // generator, no sampling inside the sweep, so the donor baseline is
    // deterministic by construction rather than by seeding discipline. Windows
    // are all T tokens, so a donor always matches the shape it replaces.
    std::vector<std::size_t> starts(static_cast<std::size_t>(n_prompts));
    for (auto& s : starts)
        s = static_cast<std::size_t>(gen.uniform_int(0, static_cast<std::int64_t>(hi)));

    std::vector<int> donor_win(static_cast<std::size_t>(T));
    const std::size_t stride = use_donor ? patch_floats(cfg, PatchSite::AttnBlockOut, 1, T) : 0;
    std::vector<float> repl(use_donor ? static_cast<std::size_t>(n_comp) * stride : 0);

    std::printf("ablation_stats: %s over %s\n", ckpt.c_str(), data.c_str());
    std::printf("  L%d H%d V%d | %d prompts x %d tokens | %d components, %d forwards | baseline %s\n",
                L, NH, V, n_prompts, T, n_comp, n_prompts * (n_comp + 1 + (use_donor ? 1 : 0)),
                baseline.c_str());

    const auto fill = [&](std::vector<int>& dst, std::size_t start) {
        for (int t = 0; t < T; ++t)
            dst[static_cast<std::size_t>(t)] =
                static_cast<int>(toks[start + static_cast<std::size_t>(t)]);
    };

    for (int p = 0; p < n_prompts; ++p) {
        fill(window, starts[static_cast<std::size_t>(p)]);

        if (use_donor) {
            fill(donor_win, starts[static_cast<std::size_t>((p + 1) % n_prompts)]);
            model.forward(donor_win.data(), nullptr);
            for (int i = 0; i < n_comp; ++i) {
                const Component c = component_at(cfg, i);
                capture_site(model, c.site, c.layer, c.head,
                             repl.data() + static_cast<std::size_t>(i) * stride);
            }
        }

        model.forward(window.data(), nullptr);
        softmax_into(p_base.data(), model.acts().logits + last, V);

        // component_at rather than a fourth hand-written enumeration: inspect,
        // coax_sweep and this loop must agree on which index means which
        // component, and the arithmetic written out separately in each is how
        // one of them ends up measuring something else.
        for (int i = 0; i < n_comp; ++i) {
            const Component c = component_at(cfg, i);
            if (use_donor) {
                const Patch patch{c.site, c.layer, c.head,
                                  repl.data() + static_cast<std::size_t>(i) * stride};
                model.forward(window.data(), nullptr, -1, &patch, 1);
            } else {
                const Ablation kind = (c.site == PatchSite::HeadOut)  ? Ablation::Head
                                      : (c.site == PatchSite::MlpOut) ? Ablation::Mlp
                                                                      : Ablation::AttnBlock;
                save_and_ablate(model, kind, c.layer, c.head, saved.data());
                model.forward(window.data(), nullptr);
                softmax_into(p_abl.data(), model.acts().logits + last, V);
                restore_ablation(model, kind, c.layer, c.head, saved.data());
                const double dz = kl_divergence(p_base.data(), p_abl.data(), V);
                ASSERT_MSG(std::isfinite(dz), "ablation_stats: KL is not finite");
                stat[static_cast<std::size_t>(i)].add(dz);
                continue;
            }
            softmax_into(p_abl.data(), model.acts().logits + last, V);
            const double d = kl_divergence(p_base.data(), p_abl.data(), V);
            // A non-finite divergence means the ablated forward broke; that is
            // a real result about the model, not a number to average away.
            ASSERT_MSG(std::isfinite(d), "ablation_stats: KL is not finite");
            stat[static_cast<std::size_t>(i)].add(d);
        }
    }

    std::vector<int> order(static_cast<std::size_t>(n_comp));
    for (int i = 0; i < n_comp; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return stat[static_cast<std::size_t>(a)].mean() > stat[static_cast<std::size_t>(b)].mean();
    });

    for (Samples& x : stat) std::sort(x.v.begin(), x.v.end());
    std::printf("\n  %-10s %9s %9s %9s %9s %9s\n", "component", "mean KL", "median", "p90",
                "max", "active");
    const int n_show = std::min(show, n_comp);
    for (int i = 0; i < n_show; ++i) {
        const int c = order[static_cast<std::size_t>(i)];
        const Samples& s = stat[static_cast<std::size_t>(c)];
        char name[24];
        component_label(cfg, c, name, sizeof(name));
        std::printf("  %-10s %9.4f %9.4f %9.4f %9.4f %8.0f%%\n", name, s.mean(),
                    s.quantile(0.5), s.quantile(0.9), s.quantile(1.0), 100.0 * s.active(0.01));
    }
    if (n_show < n_comp) std::printf("  ... %d more (raise --top)\n", n_comp - n_show);

    // The headline: which components are consistent, and which were a story about
    // one prompt. cv is the discriminator, not the mean.
    // The discriminator is how OFTEN a component matters, not how much it matters
    // on average -- a mean of 4.9 built from a few huge prompts and many near-zero
    // ones is a different claim from a steady 4.9.
    int always = 0, sometimes = 0, rarely = 0;
    for (const Samples& s : stat) {
        const double a = s.active(0.01);
        if (a >= 0.9) ++always;
        else if (a >= 0.25) ++sometimes;
        else ++rarely;
    }
    std::printf("\n  active on >=90%% of prompts: %d   on 25-90%%: %d   on <25%%: %d\n",
                always, sometimes, rarely);
    std::printf("  A single-prompt ablation map cannot distinguish these. Compare any component's\n"
                "  median against the value tools/inspect reports for one prompt before believing it.\n");
    return 0;
}
