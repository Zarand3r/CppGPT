// The intervention seam: replace one activation mid-forward.
//
// WHY THIS HEADER EXISTS SEPARATELY. `GPT2::forward` has to see this vocabulary,
// and `interpret.hpp` already includes `model.hpp`, so putting it there would be
// circular. Keeping it in its own header means `model.hpp` gains one include and
// no interpretability concepts of its own, and there is exactly one file to read
// to understand where the core/interpretability boundary runs.
//
// WHY A PATCH ARGUMENT AND NOT A HOOK. A callback in the per-layer loop is a
// function pointer on the numerical path — indirection the repo's doctrine
// forbids, and one the optimizer cannot see through. A POD consulted through a
// null check costs nothing and keeps the forward pass readable.
//
// WHY THE FORWARD MUST BE ENTERED AT ALL. Patching a head's output shifts
// `attproj` by a DIFFERENT amount at every position. `attprojb` is per-channel,
// so no modification of any weight can express that. This is why
// `save_and_ablate` can only ever express *zero* ablation, and why the field
// uses hooks. There is no way to get interchange interventions from outside the
// forward pass.
//
// OWNERSHIP. `Patch::replacement` is BORROWED — caller-owned, and it must
// outlive the `forward` call. Nothing here allocates.
#pragma once

#include <cstddef>
#include <cstring>

#include "cppgpt/core.hpp"
#include "cppgpt/model_config.hpp"

namespace cppgpt {

// Where an intervention can be applied. Deliberately one-for-one with
// `interpret.hpp`'s `Ablation`, so the two vocabularies cannot drift: every
// ablation is expressible as a patch of the same component, and the unit test
// pins that correspondence at zero.
//
//   HeadOut       — one head's channel block of `atty`, before attproj reads it.
//   AttnBlockOut  — a whole attention block's `attproj`, before the residual add.
//   MlpOut        — a whole MLP block's `fcproj`, before the residual add.
enum class PatchSite { HeadOut, AttnBlockOut, MlpOut };

// Floats a replacement buffer must hold for `site` at this config's B and T.
// HeadOut covers one head (C / n_head channels); the block sites cover all C.
[[nodiscard]] inline std::size_t patch_floats(const Config& cfg, PatchSite site, int B,
                                              int T) noexcept {
    const auto bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    const auto C = static_cast<std::size_t>(cfg.n_embd);
    if (site == PatchSite::HeadOut) {
        ASSERT(cfg.n_head > 0 && cfg.n_embd % cfg.n_head == 0);
        return bt * (C / static_cast<std::size_t>(cfg.n_head));
    }
    return bt * C;
}

// One intervention. POD: no allocation, no ownership, no lifetime of its own.
// `head` is ignored unless `site == PatchSite::HeadOut`.
struct Patch {
    PatchSite site;
    int layer;
    int head;
    const float* replacement;  // BORROWED; patch_floats() elements
};

// Fail fast on a patch that names something that does not exist.
//
// WITHOUT THIS the failure is SILENT: apply_patch only fires when the layer
// matches, so `Patch{HeadOut, 99, ...}` on a 3-layer model matches nothing, the
// forward runs clean, and the caller reads an UNPATCHED result as a patched
// one. Every downstream number would be wrong and nothing would say so. Found
// in the Step 1 review; the repo's rule is that an invariant violation aborts
// and a degradation is never silent.
inline void validate_patch(const Patch& patch, int n_layer, int n_head) noexcept {
    ASSERT_MSG(patch.layer >= 0 && patch.layer < n_layer, "patch: layer out of range");
    ASSERT_MSG(patch.replacement != nullptr, "patch: replacement is null");
    ASSERT_MSG(patch.site != PatchSite::HeadOut || (patch.head >= 0 && patch.head < n_head),
               "patch: head out of range");
}

// Validate a whole set, and reject two patches writing the same region.
//
// Without the duplicate check, `{HeadOut L0H1 = a, HeadOut L0H1 = b}` would
// silently apply whichever came last and report a measurement of one
// intervention as though it were two -- the same silent-wrong-answer shape as a
// patch naming a layer that does not exist.
//
// The pointer and count must agree: `forward(..., &p)` with a defaulted count of
// zero would apply NOTHING and return a clean result the caller reads as
// patched.
inline void validate_patches(const Patch* patches, int n, int n_layer, int n_head) noexcept {
    ASSERT_MSG((patches == nullptr) == (n == 0), "patch set: pointer and count disagree");
    ASSERT_MSG(n >= 0, "patch set: negative count");
    for (int i = 0; i < n; ++i) {
        validate_patch(patches[i], n_layer, n_head);
        for (int j = 0; j < i; ++j)
            ASSERT_MSG(!(patches[i].site == patches[j].site &&
                         patches[i].layer == patches[j].layer &&
                         (patches[i].site != PatchSite::HeadOut ||
                          patches[i].head == patches[j].head)) ,
                       "patch set: two patches write the same region");
    }
}

namespace detail {

// Write `patch->replacement` into `dest` when the patch targets (site, layer).
// `dest` is the layer's [B, T, C] slice of the activation being written.
//
// The block sites are a flat copy of the whole slice. HeadOut is strided: head
// h owns channels [h*hs, (h+1)*hs) of every position, which is the same slice
// `save_and_ablate` reaches through attprojw's column block.
//
// A no-op when there is no patch, when it targets another site, or when it
// targets another layer — so the call sites in the forward pass are
// unconditional and the branch is one predictable compare per layer.
inline void apply_one(const Patch* patch, PatchSite site, int layer, float* dest, int B, int T,
                      int C, int NH) noexcept {
    if (patch->site != site || patch->layer != layer) return;
    ASSERT(patch->replacement != nullptr);

    const auto bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    if (site != PatchSite::HeadOut) {
        std::memcpy(dest, patch->replacement, bt * static_cast<std::size_t>(C) * sizeof(float));
        return;
    }

    ASSERT(NH > 0 && C % NH == 0);
    ASSERT(patch->head >= 0 && patch->head < NH);
    const auto hs = static_cast<std::size_t>(C / NH);
    for (std::size_t i = 0; i < bt; ++i)
        std::memcpy(dest + i * static_cast<std::size_t>(C) + static_cast<std::size_t>(patch->head) * hs,
                    patch->replacement + i * hs, hs * sizeof(float));
}

// Read the site's current value OUT of the same slice, into `out`. The donor
// half of an interchange intervention: run a forward on the donor prompt, call
// this, then pass the buffer back as a Patch on the clean prompt.
inline void capture_from(PatchSite site, int head, const float* src, float* out, int B, int T, int C,
                         int NH) noexcept {
    const auto bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    if (site != PatchSite::HeadOut) {
        std::memcpy(out, src, bt * static_cast<std::size_t>(C) * sizeof(float));
        return;
    }
    ASSERT(NH > 0 && C % NH == 0);
    ASSERT(head >= 0 && head < NH);
    const auto hs = static_cast<std::size_t>(C / NH);
    for (std::size_t i = 0; i < bt; ++i)
        std::memcpy(out + i * hs,
                    src + i * static_cast<std::size_t>(C) + static_cast<std::size_t>(head) * hs,
                    hs * sizeof(float));
}

// Apply every patch in the set that targets (site, layer).
//
// A SET rather than a single patch because conditional co-ablation silences a
// primary component and then measures a second one, which means two live
// interventions in one forward. Order within the set does not matter: each patch
// writes a disjoint region (a distinct site, layer, or head), and a set with two
// patches on the SAME region is a caller error that validate_patches rejects
// rather than resolving by whichever happens to be written last.
inline void apply_patches(const Patch* patches, int n, PatchSite site, int layer, float* dest, int B,
                          int T, int C, int NH) noexcept {
    for (int i = 0; i < n; ++i) apply_one(&patches[i], site, layer, dest, B, T, C, NH);
}

}  // namespace detail
}  // namespace cppgpt
