// cppgpt GPT-2 model: parameter/activation arenas and the full training step.
//
// Parameters and activations each live in one Storage arena; ParamTensors /
// ActTensors are non-owning views (pointers) into them, laid out in llm.c .bin
// order so a future weight loader and PyTorch fixtures stay interchangeable.
// The classifier head is weight-tied to the token embedding (lm_head == wte), so
// there is no separate lm_head parameter.
//
// The model owns the whole training step: forward (loss, or logits-only when
// targets == nullptr), backward, gradient clipping and the AdamW update, plus
// versioned checkpoint save/load (see cppgpt/checkpoint.hpp).
#pragma once

#include <cstddef>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/model_config.hpp"
#include "cppgpt/optimizer.hpp"
#include "cppgpt/patch.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/storage.hpp"

namespace cppgpt {

// Config lives in model_config.hpp so tensors.hpp can use it without
// depending on this header.

// Non-owning pointers into the parameter buffer, in llm.c .bin order. Per-layer
// tensors are stored contiguously as [L, ...].
struct ParamTensors {
    float* wte;       // [V, C]      token embedding (also the tied classifier)
    float* wpe;       // [maxT, C]   position embedding
    float* ln1w;      // [L, C]
    float* ln1b;      // [L, C]
    float* qkvw;      // [L, 3C, C]
    float* qkvb;      // [L, 3C]
    float* attprojw;  // [L, C, C]
    float* attprojb;  // [L, C]
    float* ln2w;      // [L, C]
    float* ln2b;      // [L, C]
    float* fcw;       // [L, 4C, C]
    float* fcb;       // [L, 4C]
    float* fcprojw;   // [L, C, 4C]
    float* fcprojb;   // [L, C]
    float* lnfw;      // [C]
    float* lnfb;      // [C]
};
inline constexpr int kNumParamTensors = 16;

// Non-owning pointers into the activation buffer (sized for a fixed B, T).
struct ActTensors {
    float* encoded;    // [B, T, C]
    float* ln1;        // [L, B, T, C]
    float* ln1_mean;   // [L, B, T]
    float* ln1_rstd;   // [L, B, T]
    float* qkv;        // [L, B, T, 3C]
    float* atty;       // [L, B, T, C]
    float* preatt;     // [L, B, NH, T, T]
    float* att;        // [L, B, NH, T, T]
    float* attproj;    // [L, B, T, C]
    float* residual2;  // [L, B, T, C]
    float* ln2;        // [L, B, T, C]
    float* ln2_mean;   // [L, B, T]
    float* ln2_rstd;   // [L, B, T]
    float* fch;        // [L, B, T, 4C]
    float* fch_gelu;   // [L, B, T, 4C]
    float* fcproj;     // [L, B, T, C]
    float* residual3;  // [L, B, T, C]
    float* lnf;        // [B, T, C]
    float* lnf_mean;   // [B, T]
    float* lnf_rstd;   // [B, T]
    float* logits;     // [B, T, V]
    float* probs;      // [B, T, V]
    float* losses;     // [B, T]
};
inline constexpr int kNumActTensors = 23;

// Per-layer slice of an activation shaped [L, B, T, C]. Consumers outside
// model.cpp must use this rather than recomputing the stride: tools/inspect.cpp
// derived it as `layer * T * C` and silently dropped the B factor, which was
// correct only because that tool happens to build with B == 1. Three separate
// re-derivations of a stride the model already knows is two too many.
[[nodiscard]] inline const float* layer_slice(const float* base, int layer, int B, int T,
                                              int C) noexcept {
    return base + static_cast<std::size_t>(layer) * static_cast<std::size_t>(B) *
                      static_cast<std::size_t>(T) * static_cast<std::size_t>(C);
}

// Per-(layer, head) slice of an attention tensor shaped [L, B, NH, T, T]. Same
// reason as above, and the same bug had survived TWICE in tools/inspect.cpp on
// this shape after the residual one was fixed — announcing the class retired
// while two instances remained.
[[nodiscard]] inline const float* head_slice(const float* base, int layer, int head, int B, int NH,
                                             int T) noexcept {
    const auto per_head = static_cast<std::size_t>(T) * static_cast<std::size_t>(T);
    const auto per_layer = static_cast<std::size_t>(B) * static_cast<std::size_t>(NH) * per_head;
    return base + static_cast<std::size_t>(layer) * per_layer +
           static_cast<std::size_t>(head) * per_head;
}

class GPT2 {
public:
    // Allocates the parameter and activation arenas for `cfg` at batch dims B, T.
    GPT2(const Config& cfg, int B, int T);

    // Initialize parameters with the canonical GPT-2 scheme: weights N(0, 0.02);
    // residual-projection weights (attproj, fcproj) N(0, 0.02/√(2L)); biases 0;
    // LayerNorm gains 1, shifts 0.
    void init_weights(Generator& gen);

    // `logits_at`: compute the classifier at ONE position (0-based within each
    // batch element) instead of all T. -1 means every position, as before.
    //
    // At GPT-2's V=50257 the tied head is [T,C] x [C,V] -- 79 GFLOP at T=1024,
    // about 1.6 s -- against 77 MFLOP for a single row. Generation reads exactly
    // one row per step, so this is a requirement for usable generation rather
    // than an optimisation. It is a POSITION and not a "last" flag because the
    // token buffer is right-padded: the row that matters is prompt length - 1,
    // which is only T-1 once the context has filled.
    //
    // Rows other than `logits_at` are left STALE, not zeroed -- zeroing costs the
    // bandwidth this exists to save, and a caller reading them wants the loud
    // wrongness. `targets` must be null, since the loss needs every position.

    // Run the forward pass for `tokens` (each [B*T] for the model's fixed B, T; ids
    // in [0,V)), filling activations. If `targets` is non-null, also computes the
    // softmax + cross-entropy and records `mean_loss`; pass nullptr for inference
    // (logits only — read them from acts().logits).
    //
    // `patch` (default none) replaces ONE activation site mid-pass — the seam
    // interchange interventions need, and the only thing in this class that
    // exists for the interpretability layer. See cppgpt/patch.hpp for why no
    // weight modification can express it. With `patch == nullptr` this function
    // is bit-identical to the pre-seam build; //tests/unit:interpret_golden_test
    // and //tests/integration:parity_test both pin that.
    void forward(const int* tokens, const int* targets, int logits_at = -1,
                 const Patch* patch = nullptr);

    // Zero the parameter-gradient and activation-gradient arenas. Call before
    // backward(), since every op's backward accumulates (+=).
    void zero_grads() noexcept;

    // Backward pass for the same tokens/targets as the preceding forward(),
    // accumulating into the gradient arenas. `dwte` receives contributions from
    // both the classifier (tied head) and the embedding paths (weight tying).
    // Requires that the preceding forward() was given targets (it consumes the
    // probabilities that only a loss-computing forward fills); asserts otherwise.
    void backward(const int* tokens, const int* targets);

    // AdamW step over all parameters using the current gradients (from backward).
    // Canonical GPT-2 2-group weight decay: only weight matrices and embeddings
    // (wte, wpe, and the per-layer qkvw/attprojw/fcw/fcprojw) decay; biases and
    // LayerNorm gains/shifts do not. The optimizer moments are allocated lazily on
    // the first call (inference-only use pays no moment memory). Advances the
    // internal step counter used for bias correction.
    void update(const AdamW& opt) noexcept;

    // Clip the global gradient L2 norm to `max_norm` over the whole gradient
    // arena, in place, and return the pre-clip norm (to log/monitor). Call between
    // backward() and update(). max_norm <= 0 disables clipping. See
    // cppgpt::clip_grad_norm.
    [[nodiscard]] float clip_grad_norm(float max_norm) noexcept;

    // Write a checkpoint to `path` (atomic tmp->rename): the weights, plus the
    // AdamW moments and step counter when the optimizer has run. NOTE: the RNG
    // engine state, the dataloader cursor and the LR-schedule position are NOT
    // saved, so a resumed run restores weights and optimizer state exactly but
    // replays the data order and the schedule from step 0 — resume is
    // state-exact, not trajectory-exact. See cppgpt/checkpoint.hpp for the format.
    // `tokenizer_kind`/`vocab_fingerprint` record which tokenizer these weights
    // expect (see cppgpt/checkpoint.hpp). They default to the char tokenizer with
    // no fingerprint, which is what every pre-v3 checkpoint effectively said.
    [[nodiscard]] Result<void> save_checkpoint(const char* path,
                                               std::uint32_t tokenizer_kind = kTokenizerChar,
                                               std::uint64_t vocab_fingerprint = 0) const noexcept;

    // How much of a checkpoint to restore.
    //   Full        — weights + AdamW moments + step: resuming an interrupted run.
    //   WeightsOnly — weights only; moments and step reset to zero. This is what
    //                 fine-tuning wants: the saved moments and step belong to the
    //                 base run's schedule, and carrying them into a new run at a
    //                 different LR applies stale, differently-scaled updates.
    enum class LoadMode { Full, WeightsOnly };

    // Load a checkpoint written by save_checkpoint. Validates magic/version, that
    // the file's Config and param_count match this model, and the payload
    // checksum — returning VersionMismatch / ShapeMismatch / CorruptCheckpoint /
    // ChecksumMismatch without touching the live arenas on failure. On success
    // restores weights (and moments + step, if the file carries them). A
    // moment-less file RESETS the optimizer moments and step to zero, so the
    // state always matches what is logged (a warned, bounded degradation, never
    // a silent carry-over of a previous run's moments).
    [[nodiscard]] Result<void> load_checkpoint(const char* path,
                                              LoadMode mode = LoadMode::Full) noexcept;

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }
    [[nodiscard]] int batch() const noexcept { return B_; }    // fixed batch size
    [[nodiscard]] int seq_len() const noexcept { return T_; }  // fixed context length
    [[nodiscard]] const ParamTensors& params() const noexcept { return params_; }
    // Mutable access is for the optimizer, tests, and `interpret.hpp`'s
    // save_and_ablate — the ONLY interpretability code that writes to the model.
    // Everything else in the interpretability layer reads through the const
    // params()/acts() overloads, and new work must keep it that way: an
    // intervention belongs in the forward pass behind an explicit patch
    // argument, not in a caller reaching in here to rewrite weights.
    [[nodiscard]] ParamTensors& params() noexcept { return params_; }
    [[nodiscard]] const ParamTensors& grads() const noexcept { return grads_; }
    [[nodiscard]] const ActTensors& acts() const noexcept { return acts_; }
    [[nodiscard]] float mean_loss() const noexcept { return mean_loss_; }

    // Total float count of the parameter arena (params()/grads() are contiguous
    // blocks of this many floats, in .bin order — for weight loading / fixtures).
    [[nodiscard]] std::size_t param_count() const noexcept { return param_count_; }

private:
    // Allocate + zero the AdamW moment arenas on first use (update() or a
    // moment-carrying load). Idempotent.
    void ensure_moment_arenas() noexcept;

    Config cfg_;
    int B_, T_;
    std::size_t param_count_ = 0;
    Storage param_store_;
    Storage grad_store_;
    Storage act_store_;
    Storage act_grad_store_;
    Storage scratch_store_;  // attention backward scratch (datt, dpreatt)
    Storage m_store_;        // AdamW first moments (lazily allocated in update())
    Storage v_store_;        // AdamW second moments (lazily allocated in update())
    ParamTensors params_{};
    ParamTensors grads_{};
    ActTensors acts_{};
    ActTensors act_grads_{};
    float* datt_ = nullptr;
    float* dpreatt_ = nullptr;
    float* m_ = nullptr;  // AdamW moment arenas, same flat layout as the params
    float* v_ = nullptr;
    int adam_step_ = 0;  // 1-based step counter for bias correction
    float mean_loss_ = 0.0f;
    // Whether the last forward() computed probs/mean_loss (i.e. had targets).
    // backward() consumes acts().probs, so running it after an inference-only
    // forward would silently produce gradients from another batch's (or
    // uninitialized) probabilities.
    bool has_loss_ = false;
};

}  // namespace cppgpt
