// cppgpt GPT-2 model: parameter/activation arenas and the forward pass.
//
// Parameters and activations each live in one Storage arena; ParamTensors /
// ActTensors are non-owning views (pointers) into them, laid out in llm.c .bin
// order so a future weight loader and PyTorch fixtures stay interchangeable.
// The classifier head is weight-tied to the token embedding (lm_head == wte), so
// there is no separate lm_head parameter.
//
// This header is forward-only for now; gpt2_backward + AdamW arrive next.
#pragma once

#include <cstddef>

#include "cppgpt/random.hpp"
#include "cppgpt/storage.hpp"

namespace cppgpt {

struct Config {
    int max_seq_len;  // maximum context length (rows of wpe)
    int vocab_size;   // V
    int n_layer;      // L
    int n_head;       // NH (must divide n_embd)
    int n_embd;       // C
};

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

class GPT2 {
public:
    // Allocates the parameter and activation arenas for `cfg` at batch dims B, T.
    GPT2(const Config& cfg, int B, int T);

    // Initialize parameters with the canonical GPT-2 scheme: weights N(0, 0.02);
    // residual-projection weights (attproj, fcproj) N(0, 0.02/√(2L)); biases 0;
    // LayerNorm gains 1, shifts 0.
    void init_weights(Generator& gen);

    // Run the forward pass for `tokens`/`targets` (each [B*T], ids in [0,V)),
    // filling activations and `mean_loss`. B, T must match construction.
    void forward(const int* tokens, const int* targets, int B, int T);

    // Zero the parameter-gradient and activation-gradient arenas. Call before
    // backward(), since every op's backward accumulates (+=).
    void zero_grads() noexcept;

    // Backward pass for the same tokens/targets as the preceding forward(),
    // accumulating into the gradient arenas. `dwte` receives contributions from
    // both the classifier (tied head) and the embedding paths (weight tying).
    void backward(const int* tokens, const int* targets, int B, int T);

    // AdamW step over all parameters using the current gradients (from backward).
    // Canonical GPT-2 2-group weight decay: only weight matrices and embeddings
    // (wte, wpe, and the per-layer qkvw/attprojw/fcw/fcprojw) decay; biases and
    // LayerNorm gains/shifts do not. The optimizer moments are allocated lazily on
    // the first call (inference-only use pays no moment memory). Advances the
    // internal step counter used for bias correction.
    void update(float lr, float beta1, float beta2, float eps, float weight_decay) noexcept;

    [[nodiscard]] const Config& config() const noexcept { return cfg_; }
    [[nodiscard]] const ParamTensors& params() const noexcept { return params_; }
    [[nodiscard]] ParamTensors& params() noexcept { return params_; }  // mutable: optimizer / tests
    [[nodiscard]] const ParamTensors& grads() const noexcept { return grads_; }
    [[nodiscard]] const ActTensors& acts() const noexcept { return acts_; }
    [[nodiscard]] float mean_loss() const noexcept { return mean_loss_; }

private:
    Config cfg_;
    int B_, T_;
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
};

}  // namespace cppgpt
