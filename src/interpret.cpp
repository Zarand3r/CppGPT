#include "cppgpt/interpret.hpp"

#include "cppgpt/core.hpp"
#include "cppgpt/ops.hpp"

namespace cppgpt {

void logit_lens(const GPT2& model, int layer, float* out_logits, float* scratch) noexcept {
    const Config& cfg = model.config();
    ASSERT(layer >= 0 && layer < cfg.n_layer);
    ASSERT(out_logits != nullptr && scratch != nullptr);

    const int B = model.batch(), T = model.seq_len();
    const int C = cfg.n_embd, V = cfg.vocab_size;
    const ActTensors& a = model.acts();
    const ParamTensors& p = model.params();

    // residual3 is [L, B, T, C]; take this layer's block.
    const float* residual =
        a.residual3 + static_cast<std::size_t>(layer) * B * T * static_cast<std::size_t>(C);

    // mean/rstd go in CALLER scratch, not the model's lnf_mean/lnf_rstd: writing
    // those would silently corrupt state a caller may still be inspecting, which
    // would make the "read-only" contract above a lie.
    const auto bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    float* norm = scratch;
    float* mean = scratch + bt * static_cast<std::size_t>(C);
    float* rstd = mean + bt;
    layernorm_forward(norm, mean, rstd, residual, p.lnfw, p.lnfb, B, T, C);
    matmul_forward(out_logits, norm, p.wte, nullptr, B, T, C, V);  // tied classifier
}

}  // namespace cppgpt
