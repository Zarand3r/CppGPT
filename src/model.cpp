#include "cppgpt/model.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/core.hpp"
#include "cppgpt/log.hpp"
#include "cppgpt/ops.hpp"
#include "cppgpt/tensors.hpp"

namespace cppgpt {
namespace {

// Parameter tensor sizes, derived from the table in tensors.hpp. The table is
// the single source of truth; tensors_test proves it reproduces the layout this
// function used to hard-code.
std::size_t param_sizes(const Config& c, std::size_t s[kNumParamTensors]) {
    static_assert(kNumParams == kNumParamTensors, "table and ParamTensors disagree");
    // ...and tie that literal to the STRUCT, not just to itself. Without this,
    // adding a field to ParamTensors without touching the table or the literal
    // compiled clean and left the new pointer null.
    static_assert(sizeof(ParamTensors) == kNumParamTensors * sizeof(float*),
                  "ParamTensors field count does not match kNumParamTensors");
    std::size_t tot = 0;
    for (int i = 0; i < kNumParams; ++i) {
        s[i] = param_total(kParamSpecs[i], c);
        tot += s[i];
    }
    return tot;
}

void point_params(ParamTensors& p, float* base, const std::size_t s[kNumParamTensors]) {
    float* ptr[kNumParamTensors];
    std::size_t off = 0;
    for (int i = 0; i < kNumParamTensors; ++i) {
        ptr[i] = base + off;
        off += s[i];
    }
    p.wte = ptr[0];   p.wpe = ptr[1];   p.ln1w = ptr[2];     p.ln1b = ptr[3];
    p.qkvw = ptr[4];  p.qkvb = ptr[5];  p.attprojw = ptr[6]; p.attprojb = ptr[7];
    p.ln2w = ptr[8];  p.ln2b = ptr[9];  p.fcw = ptr[10];     p.fcb = ptr[11];
    p.fcprojw = ptr[12]; p.fcprojb = ptr[13]; p.lnfw = ptr[14]; p.lnfb = ptr[15];
}

// Activation tensor sizes, derived from the table in tensors.hpp.
std::size_t act_sizes(const Config& c, int B, int T, std::size_t s[kNumActTensors]) {
    static_assert(kNumActs == kNumActTensors, "activation table and ActTensors disagree");
    static_assert(sizeof(ActTensors) == kNumActTensors * sizeof(float*),
                  "ActTensors field count does not match kNumActTensors");
    std::size_t tot = 0;
    for (int i = 0; i < kNumActs; ++i) {
        s[i] = act_total(kActSpecs[i], c, B, T);
        tot += s[i];
    }
    return tot;
}

void point_acts(ActTensors& a, float* base, const std::size_t s[kNumActTensors]) {
    float* ptr[kNumActTensors];
    std::size_t off = 0;
    for (int i = 0; i < kNumActTensors; ++i) {
        ptr[i] = base + off;
        off += s[i];
    }
    a.encoded = ptr[0];   a.ln1 = ptr[1];      a.ln1_mean = ptr[2];  a.ln1_rstd = ptr[3];
    a.qkv = ptr[4];       a.atty = ptr[5];     a.preatt = ptr[6];    a.att = ptr[7];
    a.attproj = ptr[8];   a.residual2 = ptr[9]; a.ln2 = ptr[10];     a.ln2_mean = ptr[11];
    a.ln2_rstd = ptr[12]; a.fch = ptr[13];     a.fch_gelu = ptr[14]; a.fcproj = ptr[15];
    a.residual3 = ptr[16]; a.lnf = ptr[17];    a.lnf_mean = ptr[18]; a.lnf_rstd = ptr[19];
    a.logits = ptr[20];   a.probs = ptr[21];   a.losses = ptr[22];
}

void fill_normal(float* p, std::size_t n, float std, Generator& gen) {
    for (std::size_t i = 0; i < n; ++i) p[i] = std * gen.normal();
}
void fill_const(float* p, std::size_t n, float v) {
    for (std::size_t i = 0; i < n; ++i) p[i] = v;
}

}  // namespace

GPT2::GPT2(const Config& cfg, int B, int T) : cfg_(cfg), B_(B), T_(T) {
    ASSERT(cfg.n_embd > 0 && cfg.n_head > 0 && cfg.n_embd % cfg.n_head == 0);
    ASSERT(cfg.n_layer >= 0 && cfg.vocab_size > 0 && T <= cfg.max_seq_len);

    // Parameters and their gradients share the same layout.
    std::size_t ps[kNumParamTensors];
    const std::size_t ptot = param_sizes(cfg, ps);
    // The per-tensor op signatures take `int` counts, so the whole arena must be
    // addressable as one: past INT_MAX the casts would silently wrap and turn
    // clip_grad_norm into a no-op. Fail fast at construction instead.
    ASSERT_MSG(ptot <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
               "GPT2: parameter count exceeds INT_MAX");
    param_count_ = ptot;
    param_store_ = Storage(ptot);
    grad_store_ = Storage(ptot);
    point_params(params_, param_store_.alloc(ptot), ps);
    point_params(grads_, grad_store_.alloc(ptot), ps);

    // Activations and their gradients share the same layout.
    std::size_t as[kNumActTensors];
    const std::size_t atot = act_sizes(cfg, B, T, as);
    // Same guard as the parameters above, and MORE load-bearing: the activation
    // arena is the larger of the two at any realistic B and T, and the forward
    // pass narrows element counts to int on every op (`static_cast<int>(BTC)`).
    // Guarding params but not activations left the bigger arena unchecked. The
    // total is a conservative bound — every individual tensor is a subset of it,
    // so if the total fits in int, every per-op count does.
    ASSERT_MSG(atot <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
               "GPT2: activation count exceeds INT_MAX (reduce batch or context)");
    act_store_ = Storage(atot);
    act_grad_store_ = Storage(atot);
    point_acts(acts_, act_store_.alloc(atot), as);
    point_acts(act_grads_, act_grad_store_.alloc(atot), as);

    // Attention-backward scratch: datt, dpreatt, each [B, NH, T, T]. One block,
    // split in two (a second alloc() would re-align the bump head and overflow).
    const auto att = static_cast<std::size_t>(B) * cfg.n_head * T * T;
    scratch_store_ = Storage(2 * att);
    datt_ = scratch_store_.alloc(2 * att);
    dpreatt_ = datt_ + att;
}

void GPT2::zero_grads() noexcept {
    grad_store_.zero();
    act_grad_store_.zero();
}

void GPT2::init_weights(Generator& gen) {
    // Canonical GPT-2 init, driven by the table's Init column: weights N(0,0.02),
    // residual projections N(0, 0.02/sqrt(2L)), LayerNorm gains 1, biases 0.
    // Previously a hand-written list that had to stay in step with param_sizes,
    // point_params and kDecay by eye.
    const float w_std = 0.02f;
    const float proj_std =
        0.02f / std::sqrt(2.0f * static_cast<float>(cfg_.n_layer > 0 ? cfg_.n_layer : 1));
    std::size_t ps[kNumParamTensors];
    param_sizes(cfg_, ps);
    float* p = params_.wte;  // base of the contiguous arena, in .bin order
    std::size_t off = 0;
    for (int i = 0; i < kNumParams; ++i) {
        const std::size_t n = ps[i];
        switch (kParamSpecs[i].init) {
            case Init::Normal:     fill_normal(p + off, n, w_std, gen); break;
            case Init::NormalProj: fill_normal(p + off, n, proj_std, gen); break;
            case Init::Zero:       fill_const(p + off, n, 0.0f); break;
            case Init::One:        fill_const(p + off, n, 1.0f); break;
        }
        off += n;
    }
}

void GPT2::forward(const int* tokens, const int* targets, int logits_at, const Patch* patch) {
    const int B = B_, T = T_;
    const int C = cfg_.n_embd, L = cfg_.n_layer, NH = cfg_.n_head, V = cfg_.vocab_size;
    const auto BTC = static_cast<std::size_t>(B) * T * C;
    const auto BT = static_cast<std::size_t>(B) * T;
    const auto ATT = static_cast<std::size_t>(B) * NH * T * T;
    const auto Cz = static_cast<std::size_t>(C);
    const ParamTensors& p = params_;
    const ActTensors& a = acts_;

    if (patch != nullptr) validate_patch(*patch, L, NH);

    embedding_forward(a.encoded, tokens, p.wte, p.wpe, B, T, C, V);

    for (int l = 0; l < L; ++l) {
        const auto ll = static_cast<std::size_t>(l);
        // Input residual: encoded for layer 0, else the previous block output.
        const float* res = (l == 0) ? a.encoded : a.residual3 + (ll - 1) * BTC;
        // Per-layer parameter slices.
        const float* ln1w = p.ln1w + ll * Cz;
        const float* ln1b = p.ln1b + ll * Cz;
        const float* qkvw = p.qkvw + ll * 3 * Cz * Cz;
        const float* qkvb = p.qkvb + ll * 3 * Cz;
        const float* aprojw = p.attprojw + ll * Cz * Cz;
        const float* aprojb = p.attprojb + ll * Cz;
        const float* ln2w = p.ln2w + ll * Cz;
        const float* ln2b = p.ln2b + ll * Cz;
        const float* fcw = p.fcw + ll * 4 * Cz * Cz;
        const float* fcb = p.fcb + ll * 4 * Cz;
        const float* fcprojw = p.fcprojw + ll * Cz * 4 * Cz;
        const float* fcprojb = p.fcprojb + ll * Cz;
        // Per-layer activation slices.
        float* ln1 = a.ln1 + ll * BTC;
        float* ln1_mean = a.ln1_mean + ll * BT;
        float* ln1_rstd = a.ln1_rstd + ll * BT;
        float* qkv = a.qkv + ll * BTC * 3;
        float* atty = a.atty + ll * BTC;
        float* preatt = a.preatt + ll * ATT;
        float* att = a.att + ll * ATT;
        float* attproj = a.attproj + ll * BTC;
        float* residual2 = a.residual2 + ll * BTC;
        float* ln2 = a.ln2 + ll * BTC;
        float* ln2_mean = a.ln2_mean + ll * BT;
        float* ln2_rstd = a.ln2_rstd + ll * BT;
        float* fch = a.fch + ll * BTC * 4;
        float* fch_gelu = a.fch_gelu + ll * BTC * 4;
        float* fcproj = a.fcproj + ll * BTC;
        float* residual3 = a.residual3 + ll * BTC;

        layernorm_forward(ln1, ln1_mean, ln1_rstd, res, ln1w, ln1b, B, T, C);
        matmul_forward(qkv, ln1, qkvw, qkvb, B, T, C, 3 * C);
        attention_forward(atty, preatt, att, qkv, B, T, C, NH);
        detail::apply_patch(patch, PatchSite::HeadOut, l, atty, B, T, C, NH);
        matmul_forward(attproj, atty, aprojw, aprojb, B, T, C, C);
        detail::apply_patch(patch, PatchSite::AttnBlockOut, l, attproj, B, T, C, NH);
        residual_forward(residual2, res, attproj, static_cast<int>(BTC));
        layernorm_forward(ln2, ln2_mean, ln2_rstd, residual2, ln2w, ln2b, B, T, C);
        matmul_forward(fch, ln2, fcw, fcb, B, T, C, 4 * C);
        gelu_forward(fch_gelu, fch, static_cast<int>(BTC * 4));
        matmul_forward(fcproj, fch_gelu, fcprojw, fcprojb, B, T, 4 * C, C);
        detail::apply_patch(patch, PatchSite::MlpOut, l, fcproj, B, T, C, NH);
        residual_forward(residual3, residual2, fcproj, static_cast<int>(BTC));
    }

    const float* last = (L == 0) ? a.encoded : a.residual3 + static_cast<std::size_t>(L - 1) * BTC;
    layernorm_forward(a.lnf, a.lnf_mean, a.lnf_rstd, last, p.lnfw, p.lnfb, B, T, C);
    // The loss reads every position, so a single-position forward cannot compute
    // one; silently producing a loss over stale logits is what this rules out.
    ASSERT_MSG(logits_at < 0 || targets == nullptr,
               "forward: a single-position forward cannot compute a loss");
    ASSERT_MSG(logits_at < T, "forward: logits_at is past the end of the sequence");
    if (logits_at < 0) {
        matmul_forward(a.logits, a.lnf, p.wte, nullptr, B, T, C, V);  // tied classifier
    } else {
        const auto Vz = static_cast<std::size_t>(V);
        const auto Cz2 = static_cast<std::size_t>(C);
        for (int b = 0; b < B; ++b) {
            const auto row = static_cast<std::size_t>(b) * static_cast<std::size_t>(T) +
                             static_cast<std::size_t>(logits_at);
            matmul_forward(a.logits + row * Vz, a.lnf + row * Cz2, p.wte, nullptr, 1, 1, C, V);
        }
    }

    // Inference (targets == nullptr): stop at the logits; no probs/loss. Otherwise
    // finish with softmax + cross-entropy and record mean_loss.
    has_loss_ = (targets != nullptr);
    if (targets != nullptr) {
        for (std::size_t bt = 0; bt < BT; ++bt)
            softmax_forward(a.probs + bt * static_cast<std::size_t>(V),
                            a.logits + bt * static_cast<std::size_t>(V), V);
        cross_entropy_forward(a.losses, a.probs, targets, B, T, V);
        double sum = 0.0;
        for (std::size_t bt = 0; bt < BT; ++bt) sum += a.losses[bt];
        mean_loss_ = static_cast<float>(sum / static_cast<double>(BT));
    }
}

void GPT2::backward(const int* tokens, const int* targets) {
    // backward reads acts().probs, which only a loss-computing forward fills. An
    // inference forward (targets == nullptr) leaves them stale or uninitialized,
    // so running backward after one would yield plausible but wrong gradients.
    ASSERT_MSG(has_loss_, "backward() requires a preceding forward() with targets");
    const int B = B_, T = T_;
    const int C = cfg_.n_embd, L = cfg_.n_layer, NH = cfg_.n_head, V = cfg_.vocab_size;
    const auto BTC = static_cast<std::size_t>(B) * T * C;
    const auto BT = static_cast<std::size_t>(B) * T;
    const auto ATT = static_cast<std::size_t>(B) * NH * T * T;
    const auto Cz = static_cast<std::size_t>(C);
    const ParamTensors& p = params_;
    const ParamTensors& g = grads_;
    const ActTensors& a = acts_;
    const ActTensors& d = act_grads_;  // gradient activations (zeroed by zero_grads)

    // Loss + classifier (tied head): dlogits = (probs − onehot)/(B·T); dwte += classifier path.
    cross_entropy_backward(d.logits, a.probs, targets, B, T, V);
    matmul_backward(d.lnf, g.wte, nullptr, d.logits, a.lnf, p.wte, B, T, C, V);

    // Final LayerNorm; its input gradient is the gradient of the last block output.
    const float* last = (L == 0) ? a.encoded : a.residual3 + static_cast<std::size_t>(L - 1) * BTC;
    float* d_last = (L == 0) ? d.encoded : d.residual3 + static_cast<std::size_t>(L - 1) * BTC;
    layernorm_backward(d_last, g.lnfw, g.lnfb, d.lnf, last, p.lnfw, a.lnf_mean, a.lnf_rstd, B, T, C);

    // Transformer blocks in reverse.
    for (int l = L - 1; l >= 0; --l) {
        const auto ll = static_cast<std::size_t>(l);
        const float* res = (l == 0) ? a.encoded : a.residual3 + (ll - 1) * BTC;
        float* d_res = (l == 0) ? d.encoded : d.residual3 + (ll - 1) * BTC;  // outgoing (accumulates)
        float* d_res3 = d.residual3 + ll * BTC;                              // incoming
        // Per-layer parameter / gradient slices.
        const float* ln1w = p.ln1w + ll * Cz;
        const float* qkvw = p.qkvw + ll * 3 * Cz * Cz;
        const float* aprojw = p.attprojw + ll * Cz * Cz;
        const float* ln2w = p.ln2w + ll * Cz;
        const float* fcw = p.fcw + ll * 4 * Cz * Cz;
        const float* fcprojw = p.fcprojw + ll * Cz * 4 * Cz;
        float* g_ln1w = g.ln1w + ll * Cz;
        float* g_ln1b = g.ln1b + ll * Cz;
        float* g_qkvw = g.qkvw + ll * 3 * Cz * Cz;
        float* g_qkvb = g.qkvb + ll * 3 * Cz;
        float* g_aprojw = g.attprojw + ll * Cz * Cz;
        float* g_aprojb = g.attprojb + ll * Cz;
        float* g_ln2w = g.ln2w + ll * Cz;
        float* g_ln2b = g.ln2b + ll * Cz;
        float* g_fcw = g.fcw + ll * 4 * Cz * Cz;
        float* g_fcb = g.fcb + ll * 4 * Cz;
        float* g_fcprojw = g.fcprojw + ll * Cz * 4 * Cz;
        float* g_fcprojb = g.fcprojb + ll * Cz;
        // Per-layer forward activations (read).
        const float* ln1 = a.ln1 + ll * BTC;
        const float* qkv = a.qkv + ll * BTC * 3;
        const float* att = a.att + ll * ATT;
        const float* atty = a.atty + ll * BTC;
        const float* residual2 = a.residual2 + ll * BTC;
        const float* ln2 = a.ln2 + ll * BTC;
        const float* fch = a.fch + ll * BTC * 4;
        const float* fch_gelu = a.fch_gelu + ll * BTC * 4;
        const float* ln1_mean = a.ln1_mean + ll * BT;
        const float* ln1_rstd = a.ln1_rstd + ll * BT;
        const float* ln2_mean = a.ln2_mean + ll * BT;
        const float* ln2_rstd = a.ln2_rstd + ll * BT;
        // Per-layer gradient activations (zeroed; written here).
        float* d_ln1 = d.ln1 + ll * BTC;
        float* d_qkv = d.qkv + ll * BTC * 3;
        float* d_atty = d.atty + ll * BTC;
        float* d_attproj = d.attproj + ll * BTC;
        float* d_residual2 = d.residual2 + ll * BTC;
        float* d_ln2 = d.ln2 + ll * BTC;
        float* d_fch = d.fch + ll * BTC * 4;
        float* d_fch_gelu = d.fch_gelu + ll * BTC * 4;
        float* d_fcproj = d.fcproj + ll * BTC;
        const int btc = static_cast<int>(BTC);

        // MLP sub-block (reverse): residual3 = residual2 + fcproj.
        residual_backward(d_residual2, d_fcproj, d_res3, btc);
        matmul_backward(d_fch_gelu, g_fcprojw, g_fcprojb, d_fcproj, fch_gelu, fcprojw, B, T, 4 * C, C);
        gelu_backward(d_fch, fch, d_fch_gelu, btc * 4);
        matmul_backward(d_ln2, g_fcw, g_fcb, d_fch, ln2, fcw, B, T, C, 4 * C);
        layernorm_backward(d_residual2, g_ln2w, g_ln2b, d_ln2, residual2, ln2w, ln2_mean, ln2_rstd,
                           B, T, C);  // d_residual2 += (onto residual contribution)

        // Attention sub-block (reverse): residual2 = res + attproj.
        residual_backward(d_res, d_attproj, d_residual2, btc);
        matmul_backward(d_atty, g_aprojw, g_aprojb, d_attproj, atty, aprojw, B, T, C, C);
        attention_backward(d_qkv, datt_, dpreatt_, d_atty, qkv, att, B, T, C, NH);
        matmul_backward(d_ln1, g_qkvw, g_qkvb, d_qkv, ln1, qkvw, B, T, C, 3 * C);
        layernorm_backward(d_res, g_ln1w, g_ln1b, d_ln1, res, ln1w, ln1_mean, ln1_rstd, B, T, C);  // d_res += (onto residual contribution)
    }

    // Embedding: dwte += embedding path (so dwte = classifier + embedding — weight tying); dwpe +=.
    embedding_backward(g.wte, g.wpe, tokens, d.encoded, B, T, C, V);
}

void GPT2::update(const AdamW& opt) noexcept {
    std::size_t ps[kNumParamTensors];
    param_sizes(cfg_, ps);  // per-tensor sizes; total tracked by param_count_

    ensure_moment_arenas();  // lazily allocate m/v on the first step (zeroed)
    ++adam_step_;

    // 2-group weight decay comes from the table's `decay` column — one place,
    // not a 16-entry bool array that had to line up with everything else by eye.
    // params_, grads_, m_, v_ share one contiguous layout; walk them by offset.
    float* param = params_.wte;
    const float* grad = grads_.wte;
    std::size_t off = 0;
    for (int i = 0; i < kNumParamTensors; ++i) {
        const float wd = kParamSpecs[i].decay ? opt.weight_decay : 0.0f;
        adamw_update(param + off, grad + off, m_ + off, v_ + off, static_cast<int>(ps[i]), opt.lr,
                     opt.beta1, opt.beta2, opt.eps, wd, adam_step_);
        off += ps[i];
    }
}

float GPT2::clip_grad_norm(float max_norm) noexcept {
    // grads_.wte is the base of the contiguous [param_count_] gradient arena (.bin
    // order), the same flat block update() steps over.
    return cppgpt::clip_grad_norm(grads_.wte, static_cast<int>(param_count_), max_norm);
}

void GPT2::ensure_moment_arenas() noexcept {
    if (m_ != nullptr) return;
    m_store_ = Storage(param_count_);
    v_store_ = Storage(param_count_);
    m_ = m_store_.alloc_zeroed(param_count_);
    v_ = v_store_.alloc_zeroed(param_count_);
}

Result<void> GPT2::save_checkpoint(const char* path, std::uint32_t tokenizer_kind,
                                   std::uint64_t vocab_fingerprint) const noexcept {
    const bool has_m = (m_ != nullptr);
    const std::size_t nbytes = param_count_ * sizeof(float);

    CheckpointHeader h{};
    h.magic = kCheckpointMagic;
    h.version = kCheckpointVersion;
    h.max_seq_len = cfg_.max_seq_len;
    h.vocab_size = cfg_.vocab_size;
    h.n_layer = cfg_.n_layer;
    h.n_head = cfg_.n_head;
    h.n_embd = cfg_.n_embd;
    h.adam_step = adam_step_;
    h.flags = has_m ? kCkptHasMoments : 0u;
    h.tokenizer_kind = tokenizer_kind;
    h.vocab_fingerprint = vocab_fingerprint;
    h.param_count = param_count_;

    ByteSpan sections[3];
    std::size_t n = 0;
    sections[n++] = {params_.wte, nbytes};
    if (has_m) {
        sections[n++] = {m_, nbytes};
        sections[n++] = {v_, nbytes};
    }
    h.checksum = checkpoint_checksum(h, sections, n);  // covers the header too
    return atomic_write(path, h, sections, n);
}

Result<void> GPT2::load_checkpoint(const char* path, LoadMode mode) noexcept {
    // Header first, then OUR OWN size arithmetic, then the payload — so a bogus
    // or mismatched file can never dictate the size of an allocation here.
    ASSIGN_OR_RETURN(CheckpointFile f, CheckpointFile::open(path));
    const CheckpointHeader& h = f.header();

    if (h.max_seq_len != cfg_.max_seq_len || h.vocab_size != cfg_.vocab_size ||
        h.n_layer != cfg_.n_layer || h.n_head != cfg_.n_head || h.n_embd != cfg_.n_embd ||
        h.param_count != param_count_) {
        return err(ErrorCode::ShapeMismatch);
    }

    const bool file_has_m = (h.flags & kCkptHasMoments) != 0;
    // WeightsOnly still reads the moments off disk (they are part of the payload
    // and the checksum covers them) but does not install them.
    const bool has_m = file_has_m;
    const bool install_moments = file_has_m && mode == LoadMode::Full;
    const std::size_t nbytes = param_count_ * sizeof(float);  // bounded by OUR model
    const std::size_t n_sections = has_m ? 3 : 1;
    if (f.payload_bytes() != static_cast<std::uint64_t>(n_sections) * nbytes)
        return err(ErrorCode::CorruptCheckpoint);

    // Stage into a scratch buffer and verify the checksum before touching a live
    // arena: a corrupt file must never leave the model half-overwritten.
    std::vector<std::byte> payload(n_sections * nbytes);
    RETURN_IF_ERROR(f.read_payload(payload.data(), payload.size()));

    ByteSpan sections[3];
    for (std::size_t i = 0; i < n_sections; ++i) sections[i] = {payload.data() + i * nbytes, nbytes};
    if (checkpoint_checksum(h, sections, n_sections) != h.checksum)
        return err(ErrorCode::ChecksumMismatch);

    // Validated — commit.
    std::memcpy(params_.wte, payload.data(), nbytes);
    if (install_moments) {
        ensure_moment_arenas();
        std::memcpy(m_, payload.data() + nbytes, nbytes);
        std::memcpy(v_, payload.data() + 2 * nbytes, nbytes);
        adam_step_ = h.adam_step;
    } else {
        // Weights-only file. The moments must be reset to match, or a model that
        // has already trained would carry STALE m/v into the next step — and the
        // step-1 bias correction (1/(1-beta1) = 10x) would amplify them into a
        // large, arbitrary kick. Zero them so the state matches what we log.
        if (file_has_m)
            LOG_INFO("loading weights only; Adam moments and step reset to zero");
        else
            LOG_WARNING("checkpoint has no optimizer moments; resetting Adam state to zero");
        ensure_moment_arenas();
        std::memset(m_, 0, nbytes);
        std::memset(v_, 0, nbytes);
        adam_step_ = 0;
    }
    return {};
}

}  // namespace cppgpt
