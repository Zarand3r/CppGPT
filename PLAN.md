# cppgpt — Implementation Plan

A **canonical GPT-2** transformer implemented from scratch in C++ — both training and inference — using only the C++ standard library. **CPU-first and feature-complete in v1**; a from-scratch CUDA backend is a designed-in *future phase*, not a rewrite. Tiny model first; scale after proof of concept.

This document holds the full design. The companion `ROADMAP.md` is the short feature checklist — the running answer to "what do we implement next." Implementation hands off to the `principal-production-engineer` skill, milestone by milestone, only after this plan is approved.

---

## Reference repos surveyed

| Repo | Role | What we borrow | What we ignore |
|---|---|---|---|
| **karpathy/llm.c** | Pure C reference of GPT-2 training: `train_gpt2.c` (~1k LOC CPU fp32). PyTorch-comparison harness ("overall okay: 1"), simple `.bin` weight format (1024-byte header + raw fp32). | **Primary structural reference.** Hand-written forward+backward per op; kernel order/shapes; `.bin` weight/dataset format; the 10-Adam-step PyTorch bit-comparison harness; arena-style dense activations. | cuBLAS/cuDNN/NCCL/MPI — the exact vendor deps we disallow. |
| **karpathy/llm.cu** | The CUDA sibling of llm.c — GPT-2 from scratch on the GPU, hand-written kernels, no cuBLAS. | **The GPU-phase reference.** Each llm.c function becomes a kernel; proves the hand-written-backward + dense-array shape ports to GPU additively (our exact future-phase plan). | — (in-scope for the future GPU phase, not v1). |
| **karpathy/nanoGPT** | PyTorch reference. ~600 LOC model+train; 124M GPT-2 target; char-Shakespeare baby model. | **Recipe and code shape only:** architecture numbers (12L/768H/12-heads/1024-ctx for 124M; 6L/384H/6-heads/256-ctx baby), AdamW 2-group decay, cosine+warmup schedule, grad-clip, MFU estimator, data pipeline, generation loop. | Its *numerical* choices that deviate from stock GPT-2 — notably `nn.GELU()` (exact erf) and `bias=False`. We match **canonical GPT-2**, not nanoGPT (see "Canonical GPT-2 conformance"). |
| **ggml-org/llama.cpp** | Inference-only, multi-backend, GGUF, quantization-first. Out of scope for v1; long-horizon scaling reference. | KV-cache mechanics; "header-only dep, plain C/C++, no framework" philosophy; CLI tool split. | Multi-backend dispatch, GGUF, dozens of architectures, grammars, multimodal — scope creep for v1. |
| **GaoYusong/llm.cpp** | Modern C++ port of llm.c with a single-header autograd tape. | Loader posture for HF GPT-2 weights; modern-C++ idioms where they shorten code. | **Its autograd tape.** We chose hand-written backward over a fixed graph (see Architecture Options A) — bare-minimal and GPU-kernel-friendly. |

---

## Goal

Build `cppgpt`: a C++ implementation of a decoder-only **canonical GPT-2** transformer that can both **train from scratch** on a small dataset and **run inference** on its own or converted pretrained GPT-2 weights — using only the C++ standard library, **CPU-only in v1**. A feature-complete GPT-2: the model, the training recipe, BPE tokenization, pretrained-weight loading, and autoregressive inference are all present and correct on CPU.

**Scale, stated honestly.** CPU-vs-GPU is a *performance* axis, not a *feature* axis. v1 is feature-complete on CPU: it trains tiny/small models, and loads + infers pretrained GPT-2 124M/350M. Training GPT-2 124M to full convergence (~300B tokens), or 350M+ from scratch, is wall-clock-impractical on CPU — that is what the **future GPU phase** (a from-scratch CUDA backend targeting a single workstation GPU such as an NVIDIA RTX 6000 Ada/Pro) is for. The GPU phase adds backends behind a device seam; it does not change the model, the math, or the API.

**Software discipline, right-sized.** Bare-minimal, not gold-plated: explicit ownership at boundaries, failure modes in types (`Result<T, E>`, `[[nodiscard]]`), fail-fast on invariant violations, deterministic by construction (no global RNG), fixture-verified numerics, tests with every behavior change. We deliberately *defer* heavier production scaffolding (multi-stream RNG, SHA-256 checkpoints, fuzz harness, CI build matrix, coverage gates) until a concrete need exists — see Deferred Complexity.

## Success Metrics

| Metric | Target | How measured |
|---|---|---|
| Forward-pass numerical fidelity | Per-tensor max-abs-error ≤ 1e-4 vs a **canonical-GPT-2** PyTorch reference (tanh GELU, bias=True, fp32) on a fixed seed | A Python script dumps weights, input tokens, and per-layer activations to `.bin`; cppgpt loads, runs, asserts. |
| Backward-pass fidelity | Per-grad max-abs-error ≤ 1e-3 vs PyTorch on the same fixture | Same harness, extended to gradients. |
| 10-step training-loop fidelity | Final loss matches PyTorch to ≤ 1e-3 after 10 AdamW steps with identical init | Following llm.c's "overall okay" pattern. |
| Tiny-model training convergence | Char-Shakespeare baby GPT (6L/384H/6-heads/256-ctx, ~10M params) reaches val loss ≤ 1.6 on TinyShakespeare in a **single overnight CPU run** (target < 8 h; firm number pending a named CPU — see Open Questions) | Train-loop logs + a held-out val split. |
| Inference correctness on GPT-2 124M | Same prompt + same sampling seed yields identical tokens to HuggingFace `transformers.GPT2LMHeadModel` for ≥ 50 generated tokens | Side-by-side decode comparison (fp32, greedy + temperature). |
| Build hygiene | `bazel build //...` with `-Wall -Wextra -Wpedantic -Werror`, ASan + UBSan clean (`--config=dev`), no external runtime deps (verified by `ldd`) | One CI build (Linux). |
| Test discipline | Every op has a unit test against a Python/PyTorch reference (forward) and a finite-difference + fixture check (backward) | `bazel test //...` green. |

Non-numeric: the code reads cleanly, ownership is obvious, hot paths perform no hidden allocation, and a new contributor can locate the matmul kernel in under 30 seconds.

## Constraints

- **Std C++ only (C++23) in v1.** No third-party libraries — no Eigen, no BLAS, no nlohmann/json, no pybind, no spdlog. C++23 (not 20) so `Result<T,E>` can alias `std::expected`. Allowed: `<expected>`, `<vector>`, `<thread>`, `<atomic>`, `<random>`, `<filesystem>`, `<span>`, `<bit>`, `<charconv>`, `<regex>`, etc. POSIX syscalls (`mmap`, `pread`) allowed via `<sys/mman.h>` on POSIX builds.
- **CPU only in v1.** No `nvcc`, no CUDA runtime, no NCCL/MPI/cuBLAS/cuDNN/Thrust in v1. Single process, single machine. **The GPU phase is explicitly planned (not forbidden):** it will add a from-scratch CUDA backend behind the device seam; cuBLAS/cuDNN remain disallowed even then (from-scratch kernels only).
- **Canonical GPT-2 semantics.** Where nanoGPT or other references deviate from stock GPT-2 (HuggingFace `gpt2` / OpenAI release), we follow GPT-2 — see "Canonical GPT-2 conformance." This is what lets M3 load HF weights and match `transformers`.
- **Both training and inference.** Inference may share or specialize the forward pass; backward is training-only.
- **Tiny scale first, then scale.** v1 works on a laptop CPU in fp32 on a tiny model in minutes; feature-complete for 124M/350M *inference*. Big-model *training* is the GPU phase's job.
- **fp32 in v1.** Mixed precision (bf16/tf32) is a GPU-phase concern; `DType` exists but is `F32`-only in v1.
- **Right-sized abstractions.** Every abstraction must encode an invariant, hide real complexity, or localize a known axis of change. No speculative abstraction for capabilities we won't ship in v1. The one forward-looking exception is the **device seam** (`Storage` device tag + op dispatch + real `DType`), justified because it makes the planned GPU phase additive instead of a rewrite.

## Non-Goals (v1)

- **Vendor math/comm runtimes** — no NCCL, MPI, cuBLAS, cuDNN, Thrust, CUTLASS, MKL — ever (even in the GPU phase, which uses hand-written kernels).
- **Multi-GPU, multi-node, distributed training** — single process, single machine.
- **ZeRO / FSDP / pipeline / tensor parallelism; gradient checkpointing for VRAM relief.**
- **Mixed precision (bf16/fp16) and quantization (int8/int4)** — fp32 only in v1; mixed precision is a GPU-phase item, and int8/int4 post-training quantization is the Efficiency Track's E2/E3 (opt-in, tolerance-validated, never a parity gate).
- **Training models > ~350M from scratch** — needs the GPU phase.
- **Mixture-of-Experts, RoPE, GQA, sliding-window / sparse / linear attention** — vanilla causal MHA only. These *change the model* (approximation or new params) and so break canonical parity by construction; they are Efficiency-Track E4, on a separate architecture path. `Config::n_kv_head` exists (defaults to `n_head`) so GQA is a future config flip, but no implementation in v1. **Flash attention is not in this list** — it is numerically *exact* (same softmax, tiled) and parity-preserving, filed as E1; still out of v1 scope, but for a different reason (not needed until GPU / long-context).
- Beam search, speculative decoding, grammars, structured output.
- Multimodal anything; web server, HTTP API, gRPC, Python bindings.
- Windows/macOS before Linux works.

**Explicitly NOT a non-goal:** GPU support. It is deferred to a planned future phase (see "GPU port seam" and "Beyond v1"), and the v1 architecture is shaped to admit it without redesign.

## Requirement Audit

| Requirement | Why | Needed v1? | Failure mode if removed |
|---|---|---|---|
| Forward pass, fp32, CPU | Inference baseline; backward depends on it | **Yes** | No model output. |
| Backward pass, fp32, CPU (hand-written) | Training baseline | **Yes** | Cannot train. |
| AdamW, 2-group weight decay | GPT-2 reproduction standard | **Yes** | Cannot match reference at wd>0. |
| Cosine LR schedule + warmup, grad clip | Stable convergence (nanoGPT recipe) | **Yes (M2)** | Poor/unstable convergence. |
| Gradient accumulation | Large effective batch on small memory | **Yes (M2)** | Limited batch size. |
| Weight tying (`wte` ↔ `lm_head`) | GPT-2 param/loss parity | **Yes** | Parameter count + loss diverge from reference. |
| tanh-approx GELU | Canonical GPT-2 activation | **Yes** | >1e-4 drift vs HF; M3 parity fails. |
| Char tokenizer | Simplest — Shakespeare baby model | **Yes** | Cannot do v1 vertical slice. |
| BPE tokenizer (tiktoken-compatible) | Load pretrained GPT-2 | M3 | Pretrained inference blocked. |
| `.bin` weight loader (llm.c-compatible) | Re-use existing dumps | M3 | Must write our own converter from day one. |
| Numerical fixture harness vs PyTorch | The only way to trust correctness | **Yes (M0)** | Silent numerical bugs poison everything. |
| `std::thread` work pool | Tiny training is slow single-threaded | M2 | v1 works on micro models; slow at scale. |
| Cache-blocked matmul (auto-vec) | Hot path | M2 | Slow but correct. |
| KV cache | Inference throughput | M3 | O(n²) re-forward per generated token. |
| Checkpoint save/load (simple, versioned) | Resume training; ship models | **Yes** | Lose training progress. |
| Sampling: argmax + temperature + top-k | Minimal text generation | **Yes** | Useless inference. |
| Device seam (`Storage` device tag, op dispatch, `DType`) | Make the GPU phase additive | **Yes** (cheap) | GPU phase becomes a rewrite. |
| CUDA backend (from-scratch kernels) | Real training scale | **GPU phase** | CPU-only stays feature-complete but slow at scale. |
| Top-p / repetition penalty | Nicety | Deferred | Quality degrades modestly. |
| Dropout | Regularization | v1, default 0.0 | Tiny models overfit slightly. |

## Existing System Understanding

Repo state: greenfield at `/home/rbao/cppgpt/`. Present: `CLAUDE.md` (skill wiring), `PLAN.md` (this file), `ROADMAP.md` (the short checklist, kept in sync with this plan). No build system, no source, no commits yet.

The GPT-2 math is public and well-trodden; the verification oracle is a ~100-line PyTorch script. The hard part is not "what to compute" — it is laying out the C++ so the math stays obvious, the data stays dense, the hot path stays allocation-free, and the device seam stays clean enough that a CUDA backend drops in later.

## Canonical GPT-2 conformance (GPT-2, not nanoGPT)

The numerical oracle and the eventual pretrained-weight target are **canonical GPT-2** (OpenAI release / HuggingFace `gpt2`). Where nanoGPT deviates from stock GPT-2 for speed, we follow GPT-2, because M3's whole point is loading HF GPT-2 124M and matching `transformers` token-for-token. The four details that break a ≤1e-4 parity test if mismatched are GELU, vocab padding, bias, and precision.

- **GELU = tanh approximation (`gelu_new`):** `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. This is OpenAI/HF GPT-2. nanoGPT's `nn.GELU()` (exact erf) is **not** GPT-2 and would fail forward parity against an HF dump. The PyTorch oracle must be configured to match (`approximate='tanh'`).
- **Weight tying:** `lm_head.weight` *is* `wte.weight` (same buffer). Backward accumulates into the shared parameter from both the embedding and classifier paths. Parameter count reports it once.
- **Learned absolute position embeddings** (`wpe`), added to token embeddings. No RoPE.
- **Bias everywhere:** `bias=True` on all `Linear`s and `LayerNorm`s (stock GPT-2). nanoGPT's `bias=False` option is not used — HF weights carry biases.
- **Vocab = 50257**, unpadded. nanoGPT pads to 50304 for Tensor-Core alignment; that is a GPU optimization with no CPU benefit and it complicates HF-weight loading, so v1 uses 50257. (Padding can return as a GPU-phase option.)
- **AdamW two-group weight decay:** decay applies to ≥2D weights (matmuls + embeddings); biases and LayerNorm gains/shifts are excluded (`apply_weight_decay=false`). Omitting this is the most common GPT-2-repro bug.
- **Residual-projection scaled init:** `attn.c_proj.weight` and `mlp.c_proj.weight` init `N(0, 0.02/√(2·n_layer))`; all other weights `N(0, 0.02)`; biases zero; LayerNorm gain=1, shift=0. Keeps residual-stream variance bounded.
- **LayerNorm:** pre-norm, `eps=1e-5`, affine.
- **Cross-entropy:** over `[B·T, V]` vs `[B·T]`, mean reduction.
- **Recipe (training, M2+):** cosine LR schedule with linear warmup, `grad_clip=1.0`, betas `(0.9, 0.95)`. Slice 0's 10-step parity test uses fixed lr / no clip to match the simplest oracle.

## Architecture Decomposition

Target file layout (final names confirmed at M0). Dense arrays + free-function ops + hand-written backward, in the llm.c shape, with a device seam:

```
cppgpt/
├── MODULE.bazel · BUILD.bazel · .bazelrc   (Bazel/bzlmod; hermetic LLVM+Python)
├── README.md
├── PLAN.md                    (this file)
├── ROADMAP.md                 (short feature checklist)
├── CLAUDE.md                  (skill wiring)
├── include/cppgpt/
│   ├── core.hpp               Result<T,E>=std::expected, ErrorCode+describe(), ASSERT/TRY/MUST macros.
│   ├── log.hpp                Leveled std-only logging: LOG_INFO/WARNING/ERROR/FATAL/EVERY_N.
│   ├── random.hpp             Generator: explicit RNG (mt19937_64), no global, no default ctor.
│   ├── storage.hpp            Owning aligned buffer tagged with Device; arena/bump allocation.
│   ├── tensor.hpp             TensorView: non-owning {float*, shape, stride, rank, device}.
│   ├── ops.hpp                Forward ops (device-dispatched): matmul, layernorm, attention, gelu (tanh),
│   │                          softmax, residual, encoder (tok+pos), classifier, cross_entropy.
│   ├── ops_backward.hpp       Matching hand-derived backward ops, symmetric signatures.
│   ├── model.hpp              Config, GPT2 (params/grads/activations arenas + AdamW state), forward/backward/update.
│   ├── optimizer.hpp          AdamW (2-group decay), clip_grad_norm, cosine_with_warmup.
│   ├── tokenizer.hpp          Tokenizer interface; CharTokenizer (v1), BpeTokenizer (M3).
│   ├── dataloader.hpp         Memory-mapped uint16 .bin token stream; shuffled epochs.
│   ├── checkpoint.hpp         Simple versioned save/load (header + raw fp32), atomic via tmp+rename.
│   ├── sample.hpp             Greedy / temperature / top-k (top-p deferred).
│   └── verify.hpp             Fixture compare: max-abs-error, mismatch histogram.
├── src/                       Matching .cpp where definitions live (small lib, not header-only).
├── tools/   train.cpp · generate.cpp · bench.cpp · verify.cpp
├── tests/   unit/ · integration/ · fixtures/
├── scripts/ gen_fixtures.py · convert_gpt2.py · prepare_shakespeare.py
└── third_party/               EMPTY. Tracked to prove discipline.
```

Modules and responsibilities:

1. **Storage / TensorView.** `Storage` owns an aligned fp32 buffer **tagged with a `Device`** (arena-allocated for activations; persistent for params/grads). `TensorView` is a non-owning `{float*, shape, stride, rank, device}`, ranks up to 4 (B, T, C, H). `DType` exists (`F32` only) for forward-compat. No `std::variant`, no virtuals.
2. **Ops (forward).** Free functions, no allocation inside, caller passes output buffers. Each **dispatches on `device`** (CPU impl in v1; the seam for CUDA later). Signatures mirror llm.c: `matmul_forward(out, inp, weight, bias, B,T,C,OC)`, `layernorm_forward(out, mean, rstd, x, w, b, B,T,C)`, `attention_forward(out, preatt, att, qkv, B,T,C,NH)`, `gelu_forward` (tanh), `softmax`, `residual`, `encoder_forward`, `crossentropy_forward`.
3. **Ops (backward).** Symmetric, hand-derived. Each takes upstream gradient + saved activations, produces downstream gradient. No autograd tape — the graph is the fixed GPT-2 architecture.
4. **Model.** `struct GPT2 { Config cfg; Storage params, grads, activations, act_grads; AdamState opt; }`. Param layout matches llm.c `.bin` ordering. **Weight tying:** `lm_head` aliases `wte` (same offset in `params`); backward adds into that one region from both paths. `gpt2_forward` writes activations and sets loss; `gpt2_backward` consumes them into `grads`; `gpt2_update` runs AdamW.
5. **Optimizer.** AdamW with per-parameter `m`/`v`, bias correction, **decoupled decay only on ≥2D weights**. `clip_grad_norm` (global L2) and `cosine_with_warmup` live here. No hidden allocation.
6. **Tokenizer.** `Tokenizer` interface; v1 `CharTokenizer` (alphabet discovered from corpus); M3 `BpeTokenizer` (GPT-2 byte-level merges, regex pre-tokenizer).
7. **Dataloader.** Memory-mapped `.bin` of `uint16` tokens; cursor + shuffled epoch order via an explicit `Generator`. No threads in v1's loader.
8. **Checkpoint.** Versioned header + raw fp32 (same family as the weight format). Written atomically (`path.tmp` → fsync → rename). Loader validates magic/version/shape and returns `Result`.
9. **Sample.** Greedy, temperature, top-k. Reads logits; draws from an explicit `Generator`.
10. **Verify.** Loads a fixture, compares element-wise, reports max-abs-error and a mismatch histogram.

Build: single Bazel module (bzlmod), hermetic LLVM/Clang 20 with statically-linked libc++ and a hermetic Python 3.12 for the oracle/data scripts. One `cc_library` (`cppgpt`), tool `cc_binary`s, `cc_test`s, and `py_binary`/`py_test` for scripts. C++23; `-Wall -Wextra -Wpedantic -Werror` globally; `--config=release` (`-O3 -march=native`), `--config=dev` (`-O1 -g -fsanitize=address,undefined`).

## Core Entities and Interfaces

```cpp
enum class Device { CPU };          // CUDA reserved for the GPU phase; v1 ops assert device==CPU.
enum class DType  { F32 };          // BF16/F16 reserved; assert-on-use in v1.

// Owning, device-tagged, arena-backed.
class Storage {
public:
    Storage(size_t bytes, Device dev = Device::CPU);
    float* alloc(size_t n_floats, size_t align = 64);   // bump allocator
    void   reset();                                      // drop activations at step end
    Device device() const noexcept;
    std::span<float> as_span();
private:
    std::unique_ptr<float[], AlignedDeleter> buf;        // device memory in the GPU phase
    size_t capacity, head;
    Device dev;
};

// Non-owning view. Cheap to pass by value. Carries its device for dispatch.
struct TensorView {
    float* data;
    std::array<int, 4> shape;       // rank padded with 1s
    std::array<int, 4> stride;
    int    rank;
    Device device;
};

struct Config {
    int   max_seq_len, vocab_size;          // 50257 for GPT-2
    int   n_layer, n_head, n_kv_head, n_embd;   // n_kv_head defaults to n_head (GQA hook)
    float dropout;                          // 0.0 in v1
    // training recipe (M2+): learning_rate, weight_decay, beta1, beta2, eps,
    //   grad_clip, warmup_iters, lr_decay_iters, min_lr, max_iters, batch_size.
    // factories: baby(), shakespeare(), gpt2_124m(), gpt2_medium(); validate() -> Result<void,Error>.
};

struct GPT2 {
    Config   cfg;
    Storage  params, grads;          // persistent; lm_head region aliases wte region (weight tying)
    Storage  activations, act_grads; // per-step arenas, reset each step
    AdamState opt;                   // m, v, step
};

// All ops: pure functions, no allocation, caller passes output buffer; dispatch on device.
void matmul_forward (float* out, const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC, Device dev);
void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC, Device dev);
// ... layernorm, attention, gelu (tanh), softmax, residual, encoder, classifier, crossentropy: all symmetric.

// Top-level (fixed GPT-2 graph; hand-written backward).
void  gpt2_forward (GPT2& m, const int* tokens, const int* targets, int B, int T);  // sets loss
void  gpt2_backward(GPT2& m);
void  gpt2_update  (GPT2& m, float lr, float wd, float b1, float b2, float eps, int t);  // 2-group decay inside
float clip_grad_norm(GPT2& m, float max_norm);
float cosine_with_warmup(int it, int warmup, int decay_iters, float min_lr, float max_lr);
```

The signatures deliberately mirror **llm.c** — that repo is already bit-compared against PyTorch, so matching its shapes and `.bin` parameter ordering makes our weight converter and fixtures interchangeable and collapses verification effort. The only addition is the trailing `Device dev` on each op: in v1 it asserts `CPU`; in the GPU phase it selects a kernel. The math and the public API are otherwise device-agnostic.

## Data Flow / Control Flow

**Training step (per micro-batch of B sequences × T tokens):**

```
tokens[B,T] ─► encoder_forward ─► acts.encoded[B,T,C]
                     │
             (× n_layer transformer blocks)
                     │
                ln_f_forward
                     │
            matmul (classifier head, = wte^T via tying)
                     │
            softmax + cross_entropy ───────► loss
                     │
                     ▼
             gpt2_backward (hand-written, reverse order)
                     │
                     ▼
        clip_grad_norm → gpt2_update (AdamW)
                     │
                     ▼
        activations.reset() + act_grads.reset()  (drop transients)
```

Each transformer block:
```
x ─► ln_1 ─► attn(qkv→att→softmax→att·v→proj) ─► (+ x) ─► ln_2 ─► mlp(fc1→gelu→fc2) ─► (+ residual)
```

Gradient accumulation (M2+): run K micro-batches accumulating into `grads`, scale loss by 1/K, then a single `clip → update → reset`.

**Inference (single-token decode after prefix):**

```
prefix tokens ─► forward(B=1, T=prefix_len) ─► save K,V into cache
new token     ─► forward_step(B=1, T=1, uses kv_cache) ─► logits ─► sample ─► append
```

KV cache is a separate `Storage` of shape `[n_layer, 2, max_seq, n_head, head_dim]`. Forward-step writes the new k/v slice and reads the full prefix. Logit cropping: when generating, run the classifier head on the last position only. This `Storage` is also the insertion point for the Efficiency Track's KV-cache quantization (E2/E3) — its element type moves from fp32 to a quantized `DType` behind the reserved seam, with the fp32 cache staying the tolerance oracle (invariant 11).

## State Machines / Lifecycles

**Arena lifetime per training step:**
1. Step begins → `activations.reset()` and `act_grads.reset()` (head = 0).
2. Forward bumps `activations` as it writes acts (saved for backward).
3. Backward reads acts in reverse, writing grad-of-acts into `act_grads` and parameter grads into `grads`.
4. `clip_grad_norm` then `gpt2_update` use `grads` + `params` (both persistent) and AdamW `m`/`v`. Arenas untouched.
5. End of step → loop to (1).

Invariant: after warmup, the steady-state step allocates **zero** bytes outside arena resets. Verified by an `operator new` counter hook in dev tests.

**Model lifecycle:**
```
Config ─► allocate params, grads, AdamW state           [persistent, once]
        ─► allocate activation + act_grad arenas          [reset each step]
        ─► load_weights(...) or init_weights(gen)          [GPT-2 init + residual scaling]
        ─► (train loop: forward → backward → clip → update → reset)
        ─► save_checkpoint(...)                            [atomic, versioned]
        ─► destroy
```

**Tokenizer:** built once at startup (char map for v1; merges file for BPE). Immutable thereafter.
**KV cache:** allocated once at session start with `max_seq_len`; `t` cursor advances per generated token; reset on new prompt.

## GPU port seam (future phase — designed in, not built)

Going CPU→GPU changes the *execution and memory model*, not the math. Three seams, kept real from day one, make the future CUDA backend additive rather than a rewrite:

1. **`Storage` carries a `Device`** and owns device memory. CPU = aligned host buffer; CUDA later = `cudaMalloc`'d arena. Same bump-allocator interface; allocation happens once and is reused (mandatory on GPU where `cudaMalloc` is expensive).
2. **Ops dispatch on `device`.** `*_forward`/`*_backward` take a `Device` and select an implementation. v1 has only the CPU path; the GPU phase adds `.cu` kernels behind the same call site. The CPU implementation becomes the **numerical oracle** that validates every kernel via the existing fixture harness — exactly the llm.c → llm.cu relationship.
3. **`DType` is real.** v1 is `F32` only, but the field exists so mixed precision (bf16/tf32 — a GPU-phase item) slots in without touching op signatures.

What the GPU phase *adds* (out of v1 scope, listed so it is not forgotten): hand-written CUDA kernels (tiled matmul, fused attention, reduction-based layernorm/softmax/loss), host↔device transfer + streams, mixed precision, GPU-aware memory budgeting, and a relaxed determinism contract (GPU reductions are not bit-identical by default). None of these alter the v1 model or API.

## Architecture Options

### A. Backward / autograd strategy
| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Hand-written backward per op over the fixed GPT-2 graph (llm.c/llm.cu)** | Smallest code; matches the bit-compared reference 1:1; no tape overhead; **maps directly to hand-written CUDA kernels** | Adding a *new architecture* is a code edit, not a config flip | **Chosen.** GPT-2 is fixed; RoPE/GQA/etc. are non-goals. Bare-minimal and GPU-ready. |
| Tape-based autograd (PyTorch / llm.cpp tinytorch) | Flexible: new ops localize; `loss.backward()` decouples op authors | ~500+ LOC of tape machinery; per-op graph nodes; harder to port to from-scratch kernels | Rejected for v1 — over-built for a fixed architecture; revisit only if we start varying architectures (see Deferred). |
| Static expression-template autograd (Eigen-style) | Zero runtime overhead | Heavy metaprogramming; opaque errors; slow compile | Rejected — anti-doctrine. |

### B. Tensor representation
| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Dense `float*` + shape/stride view, owning `Storage` tagged with `Device`** | Dense; trivial to reason about; SIMD/GPU-friendly; carries device for dispatch | Manual stride math; no rank type-safety | **Chosen.** `TensorView` for clarity; `Storage{Device}` is the GPU seam; `DType` reserved for mixed precision. |
| Single `Tensor` class with dtype enum + graph fields (GGML/PyTorch) | Future-proofs quantization, carries grad/grad_fn | Branches on dtype in hot path; needs a tape to use grad_fn | Deferred — only earns its keep with a tape or quantization, neither in v1. |
| Templated `Tensor<float, Rank>` | Compile-time rank checks | Compile blowup; hard cross-TU; over-engineered | Rejected. |

### C. Parallelism (CPU)
| Option | Pros | Cons | Decision |
|---|---|---|---|
| **`std::thread` work pool** | Std-only; explicit | Write our own pool (~100 LOC) | **Chosen** at M2. Parallelize independent rows/batch so reductions stay deterministic per seed. |
| OpenMP `#pragma omp parallel for` | One-line | Compiler runtime, arguably "external" | Rejected — keep std-only. |
| Single-threaded only | Simplest | Useless above tiny scale | Acceptable through M1. |

### D. SIMD strategy
| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Compiler auto-vec (`-O3 -march=native`)** | No intrinsics in source | Hit-or-miss | **Chosen v1–M2.** Establish correctness, then measure. |
| Hand-written AVX2/NEON intrinsics | Predictable perf | Per-target codepaths | M2+ only if benchmarks demand; compile-gated. |
| Library (xsimd, highway) | Portable | External dep | Rejected. |

### E. Compute backend / scale
| Option | Pros | Cons | Decision |
|---|---|---|---|
| **CPU-first v1, from-scratch CUDA as a designed-in future phase (device seam now, kernels later)** | Feature-complete GPT-2 immediately; honest scale story; GPU port is additive; no vendor runtime | Big-model *training* waits for the GPU phase | **Chosen.** Matches "bare-minimal feature-complete GPT-2 that can later run on an RTX 6000-class GPU." |
| GPU from day one (hand-written CUDA) | Max scale sooner | nvcc + CUDA toolchain up front; harder to get numerics right without a CPU oracle | Rejected for v1 — correctness-first on CPU, then port. |
| GPU via cuBLAS/cuDNN | Fastest to scale | Vendor deps; not "from scratch" | Rejected — violates the from-scratch rule even in the GPU phase. |

### F. Tokenizer
| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Char-level for v1** | ~5 LOC; no merges file | Cannot load pretrained GPT-2 | **Chosen for M0–M2.** |
| Full BPE matching tiktoken | Loads any GPT-2 weight | Byte-level encoder + merges + regex pre-tokenization is fiddly; `std::regex` lacks `\p{L}`/`\p{N}` so the pre-tokenizer is hand-rolled | **Chosen for M3.** |
| Pre-tokenize in Python, ship uint16 | Skip writing BPE | Inference becomes Python-dependent | Rejected — need standalone inference. |

### G. Weight format
| Option | Pros | Cons | Decision |
|---|---|---|---|
| **llm.c `.bin` (1024-byte header + raw fp32, fixed param order)** | Re-use existing dumps; simple loader; fixture-interchangeable | Coupled to llm.c param ordering | **Chosen.** |
| GGUF | Industry de facto; quant-ready | Big parser; overkill for fp32 GPT-2 | Deferred. |
| Custom format | Total control | Forces a Python converter from day one | Rejected. |

## Tradeoff Analysis

**Backend / scale (resolved).** CPU-first, feature-complete v1; a from-scratch CUDA backend is a planned future phase behind the device seam. CPU-vs-GPU is performance, not features — so nothing about GPT-2 is omitted by shipping CPU-only first. The seam (device-tagged `Storage`, op dispatch, real `DType`) is the only forward-looking abstraction we pay for now; it is cheap and prevents a future rewrite.

**Backward (resolved).** Hand-written per-op backward over the fixed GPT-2 graph. For a fixed architecture this is the minimal, most directly verifiable, and most GPU-portable choice (llm.c → llm.cu). A tape would be ~500+ LOC of machinery to support architecture variation we have declared a non-goal.

**Numerical reference (resolved).** Canonical GPT-2, not nanoGPT. Match GPT-2 for anything numerical or weight-loading (tanh GELU, bias=True, vocab 50257, fp32 parity); borrow nanoGPT and llm.c only for recipe and code shape. Mixing them — e.g. an erf-GELU oracle while claiming HF parity — is the trap this section exists to prevent.

**Single-file vs multi-file (resolved).** Multi-file, per-module `.hpp/.cpp`, one library + several CLIs + one test binary. Strict include hygiene; the matmul kernel findable in seconds.

**Abstraction discipline (resolved, right-sized).** Pluggable only where a second implementation already exists or is on the v1 schedule: `Tokenizer` (char + BPE) and `Sampler` (greedy/temperature/top-k). `LayerNorm`, `GELU`, attention, MLP are concrete. `Config::n_kv_head` exists as a five-line GQA hook. Everything else is concrete until a real need appears.

## Risks and Bottlenecks

| # | Risk | Cheapest experiment | Success criterion | Fallback |
|---|---|---|---|---|
| R1 | **Numerical drift hidden until late.** A subtle indexing bug in attention/layernorm diverges silently from the reference. | Build the fixture harness *first* (M0): PyTorch (tanh GELU) dumps inputs, weights, every activation; cppgpt loads and asserts. | Per-tensor max-abs-error ≤ 1e-4. | Bisect by tensor name → offending op → focused unit test. |
| R2 | **Backward derivation errors.** Hand-derived gradients are error-prone (attention, layernorm). | Per-op finite-difference Jacobian on tiny tensors + fixture compare. | Relative error ≤ 1e-3. | Re-derive on paper; cross-check vs `torch.autograd.grad`. |
| R3 | **CPU matmul is a brick.** Naive triple-loop on 124M is unusable. | Bench naive matmul at M2 start; if < 5 GFLOP/s, plan blocking immediately. | ≥ 30 GFLOP/s single-thread fp32 on a modern x86 core. | Cache-blocked matmul; then compile-gated AVX2 intrinsics (still no external lib). |
| R4 | **BPE/tokenizer parity.** Unicode, byte fallbacks, regex pre-tokenization; `std::regex` lacks `\p{L}`/`\p{N}`. | Hand-roll the pre-tokenizer; test vs tiktoken on 1000 strings before integrating. | 100% token-stream match. | Pre-tokenize in Python for training-only paths (delays standalone inference). |
| R5 | **Activation memory.** Attention score buffers (`preatt`+`att`, each `L·B·NH·T·T` fp32) dominate: for 124M at B=8/T=1024 that pair alone is ~9.6 GB — past a laptop. **v1 trains only the ~10M baby model; 124M/350M are inference-only**, where no saved-activation graph exists. | Measure baby-model peak at M2; measure inference peak at M3/M4. | Baby training peak ≤ 2 GB; GPT-2 medium inference peak ≤ 4 GB. | Reduce B/T; recompute attention in backward; activation checkpointing → GPU phase. |
| R6 | **Abstraction creep under "production-grade".** | Allow-list of pluggable interfaces fixed up front (`Tokenizer`, `Sampler`); anything else concrete until a 2nd impl exists. | Review rejects new interfaces without ≥2 impls. | Delete the interface; inline the impl. |
| R7 | **Device seam rots / leaks CPU assumptions.** Host pointers baked in where a device tag belongs. | Every op takes `Device`; `Storage` owns the tag; a lint/grep check at M0. | No raw host-pointer arithmetic outside CPU op bodies. | Re-thread `Device` through the offending call sites. |
| R8 | **OS/build portability.** | Linux-only for v1. | `bazel build //...` green on one distro. | macOS/Windows are their own later milestones. |

## Invariants

Each maps to a test, assertion, or check.

1. **No allocation in the training-step hot path** after warm-up. *(operator-new hook, 100 steps, assert 0 after the first.)*
2. **Arena reset between steps drops all transient memory;** no use-after-free of activations across steps. *(ASan + stress run.)*
3. **Shapes are explicit at every op boundary** — no "guess from total size." *(Every op asserts B, T, C, … before work.)*
4. **Params and grads are the same size and order;** the `lm_head` region *is* the `wte` region (tying). *(Test offsets + `params.size()==grads.size()`.)*
5. **RNG is explicit** — every stochastic op takes a `Generator&`; no global default. Same seed → bit-identical logits at step 0. *(Two-run determinism test.)*
6. **Failures fail fast, never silent** — out-of-range token IDs, NaN loss, mis-shaped checkpoint → abort with a precise error. *(Assertions; `[[nodiscard]]` on loaders.)*
7. **Ownership is unambiguous** — `Storage` owns; `TensorView` borrows. No `shared_ptr<float>`. *(Review gate.)*
8. **Every op carries its `Device`; v1 asserts `CPU`.** The seam never silently assumes host memory. *(Op-entry assertion + R7 lint.)*
9. **Verification is run, not promised** — every commit touching an op runs the fixture vs PyTorch. *(CI gate.)*
10. **No third-party runtime deps** — `ldd` allow-list (libc/libm/libstdc++/libpthread). *(CI gate.)*
11. **Efficiency/approximation modes never relax the canonical gates.** Any quantized, sparse, linear, or otherwise-approximate path is opt-in and validated against the fp32 CPU path within a *documented, committed* tolerance; the fp32 CPU path remains the parity oracle. A mode that loosens, skips, or replaces a canonical parity check to "pass" is out of scope by construction. *(Applies to the whole Efficiency & Research Track; exact modes like E1 flash attention still meet the standard fp32 tolerance.)*

(Future-phase invariant, M-GPU: CPU and CUDA paths are numerically equivalent within a documented tolerance — same fixture harness, both backends.)

## Vertical Slice Strategy

The first end-to-end slice proves the hardest thing — matching PyTorch numerically — before any performance, GPU, or BPE work.

**Slice 0 — "Loss matches PyTorch on a baby model after 10 AdamW steps."**
- Model: 2 layers, 64 embed, 2 heads, 8 context, vocab 65 (TinyShakespeare char set). ~50k params.
- Init: fixed seed; matches a canonical-GPT-2 PyTorch reference (tanh GELU, bias=True) at every weight.
- Data: 8 fixed sequences of length 8, hard-coded.
- Train: 10 AdamW steps, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, wd=0.0.
- Pass: final loss within 1e-3 of the reference; intermediate per-tensor max-abs-error ≤ 1e-4.

This exercises tensor/storage, arena allocator, all forward ops, all backward ops, AdamW, save/load round-trip, the fixture harness, and the CLI. If it passes, the math is right and we scale model dimensions with confidence. If it fails, every later milestone is blocked.

## Milestone Roadmap

Each milestone: goal, deliverable, scope/non-scope, risks retired, definition-of-done. (`ROADMAP.md` is the flat checklist version of this.)

### M0 — Skeleton + fixture harness *(retires R1, R7, R8)*
**Goal:** Buildable repo; a working canonical-GPT-2 PyTorch-fixture harness; matmul forward+backward (hand-written, device-dispatched) verified vs PyTorch; "no external deps" check.
**Deliverable:** Bazel module + directory skeleton; `scripts/gen_fixtures.py` (dumps matmul fwd/bwd reference, tanh-GELU-configured for later ops); `core.hpp` (Result=std::expected, ErrorCode, ASSERT/TRY/MUST macros), `log.hpp` (Logger), `random.hpp` (Generator); `Storage{Device}`, `TensorView`, `Config`; `matmul_forward/backward` with `Device` dispatch (CPU); `tests/unit/matmul_test.cpp`; `tools/verify.cpp` (stub for full forward); one CI build with `ldd` allow-list.
**Out of scope:** other ops, attention, training loop, BPE, GPU.
**DoD:** `bazel test //...` shows "matmul forward/backward match PyTorch"; ASan+UBSan clean (`--config=dev`); a contributor reads the whole harness in < 30 min.
**Effort:** 3–5 days.

### M1 — Full forward + backward on a baby model *(retires R2)*
**Goal:** Execute Slice 0.
**Deliverable:** all ops (`layernorm`, `attention`, `softmax`, `gelu`-tanh, `residual`, `encoder`, `classifier`, `cross_entropy`) forward+backward, each fixture-tested; `GPT2` + `gpt2_forward/backward/update`; AdamW (2-group decay); weight tying; char tokenizer; `tools/train.cpp` (hardcoded baby config, 10 steps, prints loss); `tools/verify.cpp` (full fwd/bwd compare); finite-difference gradient checker test.
**Out of scope:** parallelism, real datasets, BPE, GPU, perf.
**DoD:** `tools/train` prints the same 10-step loss sequence as the PyTorch reference, within 1e-3; every op has fwd+bwd fixture tests; gradcheck passes.
**Effort:** 2–3 weeks.

### M2 — TinyShakespeare convergence + CPU performance *(retires R3, R5)*
**Goal:** Train 6L/384H/6-heads/256-ctx (~10M params) to val loss ≤ 1.6 on TinyShakespeare in an overnight CPU run.
**Deliverable:** `dataloader` (mmap uint16, shuffled epochs); `scripts/prepare_shakespeare.py`; `checkpoint` (versioned, atomic) + periodic save; `tools/generate.cpp` (temperature + top-k); cache-blocked matmul; `std::thread` work pool (deterministic row partition); gradient accumulation; cosine+warmup schedule; grad clip; `tools/bench.cpp` (GFLOP/s, step time, tokens/sec).
**Out of scope:** BPE, pretrained weights, GPU.
**DoD:** an overnight run produces a sampleable checkpoint; bench reports ≥ 30 GFLOP/s single-thread matmul; measured peak memory recorded.
**Effort:** 2 weeks.

### M3 — BPE tokenizer + pretrained GPT-2 124M inference *(retires R4)*
**Goal:** Load HF GPT-2 124M and reproduce identical generation vs `transformers` for ≥ 50 tokens at a fixed seed.
**Deliverable:** GPT-2 BPE (byte-level encoder, merges parser, hand-rolled regex pre-tokenizer); `scripts/convert_gpt2.py` (HF → `.bin`); KV cache; prefix + autoregressive decode with logit cropping; side-by-side test vs `transformers.generate` (greedy + temperature).
**Out of scope:** GPT-2-scale training; GPU.
**DoD:** `tools/generate --weights gpt2-124M.bin --prompt "Once upon a time"` matches HF transformers token-for-token; tokenizer byte-exact on 1000+ strings; KV-cache on/off identical.
**Effort:** 2 weeks (BPE is fiddly).

### M4 — Polish + GPT-2 medium (350M) inference + GPU-seam readiness *(retires R6)*
**Goal:** "Correct" → "clean," scale to GPT-2 medium inference on a workstation CPU, and verify the GPU seam is real.
**Deliverable:** leveled structured logging replaces ad-hoc `cerr`; `Result<T,E>` + `[[nodiscard]]` at every loader/parser/checkpoint boundary; a small observability set logged to CSV (tokens/sec, grad norm, peak memory, step time); GPT-2 medium inference tested for HF parity; a **GPU-seam audit** (every op carries `Device`; no host-pointer leakage — retires R7 fully); `docs/ARCHITECTURE.md` + `docs/CONTRIBUTING.md` (incl. "how to add a CUDA backend behind the seam").
**Out of scope:** the CUDA backend itself; SIMD intrinsics (measured-need only); macOS/Windows; the deferred scaffolding (fuzz/CI-matrix/SHA/multi-stream RNG).
**DoD:** a fresh engineer can clone, build, train baby Shakespeare, generate from GPT-2 124M, run GPT-2 medium inference, and read the architecture doc in under an hour.
**Effort:** 1–2 weeks.

### Beyond v1 (each its own future plan)

**GPU phase (headline).** From-scratch CUDA backend behind the device seam: hand-written kernels, host↔device transfers + streams, mixed precision (bf16/tf32), GPU memory budgeting, relaxed determinism contract; targets a single workstation GPU (e.g. NVIDIA RTX 6000 Ada/Pro) for real GPT-2-scale training. CPU stays the numerical oracle.

Other GPU-adjacent items: **SIMD intrinsics (AVX2/NEON)** for the CPU path (measured-need only); **tape autograd** (only if we start varying architectures); **top-p / repetition penalty; GGUF; macOS/Windows**.

**Efficiency & Research Track.** Ordered least→most numerical perturbation. Governed by invariant 11: each is opt-in and validated against the fp32 CPU oracle within a documented tolerance; none relax the canonical parity gates. The v1 seams they reuse (the reserved `DType` field, the device-dispatched op signatures, the KV-cache `Storage`) already exist — no new abstraction is paid for now (see "Deferred Complexity").

- **E1 · Flash attention (exact, parity-preserving).** Online-softmax tiled attention behind the existing `attention_forward/backward` call site — additive, exactly like the GPU seam. O(T) score memory instead of O(L·B·NH·T²); this is the general fix for R5, though R5 is already handled in v1 by scoping big models to inference-only. Same softmax → meets the standard fp32 tolerance, not a looser one. Primary payoff: GPU training and long-context inference. Validated against fp32 vanilla attention.
- **E2 · Post-training quantization (inference).** int8/int4 weight quantization + KV-cache quantization as an opt-in inference mode, via the reserved `DType` seam (or a parallel quantized `Storage` with dequant-on-the-fly in the matmul). Not token-exact — lives behind a flag, validated within a committed tolerance vs the fp32 path. The KV-cache `Storage` (already shaped `[n_layer,2,max_seq,n_head,head_dim]` in v1) is the quantization insertion point.
- **E3 · TurboQuant-class near-optimal quantization (research).** Data-oblivious *online* vector quantization suited to KV cache: randomly rotate vectors (inducing a concentrated Beta distribution per coordinate), then apply per-coordinate optimal scalar quantizers; a two-stage MSE quantizer + 1-bit Quantized-JL residual yields an *unbiased inner-product* estimator (the quantity attention actually needs). Reported ≈ quality-neutral at 3.5 bits/channel and marginal at 2.5, within a small constant of the information-theoretic distortion bound. Plugs into the E2 KV-cache seam; accelerator-friendly and online, so it also fits the GPU phase. Reference: TurboQuant, arXiv 2504.19874.
- **E4 · Sparse / linear / hybrid attention (research, architecture-changing).** Sub-quadratic attention: linear (kernel/recurrent, O(N) but a known reasoning / long-retrieval penalty and non-injective attention), sparse (block/pattern/routed token subsets), or **hybrid** (linear backbone with interleaved full or sparse-softmax layers) — the current consensus sweet spot, with "component collapse" (the model ignoring the linear branch) as the documented failure mode. These change the computation, so they break canonical GPT-2 token-exact parity by construction and live on a separate architecture path — the same trigger that would justify reconsidering a tape/autograd. Validated on task metrics (perplexity, long-context retrieval), not token-exact parity. References: surveys arXiv 2507.19595, 2504.17768.

**Ambition ceiling (noted, not planned).** TurboQuant-style near-optimal KV-cache quantization (E3) composed with a hybrid sparse/linear backbone (E4) is roughly the frontier a from-scratch, dependency-free GPT-2 could chase for long-context, low-memory inference. Recorded as direction only; every item above is behind invariant 11 and the "do not start" gate.

## Verification Strategy

| Layer | Method | When |
|---|---|---|
| Single op forward | Fixture compare vs canonical-GPT-2 PyTorch (≤ 1e-4) | Every commit touching the op |
| Single op backward | Fixture compare + finite-difference grad check (≤ 1e-3) | Every commit touching the op |
| Full forward | Fixture compare on all intermediate activations | M1 onward |
| Full backward | Fixture compare on all gradients | M1 onward |
| Training loop | 10-step loss match vs PyTorch | M1, every regression |
| Tokenizer | Byte-exact vs tiktoken on 1000+ strings | M3 |
| Inference parity | Token-exact vs HF transformers on N seeds | M3, M4 |
| Memory | ASan + UBSan + alloc-counter in dev builds | Every CI run |
| Determinism | Same seed → bit-identical logits across runs | Every CI run |
| Build hygiene | `ldd` allow-list; `-Werror`; no external deps | Every CI run |
| CPU↔GPU parity | Same fixture harness, both backends | GPU phase only |

Anti-pattern to avoid: "ship now, write tests later." The fixture harness is M0 deliverable #1 because every later milestone leans on it.

## Deferred Complexity

Things we explicitly do *not* build in v1, and the trigger that would change that.

| Deferred | Trigger to reconsider |
|---|---|
| Tape-based autograd | We start varying the architecture (RoPE/GQA/etc.), or > ~15 differentiable ops |
| CUDA backend | The GPU phase begins (the planned next major effort) |
| Mixed precision (bf16/tf32) | GPU phase / memory or throughput pressure |
| Multi-stream RNG, SHA-256 checkpoints, fuzz harness, CI build matrix, coverage/clang-tidy gates | A concrete failure or contributor need demands it — not pre-emptively |
| Flash attention (E1, exact) | Long-context inference, or GPU training where the O(T²) score buffer dominates. Parity-preserving, so it can enter whenever measured need appears — no architecture change. |
| Int8/int4 post-training quantization (E2) | Need to fit 124M/350M+ inference in a tighter memory/latency budget than fp32 allows. Opt-in; tolerance-validated (invariant 11). |
| TurboQuant-class KV-cache quantization (E3) | E2 exists and KV-cache memory is the inference bottleneck at long context; want near-optimal bits/channel. Research-grade. |
| Sparse / linear / hybrid attention (E4) | We deliberately trade token-exact parity for sub-quadratic long-context compute — a separate architecture path, not the canonical GPT-2 model. Also the trigger to reconsider a tape. |
| GGUF weight format | Interop with the llama.cpp ecosystem, or shipping quantized weights (pairs with E2/E3). |
| Activation checkpointing | OOM on a target model size (GPU phase) |
| Multi-GPU / distributed | Single-GPU training is genuinely throughput-bound |
| RoPE / GQA / RMSNorm / SwiGLU | We start porting a non-GPT-2 architecture (Llama-family) — would also motivate a tape |
| HTTP server / Python bindings | A downstream user asks |
| macOS / Windows build | A developer needs it |

## Recommended Next Step

Hand off to `principal-production-engineer` and start **M0** — the `Storage`/`TensorView`/`Config` skeleton, the device-dispatched matmul forward+backward, and the canonical-GPT-2 fixture harness. It is the cheapest experiment that retires the highest-impact risk (R1: silent numerical drift) and exercises the device seam (R7) from the first commit.

## Open Questions / Decision Points

| # | Question | Status |
|---|---|---|
| 1 | GPU code? | **Resolved — CPU-first v1, from-scratch CUDA as a designed-in future phase.** Device seam present now; kernels deferred. cuBLAS/cuDNN never. |
| 2 | Autograd? | **Resolved — hand-written backward over the fixed GPT-2 graph.** No tape in v1. |
| 3 | Numerical reference? | **Resolved — canonical GPT-2 (HF `gpt2`), not nanoGPT.** tanh GELU, bias=True, vocab 50257, fp32 parity. |
| 4 | OpenMP? | **Resolved — no.** `std::thread` work pool only. |
| 5 | Compiler intrinsics (`<immintrin.h>`)? | **Resolved — deferred.** Auto-vec through M2; compile-gated path only if benchmarks demand. |
| 6 | Linux-only for v1? | **Resolved — yes.** macOS/Windows post-v1. |
| 7 | Char tokenizer first, BPE at M3? | **Resolved — yes.** |
| 8 | Target CPU spec for benchmark numbers? | **Open.** Need a named CPU (cores, cache, AVX-512?) to firm up M2/M4 perf and the convergence wall-clock. Placeholders until then. |
| 9 | Public or private repo? | **Open.** Affects whether tests download HF GPT-2 weights or take a user-supplied path. |
