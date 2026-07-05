# cppgpt — Roadmap (feature checklist)

The running answer to "what do we build next." Flat list of core features, in build order. Design rationale, interfaces, risks, and DoD live in `PLAN.md`.

**Locked decisions:** CPU-first v1, feature-complete **canonical GPT-2** (tanh GELU, bias=True, vocab 50257, fp32). Hand-written backward over the fixed GPT-2 graph (no tape). Std-only C++, single Bazel module (hermetic LLVM/Clang + Python). GPU = from-scratch CUDA, designed-in future phase (device seam now, kernels later).

Legend: `[ ]` todo · `[~]` in progress · `[x]` done. **Next step = first unchecked box.**

---

## M0 — Skeleton + fixture harness
- [x] Bazel workspace: hermetic LLVM/Clang 20 (static libc++) + hermetic Python 3.12; `--config=release` / `--config=dev` (ASan/UBSan); `-Wall -Wextra -Wpedantic -Werror`; `py_binary`/`py_test` wired
- [x] Directory skeleton (`include/ src/ tools/ tests/ scripts/ third_party/`) + std-only `cc_test` harness + green `bazel test //...`
- [x] `core.hpp` errors: `Result<T,E> = std::expected` (enregisterable, C++23) · `ErrorCode` + `describe()` · `ASSERT`/`DCHECK`/`MUST`/`UNREACHABLE` · `TRY`/`ASSIGN_OR_RETURN`/`RETURN_IF_ERROR`/`TRY_OR`/`TRY_OR_CONTINUE`
- [x] `log.hpp`: leveled Logger — `LOG_INFO`/`WARNING`/`ERROR`/`FATAL` + `LOG_EVERY_N`; level threshold; swappable sink; `std::format`
- [x] `random.hpp`: explicit `Generator` (`mt19937_64`; no global, no default ctor) — `uniform`/`uniform_int`/`normal`
- [~] `Storage` (aligned, `Device`-tagged, arena/bump) **done** (`storage.hpp` + `device.hpp`); `TensorView`/`Config`/`DType` deferred (no consumer until the first multi-op/model code — ops take raw `float*`)
- [x] `matmul_forward` / `matmul_backward` (CPU, device-dispatched) — `ops.hpp` + `src/ops.cpp`
- [ ] `scripts/gen_fixtures.py` (canonical-GPT-2 PyTorch oracle, tanh GELU) · `verify.hpp` — **deferred**: no PyTorch in this env; needed for the first GPT-2-specific op (gelu-tanh/layernorm), not for matmul
- [x] `tests/unit/matmul_test.cpp` — fwd exact fixtures + bwd **adjoint identity** + finite-difference (independent of PyTorch); `storage_test.cpp` covers the arena
- [ ] CI: one Linux build + `ldd` allow-list + ASan/UBSan

## M1 — Full GPT-2 forward + backward (Slice 0)
- [x] Ops fwd+bwd, each tested (exact/property fixtures + finite-difference gradcheck — no PyTorch in env): `gelu` (tanh), `residual`, `layernorm`, `softmax`, `attention` (reuses `softmax`), `encoder` (tok+pos), `cross_entropy` (softmax-fused). `classifier` = `matmul` with the tied `wte` (no new op). Full transformer block integration-tested (`tests/integration/transformer_block_test`).
- [x] `GPT2` struct (`Config`, param/grad/activation `Storage` arenas, llm.c `.bin` layout) + **`gpt2_forward`, `gpt2_backward`, `gpt2_update` done** (`model.hpp`/`model.cpp`, end-to-end gradcheck + overfit smoke)
- [x] Weight tying (`lm_head` aliases `wte`) — forward classifier uses `wte`; `dwte` accumulates from classifier + embedding paths in `gpt2_backward` (gradcheck-verified)
- [x] **GPT-2 init + residual-proj scaling** (`init_weights`) + **AdamW** (`adamw_update` op + `GPT2::update`; 2-group decay: ≥2D weights decay, biases/LN don't; torch-parity fixture)
- [x] `CharTokenizer` (deterministic byte-level vocab, exact round-trip; BPE is M3)
- [~] `tools/train.cpp` (baby config, char-level, random-window batches — **trains end to end**, loss descends from ≈ln(V)); `tools/verify.cpp` (full fwd/bwd) next
- [x] Finite-difference gradient checker test (`model_test` end-to-end gradcheck + `tests/numeric.hpp`)
- [ ] **Gate:** 10-step loss matches PyTorch ≤ 1e-3 (`scripts/gen_fixtures.py` vs `notebooks/train_gpt2.py`)

## M2 — TinyShakespeare convergence + CPU perf
- [ ] `dataloader` (mmap uint16, shuffled epochs) · `scripts/prepare_shakespeare.py`
- [ ] `checkpoint` (versioned header + raw fp32, atomic tmp→rename) + periodic save/resume
- [ ] Gradient accumulation · cosine+warmup LR schedule · `clip_grad_norm`
- [ ] Cache-blocked matmul · `std::thread` work pool (deterministic row partition)
- [ ] Sampling: argmax / temperature / top-k
- [ ] `tools/generate.cpp` · `tools/bench.cpp`
- [ ] **Gate:** val loss ≤ 1.6 overnight; matmul ≥ 30 GFLOP/s single-thread

## M3 — BPE + pretrained GPT-2 124M inference
- [ ] BPE: byte-level encoder, merges parser, hand-rolled regex pre-tokenizer
- [ ] `scripts/convert_gpt2.py` (HF → `.bin`)
- [ ] KV cache (fp32; the `Storage` that later becomes the E2/E3 quantization seam) · prefix + autoregressive decode · logit cropping (last position)
- [ ] **Gate:** tokenizer byte-exact vs tiktoken (1000+ strings); generation token-exact vs HF for ≥ 50 tokens; KV-cache on/off identical

## M4 — Polish + GPT-2 medium inference + GPU-seam readiness
- [ ] Structured logging everywhere · `Result`+`[[nodiscard]]` at all loader/parser/checkpoint boundaries
- [ ] Observability CSV (tokens/sec, grad norm, peak memory, step time)
- [ ] GPT-2 medium (350M) inference — HF parity + measured tokens/sec
- [ ] GPU-seam audit (every op carries `Device`; no host-pointer leakage)
- [ ] `docs/ARCHITECTURE.md` · `docs/CONTRIBUTING.md` (incl. "how to add a CUDA backend")
- [ ] **Gate:** clone → build → train baby → generate 124M → infer 350M → read docs, under an hour

---

## Future phase (post-v1, do not start)

**GPU phase (headline):**
- [ ] **GPU: from-scratch CUDA backend** behind the device seam (kernels, H2D/streams, mixed precision, memory budgeting)
- [ ] SIMD intrinsics (AVX2/NEON) for the CPU path — measured-need only
- [ ] Tape autograd — only if we start varying architectures (RoPE/GQA/…)
- [ ] Deferred scaffolding (multi-stream RNG, SHA-256 ckpt, fuzz harness, CI matrix, coverage gates) — only on concrete need
- [ ] Top-p / repetition penalty · macOS/Windows

**Efficiency & Research Track** — ordered least→most numerical perturbation. Each is opt-in and validated against the fp32 CPU path within a *documented* tolerance; **none relax the canonical parity gates** (fp32 CPU stays the oracle — see PLAN invariant 11). Do not start.
- [ ] **E1 · Flash attention (exact).** Online-softmax tiled attention behind the existing `attention_forward/backward` signature; O(T) score memory vs O(T²). Numerically equivalent to fp32 vanilla attention → **preserves parity**. Mainly a GPU / long-context-inference win. (Not needed to retire R5 — big models are inference-only in v1.)
- [ ] **E2 · Post-training quantization (inference).** int8/int4 weights + KV-cache quant, opt-in inference mode via the reserved `DType` seam. Validated within documented tolerance vs fp32 — **not** token-exact; lives behind a flag.
- [ ] **E3 · TurboQuant-class near-optimal quantization (research).** Data-oblivious online vector quant (random-rotate → per-coordinate optimal scalar quantizers; 2-stage MSE + 1-bit QJL for unbiased inner products). ~2.5–3.5 bits/channel KV cache near quality-neutral. Plugs into the E2 KV-cache seam. (arXiv 2504.19874)
- [ ] **E4 · Sparse / linear / hybrid attention (research, architecture-changing).** Approximates attention → **breaks canonical GPT-2 parity by construction**; lives on a separate architecture path (also the trigger to reconsider a tape). Hybrid (linear backbone + interleaved full/sparse) is the current sweet spot; watch for "component collapse." Validated on task metrics, not token-exact. (surveys arXiv 2507.19595, 2504.17768)

**Alignment & Post-Training Track (RLHF)** — turns the base LM into an instruction/preference-aligned model. A *capability* axis, not an efficiency one: it changes what the model does, not how fast it runs. The ops (forward/backward/AdamW), tokenizer, and dataloader are reused unchanged, so ops-level parity is untouched — but this is the one track whose *success* is metric-based, not token-exact-parity-based: there is no canonical "GPT-2 RLHF" oracle (outcomes depend on the preference data). New losses are still gradient-checked vs PyTorch; invariant 11's gates are added to, never relaxed. Do not start.
- [ ] **A1 · SFT (supervised fine-tuning).** Fine-tune the pretrained LM on demonstration / chat data with the loss masked to completion tokens. Reuses forward/backward/AdamW + tokenizer; new pieces are an instruction dataloader and the prompt/completion loss mask. Cheapest; unlocks the rest.
- [ ] **A2 · Reward model.** Pairwise-preference dataset (chosen vs rejected); reward model = LM backbone + scalar head, trained with a Bradley–Terry / pairwise ranking loss. New: the scalar head and the ranking loss (fwd+bwd, gradient-checked).
- [ ] **A3 · DPO (recommended alignment step).** Direct Preference Optimization — a single offline loss over preference pairs against a frozen reference (the A1 SFT model) with an implicit KL. No reward model, no sampling loop, no value network. Reuses the SFT weights as a second frozen param `Storage`; far more tractable std-only/CPU than PPO.
- [ ] **A4 · PPO (research, ambitious ceiling).** Full online RLHF: in-loop generation (rollouts via the sampler), reward scoring (A2), value head + GAE, KL-to-reference penalty, clipped policy objective. Needs the sampler inside training and 2–3 resident model copies (policy, reference, reward). GRPO / other critic-free variants noted as simpler alternatives.
