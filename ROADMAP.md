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
- [ ] `Storage` (aligned, `Device`-tagged, arena/bump) · `TensorView` (`{float*,shape,stride,rank,device}`) · `Config`
- [ ] `matmul_forward` / `matmul_backward` (CPU, device-dispatched)
- [ ] `scripts/gen_fixtures.py` (canonical-GPT-2 PyTorch oracle, tanh GELU) · `verify.hpp`
- [ ] `tests/unit/matmul_test.cpp` (fwd+bwd ≤ 1e-4 / 1e-3)
- [ ] CI: one Linux build + `ldd` allow-list + ASan/UBSan

## M1 — Full GPT-2 forward + backward (Slice 0)
- [ ] Ops fwd+bwd, each fixture-tested: `gelu` (tanh), `layernorm`, `softmax`, `attention`, `residual`, `encoder` (tok+pos), `classifier`, `cross_entropy`
- [ ] `GPT2` struct + `gpt2_forward` / `gpt2_backward` / `gpt2_update`
- [ ] Weight tying (`lm_head` aliases `wte`)
- [ ] AdamW (2-group decay: ≥2D weights decay, biases/LN don't) · GPT-2 init + residual-proj scaling
- [ ] `CharTokenizer`
- [ ] `tools/train.cpp` (baby config, 10 steps) · `tools/verify.cpp` (full fwd/bwd)
- [ ] Finite-difference gradient checker test
- [ ] **Gate:** 10-step loss matches PyTorch ≤ 1e-3

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
- [ ] KV cache · prefix + autoregressive decode · logit cropping (last position)
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
- [ ] **GPU: from-scratch CUDA backend** behind the device seam (kernels, H2D/streams, mixed precision, memory budgeting)
- [ ] SIMD intrinsics (AVX2/NEON) — measured-need only
- [ ] Tape autograd — only if we start varying architectures (RoPE/GQA/…)
- [ ] Top-p / repetition penalty · quantization/GGUF · macOS/Windows
- [ ] Deferred scaffolding (multi-stream RNG, SHA-256 ckpt, fuzz harness, CI matrix, coverage gates) — only on concrete need
