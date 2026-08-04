# cppgpt — Roadmap (feature checklist)

**Goal (v1 / MVP):** train, sample from, and **fine-tune your own** small model end to end on your own
text — using a *canonical GPT-2 architecture*, not GPT-2's scale. **No pretrained-weight loading, no
124M/350M, no BPE.** The value kept from "canonical" is that the architecture and training step are
numerically verified against PyTorch (M1, met at ~1e-6) — it is a real GPT-2, just small.

**MVP gate (the whole product, one loop).** `prepare` takes an output *stem* and emits
`<stem>.train.bin`, `<stem>.val.bin`, `<stem>.vocab`:
```sh
# 1. tokenize your corpus, holding out a validation split
prepare  corpus.txt base --val-frac 0.1

# 2. train
train    --data base.train.bin --val base.val.bin --vocab base.vocab \
         --layers 4 --heads 4 --embd 128 --ctx 128 --steps 5000 --ckpt base.ckpt

# 3. sample from the model you just trained
generate --checkpoint base.ckpt --vocab base.vocab --prompt "..." --n 200

# 4. fine-tune on a second corpus, REUSING the first vocabulary
prepare  target.txt target --vocab base.vocab --val-frac 0.1
train    --data target.train.bin --val target.val.bin --vocab base.vocab \
         --init-from base.ckpt --lr 3e-4 --steps 500 --ckpt tuned.ckpt
generate --checkpoint tuned.ckpt --vocab base.vocab --prompt "..." --n 200
```
Binary: val loss descends on both runs; fine-tuning lowers val loss **on the target corpus** versus
the base checkpoint; both runs reproduce exactly from their seed.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done. Design rationale lives in `PLAN.md`, measured
numbers in `docs/measurements.md`, lessons in `docs/engineering-lessons.md`.

**Every milestone's acceptance gate includes one `review-codify-loop` run** — adversarial review,
triage with no silent drops, and lessons codified. A milestone whose functional gate passes but which
left no review cycle behind is not closed.

**Status:** the *engine* is done and verified. Everything below is CLI wiring plus a val/eval path —
no new subsystems.

---

## M0 — Skeleton + fixture harness
- [x] Bazel workspace: hermetic LLVM/Clang 20 (static libc++) + hermetic Python 3.12; `--config=release` / `--config=dev` (ASan/UBSan); `-Wall -Wextra -Wpedantic -Werror`; `py_binary`/`py_test` wired
- [x] Directory skeleton (`include/ src/ tools/ tests/ scripts/ third_party/`) + std-only `cc_test` harness + green `bazel test //...`
- [x] `core.hpp` errors: `Result<T,E> = std::expected` (enregisterable, C++23) · `ErrorCode` + `describe()` · `ASSERT`/`DCHECK`/`MUST`/`UNREACHABLE` · `TRY`/`ASSIGN_OR_RETURN`/`RETURN_IF_ERROR`/`TRY_OR`/`TRY_OR_CONTINUE`
- [x] `log.hpp`: leveled Logger — `LOG_INFO`/`WARNING`/`ERROR`/`FATAL` + `LOG_EVERY_N`; level threshold; swappable sink; `std::format`
- [x] `random.hpp`: explicit `Generator` (`mt19937_64`; no global, no default ctor) — `uniform`/`uniform_int`/`normal`
- [x] `Storage` (aligned, `Device`-tagged, arena/bump) **done** (`storage.hpp` + `device.hpp`); `Config` landed with the model (`model.hpp`); `TensorView`/`DType` deliberately not built — ops take raw `float*`, and `DType` waits for a second dtype (GPU / quantization phase)
- [x] `matmul_forward` / `matmul_backward` (CPU, device-dispatched) — `ops.hpp` + `src/ops.cpp`
- [x] `scripts/gen_fixtures.py` (canonical-GPT-2 PyTorch oracle, tanh GELU) · `verify.hpp` — run in the torch venv; the fixture is committed as `tests/fixtures/gpt2_parity.bin` so the C++ gate (`tests/integration/parity_test`) needs no torch
- [x] `tests/unit/matmul_test.cpp` — fwd exact fixtures + bwd **adjoint identity** + finite-difference (independent of PyTorch); `storage_test.cpp` covers the arena
- [ ] CI: one Linux build + `ldd` allow-list + ASan/UBSan

## M1 — Full GPT-2 forward + backward (Slice 0)
- [x] Ops fwd+bwd, each tested (exact/property fixtures + finite-difference gradcheck — no PyTorch in env): `gelu` (tanh), `residual`, `layernorm`, `softmax`, `attention` (reuses `softmax`), `encoder` (tok+pos), `cross_entropy` (softmax-fused). `classifier` = `matmul` with the tied `wte` (no new op). Full transformer block integration-tested (`tests/integration/transformer_block_test`).
- [x] `GPT2` struct (`Config`, param/grad/activation `Storage` arenas, llm.c `.bin` layout) + **`gpt2_forward`, `gpt2_backward`, `gpt2_update` done** (`model.hpp`/`model.cpp`, end-to-end gradcheck + overfit smoke)
- [x] Weight tying (`lm_head` aliases `wte`) — forward classifier uses `wte`; `dwte` accumulates from classifier + embedding paths in `gpt2_backward` (gradcheck-verified)
- [x] **GPT-2 init + residual-proj scaling** (`init_weights`) + **AdamW** (`adamw_update` op + `GPT2::update`; 2-group decay: ≥2D weights decay, biases/LN don't; torch-parity fixture)
- [x] `CharTokenizer` (deterministic byte-level vocab, exact round-trip; BPE is M3)
- [x] `tools/train.cpp` (baby config, char-level, random-window batches — **trains end to end**, loss descends from ≈ln(V)); full fwd/bwd verification done via the parity gate (`tests/integration/parity_test`, not a separate tool)
- [x] Finite-difference gradient checker test (`model_test` end-to-end gradcheck + `tests/numeric.hpp`)
- [x] **Gate MET:** cppgpt matches canonical GPT-2 in PyTorch — forward ≤ 1e-4, gradients ≤ 1e-3, 10-step AdamW loss ≤ 1e-3 (measured ~1e-6; `scripts/gen_fixtures.py` → committed `.bin` → `tests/integration/parity_test`). **M1 complete.**

## M2 — MVP: train · infer · fine-tune

The three verbs. Each box is wiring over code that already exists and is tested.

### Train — make it usable and knowable
- [ ] `prepare <corpus> <stem> [--val-frac F] [--vocab <existing>]` → `<stem>.train.bin`,
      `<stem>.val.bin`, `<stem>.vocab` (contiguous tail split). Today `prepare` takes an output
      *filename* and emits one `.bin` + a `.bin.vocab` sidecar — moving to a stem is what makes the
      train/val pair and the fine-tune vocab-reuse name cleanly.
- [ ] `train` CLI flags: `--layers --heads --embd --ctx --batch --lr --steps --seed --ckpt`
      (`tools/train.cpp` currently hardcodes L3/H4/C64/T32; only steps and the ckpt path are arguments)
- [ ] Eval loop: `--val <bin> --eval-interval N` logging **val loss**, tokens/s, elapsed, peak RSS.
      Without a val number you cannot tell training from memorizing.
- **Gate:** val loss descends and stays below a same-corpus bigram baseline; two runs at the same seed
  produce bit-identical loss curves.

### Infer — generate from a checkpoint you trained
- [ ] `generate --checkpoint <ckpt> --vocab <vocab> --prompt "..." --n K [--temperature --top-k --seed]`
- [ ] **Build the model from the checkpoint header.** `GPT2`'s ctor needs a `Config` *before*
      `load_checkpoint` can validate one against it — so `generate` must first peek the header
      (`CheckpointFile::open`, already public, returns it), construct from that `Config`, and only
      then load. Without this step `--checkpoint` cannot work at all: you would have to retype the
      exact architecture on the command line and a mismatch is `ShapeMismatch`.
      (today `tools/generate` calls `init_weights` and trains in-process — it can *never* read a `.ckpt`)
- [ ] **Short prompts.** `generate()` copies exactly `seq_len` tokens, so a 5-token prompt is impossible.
      Place the prompt at `[0, len)`, forward, read logits at `len-1` (causality makes the padded tail
      inert). Same fix as the shelved `M3-S1`, minus the HuggingFace gate.
- **Gate:** a checkpoint trained to low loss produces text recognizably in the corpus's style from a
  short prompt; greedy decoding (`--top-k 1`) is deterministic.

### Fine-tune — continue from a checkpoint on new text
- [ ] `prepare --vocab <existing.vocab>` — tokenize a *new* corpus with the **original** vocabulary.
      **This is the piece that makes fine-tuning possible at all:** a fresh corpus yields a different
      char vocab, hence a different `vocab_size`, hence `ShapeMismatch` on load. Must fail loudly on a
      byte outside the original vocab rather than silently remapping.
- [ ] `train --init-from <ckpt>` — load **weights only**, reset optimizer moments/step and restart the
      LR schedule (a plain `--ckpt` resume deliberately restores moments; fine-tuning wants fresh ones
      at a lower LR). The weights-only path already exists — `load_checkpoint` on a moment-less file
      resets Adam and says so.
- **Gate:** fine-tuning a base checkpoint on a target corpus lowers val loss **on the target** relative
  to the base checkpoint, without a from-scratch run beating it in the same wall-clock.

### Enough performance to iterate (not a benchmark contest)
- [ ] Re-measure and fix the build flags: `-march=native` measures **1.9× SLOWER** than plain `-O3` on
      the dominant kernel (`docs/measurements.md` M-1). Free speedup, no code change.
- [ ] `tools/bench` — merge branch `m2-bench` (PR #20) so the number has a reproducible source.
- [ ] Then reassess. At ~200 tok/s a small model is trainable in minutes-to-hours; blocking/threading
      only if iteration is actually painful. **No 30 GFLOP/s gate for the MVP** — that target existed
      to make a TinyShakespeare convergence run finish overnight, which is no longer the goal.
- [ ] *(optional)* Gradient accumulation — only if a batch you want exceeds RAM (measured: B=64 at
      T=256 costs 8.15 GB).

---

## Debt & deferred fixes

**Every outstanding item in the repo is listed here.** Other documents record *why* (`PLAN.md`),
*what was measured* (`docs/measurements.md`), *what was swept* (`docs/review-audit.md`) and *what we
learned* (`docs/engineering-lessons.md`) — but none of them hold work. If it needs doing, it is a box
on this page.

| Item | Source | Disposition |
|---|---|---|
| **5 constitution clauses have no enforcing test** — per-op PyTorch fixtures, every-intermediate-activation parity, the alloc-counter hook, NaN/Inf-loss abort, the `ldd` allow-list | `docs/review-audit.md` 2026-08-03 | ⬜ **Needs your decision.** Each is either "write the test" or "reword the clause", and `docs/constitution.md` is human-frozen — rewording is not an agent's call. The cheapest two are genuinely small: a one-line `ASSERT(std::isfinite(mean_loss_))` in `GPT2::forward`, and an `ldd` check in CI. |
| `kFnvOffset64` is the textbook FNV-1a-64 basis with a digit dropped | `docs/engineering-lessons.md` L6 | ⬜ **Deferred by design** — it still functions as a hash, and fixing it invalidates every existing checkpoint's checksum. Bundle with the next `kCheckpointVersion` bump. Tracked at the point of use in `checkpoint.hpp`. |
| `docs/M3_INFERENCE_PLAN.md` risk/invariant/slice IDs are unprefixed (`R1`, `S1`) and collide with `PLAN.md`'s `DR-n` | `docs/review-audit.md` 2026-08-03 | ⬜ Rename to `M3-Rn` / `M3-In` / `M3-Sn`. Cosmetic while the doc is shelved; do it if M3 is ever revived. |
| `write_token_bin` did tmp+rename but **no `fsync`** — the rule from the lesson whose incident is that very function | `docs/engineering-lessons.md` L5 | ✅ **Fixed** — fsync of file and containing directory. |
| `tools/bench` is written but lives on unmerged branch `m2-bench` (PR #20) | 3-axis review, 2026-08-03 | ⬜ Merge it — listed under MVP → performance above. |
| Open PRs: #20 bench · #21 review fixes + docs · #22 this planning work | — | ⬜ Merge order: #20, then #21, then #22. |

---

## Deferred — GPT-2 scale (out of MVP scope; nothing here is deleted)

Everything below was planned in detail and is **shelved, not cancelled**. The design work stands if
the goal ever changes; `docs/M3_INFERENCE_PLAN.md` remains the reference for the inference path.

- [ ] **Byte-level BPE tokenizer.** Needed only to share GPT-2's 50257 vocabulary. Char-level is the
      MVP tokenizer; it already round-trips exactly and needs no assets. (Full spec, traps and
      fixtures: `docs/M3_INFERENCE_PLAN.md` M3-S3.)
- [ ] **Load pretrained GPT-2 124M weights** (`convert_hf_gpt2.py` + `tools/import_hf`) and the
      token-exact-vs-HuggingFace gate. The mapping is fully researched and verified — Conv1D
      transposes, Q‖K‖V order, the layer-bisect fixture. (`M3-S2`, `M3-S5`.)
- [ ] **KV cache** — an inference speedup, irrelevant at MVP context lengths. (`M3-S4`.)
- [ ] **Inference-only arenas** — 124M at T=1024 costs 5.32 GB vs 2.61 GB; immaterial for a small
      model on a 120 GB host. (`M3-S6`.)
- [ ] **GPT-2 medium (350M) inference**, GPU-seam audit, observability CSV, `docs/ARCHITECTURE.md`.
- [ ] **CI** — one Linux build + `ldd` allow-list + ASan/UBSan. The only genuinely open M0 box; worth
      doing whenever, independent of scope.
- [ ] **30 GFLOP/s matmul + threading** — revisit only if training speed becomes the binding constraint.

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
- [ ] **E2 · Post-training quantization (inference).** int8/int4 weights + KV-cache quant, opt-in inference mode; introduces its own quantized `Storage` (no `DType` exists yet to reuse). Validated within documented tolerance vs fp32 — **not** token-exact; lives behind a flag.
- [ ] **E3 · TurboQuant-class near-optimal quantization (research).** Data-oblivious online vector quant (random-rotate → per-coordinate optimal scalar quantizers; 2-stage MSE + 1-bit QJL for unbiased inner products). ~2.5–3.5 bits/channel KV cache near quality-neutral. Plugs into the E2 KV-cache seam. (arXiv 2504.19874)
- [ ] **E4 · Sparse / linear / hybrid attention (research, architecture-changing).** Approximates attention → **breaks canonical GPT-2 parity by construction**; lives on a separate architecture path (also the trigger to reconsider a tape). Hybrid (linear backbone + interleaved full/sparse) is the current sweet spot; watch for "component collapse." Validated on task metrics, not token-exact. (surveys arXiv 2507.19595, 2504.17768)

**Alignment & Post-Training Track (RLHF)** — turns the base LM into an instruction/preference-aligned model. A *capability* axis, not an efficiency one: it changes what the model does, not how fast it runs. The ops (forward/backward/AdamW), tokenizer, and dataloader are reused unchanged, so ops-level parity is untouched — but this is the one track whose *success* is metric-based, not token-exact-parity-based: there is no canonical "GPT-2 RLHF" oracle (outcomes depend on the preference data). New losses are still gradient-checked vs PyTorch; invariant 11's gates are added to, never relaxed. Do not start.
- [ ] **A1 · SFT (supervised fine-tuning).** Fine-tune the pretrained LM on demonstration / chat data with the loss masked to completion tokens. Reuses forward/backward/AdamW + tokenizer; new pieces are an instruction dataloader and the prompt/completion loss mask. Cheapest; unlocks the rest.
- [ ] **A2 · Reward model.** Pairwise-preference dataset (chosen vs rejected); reward model = LM backbone + scalar head, trained with a Bradley–Terry / pairwise ranking loss. New: the scalar head and the ranking loss (fwd+bwd, gradient-checked).
- [ ] **A3 · DPO (recommended alignment step).** Direct Preference Optimization — a single offline loss over preference pairs against a frozen reference (the A1 SFT model) with an implicit KL. No reward model, no sampling loop, no value network. Reuses the SFT weights as a second frozen param `Storage`; far more tractable std-only/CPU than PPO.
- [ ] **A4 · PPO (research, ambitious ceiling).** Full online RLHF: in-loop generation (rollouts via the sampler), reward scoring (A2), value head + GAE, KL-to-reference penalty, clipped policy objective. Needs the sampler inside training and 2–3 resident model copies (policy, reference, reward). GRPO / other critic-free variants noted as simpler alternatives.
