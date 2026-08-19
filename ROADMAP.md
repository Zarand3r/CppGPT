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
- [x] CI: `.github/workflows/ci.yml` — build + test in dev and release, printed parity margins,
      the `ldd` allow-list gate, ASan/UBSan (`--config=asan`), and the viewer wiring check.

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
- [x] `prepare <corpus> <stem> [--val-frac F] [--vocab <existing>]` → `<stem>.train.bin`,
      `<stem>.val.bin`, `<stem>.vocab` (contiguous tail split). Today `prepare` takes an output
      *filename* and emits one `.bin` + a `.bin.vocab` sidecar — moving to a stem is what makes the
      train/val pair and the fine-tune vocab-reuse name cleanly.
- [x] `train` CLI flags: `--layers --heads --embd --ctx --batch --lr --min-lr --warmup --clip --steps --seed --ckpt`
      (`tools/train.cpp` currently hardcodes L3/H4/C64/T32; only steps and the ckpt path are arguments)
- [x] Eval loop: `--val <bin> --eval-interval N --eval-batches K` logging **val loss**, tokens/s, elapsed, peak RSS.
      Without a val number you cannot tell training from memorizing.
- [x] `tools/eval` — the standalone, deterministic evaluation the gate below needs. Full sequential
      pass (no shuffled subset), reporting nats/token, perplexity and bits/char against uniform,
      unigram and bigram baselines counted on the **training** split and scored on the **validation**
      split. Until this existed the gate below named a bigram baseline that was implemented nowhere
      and therefore could never be checked.
- **Gate:** val loss descends and stays below a same-corpus bigram baseline; two runs at the same seed
  produce bit-identical loss curves. **NOT met against an honest baseline.** The model beats a bigram by
  26.7% but **loses to a 5-gram by 7.9%** (1.820 vs 1.686 nats, `docs/measurements.md` M-11). The
  gate as written named a bigram, which any model that learns anything will clear; a 5-gram is the
  bar that separates a language model from a lookup table. Fix is training budget, not architecture
  — the train/val gap is only 0.17 nats, so this is undertraining, not overfitting. Determinism and the baseline ordering are enforced by
  `//tests/integration:e2e_pipeline_test`, not just measured once.

### Infer — generate from a checkpoint you trained
- [x] `generate --checkpoint <ckpt> --vocab <vocab> --prompt "..." --n K [--temperature --top-k --seed]`
- [x] **Build the model from the checkpoint header.** `GPT2`'s ctor needs a `Config` *before*
      `load_checkpoint` can validate one against it — so `generate` must first peek the header
      (`CheckpointFile::open`, already public, returns it), construct from that `Config`, and only
      then load. Without this step `--checkpoint` cannot work at all: you would have to retype the
      exact architecture on the command line and a mismatch is `ShapeMismatch`.
      (today `tools/generate` calls `init_weights` and trains in-process — it can *never* read a `.ckpt`)
- [x] **Short prompts.** `generate()` copies exactly `seq_len` tokens, so a 5-token prompt is impossible.
      Place the prompt at `[0, len)`, forward, read logits at `len-1` (causality makes the padded tail
      inert). Same fix as the shelved `M3-S1`, minus the HuggingFace gate.
- **Gate:** a checkpoint trained to low loss produces text recognizably in the corpus's style from a
  short prompt; greedy decoding (`--top-k 1`) is deterministic.

### Fine-tune — continue from a checkpoint on new text
- [x] `prepare --vocab <existing.vocab>` — tokenize a *new* corpus with the **original** vocabulary.
      **This is the piece that makes fine-tuning possible at all:** a fresh corpus yields a different
      char vocab, hence a different `vocab_size`, hence `ShapeMismatch` on load. Must fail loudly on a
      byte outside the original vocab rather than silently remapping.
- [x] `train --init-from <ckpt>` — load **weights only**, reset optimizer moments/step and restart the
      LR schedule (a plain `--ckpt` resume deliberately restores moments; fine-tuning wants fresh ones
      at a lower LR). The weights-only path already exists — `load_checkpoint` on a moment-less file
      resets Adam and says so.
- **Gate:** fine-tuning a base checkpoint on a target corpus lowers val loss **on the target** relative
  to the base checkpoint, without a from-scratch run beating it in the same wall-clock.

### Enough performance to iterate (not a benchmark contest)
- [x] Build flags fixed — root cause was **FMA contraction, not the ISA**. `--config=release` is now
      `-march=x86-64-v3 -ffp-contract=off`: ~2x on the dominant kernel (2.95 -> 5.82 GFLOP/s), builds no
      longer depend on the build host, parity holds at ~50x margin. See `docs/DECISIONS.md` D1.
- [x] `tools/bench` — merged (PR #20); `bazel run --config=release //tools:bench -- 20`.
- [x] Reassessed, and the cheap half turned out to be worth taking: `matmul_forward` now accumulates
      in 8 independent lanes, **5.71 -> 49.38 GFLOP/s** (8.6x) on the dominant kernel and
      **2908 -> 6429 tok/s** (2.21x) end-to-end on the toy config. `-ffp-contract=off` stopped the
      compiler fusing the reduction but also stopped it splitting one — eight fixed lanes break the
      dependency chain without a fast-math flag, so the order stays deterministic. The parity gate
      *improved* (logit 1.91e-6 -> 1.43e-6). See `docs/DECISIONS.md` D8, `docs/measurements.md` M-1.
- [ ] Threading is now the remaining lever (16 cores idle). Still not an MVP gate: it needs its own
      determinism argument, since a parallel reduction is where reproducibility gets genuinely hard.
- [ ] *(optional)* Gradient accumulation — only if a batch you want exceeds RAM (measured: B=64 at
      T=256 costs 8.15 GB).

---

## M5 — Interpretability: see inside the model

Type a prompt, watch it travel through the transformer, see the prediction form. Detail, prior art
and gates: [`docs/M5_INTERPRETABILITY_PLAN.md`](docs/M5_INTERPRETABILITY_PLAN.md).

Unusually cheap here because **every activation is already retained** — the arena keeps them for
backward, so `GPT2::acts()` exposes the whole forward pass with nothing to instrument.

- [x] **M5-S1** `logit_lens(model, layer, out)` — final layernorm over `residual3[layer]` through the
      tied unembedding: "what would the model predict if it stopped here?"
      *Gate:* at the last layer the lens **is** the model's own final computation, so it must equal
      `acts().logits` bit-identically — a non-circular check that catches a wrong layernorm, stride,
      or transposed unembedding.
- [x] **M5-S2** `tools/inspect` — one forward → schema-versioned JSON (tokens, selected attention,
      residual norms, per-layer lens top-k). Selection flags exist from the first commit: attention is
      `O(L·NH·T²)`, which is 0.5 MB at toy scale and **1.2 GB** at GPT-2 scale.
- [x] **M5-S3** `viewer.html` — self-contained, `file://`-openable, no external requests. Token strip,
      attention grid, logit-lens ladder, residual-norm chart. States the attention-is-not-explanation
      caveat *in the UI*.
- [ ] **M5-S4** Wire an inspect step into `examples/shakespeare`. *(Was marked done in error: the
      command appears in the example's README but `run.sh` never invokes `inspect` — the same
      claimed-but-absent pattern as the `tools/bench` incident that motivated L12.)*
- [x] **M5-S5** Per-position logit lens (`lens_grid`), attention rendered over the text, and per-head
      summary stats (entropy, mean attention distance, mass on position 0). The first is
      nostalgebraist's original lens form; the second is BertViz's insight that a T×T matrix is far
      less legible than shading the actual characters; the third turns a wall of heatmaps into
      something rankable.
- [x] **M5-S6** Interactive prompting — `tools/serve_viewer.py` + a `systemd --user` unit
      (`Restart=always`, survives logout). See `docs/DECISIONS.md` D5.
- [x] **Gate MET:** answerable from the viewer alone. On `ROMEO:\nWhat is` — the model commits at
      **layer 1** (KL-step 0.996 of 1.011 total; KL-to-output falls 1.011 -> 0.110 there); **layer 2 is
      near-identity** (0.015); and **L1H0/L1H2 are previous-token heads** (entropy ~0.5-0.7, mean
      attention distance ~1.5). The third answer is a real named motif, not an absence.
- [ ] ~~Gate:~~ superseded by the line above; original wording — at which layer does the model commit to its top-1?
      which layers are near-identity? does any head show an interpretable pattern? (The last may be
      *no* for a 4-layer model at 900 steps; recording that honestly is a result.)

**Out of scope:** activation patching, ablation, attribution. Those are *causal* analyses; M5 builds
the *observation* layer they sit on.

---

## Finish the tensor-table refactor (D6, partial)

Both arenas now derive sizes from `include/cppgpt/tensors.hpp`, `kDecay` is gone, and `layer_slice()`
removed the stride re-derivations. What remains:

- [ ] **Collapse the 68 hand-written layer-stride expressions** in `forward`/`backward` into loops over
      the table (`at(tensor, layer)` + `LayerParams`/`LayerActs` views). This is the change that takes
      `model.cpp` from 432 lines toward the projected ~375. Deferred, not abandoned: the sites are
      locally correct and parity-gated, the benefit is readability rather than correctness, and it is
      the only part of the refactor that touches the numerical path.
- [ ] **Retire `ParamTensors` / `ActTensors`** if the accessors above prove to be pure indirection.
- [ ] **Check `scripts/gen_fixtures.py`'s ordering against the table BY NAME.** This is the fifth
      correspondence and the only one still maintained by eye — both sides currently agree because two
      authors counted to 16 correctly, which is not a guarantee.

---

## M6 — Interpretability: causal methods and the rest of the viewer

M5 built the **observation** layer. These sit on top of it and are not started. Kept here rather than
in a side document because `ROADMAP.md` is the single source of truth for outstanding work.

### Not built, and why

- [ ] **Activation patching / causal tracing.** Re-run a forward with one activation replaced by its
      value from a *different* prompt, and measure how much the output moves. This is the method that
      turns "attention went here" into "this component caused the prediction" — the caveat the viewer
      currently prints. **Needs a different API than everything in M5:** every existing feature reads a
      finished dump, whereas patching must intervene *during* the forward. Expect a
      `forward_with_patch(layer, tensor, position, replacement)` seam, which is also the natural place
      a future GPU port would hook.
- [x] **Ablation.** Done 2026-08-18: `save_and_ablate`/`restore_ablation` zero the weights that
      carry a component, so no forward hook is needed. `inspect` sweeps all L·(NH+2) components.
- [x] **Attribution / direct logit attribution.** Done 2026-08-18: exact decomposition (parts sum
      to the logit, worst error 2.14e-08), zero extra forward passes.
- [ ] **Neuron-level views.** Max-activating positions over `fch_gelu` (`[L,B,T,4C]`, already in the
      arena). Cheap; deliberately skipped in M5 to keep the first viewer legible.
- [x] **Per-layer KL divergence** — `layer_kl` in the dump (`step` = divergence this layer introduced,
      `to_final` = distance remaining to the output), plus a viewer panel. On the Shakespeare
      checkpoint layer 1 contributes 0.996 nats of the 1.011 total: the model commits at layer 1 and
      layer 2 is near-identity (0.015).
- [ ] **Tuned lens** (Belrose et al. 2023). The plain lens borrows the *final* layernorm for every
      layer, so early layers are systematically distorted — the viewer says so. Fixing it properly
      needs a learned affine probe per layer, i.e. a small training loop, which is why it is a
      milestone and not a patch.

### Frontend improvements

- [ ] **The M5 gate is still unmet.** Its own criterion is answering the three interpretability
      questions *from the viewer*; they have so far been answered from JSON on the command line, which
      is not the same claim. Either the viewer needs a summary panel that states them, or the gate
      should be honestly reworded.
- [ ] **Compare two prompts side by side.** The single highest-value UI addition: nearly every
      interpretability question is differential ("what changes when I say X instead of Y").
- [ ] **Generation, not just inspection.** The viewer shows one forward pass; watching the lens
      evolve *as tokens are sampled* is the thing people actually want to see. Note this changes the
      server's threat model — `n` and temperature become client-controlled numeric knobs (D5 flags
      this as the test of whether that boundary was drawn in the right place).
- [ ] **Attention aggregated across heads/layers**, and head *clustering* by the summary stats already
      computed — BertViz's "model view" and AttentionViz's global perspective.
- [ ] **Streaming / progress** for long prompts; currently the request is atomic.
- [ ] **Deep-link a run** (prompt in the URL) so a finding can be shared, and an export button.
- [ ] **Dark-mode canvas colours** are hardcoded `rgba(59,91,219,·)` in the attention drawing rather
      than reading the theme token, so they do not adapt like the rest of the page.

### Known limitations worth stating in the UI

- [ ] The lens grid shows **top-1 only**; a position where the top two are near-tied looks as
      confident as one that is certain. Shading encodes probability, but a margin column would be
      honest.
- [ ] `head_stats` are computed over the **causal prefix of one prompt** — they characterize this
      input, not the head in general. Real head characterization needs many inputs.

---

## Deferred from the 3-axis review (2026-08-13)

- [x] **`ldd` allow-list.** All SEVEN binaries link only `libc`/`libm`. To be accurate: `libresolv`
      is gone because **glibc 2.34 merged it into libc**, not because of anything we did — the host
      here is glibc 2.43. Now enforced by `tools/check_ldd.sh` in CI, verified in both directions
      (it fails on a binary linking `libz`).
- [ ] **`noexcept load_checkpoint` terminates on `bad_alloc`** — reproduced at `RLIMIT_AS=25 MB`. The
      size *is* bounded by our own model, so checkpoint.hpp's stated defense holds; what fails is the
      repo's nothrow idiom, which `Storage` follows and this one allocation does not. Fix: nothrow +
      `ASSERT_MSG`, matching `Storage`.
- [x] **Activation-side `int` overflow.** Fixed 2026-08-18: `ASSERT_MSG` on `atot`, plus two
      narrow-before-multiply sites. Pinned by `//tests/unit:act_guard_test` with `CHECK_DIES_WITH`.
- [x] **`parse_selection` returns a multiset.** Fixed 2026-08-15: duplicates are rejected, as is
      `--top-k < 1` (it silently produced empty predictions).
- [ ] **`Device`: 17 signatures, 17 tautological asserts, zero callers.** `ops.hpp` additionally
      claims each op "dispatches on it"; there is no dispatch. Pure deletion, ~40 lines.
- [ ] **Dead code**: `DCHECK`/`UNREACHABLE`/`MUST`/`TRY_OR_CONTINUE` (0 uses, and the last is the sole
      consumer of `detail::warn`), 4 unreachable `ErrorCode` values, `Storage::reset()` (documented as
      the per-step mechanism, never called), the 249-line logging subsystem serving 2 call sites.
- [ ] **`verify.hpp` is test-only code shipped as public library API**, and it sizes six allocations
      directly from unvalidated file-header ints — an L3 violation inside the fixture loader the
      constitution's parity promise depends on. Move to `tests/` and bound the sizes.
- [ ] **~92 lines of tool duplication**; the atomic-write copy has already diverged (only one of the
      two writers checked `close()` — now fixed, but the duplication remains).
- [ ] **e2e hermeticity**: the `sh_test` shells out to 8 undeclared host binaries (`python3`, GNU
      `stat -c%s`, `cmp`, `sed`, `grep`). `//scripts` already pins a hermetic Python toolchain.
- [ ] **10 of 14 CLI flags can be ignored** and the e2e stays green — including `--clip` (clipping
      silently disabled) and `--top-k` (greedy silently becomes sampling).
- [ ] **9 `CHECK_DIES` for 96 `ASSERT` sites**, incl. the `INT_MAX` guard and the lens layer bound.
- [ ] **The document-ownership rule is violated by every document it governs** — measurements appear
      in `DECISIONS.md`, `PLAN.md`, `ROADMAP.md`, both example READMEs and both milestone plans.

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
| `tools/bench` was written but lived on an unmerged branch | 3-axis review, 2026-08-03 | ✅ **Merged** (PR #20). |
| Open PRs | — | ✅ #20, #21 merged. |

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
- [ ] **Threading** — the 30 GFLOP/s matmul half is done (D8: 49.38 GFLOP/s single-thread). Threading
      remains, and remains gated on a determinism argument rather than on throughput alone.

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


---

## Reconciliation — work done 2026-08-14..18 that this roadmap did not describe

Recorded because the roadmap is the single source of truth, and for four days it
was not. Everything below is on `main`, tested, and measured.

### Delivered, and it changed the plan
- [x] **The model now beats a 5-gram**, having lost to one. 1.8199 -> **1.5364 nats**
      (2.217 bits/char), top-1 55.0%, using 54 of its 64 context characters.
      `docs/measurements.md` M-11..M-13, `docs/EXPERIMENTS.md` E-1/E-2.
- [x] **`//tools:eval`** — the standalone quality gate. M2's convergence gate named a bigram
      baseline that had no implementation and so could never be checked; it now runs against a
      Witten-Bell n-gram ladder. **The gate was NOT met against an honest baseline until Run B.**
- [x] **`//tools:profile`** — forward-pass latency and per-op breakdown.
- [x] **`--log-csv`, `--ckpt-best`, `--sample random`** on `train`. `--ckpt-best` earned itself
      immediately: val loss is not monotone and the final checkpoint was 0.016 nats worse.
- [x] **Attention forward vectorised** — 1.68x on the op, forward 2.502 -> 2.337 ms (D8's sibling).
- [x] **Positional-encoding panels + next-token prediction** in the viewer (schema 3).
- [x] **`tools/wandb_log.py`** — W&B as a sidecar reading the CSV, so the binaries stay libc/libm.
- [x] **`tools/` utility library** — `run_report`, `audit_run`, `corpus_stats`, `mutate.sh`,
      `check_viewer`. See `tools/README.md`.

### Still open, and now better understood
- [x] **CI** — done 2026-08-18. Every gate verified in BOTH directions before being trusted: the
      `ldd` check fails on a `libz`-linked binary, and `--config=asan` catches a deliberate
      out-of-bounds read (2 findings). A gate that has never been seen to fail is not a gate.
- [ ] **Threading.** The single remaining order-of-magnitude: ~8.7k tok/s with 16 cores idle.
- [ ] **`gelu` is 22.5% of a forward pass**, now the largest single op — but canonical GPT-2 pins
      the formula, so this needs a decision, not an optimisation.
- [ ] **Run the `review-codify-loop`.** Required by `CLAUDE.md` after any review with >=3 findings;
      the 2026-08-18 audit produced five. `docs/engineering-lessons.md` has no entry for this
      period's dominant failure mode: **six verification checks that passed while measuring nothing**
      (a zero-ID grep, a fully cached test run, an empty binary path, a DOM stub that discarded
      output, two 0-byte files, and a death test satisfied by an unrelated failure).
- [ ] **Capacity sweep (Run C).** With budget and sampling both addressed, the remaining ~4.5% gap
      to the nanoGPT reference is most likely model size or context length.
- [ ] **Unit tests for the tool code.** `eval`/`profile`/`wandb_log.py` have e2e and sanitizer
      coverage only, and `eval`'s n-gram shipped with a real bug caught by output shape, not tests.
