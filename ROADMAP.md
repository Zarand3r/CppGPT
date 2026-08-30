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

## M6 — Interpretability: from observation to explanation

M5 built the **observation** layer: you can watch the prediction form. M6 answers the two questions
the viewer currently invites but cannot settle. Every item below is tagged with which one it serves.

> **Q1 — What does this layer or head *represent*?**
> **Q2 — *Why* does this component dominate the ablation sweep, and that one not at all?**

Method choices are grounded in the literature (§ References at the end of this section). Where this
repo can do something the field normally cannot, that is called out — it is the reason to build it
here rather than read a paper about it.

**Execution detail for the first three items** — the `forward_with_patch` seam, A1 and A2 — is
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): properties P1–P7, six steps with binary
acceptance gates, and five design tensions (§D) that need your call before Step 2 lands. This page
still owns the checkboxes.

### The dividing line: real-time vs offline

**Not cost — input.** At toy scale almost everything here is cheap.

- **Real-time** — needs only *this prompt* (or only the *weights*). Runs inside the request
  `tools/serve_viewer.py` already serves. Budget from M-9: one forward is **0.61 ms** at T=14 and
  **3.37 ms** at T=64; today's request is 1 forward + the `L·(NH+2)` = 24-forward ablation sweep,
  ~80 ms worst case. **Under ~50 forwards of one prompt belongs here.**
- **Offline** — needs *a corpus*, *many prompts*, or *a training loop*. Emits a versioned artifact,
  generated once and loaded by the viewer. Never run per request.

Two facts settle the borderline cases:

- **`ablation_stats` is offline because its answer is unshareable, not because it is slow** — 3,200
  forwards in ~3 s (M-16). Its result is a corpus statistic no single prompt can re-derive. That is
  the test, not the clock.
- **At GPT-2 124M the real-time budget collapses.** One forward at T=512 is **4.34 s** (M-14), ~7,000×
  the toy cost. Every real-time item is real-time *at toy scale only* until threading lands.

### Two things stand between the current state and everything below

- [x] **The intervention seam — done 2026-08-27 (D10, M-18).** `save_and_ablate` zeroes **weights**. That is why it needs no
      hook — and why it can only ever express *zero* ablation. Mean ablation, donor patching
      (a.k.a. resample ablation), and path patching are all **activation**-level interventions and none of
      them are expressible today. One seam —
      Shipped as a defaulted `(patches, count)` argument on `GPT2::forward` plus three `apply_patch`
      calls — eight substantive lines in the model layer, with `patch.hpp` a stated exception to the
      two-layer boundary the build now enforces (D10). A SET rather than one patch, because
      co-ablation holds a primary silenced while measuring a second. A1, A2 and B1 were built on it;
      A3 and A4 need no further core work. It is also where a future GPU port hooks.
- [ ] **A corpus-artifact channel for the viewer.** Every real-time feature is served by one
      per-request `inspect` dump; every offline item produces an artifact that is *not* a function of
      the prompt, and there is no convention for one — no schema, no version field, no loader, no
      place on disk. The first offline item to land will invent it and the next four will inherit it
      or fork it. Decide before A5/B3: extend the dump schema with an optional `corpus` section
      `inspect` merges in, or give the viewer a second fetch. **Needs a `docs/DECISIONS.md` entry.**

---

### Lane A — real-time (one prompt or the weights alone, inside the viewer request)

Done already: **ablation** (zero, weight-level), **direct logit attribution** (exact, sum-rule
tested), **per-layer KL**, **logit lens** + `lens_grid`, **head_stats**, **positional-encoding panel**.

- [x] **A1 · Fix the ablation mode. [Q2]** — done 2026-08-28. `inspect --donor` reports zero and
      resample baselines side by side; **21 of 24 components change rank** and `attn L0` falls from
      first to sixth (M-19). Mean ablation deliberately deferred to B1, where a corpus makes it
      meaningful — a "mean" over one donor is that donor.
      The published sweep is **zero-ablation**, which the causal-scrubbing line of work considers
      off-distribution *in an unprincipled manner*: it destroys properties of the activation
      distribution, so the ablated model can look either worse or better than it should, and the
      error has no known sign. The field's recommendation is a **corrupted-prompt donor** — patch the
      component's output with its value on a prompt that differs in one controlled way — because that
      isolates one feature while holding the rest of the machinery fixed. Zero and mean-over-donors
      stay as comparisons; showing that the baseline changes the answer *is* the point.
      This is a **correction to M-16 and M-17, not a new feature**: those numbers currently measure a
      model driven off its own distribution.
      *Cost:* free once the seam exists — same forward count, one scratch buffer, **no pool and no
      RNG**, since the donor is a named prompt rather than a sample. *Gate:* report every baseline
      side by side; a component whose rank changes between them is a finding, not noise.
- [x] **A2 · Conditional co-ablation (CoAx). [Q2]** — done 2026-08-28. Explains M-17's 22.9×:
      the L0 heads compensate for each other and `L0mlp` absorbs the block (M-20). **Exceeds the
      real-time budget by ~10×** — 600 forwards, 327 ms at toy scale versus 30 ms without — so it is
      opt-in via `--coax`, and at GPT-2 scale (43 min) it is offline outright. The lane rule below
      stands; this item does not meet it, and says so rather than being quietly reclassified.
      Self-repair: remove a head and a dormant backup takes over, so the primary measures small
      *and* the backup measures small on the intact model. Both look unimportant; neither is. CoAx
      scores each remaining unit by **how much its ablation effect grows once a primary set is
      removed** — label-free and output-grounded. Reported to raise backup-head recovery from 0.33 to
      **0.91 ROC-AUC** on GPT-2-small's IOI circuit, beating self-repair-aware gradient scores (0.82).
      **This repo is an unusually good substrate:** the field uses approximations because exhaustive
      conditional sweeps are infeasible; here there are only `L·(NH+2)` = **24 components, so all 576
      ordered pairs cost ~2 s at T=64**. Exhaustive, not sampled.
      This is the method that explains **M-17's 22.9×** — the L0 block being far more damaging than
      the sum of its heads *is* super-additivity, which is what CoAx measures directly rather than
      noting as an anomaly.
- [ ] **A3 · Path patching / direct-vs-indirect decomposition. [Q2]**
      `docs/INTERPRETING.md` §3 states that attribution (direct effect) and ablation (total effect)
      can disagree completely, and leaves the reader there. Path patching closes it: patch a component's
      effect *along a specific downstream path* and you learn **through which later component** a head
      acts. "Large ablation, small attribution" stops being a puzzle and becomes a named route.
      *Cost:* one forward per path; cap the path set the way A4 caps its grid.
- [ ] **A4 · Causal-tracing grid. [Q2]** — *the mechanism is A1's; only the UI is separate.*
      The field's own naming settles this: activation patching, interchange intervention and
      **resample ablation are the same operation**. Once A1 lands, tracing is that operation swept
      over a position × layer grid rather than over components. *Cost:* 1 forward per site, so the
      grid is `T·L` forwards — 56 (~34 ms) at T=14, 256 (~0.9 s) at T=64. **Cap the grid or it leaves
      the lane.**
- [ ] **A5 · Weight-based QK/OV circuit panels. [Q1] — the highest insight-per-line item in M6, and
      the one this repo can do that GPT-2-scale work cannot.**
      A head is a **QK circuit** (what it reads) and an **OV circuit** (what it writes). Both have
      closed forms over the vocabulary:
      - Full **OV** circuit `W_E · W_V · W_O · W_U` — "if the head attends to source char *i*, how
        much does it push output char *j*?"
      - Full **QK** circuit `W_E · W_Q · W_Kᵀ · W_Eᵀ` — "at dest char *i*, which source char does
        this head look for?"

      At GPT-2 these are 50257×50257 and nobody renders them. **Here `vocab_size` is 65, so each is a
      65×65 matrix that fits on screen** — 16 heads, computed in microseconds, **prompt-independent**,
      a property of the head rather than of one input. This is the single most direct answer to "what
      does this head represent" available in this codebase.
      Plus the two summary scores that make heads rankable: **copying score** = fraction of positive
      real eigenvalues of the OV matrix (Elhage et al.), and **prefix-matching score** (Olsson et al.,
      needs the offline probe B4). A head high on both **is** an induction head, by definition.
      *Cost:* pure weight algebra, zero forwards, no corpus. *Gate:* the OV matrix of a head the
      current viewer already calls a previous-token head should show what a previous-token head
      should show — a non-circular check, since one measurement is from weights and one from attention.
- [ ] **A6 · Neuron views for the current prompt. [Q1]** Top `fch_gelu` activations per position
      (`[L,B,T,4C]`, already in the arena). The corpus-wide twin is B3.
- [ ] **A7 · The component card. [Q1][Q2] — the consolidation the frontend needs.**
      Click a head or MLP, get one panel: OV vocabulary matrix and copying score (A5), QK preference
      (A5), corpus attention stats with **median not single-prompt** (B2), ablation effect in all
      every baseline with median and activity rate (A1 + M-16), direct effect (done), the path it acts
      through (A3), CoAx backup partners (A2), and top-activating contexts (B3). Today these numbers
      exist in different tools, different files, and two different documents. **This is the item that
      turns the viewer into an answer to Q1 and Q2 rather than a pile of panels.**
      *Design precedent:* Anthropic's **HeadVis** (2026) uses exactly this workflow — pick a head
      that is extreme on some metric, browse its patterns across dataset examples, then read the QK
      and OV attributions. Adopt the shape.
- [ ] **A8 · The M5 gate is still unmet.** Its criterion is answering the three interpretability
      questions *from the viewer*; they were answered from JSON on the command line, which is not the
      same claim. A7's summary panel satisfies it, or the gate should be honestly reworded.
- [ ] **A9 · Compare two prompts side by side.** Nearly every interpretability question is
      differential. Two requests, no new numerics — and the natural front end for A4.
- [ ] **A10 · Generation, not just inspection.** Watching the lens evolve *as tokens are sampled* is
      what people actually want. `n` forwards, so in-lane only for small `n`. Changes the server's
      threat model — `n` and temperature become client-controlled numeric knobs (D5 flags this as the
      test of whether that boundary was drawn in the right place).
- [ ] **A11 · Attention aggregated across heads/layers**, and head *clustering* by the stats already
      computed — BertViz's model view, AttentionViz's global view. Client-side, zero extra forwards.
- [ ] **A12 · Streaming / progress** — the request is atomic today, which stops being acceptable once
      A2 or A4 is in it.
- [ ] **A13 · Deep-link a run** (prompt in the URL) plus an export button.
- [ ] **A14 · Lens margin column.** The grid shows **top-1 only**; a near-tie looks as confident as a
      certainty.
- [ ] **A15 · Dark-mode canvas colours** are hardcoded `rgba(59,91,219,·)` in the attention drawing
      instead of reading the theme token.

---

### Lane B — offline (corpus passes and training loops, into a versioned artifact)

Done already: **`//tools:ablation_stats`** — 128 windows × 32 tokens, which established the
median-and-activity-rate summary the rest of this lane should copy (M-16).

- [x] **B1 · Corpus ablation under both baselines. [Q2]** — done 2026-08-29 (M-21). The offline half
      of A1. M-16's headline — the single-prompt view overstating one head by **8×** its median — was
      measured under zero ablation; whether it survives the donor baseline is unknown and is the first
      thing to find out. *(Small: ~3 s per mode.)*
- [ ] **B2 · Corpus-wide attention statistics. [Q1]** `head_stats` (entropy, mean distance, mass on
      position 0) are computed over the **causal prefix of one prompt** — they characterise that
      input, not the head. M-16 already showed what single-prompt numbers do to ablation; the same
      correction is owed here **before any head is named in the UI**. *(Small.)*
- [ ] **B3 · Max-activating examples. [Q1]** For each of the 2,048 MLP neurons, its top-activating
      contexts over the corpus. Needs **no new model** — `fch_gelu` is already in the arena. One
      corpus pass; artifact is per-neuron top-k contexts. *(Small.)*
- [ ] **B4 · Induction-head probe. [Q1]** Repeated random sequences (Olsson et al.), giving the
      **prefix-matching score** that completes A5's pair. A yes/no answer about whether a canonical
      circuit exists in 4 layers — and if the answer is *no*, that is a result about a character-level
      model, recorded, not a failure. *(Small.)*
- [ ] **B5 · Attribution patching, and the measurement of its error. [Q2] — the one item here that is
      a contribution rather than an application.**
      `effect ≈ −∇a·a` scores every site from **1 forward + 1 backward** (the estimator is real-time;
      this study is not). Edge attribution patching does the same for edges: two forwards and one
      backward for the whole graph, versus one forward per edge. **The field uses this because
      exhaustive ablation is infeasible — here it is affordable, so the approximation's error can be
      measured rather than assumed.** Registered predictions of where it fails: M-17's non-additivity
      (L0 block at 22.9× the sum of its heads), and gradient saturation, which is why EAP-IG
      integrates along a path instead of taking a single gradient. *(Medium.)*
- [ ] **B6 · Tuned lens. [Q1]** The plain lens borrows the *final* layernorm for every layer, so early
      layers are systematically distorted — the viewer says so, `interpret.hpp` says so. The fix is
      concrete: a per-layer **affine probe `ĥ = h + Wh + b`, `W` initialised to zero so it starts as
      identity, trained to minimise KL to the final-layer distribution**. That is a small training
      loop this repo can already run. Artifact is the probe weights; the lens itself stays real-time
      once trained. *(Medium.)*
- [ ] **B7 · Automated interpretation of neurons and heads. [Q1]** Generate a natural-language
      explanation from B3's top contexts, then **score it by simulation** — predict activations from
      the explanation alone and correlate with the real ones. The score is the point; unscored
      explanations are the known failure mode of this technique, and even scored, it explains
      *correlation with text*, not downstream causal effect.
      ⚠️ **Constitution question:** this needs an external LLM. Keep it strictly offline and
      out-of-band so the artifact is committed text and no binary gains a runtime dependency —
      `docs/constitution.md` is human-frozen, so **this is your call, not an agent's.** *(Medium.)*
- [ ] **B8 · Transcoders → attribution graphs. [Q1][Q2] — the field's current endpoint, and
      deliberately last.**
      A **transcoder** approximates an MLP's whole input→output map with a wider sparsely-activating
      MLP; a **cross-layer transcoder** gives each feature a separate decoder per downstream layer.
      That is what makes **attribution graphs** possible: nodes are features, token embeddings, output
      logits and **error nodes**; edges are linear attributions through a *local replacement model*
      that freezes attention patterns and layernorm denominators so feature interactions become
      strictly linear. Validated by intervention (~0.72 Spearman between graph influence and measured
      perturbation effect).
      **Read the stated limitations before committing:** frozen attention means the method does not
      explain QK circuits at all — precisely the "interesting part" for induction heads, and precisely
      what A5 *does* cover here; error nodes can hide the critical step; pruned graphs still run to
      hundreds of nodes. Add the scale caveat: SAE-family methods exist to resolve **superposition**,
      and at `n_embd=128` in a model that beats a 5-gram by 8.9% the payoff is the least certain on
      this page. For calibration, Anthropic's circuit tracing gave satisfying insight on roughly a
      quarter of prompts on a production model. *(Large — hours.)*

---

### Deliberately not doing, and why

- **Plain SAEs on the residual stream.** Transcoders (B8) dominate them for circuit work — they model
  the MLP's transformation rather than reconstructing one point — and both inherit the **dark matter**
  problem: a consistent fraction of activation variance no SAE explains, concentrated on the same
  tokens across sizes and sparsities. If any dictionary method is built here, it is B8.
- **Chasing benchmark faithfulness scores.** Circuit faithfulness metrics have been shown to be
  **not robust** — the measured faithfulness of the *same* circuit moves with methodological choices.
  Gates in this milestone are sum rules, bit-identity, and cross-method disagreement (A1, A5), not
  a leaderboard number.
- **Semantic-feature language.** `docs/INTERPRETING.md` §6 governs: character-level, 4 layers,
  `n_embd` 128. Concept-level claims are unwarranted at this scale regardless of tooling.

### References

Ablation and causality — [Causal scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) (Redwood, 2022; zero/mean ablation is off-distribution) ·
[Path patching](https://arxiv.org/pdf/2304.05969) (Goldowsky-Dill et al., 2023) ·
[The Hydra Effect](https://arxiv.org/pdf/2307.15771) (McGrath et al., 2023) ·
[Conditional Co-Ablation](https://arxiv.org/abs/2607.01940) (2026; self-repair-aware importance) ·
[Circuit faithfulness metrics are not robust](https://arxiv.org/pdf/2407.08734) (2024).
Cheap approximations — [EAP-IG / Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806) (2024).
Heads — [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Elhage et al., 2021; QK/OV, copying score) ·
[In-context Learning and Induction Heads](https://arxiv.org/pdf/2209.11895) (Olsson et al., 2022; prefix-matching score) ·
[A Primer on the Inner Workings of Transformer-based LMs](https://arxiv.org/pdf/2405.00208) (2024) ·
[HeadVis](https://transformer-circuits.pub/2026/headvis/index.html) (2026; the head-investigation UI).
Lenses — [Tuned Lens](https://arxiv.org/abs/2303.08112) (Belrose et al., 2023).
Features and circuits — [Transcoders find interpretable LLM feature circuits](https://arxiv.org/abs/2406.11944) (2024) ·
[Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Anthropic, 2025; CLTs, attribution graphs, error nodes) ·
[Decomposing the Dark Matter of Sparse Autoencoders](https://arxiv.org/html/2410.14670v1) (2024) ·
[Automatically Interpreting Millions of Features](https://arxiv.org/pdf/2410.13928) (2024; explanation scoring) ·
[Open Problems in Mechanistic Interpretability](https://arxiv.org/html/2501.16496) (2025).

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
- [ ] **Load pretrained GPT-2 124M weights** (`//tools:convert_hf` + `tools/import_hf`) and the
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

### BPE + real GPT-2 weights — delivered 2026-08-19
- [x] **Byte-level BPE tokenizer** reading `vocab.json` + `merges.txt` directly. Differential-gated
      against tiktoken AND transformers: 40/40 fixture cases and the full 1.1 MB corpus
      (338,025 tokens) byte-identical to both, round-trip exact. ASCII pre-tokenizer; non-ASCII is
      REFUSED rather than mis-split. (`docs/measurements.md` M-15.)
- [x] **`//tools:convert_hf`** — `model.safetensors` -> checkpoint, in C++, no Python
      (`docs/DECISIONS.md` D9, reversing the plan's first recommendation). Transposes HF's Conv1D
      `[in, out]`, skips the 12 causal masks, and asserts 148 + 12 == 160 before writing.
- [x] **Checkpoint v3** records which tokenizer the weights expect (kind + fingerprint), because
      `vocab_size == vocab_size` is vacuous for BPE. v2 files stay loadable.
- [x] **Forward parity at 124M** against a **fp64** reference, with a gate that cannot be loosened:
      we are *closer to truth than HuggingFace* on 3 of 4 prompts, tied on the fourth.
- [x] **Greedy decode byte-identical to HuggingFace** on real GPT-2 weights.
- [x] **Single-position classifier** (`forward(..., logits_at)`) — 1.87x per generated step,
      bit-identical. A position rather than a "last" flag, because the token buffer is right-padded.
- [ ] **S2: Unicode pre-tokenizer.** ASCII is exact today; `\p{L}`/`\p{N}` need generated property
      tables. Non-ASCII currently fails loudly, which is correct but limiting.

### Resuming in a fresh session (2026-08-26)

Everything below is on disk or on GitHub; nothing lives only in a conversation.
`CLAUDE.md` auto-loads and routes to the skills, so a new session in this
directory starts oriented.

**Read in this order.** `docs/INTERPRETING.md` (how to read the interpretability tools, and what they do not support — the reasoning behind every panel) → `ROADMAP.md` (this file — the single source of truth for
what is left) → `docs/EXPERIMENTS.md` (why the training runs were shaped as they
were, with predictions registered before results) → `docs/DECISIONS.md` D1–D9
(the architectural calls and the two that were reversed) →
`docs/engineering-lessons.md` L1–L19 (the failure modes this repo has actually
hit) → `docs/measurements.md` M-1…M-16 (every number, each with a reproduce
command).

**Open PRs, none merged.** #35 threading design (needs three decisions, no code),
#36 viewer provenance, #37 architecture map (stacked on #36), #38 ablation
variance. `git log --oneline origin/main..<branch>` shows each.

**The mech-interp series is superseded by M6 above.** Its four remaining items
(max-activating examples, attribution patching, the induction-head probe,
transcoders/SAEs) are now B3, B5, B4 and B8 in a consolidated list that adds the
methods a 2026-08-26 literature sweep turned up — resample ablation, conditional
co-ablation, path patching, and weight-based QK/OV circuit panels — each tagged
with whether it answers *what does this component represent* or *why does it
dominate the ablation sweep*. Nothing is tracked in two places; M6 is where these
live.

**Working agreement:** modular PRs for review, not direct pushes to `main`.

### Genuinely open — the honest list (2026-08-20, tag `v0.3.0-gpt2`)

Consolidated so it is not scattered across sections. Everything here is open
*with a reason*, not merely unstarted.

| # | item | why it matters now |
|---|---|---|
| 1 | **Threading** | GPT-2 generates at **5.1 s/token**, single-threaded on 16 idle cores. Correct and unusable. The binding constraint — ahead of any further single-thread work. |
| 2 | **KV cache / right-sized window** | `generate_absolute` pays a full `T=1024` forward per step regardless of prompt length. The algorithmic half of (1); they compound. |
| 3 | **S2 — Unicode pre-tokenizer** | ASCII is exact; `\p{L}`/`\p{N}` need generated property tables. Non-ASCII currently *fails loudly*, which is correct but limits the tokenizer to English-ish text. |
| 4 | **Run `review-codify-loop`** | Required by `CLAUDE.md` after any review with ≥3 findings; the 2026-08-18 audit produced five. `docs/engineering-lessons.md` still has no entry for this period's dominant failure mode. |
| 5 | **Unit tests for tool code** | `convert_hf` and `dump_logits` have **zero** tests (2026-08-20 review). `convert_hf`'s transpose and 148/160 reconciliation are covered only by the real 522 MB download. Unblocked by a small synthetic safetensors fixture — contained, not blocked. Note is at the point of use in `tools/convert_hf.cpp`. |
| 6 | **`gelu` is 22.5% of a forward** | Now the largest single op. Canonical GPT-2 pins the formula, so this needs a *decision*, not an optimisation. |
| 7 | **`notebooks/` on `origin/main`** | Committed in error (`c4aeaa8`); 3 files, 24 KB, no secrets. One command to remove. |
| 8 | **`m1-train` stale branch** | 5 commits, content long since superseded. |
| 9 | **Larger GPT-2 sizes** | 355M+ wants bf16; we are fp32-only. Out of scope, stated so it is not mistaken for an oversight. |

### Still open, and now better understood
- [x] **CI** — done 2026-08-18. Every gate verified in BOTH directions before being trusted: the
      `ldd` check fails on a `libz`-linked binary, and `--config=asan` catches a deliberate
      out-of-bounds read (2 findings). A gate that has never been seen to fail is not a gate.
- [ ] **Threading — now the binding constraint, not a nicety.** GPT-2 124M generates at **5.1 s per
      token** (12 tokens in 61 s at ctx 1024), single-threaded on a 16-core box. Correct and
      unusable. This outranks any further single-thread work. (`docs/measurements.md` M-14.)
- [ ] **`generate_absolute` pays a full T=1024 forward per step** regardless of prompt length. A KV
      cache or a right-sized window is the algorithmic half of the same problem.
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
