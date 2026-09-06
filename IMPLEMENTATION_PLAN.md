# Implementation plan — M7: representation and concept interpretability

Scope: everything needed to move from *"which component matters"* to *"what does it represent"*.
`ROADMAP.md` owns the milestone and the checkboxes; this document owns execution.

The previous plan (the intervention seam, M6-A1/A2/B1) is complete — its five steps are recorded as
`ROADMAP.md` checkboxes and M-18 through M-22.

---

## Read this first: the honest framing

Three facts should shape every step, and a coding agent that forgets them will produce confident
nonsense.

**1. This model may have no concepts to find.** Character-level, 4 layers, `n_embd` 128. Its
regularities are character n-gram statistics, capitalisation and line structure
(`docs/INTERPRETING.md` §6). A5 already found **no copying heads and therefore no induction heads**
(M-22). "This head represents nothing nameable" is a legitimate and likely outcome, and recording it
is the deliverable — not a signal to keep adjusting until something appears.

**2. A head does not have *a* concept.** The literature finds attention superposition: several
concepts share one head. "What does head 3 represent" is often malformed. Prefer "which directions
does this head read and write, and what does each do".

**3. The field has partly moved away from SAEs.** DeepMind published negative downstream results and
deprioritised the direction; tuned linear probes match or beat SAE probes; on OthelloGPT, where
ground truth exists, SAEs recovered 9 of 180 known features. So **probes before SAEs**, and SAEs only
if the cheaper methods leave something specific unexplained. See §D.

---

## Real-time or offline — the rule, and where each step lands

`ROADMAP.md`'s lane rule decides by **input**, not cost: needs only this prompt or the weights →
interactive; needs a corpus or a training loop → offline artifact.

Almost everything here is **fit offline, apply real-time**. That is the shape to build for.

| | fit | apply |
|---|---|---|
| QK/OV circuits (A5, done) | — nothing to fit | real-time; and *prompt-independent* |
| SVD of those circuits (Step 1) | — | real-time, prompt-independent |
| Max-activating examples (Step 4) | one corpus pass | artifact lookup |
| Linear probes (Step 5) | labelled corpus | one dot product |
| Causal validation (Step 6) | — | 1 forward per intervention |
| Attention SAEs (Step 7) | training loop, hours | one matmul |

**Measured (M-24).** The weight-space panel costs **63 ms** of a 90 ms request in a release build,
and its output is **byte-identical across different prompts**, verified. Two earlier figures were
wrong in opposite directions — 96 ms taken before the SVD existed, then 685 ms taken through
`bazel-bin/`, which is a symlink to whichever config built last and was pointing at the debug binary.
Time through `bazel-out/k8-opt/bin/`, never `bazel-bin/`.

---

## The steps at a glance

- [x] **Step 1** — SVD of the OV/QK circuits. Done (M-23). Gates P1, P2 green; six mutations verified.
- [x] **Step 2** — Cache the weight-space panel at server startup. Done: 90 ms → 27 ms (M-24).
- [x] **Step 3** — The corpus-artifact channel. Done (D11, M-26). Gates P3.
- [x] **Step 4** — Max-activating examples over `fch_gelu`. Done (M-26).
- [ ] **Step 5** — Linear probes for candidate directions. Gates P4.
- [ ] **Step 6** — Causal validation: a direction is a hypothesis until steering confirms it. Gates P5.
- [ ] **Step 7** — Attention SAEs, *only if* Steps 4–6 leave something specific unexplained.

```
1 ──▶ 2
      3 ──▶ 4 ──▶ 5 ──▶ 6 ──▶ 7
```

Steps 1–2 are independent of 3–7 and can land first. Step 3 blocks everything after it.

---

## Properties

### P1 — Weight-space panels are prompt-independent
**Invariant:** any panel derived from weights alone is byte-identical across prompts and across
intervening forward passes.
**Forbids:** a stray read of `acts()` turning a claim about the head into a claim about one input.
**Proved by:** Step 1, extending `circuits_test`'s existing purity check to the SVD outputs.

### P2 — The decomposition reconstructs
**Invariant:** `‖A − UΣVᵀ‖_max` is within tolerance, `U` and `V` are orthonormal, and singular values
are non-increasing.
**Forbids:** a plausible-looking basis that is not a decomposition of anything. This is the one gate
that cannot be satisfied by a wrong implementation.
**Proved by:** Step 1, on random matrices *and* on a constructed matrix with known singular values.

### P3 — Artifacts are versioned and staleness is loud
**Invariant:** an artifact records the checkpoint it was computed from; loading one that does not
match the live model fails with a named error.
**Forbids:** the failure mode this repo has hit repeatedly — a stale file rendering as current data.
**Proved by:** Step 3, with a death/error test on a deliberately mismatched artifact.

### P4 — Probe accuracy is held-out
**Invariant:** every reported probe number comes from data the probe was not fitted on.
**Forbids:** the standard way probe results become meaningless.
**Proved by:** Step 5 — a probe fitted on shuffled labels must score at chance. If it does not, the
split is leaking.

### P5 — A direction is a hypothesis until an intervention confirms it
**Invariant:** no direction is presented as a feature in the viewer unless steering along it moves
the output in the predicted direction, measured and recorded.
**Forbids:** the central failure of this field — naming a direction from correlation alone.
**Proved by:** Step 6; the panel shows the causal effect size beside the name, or shows no name.

### P6 — Negative results are recorded
**Invariant:** "we looked and found nothing interpretable" appears in `docs/measurements.md` with the
same detail as a positive result.
**Forbids:** silent iteration until something looks interesting — which given §1 above is the most
likely way this milestone goes wrong.
**Proved by:** review, not code. Called out here because it is the rule most easily skipped.

---

## Step 1 — SVD of the OV and QK circuits

**Goal:** decompose each head's circuits into singular directions, the cheapest thing in this plan
that is literally "directions".

**Why now:** weights-only, no corpus, no training, no new infrastructure. It is the natural extension
of A5 and the only step that can land immediately.

**Why this method:** individual singular directions of a head's circuits have been found to encode
distinct separable operations (*Beyond Components*, arXiv 2511.20273). Unlike an SAE it needs no
training and has an unarguable correctness test.

### Tests first
- [ ] **P2** — `A == U·diag(S)·Vᵀ` within tolerance on random matrices; `U`, `V` orthonormal to
      tolerance; `S` non-increasing and non-negative.
- [ ] A constructed matrix with **known** singular values (e.g. `diag(3,2,1)` under known rotations)
      recovers them. Random matrices alone would pass for a routine that returns *some* valid
      decomposition of the wrong thing.
- [ ] **P1** — SVD outputs identical across intervening forwards.
- [ ] Rank-deficient and square-zero inputs terminate and return zeros rather than looping.

### Implementation
- [ ] One-sided Jacobi SVD in `interpret.cpp` — ~100 lines, no dependencies, numerically sound for
      matrices this size. Iterate until off-diagonal mass is below tolerance or an iteration cap is
      hit; the cap must be an `ASSERT`, not a silent return.
- [ ] `svd_circuit(model, layer, head, which, U, S, V)` where `which` selects OV or QK.
- [ ] Emit the top-k singular directions per head, each with its singular value and the characters
      most aligned with it in embedding space.

### Integration check
- [ ] `//tools:check_viewer` renders the panel populated.
- [ ] Cost measured and recorded; it must stay in the real-time lane.

### Acceptance
- [ ] P1, P2 green, all four mutation classes above verified to fail.
- [ ] `docs/measurements.md` records what the top directions look like — **including if the answer is
      that they are not interpretable** (P6).

**Depends on:** nothing.

---

## Step 2 — Cache weight-space panels per checkpoint

**Goal:** compute prompt-independent panels once, not per request.

**Why now:** measured — the circuits section is byte-identical across prompts and costs ~96 ms every
time. Step 1 adds more of the same. This is the cheapest performance work in the plan and it is a
correctness statement too: caching is only safe *because* P1 holds, so the cache is evidence the
property is real.

### Tests first
- [ ] A cached panel and a freshly computed one are byte-identical.
- [ ] Changing the checkpoint invalidates the cache — a stale hit is the whole risk.

### Implementation
- [ ] Key the cache on the checkpoint's existing payload checksum, which already exists in the header.
- [ ] `serve_viewer.py` reuses it across requests.

### Acceptance
- [ ] Request latency with circuits enabled returns to the `--circuits 0` baseline after the first.
- [ ] `docs/measurements.md` updated with before/after.

**Depends on:** Step 1 (so both panels are cached by one mechanism).

---

## Step 3 — The corpus-artifact channel

**Goal:** a versioned file an offline pass writes and the viewer reads.

**Why now:** it blocks Steps 4–7 and B2/B3/B4 in `ROADMAP.md`. It has been flagged as needing a
`docs/DECISIONS.md` entry since before M6-A1 landed, and four items are now waiting on it.

**Decided — `docs/DECISIONS.md` D11.** A **binary** envelope with a checkpoint-style header
(magic, version, producing checkpoint's checksum, payload), merged by `inspect` so the viewer still
opens one file. Not JSON: there is no JSON parser in this repo, `convert_hf` only manages a flat
header, and a general parser is hundreds of lines of new surface for a machine-to-machine format
nobody reads by hand. The checkpoint format already solves this and is tested.

**Ship this step together with Step 4** so the channel has a real consumer. A loader with no producer
is a speculative abstraction, which this repo's doctrine forbids.

### Tests first
- [ ] **P3** — an artifact whose checkpoint checksum does not match the live model is refused with a
      named error, and no panel renders stale numbers.
- [ ] A missing artifact is a stated absence in the UI, never an empty chart.
- [ ] Schema version bump; an old artifact fails loudly.

### Implementation
- [ ] `Result<void, ErrorCode>` on load, following `save_checkpoint`'s shape — the reason matters.
- [ ] Artifact carries: schema version, checkpoint checksum, the corpus it was built from, and the
      command that produced it.

### Acceptance
- [ ] P3 green, verified by mutation.
- [ ] `docs/DECISIONS.md` D11 written and merged in the same PR.

**Depends on:** nothing, but do it before Step 4.

---

## Step 4 — Max-activating examples

**Goal:** for each of the 2,048 MLP neurons, the corpus contexts that most activate it.

**Why now:** highest insight-per-line on the offline side, needs no new model — `fch_gelu` is already
in the arena — and it is the first real consumer of Step 3's channel.

### Tests first
- [ ] Deterministic: same corpus and checkpoint, byte-identical artifact.
- [ ] A neuron that never activates is reported as such, not omitted.
- [ ] The recorded activation for a context, recomputed by a forward, matches.

### Implementation
- [ ] One corpus pass, streaming top-k per neuron; no full activation history in memory.
- [ ] Viewer: click a neuron, see its contexts.

### Acceptance
- [ ] The artifact loads, the panel renders, and **P6** — the write-up says plainly whether the top
      contexts are interpretable or just frequent character patterns.

**Depends on:** Step 3.

---

## Step 5 — Linear probes for candidate directions

**Goal:** given a labelled property (is-uppercase, is-line-start, is-vowel, follows-speaker-label),
find the direction in the residual stream that predicts it.

**Why probes and not SAEs:** the evidence in §D. Probes are simpler, need less data, and the field
now reports them matching or beating SAE probes.

**Choose labels the model could plausibly encode.** Character-level and 4 layers: capitalisation,
line structure, punctuation, speaker labels. Not sentiment, not semantics.

### Tests first
- [ ] **P4** — a probe fitted on **shuffled labels** scores at chance. If it does not, the split leaks.
- [ ] Train/test split is by corpus position, not random rows — adjacent characters are not
      independent samples.
- [ ] A probe's reported accuracy is reproducible from the recorded seed.

### Implementation
- [ ] Logistic regression by gradient descent, in the training loop this repo already has.
- [ ] Artifact per (layer, property): the direction, held-out accuracy, and the label definition.

### Acceptance
- [ ] Every reported number held-out; the shuffled-label control recorded next to each.
- [ ] Properties that do **not** decode are listed with their accuracies (P6).

**Depends on:** Step 3.

---

## Step 6 — Causal validation

**Goal:** confirm a direction *causes* what it appears to encode, using the patch seam that already
exists.

**Why now:** without it Step 5 produces correlations with names attached, which is the central failure
mode of this field. **This step is what makes the previous one publishable rather than suggestive.**

### Tests first
- [ ] **P5** — steering along a direction moves the output in the predicted direction, measured; a
      random direction of the same norm does not. The random control is the gate.
- [ ] Steering with zero magnitude is a no-op, bit-identical (this is P1 of the seam, reused).

### Implementation
- [ ] Add the direction to the residual at a chosen layer via a `Patch`, scaled.
- [ ] Report effect size, and the off-target cost — the literature's named risks are ripple effects
      and over-steering.

### Acceptance
- [ ] No direction is named in the viewer without a recorded causal effect beside it (P5).
- [ ] Directions that fail the causal test are kept in the artifact, marked failed (P6).

**Depends on:** Step 5.

---

## Step 7 — Attention SAEs, conditionally

**Do not start this step unless Steps 4–6 leave a specific, named thing unexplained.** Write down
what it is first. See §D for why the bar is this high.

If it is justified: train on concatenated head outputs (`z`), then use weight-based head attribution
to assign features back to heads, since features are not head-local.

**Acceptance:** measured against the Step 5 probes on the same properties. If probes match it, the
SAE has not earned its complexity and the honest result is to say so.

**Depends on:** Steps 4–6, and an explicit decision to proceed.

---

## §A — Golden path

The synthetic `//tests/unit:interpret_golden_test` continues to cover every library number. Each step
extends it rather than adding a parallel golden. Real-model numbers go in `docs/measurements.md` with
a reproduce command — the repo's existing convention, and the reason there is no second target.

## §B — Iteration loop

Read the failing assertion. Decide whether the test's invariant or the implementation is wrong before
changing either. Minimum change, or rewrite the file fresh against the test — whichever is faster.
Stuck more than 30 minutes: write expected vs observed in the PR, print actual values, re-read the
step's acceptance block, consider rewriting from scratch. Do not start the next step.

**Every new gate is mutation-tested before it is trusted.** This is not optional here: four gates in
M6 passed against deliberately broken code before being fixed, and two of them were caught only
because the mutation was run.

## §C — Out of scope

Threading and the KV cache (`ROADMAP.md`); everything at GPT-2 124M scale, where one forward is
4.34 s and these sweeps are hours; tuned lens (B6) and transcoders/attribution graphs (B8), which are
tracked separately.

## §D — Why probes before SAEs

Recorded so a coding agent does not reach for the more famous tool first:

- DeepMind published [negative results on SAEs for downstream tasks and deprioritised the direction](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9).
- Tuned linear probes match or exceed SAE probes, including under label noise and covariate shift.
- On OthelloGPT, where ground truth exists, [SAEs recovered 9 of 180 known board-state features](https://www.lesswrong.com/posts/BduCMgmjJnCtc7jKc/research-report-sparse-autoencoders-find-only-9-180-board) — the closest published analogue to this repo's situation, a small model with knowable structure.
- Attention superposition means features are not head-local, so an SAE alone does not answer "what
  does this head represent" without [weight-based head attribution](https://arxiv.org/pdf/2506.17052) on top.

Supporting method references: [Interpreting Attention Layer Outputs with SAEs](https://arxiv.org/pdf/2406.17759) ·
[Beyond Components: singular-vector interpretability](https://arxiv.org/pdf/2511.20273) ·
[Analysing the safety pitfalls of steering vectors](https://arxiv.org/html/2603.24543).
