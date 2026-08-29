# Implementation plan — the intervention seam

Scope: the `forward_with_patch` seam and the two measurements it unlocks — **M6-A1** (fix the
ablation baseline) and **M6-A2** (conditional co-ablation). `ROADMAP.md` owns the milestone;
this document owns execution. Everything else in M6 is out of scope (§C).

**Migrate, not rewrite.** The observation layer is aligned and already gated by non-circular tests.
One capability is missing — intervening *during* a forward — and nothing has to be torn out to add it.

---

## Decisions made (was §D)

A literature check settled every open tension, and two of the answers deleted work rather than adding it.

**1 · Resample ablation and activation patching are the same mechanism.** The field's own naming makes
this explicit: activation patching is *also called* interchange intervention, causal tracing, and
**resample ablation** — replacing an activation with one cached from a different run. So M6-A1 and
M6-A4 are not two features. One seam plus a choice of donor value gives both.

**2 · The baseline is a corrupted-prompt donor, not a synthetic value.** Heimersheim & Nanda
recommend corrupted-prompt noising/denoising over zero- or mean-ablation: ablations push the model
off-distribution, while a donor prompt isolates one feature and holds the rest of the machinery
fixed. ACDC does the same and calls it an interchange intervention. **Decided:** the donor value is
the default; zero and mean-over-donors remain available as comparisons, since showing that the mode
changes the answer is the entire point of A1.

**3 · No activation pool, and no RNG.** The standard workflow is three forward passes — clean,
donor, donor-patched — with the site cached from the donor run. That is a `[B,T,·]` scratch buffer,
not a data structure. **Decided:** donors are named prompts the caller passes, or chosen by a fixed
deterministic rule in the corpus sweep. Nothing is sampled, so determinism is structural rather than
tested-for, and the whole `ActPool` / seeding / pool-provenance apparatus is deleted from this plan.

**4 · The metric stays KL.** Logit difference is the recommended metric in the literature, but it
needs a two-way contrast (Mary vs John) that a character-level model does not naturally have, and
every published number in this repo is already in nats. ACDC's default is also KL. **Decided:** KL
only. Logit difference is rejected, with reason, rather than deferred.

**5 · Noising, not denoising, and say so.** The two directions are asymmetric — denoising asks
whether a component is *sufficient*, noising whether it is *necessary*. The existing sweep is
noising. **Decided:** stay with noising, and state the asymmetry in `docs/INTERPRETING.md` rather
than building both.

**6 · `save_and_ablate` stays.** It was going to be retired as duplication. It is the independent
second implementation that makes P2 non-circular, it is already written and tested, and deleting it
would remove a gate. **Decided:** keep, and stop calling it a tension.

**Still yours, not mine:** M6-B7 (auto-interp) needs an external LLM, and `docs/constitution.md` is
human-frozen. Out of scope here; flagged in `ROADMAP.md` so it is not lost.

### The fact that shapes the design

`save_and_ablate` zeroes **weights** (`attprojw`'s column block for a head), and `interpret_test`
already proves that equals zeroing the head's channels of `atty`. The existing tested path and the
new activation path therefore meet at exactly one point: zero. That makes **P2** a genuinely
non-circular gate — two implementations sharing no code must agree bit-for-bit — and it is why the
seam can be trusted before any new number is believed.

### Already aligned, do not rebuild

| | |
|---|---|
| `src/model.cpp:169–214` | the per-layer block is a **flat loop over arena slices**. Patch points are lines in a loop body, not a call graph. |
| `layer_slice` / `head_slice` | the strides a patch site needs, already centralised (three past bugs) |
| `tools/inspect.cpp:601–621` | the `sweep(kind, name, layer, head)` lambda — Steps 2 and 3 extend it |
| `tools/ablation_stats.cpp:159–171` | the corpus loop; Step 4 parameterises it |
| `include/cppgpt/interpret.hpp` | `kl_divergence`, `Ablation`, the doc conventions to match |
| `tests/unit/interpret_test.cpp` | the house idiom: one load-bearing check per feature, chosen so it cannot pass for a broken implementation |

---

## The steps at a glance

- [ ] **Step 0** — Stamp a synthetic golden. A reference for every later step.
- [ ] **Step 1** — The seam: `forward` takes an optional `Patch`. Gates P1–P4.
- [ ] **Step 2** — A1 end-to-end: donor / mean / zero baselines, library → JSON → viewer.
- [ ] **Step 3** — A2 end-to-end: exhaustive conditional co-ablation. Gates P5.
- [ ] **Step 4** — B1: corpus study re-run; correct M-16 and M-17.

```
0 ──▶ 1 ──▶ 2 ──▶ 3 ──▶ 4
```

Strictly serial — each step consumes the previous step's measurement. That is a property of the work,
not an oversight. There is no separate "add CI guards" step: each step's acceptance gates go into CI
with the step, because a guard added later turns main red on merge.

---

## Properties

Five, each `==` rather than tolerance-based, each tied to one step.

### P1 — Patch identity
**Invariant:** patching any site with the value it already holds is bit-identical to not patching.
**Forbids:** a wrong stride, a wrong offset, a partial block, a perturbed neighbouring channel.
**Proved by:** Step 1 — `forward` with and without the patch, `==` over the whole logits buffer.

### P2 — Zero-patch equals weight ablation
**Invariant:** patching a head's `atty` channels with zeros is bit-identical to
`save_and_ablate(Ablation::Head)` + `forward`.
**Forbids:** two ablation truths in the repo with no way to tell which produced a published number.
**Proved by:** Step 1. Non-circular — one path goes through `attprojw`, the other through `atty`.
**Caveat found in review:** the two paths compute `atty·0` and `0·attprojw` respectively, which agree
bit-for-bit **only while both operands are finite** (`inf·0` is NaN, `0·w` is 0). That is already a
fail-fast condition, but if P2 ever fails mysteriously, check for a non-finite activation before
suspecting the strides.

### P3 — The unpatched forward is untouched
**Invariant:** `forward` with no patch is bit-identical to the pre-change build.
**Forbids:** any change to the numerical path — training and the canonical-GPT-2 parity gate both run
through this function.
**Proved by:** Step 1, via the **existing** `//tests/integration:parity_test` plus Step 0's golden.
No new test; the gate already exists and this is what it is for.

### P4 — The seam never mutates parameters
**Invariant:** after any patched forward, the parameter arena is bit-identical to before it.
**Forbids:** the `save_and_ablate` mutate-then-restore pattern leaking into the new path, where one
missed restore silently corrupts every later measurement in a sweep.
**Proved by:** Step 1 — checksum the parameter arena before and after.

### P5 — CoAx reduces to marginal ablation at |S| = 0
**Invariant:** the conditional score with an empty primary set equals the marginal sweep, bit-for-bit.
**Forbids:** a conditional score silently measuring something else while looking plausible.
**Proved by:** Step 3.

> Determinism is **not** a property here. With donors named rather than sampled there is no RNG in
> this code, so there is nothing to seed and nothing to test. That is the design doing the work
> instead of a gate.

---

## How to execute

1. **Contract first.** Step 1 locks the patch-site vocabulary; nothing downstream starts until P1–P4 are green.
2. **One measurement per step, end-to-end.** A step is done when the number reaches the viewer, not when the library function compiles.
3. **Rewrite from scratch when easier, but verify before deleting.** The gate is the test, not the provenance.
4. **A failing test is information.** Read the assertion before changing anything.
5. **Deletions live in Acceptance, never in Implementation.**
6. **Properties are merge gates.**

---

## Step 0 — Stamp a synthetic golden

**Goal:** a committed record of today's behaviour, so every later step proves it changed only what it meant to.

**Why now:** P3 compares against "the pre-change build" — that needs an artifact, and after Step 1
lands it is too late to capture one.

**Constraint found in pre-flight:** `data/` is **gitignored**; `data/shakespeare.ckpt` is 9.7 MB and
not committed. A golden built on it could not run on a clean checkout — exactly what
`docs/engineering-lessons.md` **L11** forbids. So the golden is synthetic, and real-model numbers go
where this repo already puts them: `docs/measurements.md`, with a reproduce command.

### Tests first
- [ ] `//tests/unit:inspect_golden_test` — builds a synthetic model in-process (the `interpret_test`
      idiom: `n_layer=3, n_head=2, n_embd=16, vocab=11`, `Generator(31337)`), runs the same code path
      `inspect` runs, canonicalises the JSON, compares to a committed golden.
- [ ] The golden contains the full `L·(NH+2)` sweep, so a change in ablation semantics shows up as a
      diff rather than a number nobody re-read.

### Implementation
- [ ] `tests/fixtures/inspect_golden.json`, plus the generator script committed beside it.
- [ ] The canonicaliser lives in one place (`tools/` already carries ~92 lines of duplication).

### Integration check
- [ ] `bazel test //...` green, 32/32.

### Acceptance
- [ ] Passes with `data/` moved aside. **Verify by actually moving it**, not by reasoning about it.
- [ ] Perturbing one weight makes it **fail**. A gate never seen to fail is not a gate — this repo has
      shipped six checks that passed while measuring nothing.

**Depends on:** nothing.

---

## Step 1 — The seam

**Goal:** run a forward with one named activation site replaced, without mutating parameters and
without changing the unpatched path.

**Why now:** every remaining item in this plan is an activation-level intervention, and none are
expressible today. This is M6's only new API.

**Note:** the site vocabulary is the contract — lock it before Step 2, because the JSON schema and
the viewer both name these sites.

### Tests first
- [ ] **P1** — patch with the current value vs no patch, `==` over the logits buffer.
- [ ] **P2** — zero-patch a head vs `save_and_ablate(Head)` + forward, `==`.
- [ ] **P4** — parameter-arena checksum before `==` after.
- [ ] A patch at layer `l` leaves activations at layers `< l` bit-identical. Catches a patch applied
      at the wrong point in the block.
- [ ] `CHECK_DIES_WITH` on an out-of-range layer, head, or position. *(9 `CHECK_DIES` for 96 `ASSERT`
      sites is the current ratio; this is not the step to widen it.)*

### Implementation
- [ ] `enum class PatchSite { HeadOut, MlpOut, AttnBlockOut }` — exactly the three existing `Ablation`
      kinds, one for one, so the two vocabularies cannot drift. This matches the field's own "site"
      abstraction (a head, a block, a position).
- [ ] `struct Patch { PatchSite site; int layer; int head; const float* replacement; }` — POD,
      caller-owned buffer, no allocation.
- [ ] `void GPT2::forward(const int* tokens, const int* targets, int logits_at, const Patch* patch)`
      — one `if (patch && patch->layer == l)` per layer, applied straight after the op that writes the
      site. **Not a callback:** a function pointer in this loop is indirection the doctrine forbids
      and the profiler would notice.
- [ ] `capture_site(const GPT2&, PatchSite, int layer, int head, float* out)` — the donor half. Reads
      the site out of the arena after a forward. This is the whole of "caching the donor run".

### Integration check
- [ ] `//tests/integration:parity_test` green — the canonical-GPT-2 gate runs through this function.
- [ ] Step 0's golden unchanged.
- [ ] `//tools:profile` — unpatched forward within noise of M-9's 3.37 ms at B=1, T=64. If a null
      check costs measurably more than nothing, the placement is wrong.

### Acceptance
- [ ] P1–P4 green.
- [ ] `grep -n "float\* saved" src/interpret.cpp` shows the new path has **no** save buffer. If it has
      one it is mutating parameters and P4 is a lie.
- [ ] `docs/measurements.md` records patched-vs-unpatched forward cost.

**Depends on:** Step 0.

---

## Step 2 — A1: donor-based ablation, end-to-end

**Goal:** `inspect` reports every component's effect under a **donor** baseline as the default, with
zero and mean-over-donors alongside, and the viewer shows all three and names which is shaded.

**Why now:** this is a correction to published numbers, not a new feature. M-16 and M-17 currently
measure a model driven off its own activation distribution.

### Tests first
- [ ] Patching a site from a donor whose value at that site equals the clean value reduces to P1 —
      a third independent route to the same identity.
- [ ] Mean over a single donor equals that donor. Catches an accumulator that never divides.
- [ ] Schema: each ablation entry carries a `baseline` field, and an unknown value is **rejected
      loudly** rather than defaulted.

### Implementation
- [ ] `--donor "<prompt>"` on `inspect`. Absent, the donor is a fixed deterministic transform of the
      prompt, recorded in the dump — never an implicit or hidden choice.
- [ ] Streaming mean over donors: one accumulator buffer, no pool.
- [ ] Extend `inspect.cpp`'s `sweep` lambda over the three baselines.
- [ ] JSON: `ablation[].effect` → `ablation[].effects{donor,mean,zero}`. **Bump the schema version** —
      a viewer reading the old field must fail loudly, not render an empty bar.
- [ ] `viewer.html`: three bars per component; the caveat text names the shaded baseline.

### Integration check
- [ ] Golden regenerated **once**, here, with the schema bump. This is the intended artifact change.
- [ ] Viewer opens from `file://`, renders the new dump, no network.

### Acceptance
- [ ] The dump records **at least one component whose rank changes between zero and donor baseline** —
      or, if none does, that is recorded in `docs/measurements.md` as the finding. Both are results;
      silence is not.
- [ ] `docs/INTERPRETING.md` §4a updated from "tracked in M6-A1" to what was measured, and the
      noising/denoising asymmetry stated (decision 5).

**Depends on:** Step 1.

---

## Step 3 — A2: exhaustive conditional co-ablation

**Goal:** for every ordered pair of the 24 components, how much the second one's effect **grows**
once the first is silenced — and the viewer names each component's backup partners.

**Why now:** the direct answer to "why does this head dominate the sweep and that one not at all",
and the item where this repo does what the field approximates: 24 components → 576 ordered pairs →
~2 s at T=64, exhaustive rather than sampled.

### Tests first
- [ ] **P5** — empty primary set reproduces the marginal sweep, `==`.
- [ ] Both orders are computed; the test asserts one is not silently reused for the other.
- [ ] A component co-ablated with itself is handled explicitly, not left as a meaningless diagonal.

### Implementation
- [ ] `coax_sweep(model, baseline, out[24*24])` — two nested loops over the existing component
      enumeration, reusing Step 2's baselines. The baseline is a parameter: CoAx under zero ablation
      inherits zero ablation's off-distribution problem.
- [ ] JSON: a `coax` matrix plus, per component, its top-k backup partners.
- [ ] `viewer.html`: on the component card — "silencing this makes *these* grow."

### Integration check
- [ ] Golden regenerated with the `coax` section.
- [ ] Runtime measured. If it exceeds the ~50-forward real-time budget at the served prompt length,
      it moves to the offline lane and `ROADMAP.md` is corrected.

### Acceptance
- [ ] P5 green.
- [ ] **The L0 result is stated explicitly.** M-17 records the L0 block at 22.9× the sum of its heads.
      CoAx either reproduces that as measured super-additivity or it does not, and the number goes in
      `docs/measurements.md` either way. *This is the step's real gate — an unexplained 22.9× is what
      motivated the method.*
- [ ] `docs/INTERPRETING.md` §4b updated from prediction to result.

**Depends on:** Step 2.

---

## Step 4 — B1: the corpus study, re-run

**Goal:** M-16's corpus statistics re-measured under the donor baseline.

**Why now:** M-16's headline — the single-prompt view overstating one head by 8× its median — was
measured under zero ablation, and it is cited in two documents.

### Tests first
- [ ] `--baseline` accepted and an unknown value rejected loudly.
- [ ] Donor selection is a deterministic rule over the window index; two runs produce identical output
      with no seed involved.

### Implementation
- [ ] Parameterise `ablation_stats.cpp`'s loop by baseline. Donor for window `i` is a fixed function
      of `i` — no RNG, no pool.

### Integration check
- [ ] Full run at 128 windows × 32 tokens per baseline; ~3 s each per M-16.

### Acceptance
- [ ] `docs/measurements.md` M-16 carries every baseline, with the 8× claim confirmed or corrected.
- [ ] If the ranking of "large *and* consistent" components changes, `docs/INTERPRETING.md` §4 is
      rewritten — it currently names L0 MLP and L0 attn on zero-ablation evidence.
- [ ] Run the `review-codify-loop` — required by `CLAUDE.md` at a milestone boundary, and overdue
      from the 2026-08-18 audit.

**Depends on:** Step 3.

---

## Definition of done

- [ ] P1–P5 green in CI.
- [ ] `inspect` reports donor, mean and zero baselines, plus the 576-pair CoAx matrix.
- [ ] The viewer's component card answers, for a clicked head: how much it matters, under which
      baseline, whether that survives across the corpus, and which components back it up.
- [ ] M-16 and M-17 confirmed under the donor baseline, or corrected.
- [ ] `ROADMAP.md` M6-A1, M6-A2, M6-B1 checked, with measured numbers.

## §A — Golden path

```
GIVEN  a seeded synthetic model (n_layer=3, n_head=2, n_embd=16, vocab=11,
       Generator(31337)) and a fixed token sequence
WHEN   the inspect code path runs at the current schema version
THEN   the canonicalised JSON matches tests/fixtures/inspect_golden.json byte-for-byte
```

Runs after every step. Regenerated only in Steps 2 and 3, where the schema intentionally changes.
Real-model numbers are not a test — they go in `docs/measurements.md` with a reproduce command,
which is this repo's existing convention and needs no second target.

## §B — Iteration loop

```
Read the failing assertion verbatim
        │
        ▼
Is the test's invariant correct?
   No ◀─┴─▶ Yes
   │         │
   ▼         ▼
 Fix test  Fix impl — minimum change OR rewrite the file
 + note      fresh against the test, whichever is faster
 in PR       │
        ▼
Re-run the failing test → run §A → green ⇒ done
```

**Stuck > 30 min on the same failure:** stop coding; write expected vs observed in the PR draft;
print actual values; re-read the step's Acceptance; consider rewriting the file from scratch against
the acceptance test. Still stuck — escalate in the PR. **Do not start the next step.**

## §C — Out of scope

A3 path patching, A5 QK/OV panels, A6–A15 viewer work, B2–B8. Also out of scope: threading, the KV
cache, and anything at GPT-2 124M scale — at 4.34 s per forward the 576-pair sweep is 40 minutes,
which is a different plan.

**A4 is not out of scope so much as absorbed:** decision 1 makes activation patching the same
mechanism as the donor baseline. What Step 2 does not build is the *UI* for choosing arbitrary patch
sites interactively; the library supports it the moment Step 1 lands.

## §D — References for the decisions above

[How to use and interpret activation patching](https://www.lesswrong.com/posts/FhryNAFknqKAdDcYy/how-to-use-and-interpret-activation-patching) (Heimersheim & Nanda, 2024) — corrupted-prompt over ablation; noising vs denoising asymmetry; metric pathologies; backup heads ·
[Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) (Zhang & Nanda, ICLR 2024) — hyperparameter choices change the answer ·
[ACDC](https://www.emergentmind.com/papers/2304.14997) (Conmy et al., 2023) — interchange interventions, KL as default metric ·
[Causal scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) (Redwood, 2022) — why zero and mean go off-distribution ·
[Conditional Co-Ablation](https://arxiv.org/abs/2607.01940) (2026) — the Step 3 method ·
[nnpatch](https://github.com/jkminder/nnpatch) — the "site" abstraction adopted in Step 1.
