# cppgpt — executive summary

*Kept current after every significant change. Concise by design; detail lives in the documents it points to.*

---

## Objective

cppgpt is a GPT-2 implementation in dependency-free C++ that trains, runs inference, loads real
OpenAI GPT-2 weights, and — the part that makes it unusual — lets you look inside the model while it
runs. Because every activation is already kept in an arena for the backward pass, interpretability
needs no instrumentation: a browser viewer reads a dump of one forward pass and shows where a
prediction is formed, which components carry it, and what breaks when you silence them. The model is
verified against PyTorch step for step, so what the tools measure is canonical GPT-2 rather than a
lookalike.

---

## Requirements

- **Numerically canonical.** Forward, backward and optimizer match PyTorch's GPT-2 from identical
  weights, and real OpenAI weights load and decode byte-identically.
- **No runtime dependencies.** Every binary links only `libc`/`libm`; the viewer opens from `file://`
  with no network. One pinned hermetic toolchain.
- **Deterministic.** Same inputs, same seed, same bytes out.
- **Fails loudly.** Invalid state aborts with a reason; invalid *artifacts* are refused with an error
  code. Never a silent fallback.
- **`bazel test //...` green on a clean checkout.** No test may need an uncommittable artifact.
- **Every claim measured.** Numbers live in `docs/measurements.md` with a reproduce command; every
  gate is mutation-tested before it is trusted.
- **Interpretability is honest.** Panels state what they do *not* support; a single-prompt number is
  never presented as a fact about the model.

---

## System design

```
   corpus ─▶ prepare ─▶ .bin ─▶ train ─▶ checkpoint ─┬─▶ generate      (sampling)
                                                     ├─▶ eval          (loss, baselines)
   HF safetensors ─▶ convert_hf ────────────────────▶┤
                                                     └─▶ inspect ─▶ run.json ─▶ viewer.html
                                                          ▲                        ▲
                                                          │                   serve_viewer.py
                                                     ablation_stats           (localhost only)
                                                     (corpus statistics)
```

### Library — two layers, enforced by the build (D10)

- **`//:cppgpt_model`** — train, inference, load. `GPT2` owns parameter and activation arenas;
  `ops.cpp` holds the kernels; `checkpoint` handles versioned, checksummed, atomic save/load.
- **`//:cppgpt_interp`** — logit lens, direct logit attribution, KL, ablation, donor capture,
  conditional co-ablation, and the arena-indexing helpers.
- The model may not reach the interpretability layer. That is a **compile error**, not a review
  comment — verified by a probe that used to build clean.
- One stated exception: `patch.hpp` sits model-side because `forward` calls it. It is an enum, a POD,
  and three functions of `memcpy` and asserts.

### Key interfaces

- `GPT2::forward(tokens, targets, logits_at, patches, n_patches)` — the only core API the
  interpretability layer needed: an optional **set** of activation substitutions, applied mid-pass.
  With none, bit-identical to the pre-seam build.
- `capture_site(...)` + `Patch` — the donor half. Together they express activation patching,
  interchange intervention, and resample ablation, which are the same operation.
- `coax_sweep(...)` — all `n²` ordered component pairs; the baseline is a parameter, not a constant.
- `run.json` (schema 6) — the single contract between `inspect` and the viewer. Versioned; an old
  viewer fails loudly rather than mislabelling numbers.

### Design decisions

- **Failure model.** Programmer error aborts (`ASSERT_MSG`, with the reason). Invalid *artifacts*
  return `Result<T, ErrorCode>` — `save_checkpoint` refuses to write non-finite parameters, because
  such a file is valid-looking, checksums correctly, and generates garbage.
- **Interventions are data, not callbacks.** A `const Patch*` consulted per layer, never a function
  pointer on the numerical path.
- **Ablation baseline is explicit.** Zero ablation is off-distribution; a donor prompt is the field's
  recommendation. Both are reported and the viewer names which one it shades — because the choice
  reorders the answer (M-19, M-21).
- **The viewer is a static file.** No framework, no build step, no network. `serve_viewer.py` is a
  localhost front door that clamps every knob.
- **Real-time vs offline** is decided by *input*, not cost: needs only this prompt → interactive;
  needs a corpus or a training loop → offline artifact.

---

## Roadmap

**Done** — M0–M2 (skeleton, full GPT-2 forward/backward, train/infer/fine-tune), M5 (observation
layer: lens, attention, residual norms, KL, viewer, live server), GPT-2 124M weights + BPE, CI.

**M6 — interpretability, in progress.** Two questions: *what does a component represent* (Q1) and
*why does it dominate the ablation sweep* (Q2).

| | item | status |
|---|---|---|
| ✅ | Intervention seam; two-layer boundary enforced by the build | merged (#40) |
| ✅ | Donor (resample) ablation, baseline named in the UI | merged (#41) |
| ✅ | Conditional co-ablation — explains M-17's 22.9× as self-repair | merged (#42) |
| ✅ | Corpus study under both baselines; corrected M-19 | PR #43 open |
| ⬜ | **A5 — QK/OV circuit panels.** Best value left: prompt-independent, zero forwards, and a 65×65 matrix that fits on screen only because this model is small | next |
| ⬜ | **A7 — component card.** Consolidates numbers that today live in four outputs | next |
| ⬜ | A3 path patching · A4 causal-tracing grid · A6 neuron views · A8–A15 viewer work | |
| ⬜ | B2 corpus attention stats · B3 max-activating examples · B4 induction probe | needs the artifact channel |
| ⬜ | B5 attribution patching + error study · B6 tuned lens · B8 transcoders | |

**Blocking several offline items:** there is no channel for a corpus artifact to reach the viewer —
no schema, no loader, no place on disk. Needs a `DECISIONS.md` entry before B2/B3.

**Open, not interpretability:**

- **Threading** — GPT-2 124M generates at 5.1 s/token on 16 idle cores. Correct and unusable; the
  binding constraint. KV cache is the algorithmic half.
- **Verification debt** — 3 constitution clauses still have no enforcing test (per-op fixtures,
  intermediate-activation parity, alloc-counter hook); training parity is proven only at `n_embd=8`;
  `dump_logits` has no test.
- **`review-codify-loop`** — overdue, and this period produced material: three checks that measured
  nothing, and a written caution that did not prevent the mistake it described.

**Needs a human decision:** whether B7 (auto-interp) may use an external LLM offline —
`docs/constitution.md` is human-frozen; and whether the donor baseline becomes the viewer's default.

---

**Where the detail lives.** `ROADMAP.md` (the work) · `IMPLEMENTATION_PLAN.md` (execution) ·
`docs/INTERPRETING.md` (how to read the tools, and what they do not support) ·
`docs/INTERPRETING_EXAMPLE.md` (a worked walkthrough, panel by panel) ·
`docs/measurements.md` (every number, with a reproduce command) · `docs/DECISIONS.md` (D1–D10) ·
`docs/engineering-lessons.md` (L1–L19) · `docs/constitution.md` (human-frozen deal-breakers).
