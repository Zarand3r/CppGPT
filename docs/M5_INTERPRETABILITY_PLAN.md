# M5 — Interpretability: see inside the model

**Status:** active. `ROADMAP.md` owns the checkboxes; this document holds the execution detail.
IDs are `M5-*` (the shelved GPT-2-scale plan owns `M3-*`; numbering skips to 5 so nothing collides).

---

## 0. Goal

> **Type a prompt, watch it travel through the transformer, and see the prediction form.**

Not a demo — a tool that answers a question you cannot otherwise answer about your own model:
*where* does the prediction get made, and *what* is each layer contributing?

## 1. Why this repo is an unusually good substrate

In PyTorch, capturing intermediates needs forward hooks. Here **every activation is already
retained**: `GPT2::acts()` exposes `encoded`, `ln1`, `qkv`, `atty`, `preatt`, `att`, `attproj`,
`residual2`, `ln2`, `fch`, `fch_gelu`, `fcproj`, `residual3`, `lnf`, `logits`, `probs` — the arena
keeps them all because backward needs them. Nothing to instrument, nothing to bolt on.

In particular `att` is `[L, B, NH, T, T]`: every head's attention matrix, already materialised.

## 2. What to show, in order of value

Ranked by *insight per unit of work*, not by visual appeal.

1. **Logit lens** — project the residual stream at layer *l* through the final layernorm and the
   tied unembedding: "what would the model predict if it stopped here?" You watch the prediction
   sharpen layer by layer. This is the feature that actually answers the goal question, and it is
   ~15 lines because every primitive exists (`layernorm_forward`, `matmul_forward`, weight tying).
2. **Residual-stream norms** — ‖residual‖ per layer per position. Shows which layers move the
   representation at all, and exposes the outlier-channel behaviour GPT-2 is known for.
3. **Attention maps** — per layer, per head, `[T, T]`. Visually the most striking, but the
   literature is emphatic that attention is *routing, not explanation* (Jain & Wallace 2019;
   Wiegreffe & Pinter 2019). The UI must say so rather than implying causation.
4. **Top-k at the final position** — the ordinary output distribution, for grounding.

**Deliberately out of scope for M5:** activation patching, ablation, neuron max-activating
examples, attribution. Those are *causal* analyses; this milestone builds the *observation* layer
they would sit on. Adding them before the viewer works would be building the second floor first.

## 3. Prior art, and what each contributes

| Work | Contribution | What we take |
|---|---|---|
| [bbycroft.net/llm](https://bbycroft.net/llm) | 3D walkthrough of every matrix op in GPT inference | The layout metaphor: tensors as objects you scrub through |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | Live GPT-2 in-browser, prompt editing, animated flow | Proof the interactive version is feasible; the panel decomposition |
| BertViz (Vig 2019) | head / model / neuron attention views | The model view — all heads as a grid — is the density sweet spot |
| Logit lens (nostalgebraist 2020) | residual stream → unembedding at each layer | The core analysis, adopted directly |
| Tuned lens (Belrose et al. 2023) | learned affine probe per layer, fixes logit-lens bias | Noted as the honest caveat; not implemented |
| TransformerLens (Nanda) | hooks, patching, ablation | The causal analyses M5 deliberately defers |
| CircuitsVis | React components for attention | Component idioms |

## 4. Architecture

Three candidate shapes were considered; the decision and its costs are **`docs/DECISIONS.md` D4**.

**Adopted: dump → self-contained viewer.** `tools/inspect` runs one forward and writes a versioned
JSON file; a single HTML file with no external requests renders it. Zero runtime dependencies, zero
new toolchain, works from `file://`, and the schema is exactly what a server or a WASM build would
serve later — so it is not throwaway work.

Rejected for now: a hand-rolled HTTP server (adds a socket and parsing surface to a codebase with
none, for a refresh-loop convenience), and Emscripten/WASM (a second toolchain in a repo whose
defining property is one hermetic pinned toolchain).

### Size is a first-class constraint

Attention is `O(L · NH · T²)`. At the toy scale that is small; at GPT-2 scale it is not:

| config | attention floats | as JSON |
|---|---|---|
| L4 NH4 T64 (toy) | 65,536 | ~0.5 MB |
| L12 NH12 T1024 (GPT-2) | 151,000,000 | ~1.2 GB |

So selection is **not** a later refinement — `--layers`, `--heads` and a position cap exist from the
first commit, and the writer refuses to emit a dump beyond a size ceiling rather than producing a
file no browser can open.

## 5. Slices

### M5-S1 — Logit lens in the library
`logit_lens(const GPT2&, int layer, float* out_logits)` — final layernorm applied to
`residual3[layer]`, then the tied `wte` classifier.

**Gate (non-circular):** at `layer == n_layer-1` the lens *is* the model's own final computation, so
its output must equal `acts().logits` **bit-identically**. A wrong layernorm, a wrong stride, or a
transposed unembedding all fail this. Plus: lens distributions are valid (sum to 1 after softmax),
and top-1 agreement with the final layer rises monotonically-ish with depth on a trained model.

### M5-S2 — `tools/inspect`
`inspect --checkpoint X --vocab V --prompt "..." [--layers a,b] [--heads a,b] [--top-k K] --out run.json`

Emits schema-versioned JSON: prompt, token ids + decoded strings, per-layer attention for the
selected heads, per-layer per-position residual norms, per-layer logit-lens top-k, final top-k.

**Gate:** output parses as JSON; every attention row sums to 1 ± 1e-5; the last layer's lens entry
matches the model's actual top-1; a dump exceeding the size ceiling is refused with a clear message
naming the flags that would shrink it.

### M5-S3 — The viewer
One self-contained `viewer.html`: token strip, attention grid (layer × head), the logit-lens ladder
showing the prediction sharpening, residual-norm chart. No external requests — the CSP-safe,
`file://`-openable constraint is the same one the repo applies everywhere else.

**Gate:** opens from `file://` with no network; renders the committed sample dump; the attention
caveat is visible in the UI, not buried in a README.

### M5-S4 — Wire into the example
`examples/shakespeare` gains an inspect step against the checkpoint the example already trains.

## 6. Success criteria

The milestone is done when, using only the viewer, you can answer:

- At which layer does the model commit to its top-1 prediction?
- Which layers change the residual stream most, and which are near-identity?
- Does any head show an interpretable pattern (previous-token, delimiter-attending)?

The first two are mechanical. The third may simply be *no* on a 4-layer model trained 900 steps —
and recording that honestly is a result, not a failure of the tool.
