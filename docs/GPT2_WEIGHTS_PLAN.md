# Design proposal — byte-level BPE and real GPT-2 weights

**Status: proposal. Nothing here is implemented.** Per `CLAUDE.md`, architecturally
significant work stops at the roadmap for review.

## Why this, and what "done" means

The repo's strongest claim is *canonical GPT-2*, and today it is verified against a
**fixture** — small random weights at toy scale, with a character-level tokenizer.
The math is well evidenced (parity 1.43e-06, mutation-tested, ASan-clean). The
*system* claim is not: real GPT-2 weights have never been loaded, and no BPE
tokenizer exists.

**Done** = `generate --checkpoint gpt2-124M.ckpt --prompt "…"` produces the same
tokens HuggingFace produces, from HF's own published weights, with the agreement
measured rather than eyeballed.

This is the single highest-leverage item for confidence in the repo, and it is
also where the subtle bugs live.

---

## The eight decisions

### D-A. BPE parsing: C++ or Python?

GPT-2's tokenizer ships as `vocab.json` + `merges.txt` (HF's names; OpenAI's original
release called them `encoder.json` + `vocab.bpe`), or as the single nested `tokenizer.json`.
All are JSON; weights ship as pickle or safetensors.

| option | cost |
|---|---|
| Parse JSON in C++ | A JSON parser is a dependency (forbidden) or ~600 lines of hand-rolled parsing **over untrusted input** — a real attack surface for a tool that already sits behind a public Funnel |
| Parse pickle in C++ | Pickle is an arbitrary-code-execution format. Not negotiable. |
| **Python converts → flat binary; C++ reads binary** | One-time cost, auditable, no runtime deps |

**Recommend the third.** It is the pattern `scripts/gen_fixtures.py` already uses,
and it keeps the libc/libm invariant that justified making W&B a sidecar.

The C++ side then reads a format it fully controls: a header, a merge-rank table,
and a vocab blob — the same shape as the existing `.vocab`/`.bin` split.

### D-B. The pre-tokenizer — the highest-risk component

GPT-2 splits text *before* BPE with:

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

`\p{L}` and `\p{N}` are Unicode property classes. **`std::regex` does not support
them**, and it operates on the original UTF-8 text, not on the byte-mapped
alphabet — so this is not reducible to a 256-entry table.

| option | verdict |
|---|---|
| `std::regex` | Cannot express `\p{L}`/`\p{N}`. Also notoriously slow. |
| Vendor a regex engine | New dependency. Rejected. |
| **Hand-rolled splitter + generated Unicode tables** | Mechanical, testable, no deps |

**Recommend the third, in two slices:** first an **ASCII-only** splitter that
*fails fast* on non-ASCII input (correct for most English, honest about its
limit), then a generated `L`/`N` property table emitted by the same Python
converter that builds the vocab.

This is where I expect bugs. The gate in D-G is aimed squarely at it.

### D-C. Weight conversion and the transpose that always bites

HF GPT-2 uses `Conv1D`, which stores weights **`[in, out]`** — the transpose of
`nn.Linear`'s `[out, in]`, which is what `matmul_forward` expects. Every
`c_attn`, `c_proj`, `c_fc` needs transposing on conversion.

This is the classic failure: it produces a model that runs, emits plausible-looking
logits, and is wrong. Only a numerical comparison against HF catches it.

**Recommend:** the converter emits tensors in the repo's `.bin` order, and it
derives that order **from `include/cppgpt/tensors.hpp`** rather than restating it.
`docs/engineering-lessons.md` already has a lesson about the layout correspondence
being re-derived in five places; this must not become the sixth.

### D-D. Two tokenizers — no base class

`CharTokenizer` stays (the toy path works and is cheap). BPE is a second concrete
type. The choice is made **once**, at startup, from the checkpoint header.

**Recommend: no virtual interface.** `CLAUDE.md` forbids speculative abstraction,
and a vtable for a decision made once per process is exactly that. Each tool
branches at construction. If a third tokenizer ever appears, revisit — with two,
an interface costs indirection and buys nothing.

### D-E. Checkpoint v3 — vocabulary identity

Today `inspect`/`generate` check `tok.vocab_size() == cfg.vocab_size`. For BPE
that is **vacuous**: `50257 == 50257` says nothing about *which* 50257.

**Recommend:** spend the existing spare header fields — no header growth:

| field | use |
|---|---|
| `reserved0` (u32) | `tokenizer_kind` (0 = char, 1 = gpt2-bpe) |
| `reserved1` (u64) | FNV-1a-64 over the tokenizer file |

Version bumps to 3. A mismatched vocab then fails loudly instead of silently
decoding garbage.

### D-F. `V = 50257` makes the classifier a performance *requirement*

At `T=1024`, the tied head is `[1024,768] × [768,50257]`:

| | |
|---|---|
| full-sequence lm_head | **79.0 GFLOP → ~1.6 s** per forward at 49 GFLOP/s |
| last position only | **77.2 MFLOP → ~1.6 ms** |
| `logits` `[1,1024,V]` | **206 MB** (and `probs` another 206 MB) |

Generation needs only the last row. **A last-position-only classifier path is not
an optimisation here; without it generation is unusable** — 1.6 s per emitted
token, and 412 MB of logits/probs that are then discarded.

### D-G. Memory, and the guard we just added

`docs/measurements.md` M-4: 124M inference-only is **2.61 GiB**, training-shaped
**5.32 GiB** at `B=1,T=1024`.

The activation arena at `B=1,T=1024` is ~1.43 G floats against the `INT_MAX` cap
of 2.147 G added on 2026-08-18. It fits — but **`B=2` does not**, and the guard
will (correctly) refuse it.

**Recommend:** accept `B=1` for the first slice, and treat the *inference arena*
(rolling per-layer buffers, no `preatt`, no `probs`, no activation gradients —
M-4 estimates ≈0.7 GiB) as a follow-on, not a prerequisite.

### D-H. Verification — the reason to do any of this

Four gates, in dependency order. Each is binary and none can pass vacuously.

1. **Tokenizer differential.** Encode ≥10 MB of text with HF's tokenizer and with
   ours; require **byte-identical token streams**. Not a similarity score — an
   equality. Plus a targeted corpus for the cases that break BPE implementations:
   leading spaces, contractions (`'s`, `'ll`), runs of whitespace, digits,
   non-ASCII, and the empty string.
2. **Round-trip.** `decode(encode(x)) == x` for every test input.
3. **Forward parity at 124M.** Our logits vs HF's on the same tokens, ≤1e-4.
4. **Greedy decode.** N tokens identical to HF. The prompt is chosen by *measured
   margin*, not taste — M-6 already did this analysis: worst top1−top2 margin over
   250 steps is **0.00716**, 5th percentile **0.106**. Against a 1e-4 logit error
   that is 70× headroom at the worst step, which is why the gate prompt must come
   from that table.

---

## Staged plan — vertical slices, each independently verifiable

| slice | delivers | gate |
|---|---|---|
| **S1** | Python converter → binary tokenizer format; C++ reader; merge application; **ASCII-only** pre-tokenizer | differential test on an ASCII corpus is byte-identical |
| **S2** | Generated Unicode `L`/`N` tables; full pre-tokenizer | differential test on the full corpus, incl. non-ASCII |
| **S3** | HF → checkpoint v3 converter (order derived from `tensors.hpp`); `tokenizer_kind` + vocab hash | shape/name mapping checked against the tensor table by name |
| **S4** | Load 124M and run a forward | logits match HF ≤1e-4 |
| **S5** | Last-position classifier path | greedy decode matches HF for N tokens; generation latency measured |

S1–S2 and S3 are independent and can proceed in either order; S4 needs both.

## Risks, ranked

1. **The pre-tokenizer.** Subtle, high blast radius, and a near-miss still
   produces fluent-looking output. Mitigated only by gate 1 being an *equality*.
2. **The Conv1D transpose.** Produces a model that runs and is wrong. Caught only
   by gate 3.
3. **Scope creep into an inference arena.** D-G says defer it; `B=1` is enough to
   prove the claim.
4. **fp32 only.** HF publishes 124M in fp32, so this is not blocking — but it caps
   us at the smallest model. 355M+ would want bf16 and is out of scope.

## What I need decided before starting

- **Is the Python conversion step acceptable** as a setup-time dependency? It is
  the crux of D-A, and it is already true for fixtures.
- **ASCII-only first (S1) or hold for full Unicode (S2)?** S1 is genuinely useful
  and much smaller; it just needs its limitation stated in the tool's output.
- **Which model?** 124M is the only one that fits the fp32/`B=1` envelope cleanly.


---

## Addendum — verified against the real artifacts (2026-08-19)

Fetched from `openai-community/gpt2` rather than recalled. Three corrections and
one new requirement.

**Filenames.** HF uses `vocab.json` and `merges.txt`. `encoder.json` and
`vocab.bpe` are OpenAI's original names and appear nowhere in the repo.

**The safetensors header is exactly as D-A assumed, confirmed by reading it:**
8-byte little-endian length (14,283 bytes here), then flat JSON, depth 2, with
precisely three fields per tensor — `dtype`, `shape`, `data_offsets` — and every
tensor `F32`. No nested metadata. The ~150-line C++ estimate holds.

**The Conv1D transpose is confirmed, not theoretical.** `h.0.attn.c_attn.weight`
has shape **`[768, 2304]`** — that is `[in, out]`. `matmul_forward` requires
`[OC, C]` = `[2304, 768]`. Every `c_attn`, `c_proj` and `c_fc` must be transposed
on conversion.

**New requirement the plan missed.** The file holds **160** tensors; we have
**148** parameters (12 per layer × 12, plus `wte`, `wpe`, `lnfw`, `lnfb`). The
difference is exactly 12: one `h.{i}.attn.bias` per layer, shape
`[1, 1, 1024, 1024]`. Those are **causal masks, not parameters** — our attention
masks in code. The converter must skip them, and the 148/160 reconciliation is
the cheapest possible check that it mapped everything else.

**`config.json` agrees with our implementation on every field that matters:**
`n_layer 12, n_head 12, n_embd 768, n_positions 1024, vocab_size 50257,
activation_function "gelu_new", layer_norm_epsilon 1e-05`. It should be *read and
validated* against the checkpoint header rather than assumed.

**Files actually needed: three.** `model.safetensors`, `vocab.json`, `merges.txt`
— plus `config.json` to validate. Everything else in the repo is another
framework's export (`tf_model.h5`, `flax_model.msgpack`, `rust_model.ot`,
`*.tflite`, `onnx/`) or redundant (`tokenizer.json` restates vocab+merges in a
nested form).
