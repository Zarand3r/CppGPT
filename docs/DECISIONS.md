# Decisions

Locked decisions, newest last. Each records **what was decided, the evidence, and what was given
up** — so a future reader can tell whether the reasoning still holds, and reopen it if the premises
change. A decision without evidence is a preference; a decision without a stated cost is a wish.

`ROADMAP.md` owns *what to do*. This file owns *why it is that way and not the obvious alternative*.

---

## D1 — Release builds use `-march=x86-64-v3 -ffp-contract=off`, replacing `-march=native`

**Date:** 2026-08-04 · **Status:** adopted · **Supersedes:** the original `-O3 -march=native`

### Decision
`--config=release` now compiles with `-O3 -march=x86-64-v3 -ffp-contract=off`.

### Evidence
Measured on the real kernel via `//tools:bench` (mlp_fc shape `BT=1024, C=768, OC=3072`, best-of-12,
hermetic clang 20.1.8, Ryzen 9 9950X3D). Full table in `docs/measurements.md` M-1.

| flags | GFLOP/s |
|---|---|
| `-march=native` **(old default)** | 2.87 |
| `-march=x86-64-v3` (AVX2+FMA) | 2.95 |
| `-march=x86-64-v2` (SSE4.2) | 2.95 |
| `-march=x86-64` (SSE2) | 2.93 |
| **`-march=native` + `-ffp-contract=off`** | **5.79** |
| **`-march=x86-64-v3` + `-ffp-contract=off`** | **5.82** ← adopted |
| `-march=x86-64-v2` + `-ffp-contract=off` | 5.63 |

Two independent findings:

1. **The bottleneck was FMA contraction, not the ISA.** Every `-march` level sits at ~2.9 with
   contraction on, and jumps to ~5.8 with it off. Clang's default `-ffp-contract=fast` fuses the
   inner `acc += inp[c] * w_oc[c]` into a single `vfmadd`, which serializes the reduction on FMA
   latency. With contraction off the multiply and the add issue separately and overlap better.
   *(Mechanism inferred from the measurement, not yet confirmed by disassembly — see cost below.)*
2. **`-march=native` was buying nothing.** `x86-64-v3` matches it (5.82 vs 5.79). So the
   machine-dependent flag can go at zero cost.

End-to-end: `//tools:train`, 40 steps, 2.10 s → 1.62 s wall.

### Why not the alternatives
- **Keep `-march=native`.** It makes every release binary depend on the build host's CPU, which
  contradicts the repo's hermetic-toolchain premise (`README.md`: Bazel, clang and Python are all
  pinned precisely so builds are reproducible) — and it measures no faster. It also makes any
  future token-exact gate host-dependent, since a different vector ISA rounds differently.
- **`-ffast-math` / `-funsafe-math-optimizations`.** Would permit reassociating the reduction and
  likely beat both. Rejected: it also enables finite-math-only and flush-to-zero, silently breaking
  the `std::isfinite` guards that `docs/engineering-lessons.md` L2 exists to enforce. Fast math is a
  global semantic change bought for a local speedup.
- **`x86-64-v2` instead of v3.** 3% slower, and buys portability back to ~2009 CPUs. Rejected as
  irrelevant for a project whose stated target is a modern x86 core; revisit if that changes.
- **Leave it alone and fix only the code.** The blocked/multi-accumulator rewrite is still the right
  next step and is still on the roadmap — but it is a day of work, and this was two lines for ~2×.

### What this costs
- **Numerics change.** Removing FMA means each multiply and add rounds separately, so results shift
  very slightly. The canonical-GPT-2 parity gate still passes with ~50× margin: forward logit error
  moves from 9.54e-07 to **1.91e-06**, against the **1e-5** tolerance the gate actually enforces —
  a **5.2x** margin, not the ~50x an earlier version of this entry claimed against a 1e-4 figure that
  was never the enforced value. A decision justified by a safety factor ten times larger than the real
  one is worse than no justification, because it is the number the next person reuses (gradients 2.98e-07, loss 9.54e-07).
  Note the direction — contraction *off* is marginally **less** accurate, because FMA rounds once
  instead of twice. Both are far inside the gate.
  *Upside:* results no longer depend on whether the target ISA happens to offer FMA, which makes the
  determinism invariant easier to hold across machines.
- **Binaries now require AVX2** (Haswell 2013+ / Zen 1+). Previously `native` implied the build host
  anyway, so this is strictly more portable than what it replaces.
- **The mechanism is inferred, not proven.** The measurement is reproducible and controlled, but the
  claim "FMA latency serializes the chain" has not been confirmed by reading the generated assembly.
  If the blocked-matmul work behaves unexpectedly, disassemble before theorising.

### Revisit when
The matmul is rewritten with multiple accumulators. Once the reduction is no longer a single serial
chain, FMA contraction should become a *win* rather than a loss, and `-ffp-contract=fast` may want to
come back. Re-measure both ways at that point — do not assume this decision still holds.

---

## D2 — `prepare --vocab` fails hard on an unknown byte; the vocabulary never grows

**Date:** 2026-08-04 · **Status:** adopted

### Decision
`prepare --vocab <existing.vocab>` re-tokenizes a new corpus with a previous run's vocabulary. If the
corpus contains any byte absent from that vocabulary, `prepare` reports **every** offending byte with
its count and exits 1. It never drops, remaps, or appends to the vocabulary.

### Why this exists at all
It is the piece the whole fine-tuning story rests on, and it is not obvious. A second corpus yields a
different character set → a different `vocab_size` → a different `wte` shape → `ShapeMismatch` from
`load_checkpoint`. Without vocabulary reuse, fine-tuning is not awkward, it is **impossible**.

### Why not the alternatives
- **Grow the vocabulary with the new bytes.** Changes `vocab_size`, so the base checkpoint no longer
  loads — it defeats the purpose. (Growing *and* resizing `wte` in place is a real feature, but it is
  a model-surgery feature, not a tokenizer one, and nothing in the MVP needs it.)
- **Map unknown bytes to a sentinel / drop them.** Silent data corruption: the model would train on
  text that differs from the file on disk, and nothing downstream could detect it. Directly against
  the fail-fast doctrine, and the exact shape of `docs/engineering-lessons.md` L2/L5.
- **Abort on the first bad byte** (what `CharTokenizer::encode`'s `ASSERT` would do). Correct but
  unhelpful: the user cannot tell one stray character from a fundamentally wrong alphabet without
  re-running repeatedly. Pre-scanning and reporting all of them costs one pass over the corpus.

### What this costs
The user must clean the corpus, or accept training from scratch with a fresh vocabulary. That is the
honest trade — the alternative is a model quietly trained on the wrong bytes.

### Note
`prepare` now takes an output **stem** rather than a filename, emitting `<stem>.train.bin`,
`<stem>.val.bin` and `<stem>.vocab`. `train` finds the sidecar by stripping the data suffix, and
accepts an explicit `--vocab` override.

---

## D3 — `--init-from` loads weights only; `--ckpt` resume restores the optimizer

**Date:** 2026-08-04 · **Status:** adopted

### Decision
`GPT2::load_checkpoint` takes a `LoadMode`. `Full` (default) restores weights, AdamW moments and the
step counter. `WeightsOnly` restores weights and **resets** moments and step to zero. `train
--init-from` uses `WeightsOnly`; `train --ckpt <existing>` uses `Full`. They are mutually exclusive:
`--init-from` starts a new run, so it does not also resume from `--ckpt`.

### Why two modes rather than one
They answer different questions.
- **Resume** continues an interrupted run: the moments and the step index belong to *that* run's
  schedule, and dropping them would restart bias correction at `t=1`, where the correction factor is
  `1/(1-β₁) = 10×` — a large, arbitrary kick. This is the failure mode already recorded as a real
  incident in `docs/engineering-lessons.md` L1.
- **Fine-tune** starts a *new* run, usually at a lower LR on different data. The base run's moments
  are second-moment estimates of gradients on the *old* corpus, scaled for the *old* learning rate.
  Carrying them in applies stale, differently-scaled updates on the first steps — exactly the same
  kick, just harder to notice because the loss is expected to move anyway.

### Why not the alternatives
- **One mode, always full.** Fine-tuning inherits the base schedule's optimizer state; the first
  steps are governed by the old run's LR, not the one the user passed.
- **One mode, always weights-only.** Breaks resume — the case L1 was written about.
- **Strip the moments from the file instead** (save a weights-only checkpoint for fine-tuning). Makes
  the *producer* decide how the consumer will use it, and a base checkpoint is legitimately both a
  thing to resume and a thing to fine-tune from.

### What this costs
One more concept in the API, and a caller who picks the wrong mode gets a subtly worse first few
steps rather than an error. Mitigated by making `Full` the default (resume is the more common and
more dangerous-to-get-wrong case) and by logging which one happened.

### Evidence
`checkpoint_test` verifies both: a `Full` load reproduces the base trajectory step-for-step, while
after a `WeightsOnly` load an AdamW step with **zero gradients** leaves the weights bit-identical —
the same probe that caught the original stale-moment bug.
End to end: fine-tuning a base checkpoint onto a different corpus moved val loss on that corpus from
**2.3775 → 0.5398** in 150 steps.

---

## D4 — Interpretability ships as a dump + static viewer, not a server or a WASM build

**Date:** 2026-08-11 · **Status:** adopted

### Decision
`tools/inspect` runs one forward and writes a schema-versioned JSON file. A single self-contained
`viewer.html` renders it, opened from `file://`, making no external requests. No server, no new
toolchain.

### Why not the alternatives
- **Hand-rolled local HTTP server** (~150 lines of std-only sockets) would give live prompt editing
  without a refresh. Rejected: it adds a socket-accept and request-parsing surface to a codebase that
  currently has neither, for what is a convenience rather than a capability. Parsing untrusted bytes
  off a socket is the highest-risk code in most programs, and this repo's whole failure model is
  built around trusted local files. If the refresh loop becomes genuinely painful, this is the first
  thing to revisit — the JSON schema is already the wire format it would serve.
- **Emscripten / WASM**, as Transformer Explainer does. Genuinely the nicest end state: one
  implementation of the model, fully interactive, shareable as a static page. Rejected *for now*
  because it introduces a second toolchain into a repo whose defining property is a single hermetic
  pinned one (`MODULE.bazel` pins clang, Python and every ruleset precisely so builds are
  reproducible). The cost is not writing the code, it is owning that toolchain forever.
- **Python/Gradio front end.** Fastest to build, and rejected hardest: it would either reimplement
  the model in Python — creating exactly the second implementation that `docs/engineering-lessons.md`
  and the one-tokenizer rule exist to prevent — or just read our dumps, in which case it is option A
  with a dependency stack bolted on.

### What this costs
Changing the prompt means re-running `inspect` and refreshing the page, rather than typing live.
That is the whole cost, and it buys zero new dependencies, zero new toolchains, and a viewer that
still opens in five years.

### How we will know it was right
- The JSON schema survives unchanged if a server or WASM build is added later (it is the same wire
  format). If the first attempt at either forces a schema redesign, this decision was wrong about
  reusability.
- Dump size stays inside what a browser opens comfortably at the scales we actually use. The plan
  already records the failure point: 1.2 GB at GPT-2 scale, which is why selection flags exist from
  the first commit rather than being retrofitted.

---

## D5 — A dev-only Python front door for interactive prompting, revising D4

**Date:** 2026-08-11 · **Status:** adopted · **Revises:** D4

### What changed
D4 chose "dump + static viewer" and deferred interactivity, reasoning that live prompting was a
convenience rather than a capability. Using it proved that wrong: a viewer you can only feed by
re-running a CLI and re-picking a file is a *file inspector*, not a way to explore a model. The
question "what does it do if I phrase it this way instead?" is the entire point, and D4's shape
made that question expensive to ask.

`tools/serve_viewer.py` now serves the viewer and exposes `POST /api/inspect`, which runs
`//tools:inspect` on a submitted prompt and returns the dump.

### Why Python, when D4 rejected a Python front end
D4 rejected Python because it would either duplicate the model (the second-implementation hazard the
one-tokenizer rule exists to prevent) or be option A with a dependency stack. This is neither: it
**shells out to the same C++ binary**, so there remains exactly one implementation of the model, and
it is dev tooling — the shipped binaries still link only libc/libm, leaving the no-runtime-dependency
invariant untouched. What actually changed is that a hand-rolled C++ HTTP server is no longer the
only way to get interactivity, and it was the socket-and-parser surface that D4 was avoiding.

### What this costs, stated plainly
This endpoint is exposed to the public internet via Tailscale Funnel, so it is the first untrusted
input this project has ever accepted. Mitigations, each verified by running an attack rather than
asserted:
- **No shell.** `subprocess.run` with a list argument; the prompt is one argv element. Verified:
  `a; touch PWNED3 &` was tokenized as 17 characters of text and created no file.
- **Vocabulary validation before exec.** Out-of-vocab bytes are rejected with an explanation. This is
  not cosmetic: `CharTokenizer::encode`'s contract is to *abort* on an unknown byte, so without this
  every prompt containing an emoji would crash the process. Verified with `hello 🚀`.
- Prompt length capped, concurrency bounded to 2, each run wall-clock bounded, output path chosen by
  the server, and no client control over layer selection or size ceilings.

### How we will know it was right
- If the endpoint is still trivially safe once someone asks for generation (not just inspection),
  the boundary was drawn in the right place. Generation takes an `n` and a temperature, which are
  *numeric* knobs — if adding them requires new validation categories, revisit.
- If the C++ side ever needs to change to support the front end, this decision failed: the whole
  point is that the front door is replaceable and the model is not.

### Note
D4's static path still works unchanged — `viewer.html` opens from `file://` and takes a dump through
the file picker with no server at all. The server is additive, not a replacement, which is why the
viewer remains a single file rather than forking into hosted and local variants.

---

## D6 — A constexpr tensor table replaces the hand-maintained layout correspondence

**Date:** 2026-08-11 · **Status:** adopted (steps 1–2, 5–6 of 7)

### The problem
`src/model.cpp` maintained a **five-way correspondence entirely by hand with no compile-time
enforcement**: `ParamTensors` field order == `param_sizes` index == `point_params` index == `kDecay`
index == the `.bin` order that `scripts/gen_fixtures.py` reproduces as a *separate Python list*.

That is not merely verbose. **Six of the sixteen parameter tensors have the identical size `L*C`**
(`ln1w, ln1b, attprojb, ln2w, ln2b, fcprojb`), and two more share `C`. Transposing any two of them
changes no size, no offset, and no allocation — only meaning. The sole guard was one whole-model
numeric comparison at one tiny config, which reports "wrong" without saying where. And the layer
stride was re-derived in three places outside `model.cpp`, one of them (`tools/inspect.cpp`) with the
`B` factor missing — correct only because that tool happens to build with `B == 1`.

### What was adopted
`include/cppgpt/tensors.hpp`: one `constexpr` row per tensor carrying its **name**, per-layer element
count, `[L, ...]` stacking flag, AdamW decay flag, and init scheme. `param_sizes`, `init_weights` and
the decay mask are now loops over that table; the `kDecay` array and the hand-written size and init
lists are gone. `Config` moved to `model_config.hpp` so the table need not depend on the model class.

### What it explicitly is NOT
Not a `Tensor` class. No runtime shapes, strides, dtypes, or dispatch. **`src/ops.cpp` is not touched**,
the op signatures keep raw `float*` + `int` dims, and the 10-argument shapes that mirror llm.c for
parity are unchanged. This was the main risk and it was avoided deliberately.

### Evidence it is layout-preserving
Migration step 1 was a pure equivalence proof *before* any deletion: `tensors_test` asserts
table-derived per-tensor sizes **and running offsets** equal the hand-written values across 5 configs
— the unit baby, the parity fixture, the Shakespeare toy, GPT-2 124M, and `L == 0` — plus the external
anchor `params_total(124M) == 124,439,808`. Then the full suite (29/29, including the M1 PyTorch parity
gate, which is a byte-order gate on the arena) and an end-to-end `inspect` diff: logit-lens top-1
sequence, residual norms and final distribution all identical.

### Cost
| | before | after |
|---|---|---|
| edit sites to add a parameter tensor | 8 | **1 table row + 1 name** |
| `matmul_forward` (mlp_fc) | 5.82 GFLOP/s | 5.68 (within run-to-run noise; `ops.cpp` unmodified) |
| memory profile | — | byte-identical (same offsets, same arenas) |

**Update (second pass).** The activation table now exists too, so *both* arenas derive their sizes
from `tensors.hpp`; the dead `kDecay` array is removed; and `layer_slice()` replaces the three
independent re-derivations of the `[L,B,T,C]` stride that produced the `inspect` B-factor bug.

Still outstanding, and now the honest remainder: the **68 hand-written layer-stride expressions inside
`forward`/`backward` are unchanged**, so `model.cpp` is 432 lines rather than the ~375 the proposal
projected. Those sites are locally correct and covered by the parity gate; collapsing them is a large
mechanical edit whose benefit is readability rather than correctness, and it is the one part of this
refactor that touches the numerical path. Tracked in `ROADMAP.md`.

Remaining migration steps (3, 4, 7 — `Arena::bind`, `at(tensor, layer)` for the forward/backward block
bodies, retiring `ParamTensors`) are deliberately **not** taken yet: they rewrite the code the parity
gate protects, and that gate only became trustworthy in the preceding commit (it had been passing on
all-NaN gradients). Sequencing them after a period of the tightened gate being green is the point.

### How we will know it was right
- Adding the next tensor touches one row. If it touches more, the table is not actually the source of
  truth and this is half-done.
- `gen_fixtures.py`'s ordering should become checkable **by name** against the table rather than by
  both authors having counted to 16 correctly. Until that check exists, the fifth correspondence is
  still manual.
- If `ops.cpp` ever has to change to accommodate the table, the boundary was drawn wrong.

---

## D7 — `Device` deleted rather than kept as a GPU seam

**Date:** 2026-08-13 · **Status:** adopted · **Supersedes:** `PLAN.md`'s "device seam" premise

### Decision
`enum class Device { CPU }` and its trailing parameter on all 17 op signatures are removed, along
with the 17 `ASSERT(dev == Device::CPU)` entry checks and `Storage`'s `dev_` tag.

### Evidence
Measured before removal: **17 signatures carried it, 17 implementations asserted it, and zero call
sites anywhere passed it.** The only non-default reference in the tree was a test asserting a
one-value enum equalled its one value. `ops.hpp`'s header additionally claimed each op "dispatches on
it" — there was no dispatch; every entry point asserted, then unconditionally called the one `_cpu`
implementation. That is an L1 violation (a comment describing behaviour the code does not have) that
survived from M0.

### Why not keep it for the CUDA phase
`PLAN.md` justified it as "the one forward-looking abstraction we pay for now… cheap, and prevents a
future rewrite." Two problems. It is not cheap — it is a defaulted parameter on every op, the only
one in the API, and defaults that hide meaning are exactly what the repo's own rules warn about. And
it is the **wrong granularity**: a GPU port moves the arenas and the whole step, not a branch inside
`residual_forward`. When a second backend is real, the seam belongs at the arena that owns the
memory. Deleting now and re-adding there costs less than carrying 17 signatures until then.

This is the doctrine applied to itself: `CLAUDE.md` forbids speculative abstraction, and this was one,
protected by having been written early.

### What this costs
If a CUDA backend arrives, the seam must be reintroduced — at the arena, deliberately, informed by
what the kernels actually need. That is the intended outcome, not a regression.

### How we will know it was right
If a GPU port is ever attempted and its first move is to add a per-op device branch, this was wrong.
If its first move is to swap the arena's allocator and the step driver — which is what every
CPU→GPU port of this shape does — it was right.
