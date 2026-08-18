# Measurements

**The single owner of every measured number in this repo.** No other document may state a
measurement; they link here. This exists because the `≈3.0 GFLOP/s` matmul figure was copied into
five sites across four documents, cited a tool that was not on the branch, and was about to be
invalidated by the very work it motivates.

Every row records **the command that reproduces it**. A number without a working reproduce command
is not a measurement, it is a rumour.

**Host (all rows unless stated):** AMD Ryzen 9 9950X3D — 16 cores / 32 threads, boost ~5.66 GHz,
L1d 48 KiB/core, L2 1 MiB/core, **L3 192 MiB (X3D V-cache)**, 120 GB RAM.
**Toolchain:** hermetic LLVM/Clang 20.1.8 (pinned in `MODULE.bazel`), libc++ static.

---

## M-1 · Matmul forward throughput (single-thread)

The dominant cost of both training and inference, and the number M2's perf gate is written against.
Kernel: `matmul_forward_cpu`, `src/ops.cpp`. Shape `BT=1024, C=768, OC=3072` (GPT-2 124M mlp_fc),
best-of-8 after warmup.

| Build flags | GFLOP/s | Date |
|---|---|---|
| `-O3 -march=native` — *the old `--config=release`* | 2.87 | 2026-08-04 |
| `-O3 -march=x86-64-v3` (AVX2+FMA) | 2.95 | 2026-08-04 |
| `-O3 -march=x86-64-v2` (SSE4.2) | 2.95 | 2026-08-04 |
| `-O3 -march=x86-64` (SSE2) | 2.93 | 2026-08-04 |
| `-O3 -march=native -ffp-contract=off` | 5.79 | 2026-08-04 |
| **`-O3 -march=x86-64-v3 -ffp-contract=off`** — **the current `--config=release`** | **5.82** | 2026-08-04 |
| `-O3 -march=x86-64-v2 -ffp-contract=off` | 5.63 | 2026-08-04 |
| **the same, + 8 independent accumulator lanes** (D8) | **49.38** | 2026-08-14 |

> **Resolved — see `docs/DECISIONS.md` D1.** The bottleneck was **FMA contraction, not the ISA**:
> every `-march` level sits at ~2.9 with contraction on and jumps to ~5.8 with it off, because
> clang's default `-ffp-contract=fast` fuses the inner `acc += a*b` into one `vfmadd` and serializes
> the reduction on FMA latency. `-march=native` was buying nothing (`x86-64-v3` matches it), so it
> was replaced with an explicit baseline for reproducible builds. Net ~2x, and the canonical-GPT-2
> parity gate still holds at ~50x margin (forward logit error 9.54e-07 -> 1.91e-06 vs a 1e-4 gate).
> End-to-end `//tools:train` 40 steps: 2.10 s -> 1.62 s.
>
> The serial-dependency diagnosis stands and the blocked/multi-accumulator rewrite is still the real
> fix; this was two lines for half of it. **True baseline is now 5.82**, so the gap to a 30 GFLOP/s
> target is ~5x.

> **Gate met — see `docs/DECISIONS.md` D8.** The multi-accumulator rewrite the note above predicted
> was done on 2026-08-14: **5.71 → 49.38 GFLOP/s** at this shape (8.6x), and 41–57 across the four
> projection shapes. `-ffp-contract=off` stopped the compiler *fusing* the reduction, but it also
> stopped it *splitting* the reduction — float addition is not associative, so one accumulator can
> only ever be a serial fp-add latency chain. Eight fixed lanes break the chain without a fast-math
> flag, so the order stays deterministic.
>
> The parity gate **improved**: logit err 1.91e-06 → 1.43e-06, grad err 2.98e-07 → 2.38e-07. Eight
> short chains round less than one long one, and PyTorch sums in blocks too, so this moved toward the
> reference. Generation from the trained checkpoint is byte-identical (278 bytes, same seed).

**Reproduce** (standalone, isolates codegen from Bazel):
```sh
CLANG=$(find ~/.cache/bazel -path '*llvm_toolchain_llvm/bin/clang++' | head -1)
# extract matmul_forward_cpu from src/ops.cpp into a main() that times best-of-8 at BT=1024,C=768,OC=3072
$CLANG -O3 -march=native mm.cpp -o mmb && ./mmb
$CLANG -O3               mm.cpp -o mmb && ./mmb
```
Once `//tools:bench` is merged (on `main`):
```sh
bazel run --config=release //tools:bench -- 20
```

## M-2 · Matmul backward throughput (single-thread)

`matmul_backward_cpu`'s inner loop carries no reduction, so it auto-vectorizes.

| Shape | GFLOP/s | Date |
|---|---|---|
| `BT=1024, C=768, OC=3072` | **~37–42** | 2026-08-03 |

> **This already exceeds the 30 GFLOP/s gate.** M2's gate as currently worded — "matmul ≥ 30 GFLOP/s
> single-thread" — is therefore **half-passed today with zero work**. The gate must name the
> *direction* (forward), the *shape*, and the *build config*. See `ROADMAP.md` M2.
> *(Measured by an independent reviewer with `g++ 15.2 -O3 -march=native`; not yet re-run under the
> hermetic clang, where forward is 1.9× slower — treat as indicative, re-measure before gating.)*

## M-3 · End-to-end training step (M2 target config)

6L / 384C / 6H, 10,770,816 params, T=256. Links the repo library unmodified.

| Batch | s/step | tokens/s | peak RSS |
|---|---|---|---|
| B=12 | 15.56 | 197 | 1.67 GB |
| B=64 | 83.95 | 195 | 8.15 GB |

Breakdown at B=12: forward matmul ≈76%, backward matmul ≈21%, everything else ≈3%.

> **Consequence for M2's convergence gate.** Char-Shakespeare val ≤ 1.6 needs ~25–40M tokens
> (nanoGPT's reference reaches 1.47 at 82M). At 197 tok/s that is **35–56 h**; even at the post-gate
> 494 tok/s it is **14–22 h**. *"val loss ≤ 1.6 overnight" is not reachable single-threaded* — the
> threading work is load-bearing and currently has no throughput gate. ~2,000 tok/s is the number the
> convergence gate actually needs.
> Note B=64 costs **8.15 GB**, 4× over the ≤2 GB baby-training budget in `PLAN.md` — which is the
> real argument for gradient accumulation (throughput is flat across B, so it is free).

## M-4 · Model sizes and memory

Derived from `param_sizes()` / `act_sizes()` in `src/model.cpp`; cross-checked against HF.

| | params | fp32 | training-shaped @ B=1,T=1024 | inference-only |
|---|---|---|---|---|
| GPT-2 124M | **124,439,808** | 0.46 GiB | **5.32 GiB** | **2.61 GiB** |
| GPT-2 350M | 354,823,168 | 1.32 GiB | **12.93 GiB** | **6.40 GiB** |

124M's param count equals HuggingFace's `named_parameters()` sum exactly. Largest activation
tensors at T=1024: `preatt` and `att`, 0.5625 GiB each; `logits`+`probs` ≈ 412 MB (decimal).
A *true* inference arena (rolling per-layer buffers, no `preatt`, no `probs`) would be ≈0.7 GiB.
**All figures fit comfortably in this host's 120 GB.**

## M-5 · Numerical parity vs canonical GPT-2 (PyTorch)

`//tests/integration:parity_test`, fixture `tests/fixtures/gpt2_parity.bin`.

| Quantity | Gate | Measured |
|---|---|---|
| Forward logits | ≤ 1e-4 | **1.43e-6** |
| Gradients | ≤ 1e-3 | **2.38e-7** |
| 10-step AdamW loss | ≤ 1e-3 | **9.54e-7** |

Logit and gradient error *improved* on 2026-08-14 (from 1.91e-6 and 2.98e-7) when `matmul_forward`
moved to 8 accumulator lanes — shorter summation chains round less, and PyTorch sums in blocks too.
See D8.

**Reproduce:** `bazel test //tests/integration:parity_test`

## M-6 · GPT-2 124M greedy-decode margins (for the M3 gate)

Top1−top2 logit margin over 250 greedy steps (5 prompts × 50 tokens), HF fp32 reference:

| | margin |
|---|---|
| worst step (any prompt) | **0.00716** |
| 5th percentile | 0.106 |
| median | 2.81 |
| `"def fibonacci(n):"` worst | **0.105** |

Against an fp32 logit error of ~1e-4, that is only **7–70× headroom at the worst step** — which is
why the M3 gate prompt must be chosen by *measured* margin, not by taste. See
`docs/M3_INFERENCE_PLAN.md` §M3-S5.

## M-8 · Toy example — char-level Shakespeare end to end

`examples/shakespeare/run.sh`, `--config=release`, L4 H4 C128, ctx 64, batch 32,
lr 3e-3, 900 steps over TinyShakespeare (1,115,394 chars, vocab 65, 10% val split).

| | |
|---|---|
| wall time | **634 s** (10.6 min) |
| throughput | **2908 tok/s** (1.84 M tokens) |
| peak RSS | **204 MB** |
| val loss | 2.42 → 2.15 → 2.03 → 1.95 → 1.86 → **1.81** (steps 150…900) |

Monotonic across all six evaluations. Reaching the nanoGPT char-Shakespeare
reference (~1.47) needs roughly 10× more tokens; nothing about the model or the
pipeline changes, only `--steps`.

**Reproduce:** `examples/shakespeare/run.sh`

## M-7 · Test suite

29 targets (23 unit + 5 integration + 1 py_test), green under `--config=dev` (ASan/UBSan) and
`--config=release`, as of 2026-08-03.

**Reproduce:** `bazel test //... --config=dev`

> Do not hard-code this count in other documents — it changes with every added test. Say
> "the full `//...` suite is green."

## M-9 · Forward-pass latency and per-op breakdown

`//tools:profile`, `--config=release`, the toy checkpoint (L4 H4 C128 V65 ctx64), best-of-N after
warmup. This is the budget for interactive interpretability: ablation and patching each cost exactly
one forward pass, so the `inspect` sweep is L·(NH+2) = 24 of these.

| B, T | forward | tokens/s | GFLOP/s |
|---|---|---|---|
| B=1, T=64 (full context) | **3.37 ms** | 18,982 | 31.4 |
| B=1, T=14 (typical viewer prompt) | **0.61 ms** | 22,871 | 36.7 |

Adding the loss (softmax + cross-entropy over `[B,T,V]`) costs under 1% at this size — below the
run-to-run noise floor, which the tool exposes by printing best *and* median rather than hiding it.

Per-op share of a forward pass at B=1, T=64 (each op timed standalone at the model's own shapes, so
these are independent measurements that do not sum to the end-to-end figure — the tool prints the
gap, here −12%, instead of implying agreement):

| op | ms/pass | share |
|---|---|---|
| matmul fc `[C→4C]` | 0.632 | 21.3% |
| matmul fcproj `[4C→C]` | 0.624 | 21.1% |
| attention | 0.505 | 17.0% |
| matmul qkv `[C→3C]` | 0.491 | 16.6% |
| **gelu** | 0.455 | **15.4%** |
| matmul attproj `[C→C]` | 0.164 | 5.5% |
| layernorm | 0.055 | 1.9% |
| matmul lm_head `[C→V]` | 0.020 | 0.7% |
| residual add | 0.017 | 0.6% |

> **Where the next win is.** Matmuls are now 65% and attention 17%, which is the healthy shape. The
> outlier is **gelu at 15.4%** — absurd for an elementwise op, and it is there because `gelu_new`'s
> tanh is a scalar libm call that cannot vectorise. It is deliberately left alone (D8): canonical
> GPT-2 pins the exact formula, and an approximation would trade parity for speed. The real remaining
> lever is threading — 16 cores sit idle — and that needs its own determinism argument, because a
> parallel reduction is where reproducibility gets genuinely hard.

**Reproduce**
```sh
bazel run --config=release //tools:profile -- --checkpoint data/shakespeare.ckpt --seq 64 --reps 100
```

## M-10 · Training throughput after D8

Same config as M-8 (L4 H4 C128 ctx64, batch 32), `--config=release`.

| | before D8 | after D8 |
|---|---|---|
| throughput | 2908 tok/s | **6429 tok/s** |
| speedup | — | **2.21×** |

Consistent with M-3's breakdown (forward matmul ≈76% of a step): making it ~8.6× faster leaves
backward and the elementwise ops as the new floor. The M-8 toy run should now take ~290 s rather
than 634 s.

## M-11 · Model quality vs baselines

`//tools:eval`, `--config=release`, the M-8 toy checkpoint (L4 H4 C128 V65 ctx64) over the full
TinyShakespeare validation split. Deterministic: sequential non-overlapping windows, 110,592 of
111,539 tokens scored (947 dropped as a partial trailing batch, and reported rather than hidden).

Baselines are Witten-Bell interpolated n-grams counted on the **training** split and scored on the
**validation** split, with exactly the context the model has: within-window, truncated at the window
start. Both restrictions matter — counting on val would let the baseline memorise what it is scored
on, and giving it context the model lacks would be the same dishonesty pointing the other way.

| | nats/token | perplexity | bits/char | vs model |
|---|---|---|---|---|
| **model** | **1.8199** | **6.17** | **2.626** | — |
| 7-gram | 1.7879 | 5.98 | 2.579 | −1.8% |
| 6-gram | 1.7129 | 5.54 | 2.471 | −5.9% |
| **5-gram (best)** | **1.6860** | **5.40** | **2.432** | **−7.4%** |
| 4-gram | 1.7874 | 5.97 | 2.579 | −1.8% |
| trigram | 2.0544 | 7.80 | 2.964 | +12.9% |
| bigram | 2.4850 | 12.00 | 3.585 | +36.5% |
| unigram | 3.3468 | 28.41 | 4.828 | +83.9% |
| uniform | 4.1744 | 65.00 | 6.022 | +129.4% |

> **The model LOSES to a 5-gram counting table by 7.9%.** This is the headline result and it
> reverses the earlier conclusion drawn from a bigram-only comparison, which the model beat by 26.7%
> and which flattered it. Beating a bigram is not evidence of having learned anything; a 5-gram is
> the bar that distinguishes a language model from a lookup table, and this checkpoint is under it.
> The cause is training budget, not architecture — see M-10 and the context curve below.

Supporting numbers: top-1 accuracy **45.6%**, calibration error **0.024** (well calibrated — the
model's confidence matches its accuracy, so it is not bluffing, merely weak). Train/val gap is
1.6474 vs 1.8199 nats, i.e. **0.17 nats**, so this is undertraining rather than memorisation.

Loss by context length (nats), position *t* predicting with *t*+1 characters:

| chars | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| loss | 2.53 | 2.15 | 2.00 | 1.92 | 1.86 | 1.82 | 1.80 | 1.80 | 1.77 | 1.77 |

Most of the gain is in the first ~6 characters; it reaches within 2% of its best by 18. So the model
does use more than a 5-gram's worth of context — it just uses it *worse* than counting does, which is
what an undertrained model looks like.

**Two independent cross-checks.** Uniform is 4.1744 = ln(65) exactly, confirming the vocabulary the
comparison is scaled by. The bigram row (2.4850, Witten-Bell) agrees with an independently written
add-one-smoothed bigram (2.4812) — the two were implemented separately and agree to 0.15%.

> **A bug this caught.** The first version anchored the n-gram context at the window *base* rather
> than at the token before the target, so every lookup used text that never precedes the token being
> predicted. The tell was bigram scoring *worse than uniform* and higher orders degrading
> monotonically — a baseline that gets worse with more context is broken, not weak.

**Reproduce**
```sh
bazel run --config=release //tools:eval -- \
  --checkpoint $PWD/data/shakespeare.ckpt \
  --data $PWD/data/shakespeare.val.bin --train $PWD/data/shakespeare.train.bin --order 6
```

## M-12 · Run A — controlled 10x scale-up

`//tools:train` at the M-8 config (L4 H4 C128, ctx 64, batch 32, lr 3e-3), 900 -> 9000 steps.
18,432,000 tokens (18.4 epochs) in 2267 s at 8130 tok/s, peak RSS 203 MB. Scored by `//tools:eval`
on the full validation split. See `docs/EXPERIMENTS.md` E-1.

| | M-8 | Run A | change |
|---|---|---|---|
| val loss (nats) | 1.8199 | **1.6133** | −0.207 |
| bits/char | 2.626 | **2.328** | −0.298 |
| perplexity | 6.17 | **5.02** | −1.15 |
| top-1 accuracy | 45.6% | **54.1%** | +8.5 pts |
| calibration error | 0.024 | 0.070 | **worse** |
| vs best n-gram (5-gram, 1.6860) | loses 7.9% | **beats 4.3%** | — |

The calibration regression is the overfitting showing up where the loss alone hides it: the model
became more confident faster than it became more correct.

**Overfitting.** Best val was **1.5514 at step 6250** (12.8 epochs); by step 9000 it had regressed to
1.6151 while train loss fell to 1.1196 — a train/val gap of 0.50 nats, up from 0.17 at 900 steps.
Only the final checkpoint was saved, so the best model was lost; `--ckpt-best` now fixes that.

Cross-check: the training loop's own eval reported 1.6151 and `//tools:eval` reported 1.6133 — 0.1%
apart, from independent implementations (shuffled subset vs full sequential pass).
