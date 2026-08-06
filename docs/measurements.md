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

**Reproduce** (standalone, isolates codegen from Bazel):
```sh
CLANG=$(find ~/.cache/bazel -path '*llvm_toolchain_llvm/bin/clang++' | head -1)
# extract matmul_forward_cpu from src/ops.cpp into a main() that times best-of-8 at BT=1024,C=768,OC=3072
$CLANG -O3 -march=native mm.cpp -o mmb && ./mmb
$CLANG -O3               mm.cpp -o mmb && ./mmb
```
Once `//tools:bench` is merged (currently **branch `m2-bench`, PR #20 — NOT on `main`**):
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
| Forward logits | ≤ 1e-4 | **~1.9e-6** |
| Gradients | ≤ 1e-3 | **~3.0e-7** |
| 10-step AdamW loss | ≤ 1e-3 | **~9.5e-7** |

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

25 targets (20 unit + 4 integration + 1 py_test), green under `--config=dev` (ASan/UBSan) and
`--config=release`, as of 2026-08-03.

**Reproduce:** `bazel test //... --config=dev`

> Do not hard-code this count in other documents — it changes with every added test. Say
> "the full `//...` suite is green."
