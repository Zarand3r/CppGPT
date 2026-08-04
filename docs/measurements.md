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
| `-O3 -march=native` — **the current `--config=release`** | **2.82** | 2026-08-03 |
| `-O3` (no ISA flags; SSE2 baseline) | **5.39** | 2026-08-03 |
| `-O3 -mavx2 -mfma` | 2.81 | 2026-08-03 |

> ⚠️ **`-march=native` currently costs ~1.9× on this kernel.** Enabling AVX2/FMA makes it *slower*,
> not faster — the opposite of the usual assumption, and the opposite of what an earlier draft of
> `PLAN.md` asserted ("the vector ISA was available and the compiler still could not vectorize it;
> the fix is the code shape, not a compiler flag"). Half of that is right — the serial `acc +=`
> dependency chain is still the root cause and blocking/multiple-accumulators is still the real fix —
> but the flag is not neutral, it is harmful. **Investigate before the blocking work**, and re-measure
> after: the true naive baseline is **5.39**, so the gap to the 30 GFLOP/s gate is ~5.6×, not ~10×.
> Mechanism not yet diagnosed (suspect FMA contraction serializing the reduction at ~4-cycle latency);
> confirm by disassembly before acting.

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

## M-7 · Test suite

25 targets (20 unit + 4 integration + 1 py_test), green under `--config=dev` (ASan/UBSan) and
`--config=release`, as of 2026-08-03.

**Reproduce:** `bazel test //... --config=dev`

> Do not hard-code this count in other documents — it changes with every added test. Say
> "the full `//...` suite is green."
