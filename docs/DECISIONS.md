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
  moves from 9.54e-07 to **1.91e-06**, against a 1e-4 tolerance (gradients 2.98e-07, loss 9.54e-07).
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
