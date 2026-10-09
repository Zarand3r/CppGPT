# Design proposal — threading

**Status: proposal. No implementation.** Per `CLAUDE.md`, architecturally
significant work stops at the roadmap for review. The determinism argument below
is the part that most needs scrutiny before any code exists.

## The problem, measured

| | |
|---|---|
| GPT-2 124M generation | **5.1 s/token** (12 tokens in 61 s at ctx 1024) |
| forward, T=512 | 4.34 s — 131 GFLOP at **30 GFLOP/s effective** |
| matmul, single thread | **59–61** GFLOP/s (`docs/measurements.md` M-30) |
| cores available | **32** (16 physical) |

Correct and unusable.

> **Revised 2026-09-17.** This section previously read *"every other optimisation
> on the table is worth a few percent; this one is worth most of an order of
> magnitude."* **That was false, and it is the sentence the staging below rested
> on.** Register blocking over `BT` rows measures **~2.1x** (1.93x on `mlp_proj`),
> is **bit-identical** to the shipping kernel, and needs no determinism argument,
> no pool, no TSan config and no `ldd` allow-list change — none of the four things
> threading needs. It was already in `PLAN.md`'s M2 sequence, between the
> multi-accumulator work (D8, done) and the thread partition, and it was skipped.
> `docs/measurements.md` M-30 has the numbers, the bit-identity check and the
> reproduce command. It is now **T0** below.
>
> The line quoting matmul at 49 GFLOP/s was also stale: `matmul_forward_cpu` is
> byte-identical since D8 and the same command reads 59–61 today.

## The determinism argument — the crux

Threading is only acceptable here if results do not change. The repo's
constitution promises numerical parity and determinism, and `docs/DECISIONS.md`
D8 already spent the "summation order changes" budget once, deliberately and
with evidence. Spending it again — accidentally, and with a result that varies
by thread count or scheduling — would be a different and much worse thing.

**The forward pass is deterministic under threading by construction.** Not by
convention, not by care — by the shape of the code:

```
matmul_forward:   out_bt[oc] = bias[oc] + s;     // s reduced entirely locally
attention:        out_row[i] += a * v[i];        // per (b, t, h), no sharing
gelu/residual:    elementwise
layernorm:        per row
```

Every output element is written **exactly once**, by whichever thread owns it,
and every reduction that produces it happens **entirely inside that thread**. So
partitioning the output space changes *who* computes a value, never *how* it is
computed. The consequence is the strong property worth gating on:

> **Bit-identical output for any thread count, including 1.**

That is testable, and it is a far better gate than "close enough": it cannot be
satisfied by a racy implementation, and it does not require choosing a tolerance.

**The backward pass does not get it from a SINGLE GLOBAL RULE.**

> **Revised 2026-09-17.** This paragraph previously read "The backward pass is
> NOT", over a table of three ops, and was read — including by its author — as
> *the backward cannot be threaded deterministically*. That is not what the code
> says. Every op below has an axis on which its writes are disjoint; what the
> backward loses is the *one rule that works everywhere*, and in two cases the
> safe axis is not the fast one. The corrected table also has **five** rows, not
> three: `layernorm_backward` and `attention_backward` were missing.

The forward's property comes from every op having the same shape, so one rule —
partition the output space — covers all of them, including ops written later. The
backward has no such single rule, because **weight sharing** forces reductions the
forward never performs: a weight is used at every position, so `dweight` sums over
every position. Each destination must therefore pick its own axis.

| op | racy axis, and why | conflict-free axis | cost of taking it |
|---|---|---|---|
| `matmul_backward` → `dinp` | — | **`bt`** | none; `dinp[bt][c]` rows are disjoint, the reduction over `oc` is thread-local |
| `matmul_backward` → `dweight`, `dbias` | `bt`: `dw_oc[c] += d*inp_bt[c]` sums over `bt` | **`oc`** | the loop is **fused** with `dinp`, which wants `bt`. Must be split into two loops → a second pass over `dout` and `inp` |
| `layernorm_backward` → `dweight`, `dbias` | `bt`: both are `[C]` accumulated over every row | **`c`**, or per-thread `[C]` partials | partials are 3 KB/thread at C=768, combined in a fixed index order |
| `attention_backward` → `dinp` | `t`: for fixed `(b,h)` every `t` writes `dk`/`dv` at all `t2 <= t` | **`(b, h)`** | none; each head owns a disjoint channel block. Note this is *narrower* than the forward's `(b,t,h)` |
| `embedding_backward` → `dwte` | `bt`: the destination row is the token id, so the same token in two threads collides | **`c`** | every thread reads all of `dout` — fine here (tiny op), ruinous for matmul |
| `clip_grad_norm` | every axis — the output is one scalar, so **no output axis exists** | none | genuinely needs partials. Fix the partial *count* independent of `P` (e.g. always 64 chunks) and combine in index order, or the result changes with thread count |

Read the table as a **price**, not a barrier. Four of the six are "pick the other
axis". One needs a loop split. One needs partials. What none of them are is
impossible — and `clip_grad_norm`, the only one with no output axis at all, still
keeps bit-identity across `P` if the partial count is fixed by the problem shape
rather than by the thread count.

**Safe is not always fast, and that is the real cost.** In the fused
`matmul_backward` loop, `c` *is* a conflict-free axis — it indexes both `dinp` and
`dweight` — so it is available without splitting anything. It is also the
innermost, contiguous, vectorised dimension, so cutting it shreds the access
pattern and puts thread boundaries mid-row. The fast axes (`bt`, `oc`) are each
unsafe for one destination. In the forward, safe and fast are the same axis; in the
backward you choose, and a wrong choice compiles, runs, and produces plausible
gradients.

**Recommendation: thread the forward only, in the first change.** That is not a
compromise — it is where the measured problem is, and it is the half where the
determinism gate is free rather than argued. Generation is pure forward, and
5.1 s/token is a forward-pass number. Backward threading is a separate decision
with a separate determinism argument (per-thread partials combined in a fixed
order), and should not ride along.

## What to thread, and how

**Partition the output rows, not the reduction.** For `matmul_forward` the unit
is `(bt, oc)`; for attention it is `(b, t, h)`. Static, contiguous ranges
computed as `[n * i / P, n * (i+1) / P)` — deterministic, no work stealing, no
dependence on scheduling.

**A fixed pool, created once.** Not `std::async` per call: the forward runs 12
matmuls per layer per token, and thread creation at that rate would cost more
than it saves. `std::thread` is std-only, so no dependency is added — but see
the allow-list note below.

**Explicit, never implicit.** `CLAUDE.md` forbids hidden threads behind innocent
names. The thread count is an explicit constructor argument, defaulting to **1**,
so nothing becomes concurrent because a library decided it should be.

**Do not thread:** anything under a few hundred microseconds. At toy scale
(L4 C128) a forward is 0.6 ms and the per-op work is ~10 µs — synchronisation
would dominate. The pool should have a minimum-work threshold below which it runs
inline, and that threshold must be *measured*, not guessed.

## Verification

1. **Bit-identical across thread counts.** Same input, `P ∈ {1, 2, 4, 8, 16}`,
   logits compared with `==`. This is the whole determinism claim in one test.
2. **TSan.** A race that does not change the answer today will change it on
   another machine. `--config=tsan` alongside the existing `--config=asan`.
3. **The parity gate unchanged.** GPT-2 logits must stay within the D9 budget
   (at least as close to fp64 truth as HF fp32), threaded or not.
4. **Speedup measured, not assumed**, at both scales — GPT-2 124M and the toy
   model — since the toy model may get *slower* and that is the expected result.

## Risks

1. **`libpthread` and the allow-list.** `tools/check_ldd.sh` permits libc/libm.
   On glibc ≥ 2.34 pthread is merged into libc, so nothing changes *here* — but
   this repo already learned (D-addendum, `libresolv`) that such a fact is a
   property of the host, not of us. The gate may need an explicit entry, and that
   should be a deliberate widening with a reason, not a surprise in CI.
2. **False sharing** at partition boundaries. Contiguous ranges keep it to the
   edges; worth measuring with `perf c2c` rather than assuming.
3. **~~Memory bandwidth, not cores, may bind.~~ WITHDRAWN — the premise was
   wrong.** This read: *"at 49 GFLOP/s single-threaded the matmul is already
   partly bandwidth-bound; 16 threads will not give 16x."* Measured
   (`//tools:bench --footprint`, M-30): holding FLOPs fixed and sweeping the
   weight matrix from 48 KiB to 36 MiB — a 768x range spanning L1, L2 and L3 —
   throughput is **flat at 53–67 GFLOP/s**. It is not bound on cache capacity or
   bandwidth. The *caution* may still hold; the stated reason does not, and L13
   says a claim that a resource binds is a measurement rather than a deduction.

   What is actually binding was **not** determined: `perf` is unavailable on this
   host (`perf_event_paranoid = 4`). Ruled out so far: footprint/bandwidth
   (above), and accumulator latency (raising `kLanes` 8 -> 16 -> 32 moves `mlp_fc`
   62.2 -> 64.6 -> 64.5). The plan must still state a *measured* target, and none
   is predicted here.
4. **Scope creep into backward.** Explicitly out.

## Staged plan

| slice | delivers | gate |
|---|---|---|
| **T0** | **Register blocking in `matmul_forward` (R=8).** No threads. Bit-identical, so no new determinism argument | `max\|diff\| == 0` vs the current kernel at all four shapes; parity gate unchanged; **packed-vs-scalar op counts checked in the emitted code** (L22 — the win is TU-dependent and evaporates silently); ~2x measured through `//tools:bench --blocked` |
| **T1** | Fixed pool, explicit count, default 1, inline below a measured threshold | pool unit tests; no behaviour change at P=1 |
| **T2** | `matmul_forward` partitioned over output rows | bit-identical for P ∈ {1,2,4,8,16}; speedup measured |
| **T3** | `attention_forward` partitioned over (b, t, h) | same |
| **T4** | End-to-end: generation latency at GPT-2 124M | parity gate unchanged; new M-16 |
| — | *backward threading* | **out of scope**, separate decision — priced in the table above, not blocked |

## What I need decided

- **Does T0 land before T1?** I believe yes. It is ~2x, bit-identical, already in
  `PLAN.md`'s M2 sequence, and it needs none of the four things threading needs.
  Threading is still the larger win; it is no longer true that nothing else is
  worth doing first. *(New question, 2026-09-17.)*
- **Forward-only first?** I believe yes — it is where the measured problem is,
  and it is the half that is deterministic for free.
- **Default thread count.** I propose **1**, with opt-in. Silent parallelism
  changes the performance and failure characteristics of every existing caller.
- **Is bit-identity across thread counts the right gate,** or too strict? It
  forecloses future work-stealing and split reductions. I think it is worth it
  here, because it is the only gate that cannot be quietly loosened.
