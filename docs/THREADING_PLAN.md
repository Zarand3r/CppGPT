# Design proposal — threading

**Status: proposal. No implementation.** Per `CLAUDE.md`, architecturally
significant work stops at the roadmap for review. The determinism argument below
is the part that most needs scrutiny before any code exists.

## The problem, measured

| | |
|---|---|
| GPT-2 124M generation | **5.1 s/token** (12 tokens in 61 s at ctx 1024) |
| forward, T=512 | 4.34 s — 131 GFLOP at **30 GFLOP/s effective** |
| matmul, single thread | 49 GFLOP/s (`docs/measurements.md` M-1) |
| cores available | **32** (16 physical) |

Correct and unusable. Every other optimisation on the table is worth a few
percent; this one is worth most of an order of magnitude.

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

**The backward pass is NOT.** Three places accumulate across the natural
partition axis, and each fails differently:

| op | what breaks |
|---|---|
| `matmul_backward` | `dw_oc[c] += d * inp_bt[c]` sums over `bt`. Partitioning by `bt` makes this a cross-thread reduction — a data race, and non-deterministic even if made atomic. |
| `embedding_backward` | `dtok_row[c] += dout_bt[c]` scatters into a row keyed by token id. The same token at two positions lands in two threads — race. |
| `clip_grad_norm` | one global `sumsq` over every gradient. |

**Recommendation: thread the forward only, in the first change.** That is not a
compromise — it is where the measured problem is. Generation is pure forward, and
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
3. **Memory bandwidth, not cores, may bind.** At 49 GFLOP/s single-threaded the
   matmul is already partly bandwidth-bound; 16 threads will not give 16×. The
   plan must state a *measured* target, and I will not predict one here.
4. **Scope creep into backward.** Explicitly out.

## Staged plan

| slice | delivers | gate |
|---|---|---|
| **T1** | Fixed pool, explicit count, default 1, inline below a measured threshold | pool unit tests; no behaviour change at P=1 |
| **T2** | `matmul_forward` partitioned over output rows | bit-identical for P ∈ {1,2,4,8,16}; speedup measured |
| **T3** | `attention_forward` partitioned over (b, t, h) | same |
| **T4** | End-to-end: generation latency at GPT-2 124M | parity gate unchanged; new M-16 |
| — | *backward threading* | **out of scope**, separate decision |

## What I need decided

- **Forward-only first?** I believe yes — it is where the measured problem is,
  and it is the half that is deterministic for free.
- **Default thread count.** I propose **1**, with opt-in. Silent parallelism
  changes the performance and failure characteristics of every existing caller.
- **Is bit-identity across thread counts the right gate,** or too strict? It
  forecloses future work-stealing and split reductions. I think it is worth it
  here, because it is the only gate that cannot be quietly loosened.
