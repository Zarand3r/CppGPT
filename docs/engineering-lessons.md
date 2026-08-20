# Engineering lessons — cppgpt

Rules distilled from **real defects observed in this repository**. Every rule cites the incident
that motivated it; that citation is what makes the rule falsifiable rather than a platitude. If you
cannot name an incident, it is not a lesson yet.

**Maintenance:** if a new incident shows an existing rule failed to prevent a recurrence,
*strengthen that rule in place* and say why it failed — do not add a near-duplicate. When this list
changes materially, refresh the pointer in `CLAUDE.md`.

Produced by the `review-codify-loop` skill. See `docs/review-audit.md` for mechanical sweeps.

---

## L1 — A comment, log line, or doc that states a postcondition must be enforced by the code

**Incident (2026-07, `src/model.cpp`).** `load_checkpoint` on a weights-only file logged
*"resume starts Adam from zero"* and `model.hpp` promised *"leaves the optimizer state at zero"* —
but the `else` branch set only `adam_step_ = 0` and never zeroed `m_`/`v_`. A model that had already
trained carried stale moments into the next step, where step-1 bias correction (10×) amplified them:
**weights moved under zero gradients.** The code and its own log message disagreed, and the log was
the thing people would trust.

**Rule.** When a comment or log asserts a postcondition, either implement it or assert it in the same
change. Prose is not a substitute for a `memset`. Treat "the doc says X" as a test case to write.

## L2 — Every float guard must handle NaN explicitly

**Incident (2026-07, `src/optimizer.cpp`, `src/sample.cpp`).** `clip_grad_norm` guarded with
`norm > max_norm`, which is **false for NaN**, so diverged gradients were neither clipped nor
reported. In `sample()`, every `logits[i] >= thresh` is false for NaN, so nothing was eligible,
`sum == 0`, and the sampler **silently returned token 0 forever**. `ErrorCode::NanOrInf` existed in
`core.hpp` and was used nowhere.

**Rule.** A comparison is not a validity check. Any code branching on a float derived from a
reduction must test `std::isfinite` explicitly and fail fast — non-finite values mean upstream
numerics already broke, which is semantic corruption, not a degraded mode.

## L3 — Never size an allocation from untrusted file contents

**Incident (2026-07, `src/checkpoint.cpp`).** `load_checkpoint` read the whole file into a
`std::vector` sized by `st_size`, inside a `noexcept` function, *before* validating anything — so a
bogus file dictated an unbounded allocation and `std::terminate` on `bad_alloc`. Reachable by
typo'ing a dataset path into the checkpoint argument.

**Rule.** Read and validate a fixed-size header first, then derive every buffer size from **our own**
invariants (the model's `param_count`, not the file's). Cross-check the file's claim against ours and
fail with a typed error. This is doubly binding inside `noexcept`.

## L4 — Test the degenerate case, or the equivalence claim is untested

**Incident (2026-07, `src/sample.cpp`).** `top_k == 1` was documented as greedy/argmax, but the
threshold rule keeps every logit `>= kth`, so a **tied** maximum left several tokens eligible and
softmax spread over them. `sample_test` passed because every vector it tested was tie-free.

**Rule.** For any "X is equivalent to Y" claim, the test must include the degenerate input — ties,
zeros, empty, single-element, all-equal. A test over generic inputs does not establish an equivalence
at the boundary, which is exactly where equivalences break.

## L5 — Any file another tool consumes is written tmp-then-rename

**Incident (2026-07, `src/dataloader.cpp`).** `write_token_bin` opened the destination with
`O_TRUNC` and, on a partial write, returned `IoError` while leaving a **truncated** `.bin` at the
target path. Token files carry no length and no checksum, so `DataLoader::open` loaded it happily and
training silently used less data than intended. The checkpoint writer in the same tree already did
this correctly.

**Rule.** Producer writes `path.tmp`, `fsync`s, `rename`s, and `unlink`s the temp on any error —
plus `fsync` the containing directory, without which the rename itself is not durable. If a format
cannot detect its own truncation, atomic replacement is the only thing standing between a failed
write and silent data loss.

## L6 — A magic constant's comment must state a verified identity

**Incident (2026-08, `include/cppgpt/checkpoint.hpp`).** `kFnvOffset64 = 1469598103934665603` is
commented as the FNV-1a-64 basis. It is the textbook value `14695981039346656037` **with the last
digit dropped** — a typo. It still functions as a hash so nothing broke, and it survived code review;
worse, a plan draft then described the deviation as *deliberate*, which would have frozen a typo into
the design forever. A sweep for the same signature found a second instance: `verify.hpp` and
`gen_fixtures.py` both comment `0x43475446` as `"CGTF"`, but its little-endian bytes spell `"FTGC"`.

**Rule.** When a constant claims a canonical identity (a named standard's basis/prime, a
mathematical constant, a four-character code), the comment must state the *verified* identity, and
the value must be checked against the source — mechanically where possible (see
`docs/review-audit.md`). A constant that merely "works" is not thereby correct. Never document a
suspected mistake as intentional; file it.

## L7 — An acceptance gate that a wrong implementation can pass is not a gate

**Incident (2026-08, `docs/M3_INFERENCE_PLAN.md`, pre-merge).** The plan's S1 gate asserted that
`generate_absolute(n_new=1)` equals a forward plus argmax at `len−1` — which *is* the implementation
restated, so it tests nothing, and a left-padding implementation (the exact bug S1 exists to prevent)
would pass. The same draft proposed a per-tensor checksum manifest generated by the very script it
was meant to validate.

**Rule.** For every gate, ask: *could a wrong-but-self-consistent implementation pass this?* If yes
it is a smoke test, not a gate. Gates must compare against something the implementation did not
produce — an external oracle, or an invariant stated independently of the code (e.g. "these two
differently-shaped runs must be bit-identical").

## L8 — Do not declare two workstreams independent without naming the quantity they share

**Incident (2026-08, `docs/M3_INFERENCE_PLAN.md`, pre-merge).** The plan twice called M2's matmul
optimization "independent" of M3. They are coupled through fp32 error: M3's right-pad protocol is
bit-identical *only because* the current matmul uses a per-row serial accumulator, and M3's greedy
gate has ~7–70× headroom at ~1e-4 — so a cache-blocked or threaded reduction can flip a token and
break M3 without touching a line of M3 code.

**Rule.** "Independent" is a claim requiring evidence. Name the quantities both streams read or
write — numerical error, memory, a file format, a shared invariant — and show they do not overlap.
Absent that, state the coupling and sequence the work.

## L9 — A plan may not depend on a capability the codebase lacks without a slice to build it

**Incident (2026-08, `docs/M3_INFERENCE_PLAN.md`, pre-merge).** Two invariants and the failure model
required verifying an asset's **sha256 at runtime**. There is no SHA-256 anywhere in this std-only
tree (only `fnv1a_64`), and `PLAN.md` explicitly defers SHA-256 to "a concrete need" — so the plan
silently assumed ~200 lines of unlisted crypto plus its test vectors.

**Rule.** Before a plan depends on a primitive, grep for it. If it does not exist, either add a slice
with its own gate, or re-specify using what does exist. Split roles when they differ: provenance
checks can run dev-time in Python; runtime integrity uses the hash the codebase already ships.

## L10 — Execute third-party API contracts; do not recall them

**Incident (2026-08, `docs/M3_INFERENCE_PLAN.md`, pre-merge).** The plan specified a layer-bisect
fixture comparing "after the embedding and after each of the 12 blocks". Executed against
transformers 5.12, `hidden_states` has `L+1 = 13` entries whose **last is post-`ln_f`** — block 11's
raw output is never exposed. The mapping was off by an entire LayerNorm, on the fixture whose sole
purpose was localizing bugs.

**Rule.** Any claim about an external library's output shape, ordering, or defaults is verified by
running it and printing the result, and the printed evidence goes in the document. This is how the
Conv1D transpose and Q‖K‖V ordering were pinned correctly in the same plan — apply it uniformly.
Corollary: verify invocation contracts too (`bazel run` sets cwd to the runfiles dir, so the plan's
headline command could not have found its own weights).

## L11 — `bazel test //...` stays green on a clean checkout

**Incident (2026-08, `docs/M3_INFERENCE_PLAN.md`, pre-merge).** The plan required a gate depending on
a 497.8 MB weight file that is deliberately never committed, while also mandating "fail loudly, never
skip silently". Together those turn every fresh clone red — and `CLAUDE.md` makes a green
`bazel test //...` the prerequisite for the autonomous harness.

**Rule.** A test needing an uncommittable artifact is a **separate target tagged `manual`**, excluded
from the `//...` wildcard and invoked explicitly. Exclusion-by-tag is not a runtime skip: once
invoked, it still fails loudly when the artifact is missing. Never make a wildcard test conditionally
green.

## L12 — A document may not state a fact about the tree that a command cannot confirm

**Incident (2026-08, `ROADMAP.md`, `README.md`, `PLAN.md`).** `tools/bench` was marked `[x]` done,
documented as a runnable command (`bazel run //tools:bench -- 20`), and cited in four documents as the
source of the repo's most-referenced measurement — while **never having been on the branch**. It lives
on the unmerged `m2-bench` branch. The checkbox was ticked from a branch that did not contain the
file. `bazel query //tools:all` disproves all five sites in under a second. Three independent
reviewers each found it first.

**Rule.** Before writing that something exists, is done, or is runnable, run the command that proves
it — `bazel query`, `ls`, `git ls-tree <branch>`. When work spans branches, a checkbox describes
**`main`**, not your working tree. Any documented command must name a target that exists on the branch
the reader will check out.

## L13 — Verify the direction of a performance claim before acting on it

**Incident (2026-08, `.bazelrc` / `PLAN.md`).** A plan asserted that because `--config=release`
already passes `-march=native`, "the vector ISA was available and the compiler still could not
vectorize it — the fix is the code shape, not a compiler flag." Measured on the real kernel with the
pinned clang: `-march=native` **2.82** GFLOP/s vs plain `-O3` **5.39** GFLOP/s. The flag was not
neutral, it was **costing ~1.9×** — so a free speedup sat behind *removing* a flag while the plan
argued only for a code rewrite. The root-cause diagnosis (serial `acc +=` chain) was right; the
conclusion drawn from it was not.

**Rule.** A claim that a flag, setting, or configuration is *neutral or beneficial* is a measurement,
not a deduction — benchmark both ways before building a plan on it. Reasoning correctly about a
bottleneck does not license an untested claim about what does or does not affect it. Record the
numbers in `docs/measurements.md` with a reproduce command.

## L14 — Mutation-test a new gate before trusting it, especially one you wrote to be strict

**Incident (2026-08, `tests/unit/interpret_test.cpp`).** A check written specifically to enforce
`logit_lens`'s read-only contract — snapshot the activation arena, lens every layer, assert nothing
changed — **passed against a mutant that deliberately scribbled into `lnf_mean`/`lnf_rstd`.** The
loop ran layers in ascending order and therefore *ended* on the last layer, where the lens recomputes
the model's own final layernorm and writes back byte-identical values. The violation was real and
the assertion was correct; the traversal order made it unobservable. Reversing the loop so it ends on
layer 0 catches the mutant immediately.

Two of the four checks in that file caught their mutants on the first try. This one did not, and
nothing about reading it suggested a problem.

**Rule.** A new gate is a hypothesis until a mutant kills it. Break the implementation in the
specific way the gate exists to prevent and watch it go red — cheap (one `sed`, one test run) and the
only evidence that distinguishes a strict test from a strict-looking one. This applies *most* to
tests written deliberately to be rigorous, because those are the ones nobody re-examines.

## L15 — Sweep a codified lesson across the whole tree, not just its incident site

**Incident (2026-08, four sites at once).** A three-axis review found four *recurrences* of already-codified
lessons at places the original fix never visited:

- **L2 (NaN guards)** — `max_abs_diff` in `verify.hpp` used `std::fmax`, which returns `x` for
  `fmax(x, NaN)` **by definition**. So the M1 parity gate — the flagship numerical-correctness gate and a
  `constitution.md` deal-breaker — **passed with all 2024 gradients set to NaN**, exit 0.
- **L2 again** — `sample()`'s `top_k == 1` fast path returned before the finite guard that L2 added, so
  greedy decoding returned token 0 forever on a NaN model. The *L4* fix (route `top_k==1` to `argmax`)
  reopened the *L2* defect on that branch. Two lessons' remedies collided and nothing tested the seam.
- **L5 (atomic writes)** — `prepare` wrote its `.vocab` with plain `ofstream(trunc)`, while the `.bin`
  files beside it went through tmp→fsync→rename. Demonstrated end to end: a torn write yields a
  *size-matched but semantically wrong* vocabulary that `train` accepts and `generate` decodes against.
- **L3 (bounded allocation)** — applied to checkpoint *sizing* but not to the `std::vector` that holds
  the payload, inside a `noexcept` function.

Each original fix was correct. Each was applied only where the bug was found.

**Rule.** When a lesson is codified, immediately grep the tree for its *signature* and fix every site in
the same change: `fmax(`/`fmin(` used as a guard, `>`/`<` on a value that can be NaN, `ofstream(...trunc)`
on a file another tool reads, container construction inside `noexcept`. Record the sweep in
`docs/review-audit.md`. A lesson applied at one site is a bug fix; applied across the tree it is a lesson.
**Corollary:** when two lessons touch the same function, add a test at their intersection — that is where
the next defect lives.

## L16 — An automated edit must assert its anchor matched

**Incident (2026-08, twice in one session).** A scripted edit added a prompt UI to `viewer.html` and wired
its button handler; the HTML replacement matched, the JavaScript one did not (the anchor comment had been
reworded in an earlier commit). `str.replace` returns the input unchanged on no match, so the button
rendered and did nothing, and the script *looked* correct. Diagnosed by cross-referencing every `$('#id')`
against every `id=` in the file.

Then the same failure corrupted a mutation-testing run: three mutants reported GREEN (survived). Re-running
with an anchor assertion showed one was an **anchor miss** — the mutation never applied — and the test had
been catching it all along. Unverified negative results are worse than no results.

**Rule.** Every scripted edit asserts the anchor exists before replacing and asserts the intended text is
present afterwards. For mutation testing specifically, a mutant that reports "survived" without proof the
file changed is not evidence — verify the diff, then run.

## L17 — A self-referential equivalence test is worse than no test, and looks better

**Incident (2026-08, `tests/unit/tensors_test.cpp`).** The parameter half of the tensor-table test
compared the table against an independently transcribed `reference_sizes()` list and killed its
mutants. Written an hour later, the *activation* half did this instead:

```cpp
for (int i = 0; i < kNumActs; ++i) tot += act_total(kActSpecs[i], c, B, T);
CHECK(tot == acts_total(c, B, T));   // acts_total IS that loop
```

`acts_total()` is the same summation, so the assertion is `x == x`. Doubling `losses`, doubling
`fch_gelu`, and flipping `encoded` to per-layer all passed. It reported "60 checks pass", which was
used as evidence the refactor was layout-preserving.

The failure is not ignorance — the correct pattern existed twenty lines above, written by the same
hand in the same file. It is that summing the thing under test and comparing to its own sum *reads*
like verification.

**Rule.** An equivalence test must compare against something the implementation did not produce: a
transcribed reference, an external oracle, or an invariant derived independently. Before trusting one,
name the two sides out loud — if both trace back to the same function, there is one side. This is L7
("a gate a wrong implementation can pass is not a gate") applied to *data* rather than behaviour, and
it needs its own entry because the shape is different: L7's failure looks circular on inspection;
this one looks like arithmetic.

## L18 — Counting is not identity

**Incident (2026-08, `tests/unit/tensors_test.cpp`).** The canonical GPT-2 2-group weight-decay split
— a `docs/constitution.md` claim — was pinned by `CHECK(decaying == 6)`. Swapping `ln1w` to decaying
and `attprojw` to non-decaying keeps the count at exactly 6 and passed the whole suite. The parity
gate cannot help: it runs with `weight_decay = 0`.

The count was written *because* the flags had just moved into a table to stop them drifting. It
measured the wrong property of the thing it was protecting.

**Rule.** When a test guards a set membership — which tensors decay, which flags are set, which files
are written — assert the *membership by name*, not the cardinality. A count is invariant under every
permutation, which is precisely the mutation class these tests exist to catch.

## L19 — A test whose target already holds the right answer cannot fail

**Incident (2026-08-20, `tests/unit/model_test.cpp`).** `forward(..., logits_at)` computes the
classifier at one position instead of all T. The test ran the full forward, saved row `T-1`, ran the
single-position forward, and compared row `T-1`. A mutation making it compute row `T-2` **survived**:
the full forward had already left the correct value at `T-1`, so the test could not distinguish *"the
single-position path wrote the right answer"* from *"it wrote nothing, and the stale value from the
previous run happened to be right."*

**Rule.** When a test checks that code WROTE something, the destination must hold a **different**
value first, and that difference must itself be asserted. Poison, then verify the poison took, then
run, then compare.

This is the third instance of the same shape, which is why it gets its own entry rather than a note
on L7:

| where | what made the test pass without the code working |
|---|---|
| `act_guard_test` | the oversized config died in the **allocator**, so the guard under test never ran — a bare `CHECK_DIES` is satisfied by any failure (fixed by `CHECK_DIES_WITH`, matching the message) |
| `eval`'s e2e contract | perplexity and bits/char are both **derived from** the nats figure, so a wrong nats satisfied all three together (fixed by an external anchor: `train`'s independent eval) |
| `logits_at` | the destination **already held** the right value (fixed by poisoning the arena first) |

The unifying failure is that the assertion had a second way to be satisfied. Ask of every check:
*what else could make this pass?*

**Corollary for optimisations.** A change whose only purpose is speed needs a test that can see the
speed being lost. `logits_at` had a second survivor — a mutation ignoring the parameter and computing
every row — which was *correct, merely slower*. That was closed by asserting the documented contract
("rows other than `logits_at` are left stale"), which until then was a claim with no test. **A
performance feature verified only for correctness is unverified.**

**Mechanical check.** `tools/mutation_suite.sh` runs a curated battery over the load-bearing files and
reports survivors. Prose did not prevent recurrence here — the same author wrote all three — so the
rule ships with something that runs.
