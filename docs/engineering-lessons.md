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
