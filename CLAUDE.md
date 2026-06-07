# Repository Instructions for Claude

This repository follows a set of skills that encode the engineering doctrine, workflow, and language-specific reference material to apply here. Load and apply them as described below — do not paraphrase the doctrine; invoke the skill so its full content is in context.

## Skills to use

### Core behavior

- **`principal-production-engineer`** — load for any implementation, code review, refactoring, performance work, or production hardening. This is the default behavior skill for non-trivial work in this repo.
- **`karpathy-guidelines`** — keep in mind for all writing, reviewing, or refactoring of code: behavioral guardrails against common LLM coding mistakes (overcomplication, non-surgical changes, unstated assumptions, missing success criteria). Biases toward caution over speed; use judgment on trivial tasks.

### Planning → execution chain

- **`strategic-engineering-planner`** — load *before* implementation when the task is architecturally significant, ambiguous, multi-file, distributed, performance-sensitive, concurrency-heavy, safety-critical, or likely to need multiple passes. Produces the architectural roadmap; hand off to execute.
- **`implementation-plan`** — load *after* the design/roadmap is locked and *before* writing implementation code, to turn it into a checklist-first `IMPLEMENTATION_PLAN.md` with vertical-slice steps, binary acceptance gates, property tests tied to invariants, a golden-path integration test, and an explicit iteration loop. Feeds `principal-production-engineer` for execution.

### C++ reference

- **`cpp-systems-internals`** — load when touching C++ runtime mechanics, API design, ownership vocabulary, data-oriented layout, codegen cost, hardware behavior, or OS-level memory (`mmap`/`madvise`/paging). This repo is C++; expect to load relevant reference files (`data_oriented_design.md`, `cpp_api_style.md`, `cpp_ownership_and_arenas.md`, `cache_lines_and_alignment.md`, `vtables_and_polymorphism.md`, `templates_and_codegen.md`, `memory_mapping.md`, etc.) for the specific task at hand.

### Autonomous & measurement modes

- **`auto-research`** — load when iteratively optimizing a measurable outcome (training loss/val_bpb, MFU, latency percentiles, throughput, memory/binary size, compile time) overnight or unattended. Enforces a fixed evaluation harness, a single mutable code surface, time-boxed runs on a dedicated branch, an append-only TSV results log, keep-on-improvement / git-reset-on-regression, and an autonomous never-stop loop.
- **`elves`** — load for long unattended multi-batch execution ("run overnight", "I'm going offline", "implement this plan end-to-end", "keep going without me"). Breaks a plan into sprint-sized batches, implements with testing and PR-based review, and documents everything for compaction recovery.

Skip the heavy workflow for small mechanical edits, but preserve the non-negotiable rules below.

## Non-negotiable engineering rules

- Prefer simple, direct code; flat, explicit control flow.
- No speculative abstractions. Every abstraction must encode an invariant, hide real complexity, or localize change.
- Shape hot code around data flow and dense memory; prefer arrays/spans/IDs over pointer graphs.
- Make ownership explicit in API and types. Prefer values/references/spans before pointers; unique ownership before shared.
- No hidden allocation, I/O, blocking, threads, throwing, or retries behind innocent-looking names.
- In C++ systems APIs: `[[nodiscard]] bool noexcept` for simple expected failure; `Result<T, E> noexcept` when the reason matters.
- Fail fast on invariant violations and semantic corruption. Degrade only through designed, bounded, observable fallback. Never silently fallback.
- Add or update tests with every behavior change; add benchmarks with every performance claim.
- Report verification truthfully — including what was not run, and why.

## Required workflow for complex changes

Explore → Map → State invariants → State failure model → Plan minimal change → Implement narrowly → Verify → Self-review → Simplify → Report.

For architecturally significant work, run the `strategic-engineering-planner` planning loop first and stop after the roadmap for review. Once the design is locked, use `implementation-plan` to produce an executable `IMPLEMENTATION_PLAN.md` before writing code. For overnight or unattended runs, drive execution through `elves` (batched plan execution) or `auto-research` (metric optimization loop).

## Before implementing nontrivial code — state briefly

- **Invariants** — what must always hold.
- **Ownership model** — for each value crossing a boundary: owned, borrowed (required/optional), shared, transferred, arena-scoped.
- **Failure model** — invariant violation → fail fast; semantic corruption → fail the operation; expected external failure → bounded retry/degrade with budget, log, metric.
- **Performance model** (hot paths only) — allocations, copies, locks/atomics, syscalls/I/O, cache locality, batching, complexity, measurement plan.

## Review output

Verdict; top risks (correctness, safety, performance, operability, maintainability); invariant gaps; ownership/lifetime issues; failure semantics; data/performance risks; complexity issues; test/benchmark gaps; minimal staged redesign; exact verification required before merge. No style nits.

## Verification discipline

Use the strongest available local signal: type checker, focused unit/regression tests, property tests for invariants, fuzzing for parsers/protocols, sanitizers (ASan/UBSan/TSan) for unsafe code, integration tests for boundaries, benchmarks for hot paths. Never claim verification not performed — if a check cannot be run, say why and give the exact command for the user to run.
