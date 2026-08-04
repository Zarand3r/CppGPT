# cppgpt

A canonical **GPT-2** transformer implemented from scratch in C++ — train, sample,
and fine-tune your own model — using only the C++ standard library.

**Scope (v1 / MVP):** GPT-2's *architecture*, not its scale. The forward pass,
backward pass and AdamW step are verified against PyTorch to ~1e-6, so the model
is a real GPT-2 — just small, and trained on your own text with a character-level
tokenizer. **Loading OpenAI's pretrained 124M/350M weights and the byte-level BPE
tokenizer are deliberately out of MVP scope** (designed and shelved, see
[`ROADMAP.md`](ROADMAP.md) → Deferred). CPU-first; a from-scratch CUDA backend is a
designed-in future phase.

See [`ROADMAP.md`](ROADMAP.md) for what is done and what is next, and
[`PLAN.md`](PLAN.md) for the design rationale.

## Build & test

The build is hermetic: Bazel is pinned in `.bazelversion` (fetched by
`bazelisk`), the C/C++ toolchain is a pinned LLVM/Clang 20 with statically-linked
libc++, and Python is a pinned 3.12 interpreter — none come from the host.

```sh
bazel build //...                 # build everything
bazel test  //...                 # build + run tests
bazel test  //... --config=dev    # with ASan + UBSan
bazel build //... --config=release # -O3, explicit ISA baseline (see docs/DECISIONS.md D1)
bazel run   //scripts:env_info    # show the hermetic Python interpreter
```

No external runtime dependencies: the shipped binary links only `libc`, `libm`,
and the dynamic loader (libc++ is static). Verify with `ldd bazel-bin/...`.

## Run

> `bazel run` sets the working directory to the target's **runfiles** dir, not the workspace root,
> so paths passed after `--` must be absolute (`"$PWD"/...`). A bare relative path silently reads or
> writes inside the runfiles tree.

```sh
# 1. get a corpus (downloads TinyShakespeare, ~1 MB; does not tokenize)
scripts/prepare_shakespeare.py data/shakespeare.txt

# 2. tokenize it -> uint16 token .bin + a .vocab sidecar
bazel run //tools:prepare -- "$PWD"/data/shakespeare.txt "$PWD"/data/shakespeare --val-frac 0.1

# 3. train from the mmap'd tokens, checkpointing (resumes if the .ckpt exists)
bazel run //tools:train -- --data "$PWD"/data/shakespeare.train.bin \
    --val "$PWD"/data/shakespeare.val.bin --layers 4 --heads 4 --embd 128 --ctx 128 \
    --steps 2000 --eval-interval 200 --ckpt "$PWD"/data/shakespeare.ckpt

# train from a plain text file instead (tokenized in memory), or "" for a
# small built-in corpus:
bazel run //tools:train -- --data "$PWD"/data/shakespeare.txt --steps 200

# 4. train a baby model in-process and sample text from it
bazel run //tools:generate -- "$PWD"/data/shakespeare.txt 400 256

# matmul throughput (always measure in release)
bazel run --config=release //tools:bench -- 20
```

`prepare` writes `<out.bin>` plus a `<out.bin>.vocab` sidecar; `train` needs the
sidecar to size the model, and never infers the vocabulary by scanning tokens.
Tokenization lives only in the C++ `CharTokenizer`, so there is no second
tokenizer that can drift out of parity.

Checkpoints carry the weights plus the AdamW moments and step counter. Resume
restores those exactly, but the RNG and dataloader position are not saved, so a
resumed run replays the data order and LR schedule from step 0.

## Prerequisites

- A C++ host able to run the LLVM toolchain (Linux/x86-64; v1 is Linux-only).
- `bazelisk` on `PATH` (reads `.bazelversion`).

### Ubuntu 26.04 (and other very new distros): `libxml2.so.2`

The pinned LLVM 20 `ld.lld` load-links `libxml2.so.2`, but recent Ubuntu ships
only `libxml2.so.16`. `lld` never *calls* libxml2 during ELF linking, so any
`libxml2.so.2` satisfies the loader. One-time fix:

```sh
sudo ln -s libxml2.so.16 /usr/lib/x86_64-linux-gnu/libxml2.so.2
```

(The hermetic toolchain cannot patch a load-time dependency of its own bundled
linker, so this host symlink is unavoidable on bleeding-edge distros. It is
harmless — the linker prints a benign "no version information available" warning
and proceeds.)

## Document map

Each fact has **one owner**; every other mention links to it. Stating the same fact twice is how the
docs drifted before (see `docs/review-audit.md`).

| Document | Owns — and only this |
|---|---|
| [`docs/constitution.md`](docs/constitution.md) | Frozen human-authored deal-breakers. **Never edited by an agent.** |
| [`ROADMAP.md`](ROADMAP.md) | **The single source of truth for what has to be done** — milestones, the MVP gate, deferred scope, and all outstanding debt. The only file allowed to contain `[x]`/`[ ]`, and the only one that holds work items. |
| [`PLAN.md`](PLAN.md) | Design rationale, options, tradeoffs, deferred-complexity triggers. Risk IDs are `DR-n`. No status, no measurements. |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Locked decisions — what was decided, the evidence, and what it cost. |
| [`docs/measurements.md`](docs/measurements.md) | **Every measured number**, each with the command that reproduces it. No other doc states a measurement. |
| [`docs/M3_INFERENCE_PLAN.md`](docs/M3_INFERENCE_PLAN.md) | Execution detail for the active milestone (slices, gates, fixtures). IDs are `M3-*`. |
| [`docs/engineering-lessons.md`](docs/engineering-lessons.md) | Rules distilled from real defects here, each citing its incident. |
| [`docs/review-audit.md`](docs/review-audit.md) | Mechanical sweeps and their results. |
| `README.md` | Build, run, layout — every command names a target that exists. |
| `CLAUDE.md` | Agent wiring, engineering rules, process obligations. |

## Layout

```
include/cppgpt/   public headers (<cppgpt/...>)
src/              library implementation (cc_library //:cppgpt)
tests/            std-only harness (//tests:check) + unit/ · integration/ · fixtures/
scripts/          Python oracle / data scripts (py_binary, py_test)
tools/            CLI binaries (train, prepare, generate)
third_party/      intentionally empty (no third-party runtime deps)
```
