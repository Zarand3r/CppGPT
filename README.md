# cppgpt

A canonical **GPT-2** transformer implemented from scratch in C++ — training and
inference — using only the C++ standard library. CPU-first and feature-complete
in v1; a from-scratch CUDA backend is a designed-in future phase. See
[`PLAN.md`](PLAN.md) for the full design and [`ROADMAP.md`](ROADMAP.md) for the
build-order checklist (the next step is the first unchecked box).

## Build & test

The build is hermetic: Bazel is pinned in `.bazelversion` (fetched by
`bazelisk`), the C/C++ toolchain is a pinned LLVM/Clang 20 with statically-linked
libc++, and Python is a pinned 3.12 interpreter — none come from the host.

```sh
bazel build //...                 # build everything
bazel test  //...                 # build + run tests
bazel test  //... --config=dev    # with ASan + UBSan
bazel build //... --config=release # -O3 -march=native
bazel run   //scripts:env_info    # show the hermetic Python interpreter
```

No external runtime dependencies: the shipped binary links only `libc`, `libm`,
and the dynamic loader (libc++ is static). Verify with `ldd bazel-bin/...`.

## Run

```sh
# 1. get a corpus (downloads TinyShakespeare, ~1 MB; does not tokenize)
scripts/prepare_shakespeare.py data/shakespeare.txt

# 2. tokenize it -> uint16 token .bin + a .vocab sidecar
bazel run //tools:prepare -- data/shakespeare.txt data/shakespeare.bin

# 3. train from the mmap'd tokens, checkpointing (resumes if the .ckpt exists)
bazel run //tools:train -- data/shakespeare.bin 2000 data/shakespeare.ckpt

# train from a plain text file instead (tokenized in memory), or "" for a
# small built-in corpus:
bazel run //tools:train -- data/shakespeare.txt 200

# 4. train a baby model in-process and sample text from it
bazel run //tools:generate -- data/shakespeare.txt 400 256

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

## Layout

```
include/cppgpt/   public headers (<cppgpt/...>)
src/              library implementation (cc_library //:cppgpt)
tests/            std-only harness (//tests:check) + unit/ · integration/ · fixtures/
scripts/          Python oracle / data scripts (py_binary, py_test)
tools/            CLI binaries (train, prepare, generate, bench)
third_party/      intentionally empty (no third-party runtime deps)
```
