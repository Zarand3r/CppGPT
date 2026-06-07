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
tests/            std-only test harness (//tests:check) + unit/ tests
scripts/          Python oracle / data scripts (py_binary, py_test)
tools/            CLI binaries (train, generate, bench, verify) — from M1
third_party/      intentionally empty (no third-party runtime deps)
```
