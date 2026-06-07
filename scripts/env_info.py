"""Reports the Bazel-provided (hermetic) Python interpreter.

Run: bazel run //scripts:env_info

This is the Python analogue of the C++ toolchain check: it proves py_binary
works on the pinned interpreter. The real oracle/data scripts (gen_fixtures.py,
convert_gpt2.py, prepare_shakespeare.py) land in this package from M0 onward.
"""

import platform
import sys


def main() -> int:
    print("== cppgpt Python toolchain check ==")
    print(f"version    : {sys.version.splitlines()[0]}")
    print(f"version_info: {tuple(sys.version_info)}")
    print(f"executable : {sys.executable}")
    print(f"platform   : {platform.platform()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
