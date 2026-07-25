#!/usr/bin/env python3
"""Download the TinyShakespeare corpus for cppgpt training.

DEV-ONLY convenience: fetches the raw ``input.txt`` (~1 MB) to a local path. It
does NOT tokenize — tokenization lives in one place, the C++ ``CharTokenizer``,
so the ``.vocab`` stays authoritative and there is no second tokenizer to drift.
After downloading, produce the training files with the C++ tool:

    scripts/prepare_shakespeare.py data/shakespeare.txt
    bazel run //tools:prepare -- data/shakespeare.txt data/shakespeare.bin
    bazel run //tools:train  -- data/shakespeare.bin 2000 data/shakespeare.ckpt

Usage: prepare_shakespeare.py [out.txt]   (default: data/shakespeare.txt)
"""

import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/shakespeare.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}\n       -> {out}")
    with urllib.request.urlopen(URL) as resp:  # noqa: S310 (trusted, fixed URL)
        data = resp.read()
    out.write_bytes(data)
    print(f"wrote {len(data)} bytes. Next: bazel run //tools:prepare -- {out} {out.with_suffix('.bin')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
