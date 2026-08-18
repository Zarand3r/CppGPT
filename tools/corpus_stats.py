#!/usr/bin/env python3
"""What does the corpus say should follow this text?

Before concluding a model is wrong, check what it was actually trained on. This
answers "should it predict X here?" from the data rather than from intuition —
the question that showed 'romeo'/'juliet' appear ZERO times in lowercase in
TinyShakespeare, so a model declining to complete 'ju' -> 'l' was correct.

Usage: corpus_stats.py <corpus.txt> <context> [more contexts...] [--top 8]
"""
from __future__ import annotations

import argparse
import collections
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("context", nargs="+")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    text = open(a.corpus, encoding="utf-8", errors="replace").read()
    print(f"{a.corpus}: {len(text):,} characters\n")
    for ctx in a.context:
        if not ctx:
            continue
        nxt = collections.Counter(
            text[i + len(ctx)] for i in range(len(text) - len(ctx))
            if text[i:i + len(ctx)] == ctx)
        total = sum(nxt.values())
        print(f"  {ctx!r}: {total} occurrences", end="")
        if total == 0:
            # The most useful answer this tool gives: the context never appears,
            # so ANY prediction after it is extrapolation.
            print("  <- never appears; any continuation is extrapolation")
            continue
        print()
        for ch, n in nxt.most_common(a.top):
            print(f"      {ch!r:<8} {n:>6}  {100*n/total:5.1f}%")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
