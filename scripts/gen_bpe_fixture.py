#!/usr/bin/env python3
"""Build the BPE differential fixture from TWO independent oracles.

Gate 1 for the C++ tokenizer is an EQUALITY on token streams, not a similarity
score: a near-miss pre-tokenizer still produces fluent-looking output, so
"close" tells you nothing.

Two oracles, and they are required to agree with each other before either is
trusted. tiktoken is OpenAI's own implementation; transformers is HuggingFace's.
They share no code. If they disagree, the fixture is wrong and no C++ result
should be believed against it.

Writes tests/fixtures/bpe_gpt2.bin:
    u32 magic, u32 n_cases
    per case: u32 text_len, text bytes, u32 n_ids, i32 ids[n_ids]
"""
import struct
import sys
from pathlib import Path

MAGIC = 0x42504531  # "BPE1"

# Cases chosen for the things that break BPE implementations, not for coverage
# of ordinary prose.
CASES = [
    "",                                   # empty
    " ",                                  # a lone space
    "\n",                                 # a lone newline
    "hello world",                        # the trivial case
    " hello",                             # leading space joins the word
    "hello ",                             # trailing space is its own token
    "Hello World",                        # capitalisation changes the merge path
    "The lover of Juliet",                # the prompt that started this
    "ROMEO:\nWhat is",                    # newline mid-prompt
    "it's",  "it'll", "it've", "it're", "it'm", "it'd", "it't",   # every contraction rule
    "IT'S",                               # the contraction rules are case-sensitive
    "123", "1234567890", " 42", "3.14159",                        # digit runs
    "a  b", "a   b", "a\t\tb", "a\n\nb",  # whitespace runs (the \s+(?!\S) branch)
    "  leading", "trailing  ",
    "!!!", "?!", "...", "(a)", "[b]", "{c}",                      # punctuation runs
    "a1b2c3",                             # letter/number alternation
    "don't stop believin'",
    "GPT-2 is a transformer.",
    "x" * 200,                            # long single word
    "The quick brown fox jumps over the lazy dog. " * 5,
    "def f(x):\n    return x + 1\n",      # code, indentation
    "a,b,c;d:e",
]


def main() -> int:
    import tiktoken
    from transformers import GPT2TokenizerFast

    tik = tiktoken.get_encoding("gpt2")
    hf = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    out = Path("tests/fixtures/bpe_gpt2.bin")
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = bytearray()
    buf += struct.pack("<II", MAGIC, len(CASES))

    disagree = 0
    for text in CASES:
        a = tik.encode(text, allowed_special=set())
        b = hf.encode(text)
        if a != b:
            # Never write a fixture the two oracles disagree on: it would gate the
            # C++ against a number neither implementation actually produces.
            print(f"  ORACLE DISAGREEMENT on {text!r}:\n    tiktoken={a}\n    hf      ={b}")
            disagree += 1
            continue
        raw = text.encode("utf-8")
        buf += struct.pack("<I", len(raw)) + raw
        buf += struct.pack("<I", len(a)) + b"".join(struct.pack("<i", i) for i in a)

    if disagree:
        print(f"  {disagree} disagreements — refusing to write a fixture built on them")
        return 1
    # Rewrite the count in case any case was skipped (it cannot be, given the
    # check above, but the file must never claim more cases than it holds).
    buf[4:8] = struct.pack("<I", len(CASES))
    out.write_bytes(bytes(buf))
    total = sum(len(tik.encode(c, allowed_special=set())) for c in CASES)
    print(f"  wrote {out} — {len(CASES)} cases, {total} tokens, {len(buf)} bytes")
    print("  both oracles agree on every case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
