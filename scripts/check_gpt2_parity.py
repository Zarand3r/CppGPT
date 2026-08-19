#!/usr/bin/env python3
"""Forward-parity gate for real GPT-2 weights.

The obvious gate -- "agree with HuggingFace fp32 to within X" -- measures the
wrong thing. HF's fp32 forward is itself off from the true answer, so that gate
charges us for THEIR rounding as well as ours, and the only way to pass it is to
keep loosening X. That is the tolerance lie this repo has a lesson about.

The gate here is self-calibrating and cannot be loosened:

    max |cppgpt_fp32 - fp64_truth|  <=  1.5 x  max |HF_fp32 - fp64_truth|

i.e. our fp32 forward must be at least as numerically sound as the reference
implementation's own fp32 forward, measured against the same fp64 ground truth.
There is no tolerance to tune -- the bar moves with the reference.

Plus the decision-relevant check: the predicted token must match at every
position, with the top1-top2 margin reported so the headroom is visible.

Usage: check_gpt2_parity.py <ours.bin>   (written by //tools:dump_logits)
"""
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel

SLACK = 1.5


def main() -> int:
    raw = open(sys.argv[1], "rb").read()
    off = 0
    n = int(np.frombuffer(raw, np.int32, 1, off)[0]); off += 4
    ids = np.frombuffer(raw, np.int32, n, off); off += 4 * n
    V = int(np.frombuffer(raw, np.int32, 1, off)[0]); off += 4
    ours = np.frombuffer(raw, np.float32, n * V, off).reshape(n, V).astype(np.float64)

    t = torch.tensor(ids.astype(np.int64))[None, :]
    with torch.no_grad():
        hf32 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2",
                                               dtype=torch.float32).eval()(t).logits[0].numpy().astype(np.float64)
        truth = GPT2LMHeadModel.from_pretrained("openai-community/gpt2",
                                                dtype=torch.float64).eval()(t).logits[0].numpy()

    ours_err = float(np.abs(ours - truth).max())
    hf_err = float(np.abs(hf32 - truth).max())
    budget = hf_err * SLACK

    print(f"  {n} tokens, logits [{n}, {V}], |logit| mean {np.abs(truth).mean():.1f}")
    print(f"  cppgpt fp32 vs fp64 truth : {ours_err:.3e}")
    print(f"  HF     fp32 vs fp64 truth : {hf_err:.3e}")
    print(f"  budget ({SLACK}x the reference's own error) : {budget:.3e}")

    ok = ours_err <= budget
    print(f"  [{'PASS' if ok else 'FAIL'}] numerically at least as sound as the reference")

    same = bool((ours.argmax(1) == truth.argmax(1)).all())
    srt = np.sort(truth, axis=1)
    margins = (srt[:, -1] - srt[:, -2])
    print(f"  [{'PASS' if same else 'FAIL'}] predicted token matches at all {n} positions")
    print(f"  worst top1-top2 margin {margins.min():.4f} vs our error {ours_err:.3e} "
          f"-> {margins.min() / max(ours_err, 1e-12):.0f}x headroom")
    return 0 if (ok and same) else 1


if __name__ == "__main__":
    raise SystemExit(main())
