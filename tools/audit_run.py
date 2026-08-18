#!/usr/bin/env python3
"""Audit a training run's metrics for bug signatures.

"The loss went down" is not evidence that training was correct. This checks the
shapes that a defect actually shows up in — a schedule that never reaches its
floor, a gradient norm pinned at the clip threshold, throughput that collapses
mid-run, a loss that starts somewhere other than ln(V) — each of which is
invisible in the headline number.

Every check prints PASS/WARN/FAIL with the measured value, so a WARN can be
judged rather than merely noticed.

Usage: audit_run.py data/runs/x.csv [--vocab 65] [--clip 1.0] [--min-lr 3e-4]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys

FAILED = 0


def say(level: str, name: str, detail: str) -> None:
    global FAILED
    if level == "FAIL":
        FAILED += 1
    print(f"  [{level:4}] {name:<26} {detail}")


def median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--vocab", type=int, default=65)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--min-lr", type=float, default=None)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv)) if r.get("step")]
    if not rows:
        print("no rows")
        return 1
    step = [int(r["step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    lr = [float(r["lr"]) for r in rows]
    gn = [float(r["grad_norm"]) for r in rows]
    tps = [float(r["tok_per_s"]) for r in rows]
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r.get("val_loss")]

    print(f"\naudit: {a.csv}  ({len(rows)} logged steps, {step[-1]} total)")

    # 1. Non-finite anywhere. A NaN that reaches the log means the run diverged
    #    and kept going, which no amount of curve-reading will reveal.
    bad = sum(1 for xs in (loss, lr, gn, tps) for x in xs if not math.isfinite(x))
    say("PASS" if bad == 0 else "FAIL", "all values finite",
        f"{bad} non-finite" if bad else "no NaN/Inf in loss, lr, grad_norm, tok/s")

    # 2. Initial loss must be ~ln(V): an untrained softmax over V symbols is
    #    uniform. Starting far below means the weights were not fresh; far above
    #    means the init or the loss is wrong.
    exp0 = math.log(a.vocab)
    d0 = abs(loss[0] - exp0)
    say("PASS" if d0 < 0.5 else "WARN", "initial loss ~ ln(vocab)",
        f"{loss[0]:.4f} vs ln({a.vocab})={exp0:.4f}  (delta {d0:.3f})")

    # 3. Steps strictly increasing and unique — duplicates mean two runs share a
    #    log and every aggregate over it is wrong.
    inc = all(b > x for x, b in zip(step, step[1:]))
    say("PASS" if inc else "FAIL", "steps strictly increasing",
        "ok" if inc else "duplicate or out-of-order steps")

    # 4. Loss spikes. Training is noisy, but a jump of >50% between logged points
    #    is instability, not noise.
    spikes = [(step[i], loss[i - 1], loss[i])
              for i in range(1, len(loss)) if loss[i] > loss[i - 1] * 1.5]
    say("PASS" if not spikes else "WARN", "no loss spikes",
        "none >50% between points" if not spikes
        else f"{len(spikes)} spikes, first at step {spikes[0][0]} "
             f"({spikes[0][1]:.3f}->{spikes[0][2]:.3f})")

    # 5. LR schedule: must warm up and then decay. A schedule stuck at its peak
    #    (or already at the floor) is a silently mis-wired cosine.
    peak_i = max(range(len(lr)), key=lambda i: lr[i])
    decayed = lr[-1] < lr[peak_i] * 0.5
    say("PASS" if decayed else "FAIL", "lr warms up then decays",
        f"peak {lr[peak_i]:.2e} at step {step[peak_i]}, final {lr[-1]:.2e}")
    if a.min_lr is not None:
        ok = abs(lr[-1] - a.min_lr) / a.min_lr < 0.05
        say("PASS" if ok else "WARN", "lr reaches its floor",
            f"final {lr[-1]:.2e} vs --min-lr {a.min_lr:.2e}")

    # 6. Gradient clipping. The logged norm is PRE-clip, so the fraction above the
    #    threshold says how often clipping bound the step. Persistent saturation
    #    means the LR is too high for the clip, and the schedule is not really in
    #    control of the step size.
    over = sum(1 for g in gn if g > a.clip)
    frac = over / len(gn)
    say("PASS" if frac < 0.25 else "WARN", "grad clipping not saturated",
        f"{100*frac:.1f}% of steps above clip {a.clip} "
        f"(median |g| {median(gn):.3f}, max {max(gn):.3f})")

    # 7. Throughput stability. A collapsing tok/s is thermal throttling, swap, or
    #    another process — and it silently changes what "9000 steps" cost.
    tail = tps[len(tps) // 4:]  # skip the warmup ramp, which is not a regression
    mu = sum(tail) / len(tail)
    sd = (sum((x - mu) ** 2 for x in tail) / len(tail)) ** 0.5
    cv = sd / mu if mu else 0.0
    say("PASS" if cv < 0.10 else "WARN", "throughput stable",
        f"{mu:.0f} tok/s +/- {100*cv:.1f}%  (min {min(tail):.0f})")

    # 8. Val must actually improve on its own starting point, or nothing was
    #    learned regardless of what the train curve did.
    if val:
        first, best = val[0][1], min(v for _, v in val)
        say("PASS" if best < first else "FAIL", "val improved",
            f"{first:.4f} -> best {best:.4f} ({100*(first-best)/first:.1f}% better)")
        gap = loss[-1] - 0
        say("PASS", "final train/val gap",
            f"{val[-1][1] - loss[-1]:+.3f} nats (train {loss[-1]:.4f}, val {val[-1][1]:.4f})")
    else:
        say("WARN", "val present", "no val_loss rows — nothing checks generalisation")

    print(f"\n  {'AUDIT PASSED' if FAILED == 0 else f'AUDIT FAILED ({FAILED} checks)'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
