#!/usr/bin/env python3
"""Summarise (and compare) training runs from --log-csv output.

Written because the same analysis was re-derived inline four times during one
session, twice with a different definition of "is it overfitting" — which is
exactly how two runs end up not being comparable.

The overfitting verdict is a TREND, deliberately, not `final > best`. For any
noisy series the final point is above the minimum, because the minimum is by
definition the luckiest sample; `final > best` is satisfied by pure noise and
proves nothing. This is not a hypothetical: that formulation reported Run B as
overfitting when its trend was flat-to-improving.

Usage: run_report.py data/runs/a.csv [data/runs/b.csv ...]
"""
from __future__ import annotations

import csv
import sys


def load(path: str):
    train, val = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                step = int(r["step"])
            except (KeyError, ValueError):
                continue
            if r.get("loss"):
                train.append((step, float(r["loss"]), r))
            if r.get("val_loss"):
                val.append((step, float(r["val_loss"])))
    return train, val


def trend(points, tail_frac: float = 0.33):
    """Least-squares slope over the last `tail_frac` of the series, in units per
    1000 steps, with the slope's OWN standard error and a t statistic.

    The standard error of the slope is the right yardstick, not the scatter of the
    points: scatter measures how noisy individual evals are, while what we are
    asking is whether the *fitted trend* is distinguishable from zero. Those
    differ by roughly sqrt(n) and by the span of the window, so comparing a slope
    to a point scatter systematically under-calls real drift on long runs and
    over-calls it on short ones.

    Returns (slope, stderr, t, n) or None when there are too few points to fit."""
    if len(points) < 6:
        return None
    cut = points[int(len(points) * (1 - tail_frac)):]
    n = len(cut)
    if n < 4:  # need n-2 >= 2 residual degrees of freedom for a usable stderr
        return None
    mx = sum(s for s, _ in cut) / n
    my = sum(v for _, v in cut) / n
    sxx = sum((s - mx) ** 2 for s, _ in cut)
    if sxx == 0:
        return None
    b = sum((s - mx) * (v - my) for s, v in cut) / sxx      # per step
    a = my - b * mx
    sse = sum((v - (a + b * s)) ** 2 for s, v in cut)
    resid_sd = (sse / (n - 2)) ** 0.5
    se_b = resid_sd / (sxx ** 0.5) if sxx > 0 else float("inf")
    t = b / se_b if se_b > 0 else 0.0
    return b * 1000.0, se_b * 1000.0, t, n


def report(path: str) -> dict:
    train, val = load(path)
    if not train:
        print(f"{path}: no usable rows")
        return {}
    last = train[-1][2]
    out = {
        "path": path,
        "steps": train[-1][0],
        "tokens": int(last.get("tokens", 0)),
        "elapsed_s": float(last.get("elapsed_s", 0.0)),
        "tok_per_s": float(last.get("tok_per_s", 0.0)),
        "final_train": train[-1][1],
    }
    print(f"\n{path}")
    print(f"  {out['steps']} steps · {out['tokens']:,} tokens · "
          f"{out['elapsed_s']/60:.1f} min · {out['tok_per_s']:.0f} tok/s")
    print(f"  final train loss {out['final_train']:.4f}")
    if not val:
        print("  no val_loss rows (train was run without --val / --eval-interval)")
        return out
    best_step, best = min(val, key=lambda x: x[1])
    out.update(best_val=best, best_step=best_step, final_val=val[-1][1],
               gap=val[-1][1] - out["final_train"])
    print(f"  best val  {best:.4f} at step {best_step}")
    print(f"  final val {val[-1][1]:.4f}   train/val gap {out['gap']:.3f} nats")
    t = trend(val)
    if t is None:
        print("  trend: too few eval points to fit")
        return out
    slope, se, tstat, n = t
    out.update(slope=slope, slope_se=se, slope_t=tstat)
    # |t| > 2 is roughly the 95% mark: the fitted trend is distinguishable from
    # a flat line given how much these points scatter.
    verdict = ("OVERFITTING" if tstat > 2 else
               "still improving" if tstat < -2 else "flat (not distinguishable from zero)")
    print(f"  trend over last {n} evals: {slope:+.4f} +/- {se:.4f} nats/1k steps "
          f"(t={tstat:+.1f}) -> {verdict}")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    runs = [r for r in (report(p) for p in sys.argv[1:]) if r]
    if len(runs) > 1:
        print("\ncomparison")
        print(f"  {'run':<28} {'best val':>9} {'final val':>10} {'gap':>7} {'tok/s':>8}")
        for r in runs:
            print(f"  {r['path'].split('/')[-1]:<28} {r.get('best_val', float('nan')):>9.4f} "
                  f"{r.get('final_val', float('nan')):>10.4f} {r.get('gap', float('nan')):>7.3f} "
                  f"{r['tok_per_s']:>8.0f}")
        base = runs[0]
        for r in runs[1:]:
            if "best_val" in base and "best_val" in r:
                d = r["best_val"] - base["best_val"]
                print(f"  {r['path'].split('/')[-1]} vs {base['path'].split('/')[-1]}: "
                      f"{d:+.4f} nats ({100*d/base['best_val']:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
