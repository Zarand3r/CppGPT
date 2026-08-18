#!/usr/bin/env python3
"""Ship a cppgpt training run to Weights & Biases.

Dev tooling, deliberately a SIDECAR rather than a feature of //tools:train. The
C++ binaries link only libc/libm and that is a constitution-level invariant; a
training loop that opens a network connection would be the first runtime
dependency in the repo. So train writes an append-only CSV (--log-csv) and this
process reads it and talks to W&B. Training does not know W&B exists, cannot be
slowed by it, and cannot fail because of it.

Usage:
    .venv/bin/python3 tools/wandb_log.py --project cppgpt -- \\
        bazel-out/k8-opt/bin/tools/train --data ... --log-csv data/runs/x.csv ...

Everything after `--` is the training command, run unchanged. Its stdout is
passed through, so you still watch the run exactly as before.

Failure model, chosen so a dashboard can never cost you a 40-minute run:
  * W&B setup failures (auth, network, bad project) abort BEFORE training starts
    — discovering them 40 minutes in is the expensive case.
  * Once training is running, a logging error is reported and the run continues.
    Training is the product; the dashboard is not.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time

# Flags worth recording as hyperparameters, with the type to read them as.
CONFIG_FLAGS = {
    "layers": int, "heads": int, "embd": int, "ctx": int, "batch": int,
    "steps": int, "lr": float, "min-lr": float, "warmup": int, "clip": float,
    "seed": int, "sample": str, "data": str, "val": str, "init-from": str,
}


def parse_train_argv(argv: list[str]) -> tuple[dict, str | None]:
    """Pull hyperparameters and the --log-csv path out of the training command."""
    cfg: dict = {}
    csv_path = None
    for i, tok in enumerate(argv):
        if not tok.startswith("--") or i + 1 >= len(argv):
            continue
        name, val = tok[2:], argv[i + 1]
        if name == "log-csv":
            csv_path = val
        elif name in CONFIG_FLAGS:
            try:
                cfg[name.replace("-", "_")] = CONFIG_FLAGS[name](val)
            except ValueError:
                cfg[name.replace("-", "_")] = val
    return cfg, csv_path


def row_to_metrics(row: dict[str, str]) -> tuple[int, dict]:
    """One CSV row -> (step, metrics). val_loss is absent on non-eval steps and is
    omitted rather than carried forward, so the W&B chart shows the points that
    were actually measured."""
    step = int(row["step"])
    m = {
        "train/loss": float(row["loss"]),
        "train/lr": float(row["lr"]),
        "train/grad_norm": float(row["grad_norm"]),
        "train/tokens": int(row["tokens"]),
        "perf/tokens_per_s": float(row["tok_per_s"]),
    }
    if row.get("val_loss"):
        m["val/loss"] = float(row["val_loss"])
    return step, m


def backfill(args) -> int:
    """Ship a completed run's CSV. Marked in the config so a backfilled run is
    never mistaken for one that was tracked live."""
    import wandb

    cfg = {"backfilled": True}
    for kv in args.config:
        k, _, v = kv.partition("=")
        for cast in (int, float):
            try:
                cfg[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            cfg[k] = v

    with open(args.backfill) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"error: {args.backfill} has no rows", file=sys.stderr)
        return 1

    run = wandb.init(entity=args.entity, project=args.project, name=args.name,
                     notes=args.notes, config=cfg)
    best = None
    last = {}
    for row in rows:
        try:
            step, m = row_to_metrics(row)
        except (ValueError, KeyError):
            continue
        run.log(m, step=step)
        last = m
        if "val/loss" in m and (best is None or m["val/loss"] < best):
            best = m["val/loss"]
    if best is not None:
        run.summary["val/best_loss"] = best
    run.summary["train/final_loss"] = last.get("train/loss")
    run.summary["train/tokens"] = last.get("train/tokens")
    print(f"  [wandb] backfilled {len(rows)} rows"
          + (f", best val {best:.4f}" if best is not None else "")
          + f" -> {run.url}", flush=True)
    run.finish()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="cppgpt")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    ap.add_argument("--name", default=None, help="run name (defaults to W&B's)")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--backfill", default=None, metavar="CSV",
                    help="ship an EXISTING csv instead of running training. For runs that "
                         "finished before this wrapper existed; the alternative is citing a "
                         "link that does not correspond to the model.")
    ap.add_argument("--config", action="append", default=[], metavar="K=V",
                    help="config entry for --backfill, repeatable (the flags cannot be "
                         "recovered from the CSV)")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    # Backfill takes no training command, so it must dispatch BEFORE the guard
    # that requires one.
    if args.backfill:
        return backfill(args)

    if not rest:
        print("error: no training command given (put it after `--`)", file=sys.stderr)
        return 2

    cfg, csv_path = parse_train_argv(rest)
    if csv_path is None:
        # This is the interface between the two processes. Without it there is
        # nothing to ship, and silently running training unlogged would defeat
        # the point of having invoked this wrapper at all.
        print("error: the training command must include --log-csv <path>; that file is "
              "what this wrapper reads", file=sys.stderr)
        return 2

    import wandb  # imported here so --help works without it installed

    # Start the W&B run BEFORE training: an auth or network failure must surface
    # now, not after a 40-minute run has already finished.
    run = wandb.init(entity=args.entity, project=args.project, name=args.name,
                     notes=args.notes, config=cfg)
    print(f"\n  W&B run: {run.url}\n", flush=True)

    # train APPENDS to the CSV, so a re-used path already holds earlier runs.
    # Seek past them: this run must log its own rows and no one else's.
    start_at = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0

    proc = subprocess.Popen(rest, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    def pump() -> None:  # pass training's own output through unchanged
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

    t = threading.Thread(target=pump, daemon=True)
    t.start()

    # Read the header BEFORE seeking past the existing rows. It lives on line 1,
    # which is inside the region we skip on a re-used path — without this, every
    # row of an appended run is discarded for having no header to zip against, and
    # the dashboard comes out silently empty. (Found by testing exactly that: the
    # first smoke test used a fresh file and passed.)
    header: list[str] | None = None
    if start_at > 0:
        with open(csv_path, "r") as f:
            first = f.readline().strip()
        if first.startswith("step,"):
            header = first.split(",")

    logged = 0
    best_val: float | None = None
    last: dict = {}
    pos = start_at
    pending = ""

    def drain() -> None:
        nonlocal header, logged, best_val, last, pos, pending
        if not os.path.exists(csv_path):
            return
        with open(csv_path, "r") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        pending += chunk
        # Only whole lines are complete rows; train flushes per row, but a read can
        # still land mid-write.
        *lines, pending = pending.split("\n")
        for line in lines:
            if not line.strip():
                continue
            fields = line.split(",")
            if fields[0] == "step":
                header = fields
                continue
            if header is None or len(fields) != len(header):
                continue
            row = dict(zip(header, fields))
            try:
                step, m = row_to_metrics(row)
            except (ValueError, KeyError):
                continue
            try:
                run.log(m, step=step)
            except Exception as e:  # noqa: BLE001 - never kill training for this
                print(f"  [wandb] log failed at step {step}: {e}", file=sys.stderr)
                return
            logged += 1
            last = m
            if "val/loss" in m and (best_val is None or m["val/loss"] < best_val):
                best_val = m["val/loss"]

    while proc.poll() is None:
        drain()
        time.sleep(1.0)
    t.join(timeout=10)
    drain()  # whatever landed after the last poll

    if best_val is not None:
        run.summary["val/best_loss"] = best_val
    if last:
        run.summary["train/final_loss"] = last.get("train/loss")
        run.summary["train/tokens"] = last.get("train/tokens")
        run.summary["perf/tokens_per_s"] = last.get("perf/tokens_per_s")
    run.summary["exit_code"] = proc.returncode
    print(f"  [wandb] logged {logged} points"
          + (f", best val {best_val:.4f}" if best_val is not None else "")
          + f" -> {run.url}", flush=True)
    run.finish(exit_code=0 if proc.returncode == 0 else 1)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
