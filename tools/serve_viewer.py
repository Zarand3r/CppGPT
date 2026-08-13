#!/usr/bin/env python3
"""Serve the interpretability viewer and run the model on submitted prompts.

Dev tooling, not part of the shipped artifact: the C++ binaries still link only
libc/libm, so the no-runtime-dependency invariant is untouched. This is a thin
front door that shells out to //tools:inspect.

Security notes, because this is intended to sit behind a PUBLIC Tailscale Funnel:

* The prompt is passed to inspect as a single argv element via exec with a list
  argument -- never through a shell -- so shell metacharacters in a prompt are
  inert. There is no string interpolation into a command anywhere in this file.
* Prompts are length-capped and validated against the model's own vocabulary
  BEFORE inspect runs. Without that check an out-of-vocab byte reaches
  CharTokenizer::encode, whose contract is to abort on an unknown byte -- a
  crash on every request containing an emoji.
* Every knob inspect exposes is fixed here or clamped; a client cannot choose
  output paths, layer selections, or size ceilings.
* Each run is wall-clock bounded and the output path is a server-chosen temp file.

Usage: serve_viewer.py --checkpoint X.ckpt --vocab X.vocab [--port 8092]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Overridden at startup from the checkpoint's own max_seq_len: a prompt longer
# than the context cannot be served, and paying a fork/exec + checkpoint load to
# discover that returned HTTP 500 for what is a client error.
MAX_PROMPT_CHARS = 256
MAX_NEW_TOKENS = 0          # inspect does not generate; kept explicit for clarity
INSPECT_TIMEOUT_S = 30
MAX_BODY_BYTES = 64 * 1024
TOP_K = 8

ARGS: argparse.Namespace
VOCAB: set[str] = set()
RUN_LOCK = threading.Semaphore(2)   # bound concurrent model runs on a shared box


def run_inspect(prompt: str) -> tuple[int, dict | str]:
    """Run one forward pass. Returns (http_status, json_or_error_message)."""
    if not prompt:
        return 400, "Empty prompt."
    if len(prompt) > MAX_PROMPT_CHARS:
        return 400, f"Prompt is {len(prompt)} characters; the limit is {MAX_PROMPT_CHARS}."
    unknown = sorted({c for c in prompt if c not in VOCAB})
    if unknown:
        shown = " ".join(repr(c) for c in unknown[:12])
        return 400, (
            f"These characters are not in this model's vocabulary: {shown}. "
            "It is a character-level model trained on one corpus, so it only knows "
            "the characters that corpus contained."
        )

    if not RUN_LOCK.acquire(timeout=20):
        return 503, "Server busy; try again."
    try:
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "run.json"
            # List form: no shell, so the prompt cannot be interpreted as syntax.
            cmd = [
                str(ARGS.inspect), "--checkpoint", str(ARGS.checkpoint),
                "--vocab", str(ARGS.vocab), "--prompt", prompt,
                "--out", str(out), "--top-k", str(TOP_K),
            ]
            try:
                p = subprocess.run(cmd, capture_output=True, timeout=INSPECT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return 504, "The model took too long."
            if p.returncode != 0 or not out.exists():
                # inspect's own stderr is the useful part; it names its own errors.
                msg = p.stderr.decode("utf-8", "replace").strip() or "inspect failed"
                return 500, msg.splitlines()[0][:400]
            return 200, json.loads(out.read_text())
    finally:
        RUN_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Without this, rfile.read()/readline() block forever and ThreadingHTTPServer
    # spawns an unbounded thread per connection: 800 half-open POSTs held 801
    # threads and 804 fds indefinitely. StreamRequestHandler.setup() applies it
    # to the socket. The Semaphore below bounds model RUNS; this bounds
    # CONNECTIONS, which is a different resource.
    timeout = 10

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html", "/cppgpt", "/cppgpt/"):
            html = (ARGS.site / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path.endswith("/run.json"):
            f = ARGS.site / "run.json"
            if f.exists():
                self._send(200, f.read_bytes(), "application/json")
            else:
                self._send(404, b'{"error":"no seed dump"}', "application/json")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.split("?", 1)[0].endswith("/api/inspect"):
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY_BYTES:
            self._send(400, b'{"error":"bad request body"}', "application/json")
            return
        try:
            req = json.loads(self.rfile.read(n))
            prompt = req.get("prompt", "")
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
        except Exception:
            self._send(400, b'{"error":"malformed JSON"}', "application/json")
            return

        status, result = run_inspect(prompt)
        if status == 200:
            self._send(200, json.dumps(result).encode(), "application/json")
        else:
            self._send(status, json.dumps({"error": result}).encode(), "application/json")

    def log_message(self, fmt: str, *a) -> None:  # quieter, and no prompt echo
        # getattr, not self.path: log_message is also reached via send_error ->
        # log_error BEFORE the request line is parsed, so a malformed request
        # raised AttributeError, killed the response, and flooded the journal
        # with tracebacks (measured 1867x amplification, no MemoryMax backstop).
        path = getattr(self, "path", "-").split("?")[0]
        print(f"  {getattr(self, 'command', '-')} {path} -> {a[1] if len(a) > 1 else ''}")


def main() -> int:
    global ARGS, VOCAB
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--vocab", type=pathlib.Path, required=True)
    ap.add_argument("--inspect", type=pathlib.Path, default=pathlib.Path("bazel-bin/tools/inspect"))
    ap.add_argument("--site", type=pathlib.Path, default=pathlib.Path("site"))
    ap.add_argument("--port", type=int, default=8092)
    ARGS = ap.parse_args()

    for p, what in ((ARGS.checkpoint, "checkpoint"), (ARGS.vocab, "vocab"),
                    (ARGS.inspect, "inspect binary"), (ARGS.site / "index.html", "site/index.html")):
        if not p.exists():
            print(f"error: {what} not found at {p}")
            return 1
    VOCAB = set(ARGS.vocab.read_text(encoding="utf-8", errors="replace"))

    # Read max_seq_len straight out of the checkpoint header (offset 8, the third
    # int32 of the 64-byte header) and cap prompts there, so an over-long prompt
    # is a 400 before any work happens.
    global MAX_PROMPT_CHARS
    try:
        import struct
        head = ARGS.checkpoint.read_bytes()[:64]
        ctx = struct.unpack_from("<i", head, 8)[0]
        if 0 < ctx < MAX_PROMPT_CHARS:
            MAX_PROMPT_CHARS = ctx
    except Exception:
        pass  # keep the conservative default; the tool re-checks anyway
    print(f"serving {ARGS.site} on 127.0.0.1:{ARGS.port} "
          f"({len(VOCAB)} vocab chars, inspect={shutil.which(str(ARGS.inspect)) or ARGS.inspect})")
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
