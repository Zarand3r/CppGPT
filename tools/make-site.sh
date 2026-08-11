#!/usr/bin/env bash
# Assemble the static site served over Tailscale Funnel.
#
# Deliberately copies into a DEDICATED directory rather than serving the repo:
# only these two files are exposed, so no source, checkpoints or corpora are.
# tools/viewer.html is the single source — this never edits it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUMP="${1:-$ROOT/data/run.json}"
OUT="$ROOT/site"
[ -s "$DUMP" ] || { echo "error: no dump at '$DUMP' (run //tools:inspect first)" >&2; exit 1; }
mkdir -p "$OUT"
cp "$ROOT/tools/viewer.html" "$OUT/index.html"
cp "$DUMP" "$OUT/run.json"
echo "site/ built from $DUMP — serve with: python3 -m http.server 8092 --bind 127.0.0.1 -d $OUT"
