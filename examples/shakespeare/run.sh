#!/usr/bin/env bash
# Character-level Shakespeare, end to end: corpus -> tokenize -> train -> sample.
# See examples/shakespeare/README.md for what to expect and how to fine-tune.
#
# Run from the repo root:  examples/shakespeare/run.sh [steps]
set -euo pipefail

STEPS="${1:-900}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BAZEL="${BAZEL:-bazel}"
DATA="$ROOT/data"
cd "$ROOT"

command -v "$BAZEL" >/dev/null || { echo "error: '$BAZEL' not found (set BAZEL=/path/to/bazel)" >&2; exit 1; }
mkdir -p "$DATA"

# 1. corpus -------------------------------------------------------------------
if [ ! -s "$DATA/shakespeare.txt" ]; then
    echo "==> downloading TinyShakespeare (~1.1 MB)"
    python3 scripts/prepare_shakespeare.py "$DATA/shakespeare.txt"
else
    echo "==> corpus already present: $DATA/shakespeare.txt"
fi

# 2. tokenize -----------------------------------------------------------------
echo "==> tokenizing (10% held out for validation)"
"$BAZEL" run -q //tools:prepare --config=release -- \
    "$DATA/shakespeare.txt" "$DATA/shakespeare" --val-frac 0.1

# 3. train --------------------------------------------------------------------
# --config=release is not optional: the default build is ~10x slower.
echo "==> training $STEPS steps (this is the slow part)"
"$BAZEL" run -q //tools:train --config=release -- \
    --data "$DATA/shakespeare.train.bin" --val "$DATA/shakespeare.val.bin" \
    --layers 4 --heads 4 --embd 128 --ctx 64 --batch 32 \
    --steps "$STEPS" --lr 3e-3 --eval-interval $(( STEPS / 6 > 0 ? STEPS / 6 : 1 )) \
    --ckpt "$DATA/shakespeare.ckpt"

# 4. sample -------------------------------------------------------------------
echo
echo "==> sampling from the trained checkpoint"
"$BAZEL" run -q //tools:generate --config=release -- \
    --checkpoint "$DATA/shakespeare.ckpt" --vocab "$DATA/shakespeare.vocab" \
    --prompt $'ROMEO:\n' --n 400 --temperature 0.8 --top-k 20

echo
echo "Checkpoint: $DATA/shakespeare.ckpt"
echo "Sample again with a different prompt, or fine-tune it — see"
echo "examples/shakespeare/README.md."
