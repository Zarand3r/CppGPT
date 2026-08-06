#!/usr/bin/env bash
# End-to-end pipeline test: prepare -> train -> generate -> fine-tune.
#
# This exercises the actual CLI binaries, which nothing else does. Every unit and
# integration test in this repo links the library directly, so the tools' own
# wiring — flag parsing, the vocab-sidecar path rule, reading a model's shape out
# of a checkpoint header — had zero automated coverage and was verified only by
# running it by hand once. Two real defects had already slipped through that gap
# (train appending ".vocab" to a data path instead of stripping the suffix; a bare
# boolean flag swallowing the following positional).
#
# Deliberately tiny: a few hundred tokens and a handful of steps, so it runs in
# seconds. It asserts the pipeline WIRES UP, not that the model learns anything —
# convergence is examples/shakespeare's job, and asserting on loss values here
# would make the test flaky for no benefit.
set -euo pipefail

PREPARE="$TEST_SRCDIR/_main/tools/prepare"
TRAIN="$TEST_SRCDIR/_main/tools/train"
GENERATE="$TEST_SRCDIR/_main/tools/generate"
WORK="$TEST_TMPDIR/work"
mkdir -p "$WORK"

fail() { echo "FAIL: $*" >&2; exit 1; }

# A corpus with enough repetition that a tiny model can move at all, and enough
# length to fill several batches.
python3 - "$WORK/base.txt" <<'PY'
import sys
words = ["alpha", "beta", "gamma", "delta"]
with open(sys.argv[1], "w") as f:
    for i in range(2000):
        f.write(words[i % len(words)] + " ")
PY

# ---------- prepare: split + vocab ----------
"$PREPARE" "$WORK/base.txt" "$WORK/base" --val-frac 0.2 > "$WORK/prep.log"
[ -s "$WORK/base.train.bin" ] || fail "prepare wrote no train split"
[ -s "$WORK/base.val.bin" ]   || fail "prepare wrote no val split"
[ -s "$WORK/base.vocab" ]     || fail "prepare wrote no vocab"
[ ! -e "$WORK/base.train.bin.tmp" ] || fail "prepare left a .tmp behind"

# The split must be a partition: train + val == the whole corpus, in tokens.
train_toks=$(( $(stat -c%s "$WORK/base.train.bin") / 2 ))
val_toks=$(( $(stat -c%s "$WORK/base.val.bin") / 2 ))
total=$(stat -c%s "$WORK/base.txt")
[ $((train_toks + val_toks)) -eq "$total" ] \
  || fail "split is not a partition: $train_toks + $val_toks != $total"

# ---------- train: flags honoured, checkpoint written ----------
"$TRAIN" --data "$WORK/base.train.bin" --val "$WORK/base.val.bin" \
         --layers 2 --heads 2 --embd 32 --ctx 16 --batch 4 \
         --steps 20 --eval-interval 20 --seed 7 --ckpt "$WORK/base.ckpt" > "$WORK/train.log"
[ -s "$WORK/base.ckpt" ] || fail "train wrote no checkpoint"
grep -q "\[eval\]" "$WORK/train.log" || fail "train --val produced no eval line"
# The model it reports must be the model we asked for, not the old hardcoded one.
grep -q "L2 H2 C32" "$WORK/train.log" || fail "train ignored the architecture flags"

# Determinism: the same seed must reproduce the run exactly.
"$TRAIN" --data "$WORK/base.train.bin" --layers 2 --heads 2 --embd 32 --ctx 16 \
         --batch 4 --steps 20 --seed 7 > "$WORK/train_a.log"
"$TRAIN" --data "$WORK/base.train.bin" --layers 2 --heads 2 --embd 32 --ctx 16 \
         --batch 4 --steps 20 --seed 7 > "$WORK/train_b.log"
grep "step" "$WORK/train_a.log" | sed -E 's/[0-9.]+ tok\/s.*//' > "$WORK/a.txt"
grep "step" "$WORK/train_b.log" | sed -E 's/[0-9.]+ tok\/s.*//' > "$WORK/b.txt"
cmp -s "$WORK/a.txt" "$WORK/b.txt" || fail "same seed produced different losses"

# ---------- generate: reads architecture from the checkpoint header ----------
# No --layers/--embd here: if the header peek is broken this cannot work at all.
"$GENERATE" --checkpoint "$WORK/base.ckpt" --vocab "$WORK/base.vocab" \
            --prompt "alpha" --n 24 --top-k 1 > "$WORK/gen_a.txt" 2>"$WORK/gen.err"
grep -q "L2 H2 C32" "$WORK/gen.err" || fail "generate did not recover the architecture from the header"
[ -s "$WORK/gen_a.txt" ] || fail "generate produced no output"
# Greedy decoding must be deterministic.
"$GENERATE" --checkpoint "$WORK/base.ckpt" --vocab "$WORK/base.vocab" \
            --prompt "alpha" --n 24 --top-k 1 > "$WORK/gen_b.txt" 2>/dev/null
cmp -s "$WORK/gen_a.txt" "$WORK/gen_b.txt" || fail "greedy generation is not deterministic"
# Output must decode within the corpus alphabet — a wrong vocab or a bad decode
# shows up as bytes that were never in the training text.
LC_ALL=C grep -qE '^[a-z ]+$' "$WORK/gen_a.txt" || fail "generated text escaped the corpus alphabet"

# ---------- fine-tune: vocab reuse then --init-from ----------
python3 - "$WORK/target.txt" <<'PY'
import sys
with open(sys.argv[1], "w") as f:
    for _ in range(2000):
        f.write("delta ")
PY
"$PREPARE" "$WORK/target.txt" "$WORK/tgt" --vocab "$WORK/base.vocab" --val-frac 0.2 > /dev/null
# Reusing a vocabulary must reproduce it byte for byte, or the checkpoint will not load.
cmp -s "$WORK/base.vocab" "$WORK/tgt.vocab" || fail "--vocab did not reuse the vocabulary"

"$TRAIN" --data "$WORK/tgt.train.bin" --val "$WORK/tgt.val.bin" --vocab "$WORK/base.vocab" \
         --layers 2 --heads 2 --embd 32 --ctx 16 --batch 4 --steps 20 --lr 3e-4 \
         --init-from "$WORK/base.ckpt" --ckpt "$WORK/tuned.ckpt" > "$WORK/ft.log"
grep -q "initialized from" "$WORK/ft.log" || fail "--init-from was ignored"
[ -s "$WORK/tuned.ckpt" ] || fail "fine-tune wrote no checkpoint"

# ---------- failure paths must fail, and say why ----------
# A byte outside the reused vocabulary is a hard error, never a silent remap.
printf 'alpha ZZZ\n' > "$WORK/bad.txt"
if "$PREPARE" "$WORK/bad.txt" "$WORK/bad" --vocab "$WORK/base.vocab" > "$WORK/bad.log" 2>&1; then
  fail "prepare accepted a byte absent from the reused vocabulary"
fi
grep -q "absent from" "$WORK/bad.log" || fail "prepare's unknown-byte error does not explain itself"

# A mistyped flag must be rejected, not silently ignored.
if "$TRAIN" --data "$WORK/base.train.bin" --stpes 5 > "$WORK/typo.log" 2>&1; then
  fail "train accepted a mistyped flag"
fi
grep -q "unknown flag" "$WORK/typo.log" || fail "mistyped flag error does not name the flag"

# The wrong vocabulary must be rejected rather than decoding gibberish.
printf 'xyz' > "$WORK/wrong.vocab"
if "$GENERATE" --checkpoint "$WORK/base.ckpt" --vocab "$WORK/wrong.vocab" \
               --prompt "a" --n 4 > "$WORK/wrongv.log" 2>&1; then
  fail "generate accepted a vocabulary that disagrees with the checkpoint"
fi

echo "PASS: prepare -> train -> generate -> fine-tune, plus 3 failure paths"
