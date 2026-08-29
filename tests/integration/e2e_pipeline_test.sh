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
INSPECT="$TEST_SRCDIR/_main/tools/inspect"
EVAL="$TEST_SRCDIR/_main/tools/eval"
ABLSTATS="$TEST_SRCDIR/_main/tools/ablation_stats"
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

# ---------- inspect: the interpretability dump ----------
# Architecture again comes only from the checkpoint header — no --layers/--embd.
"$INSPECT" --checkpoint "$WORK/base.ckpt" --vocab "$WORK/base.vocab" \
           --prompt "alpha be" --out "$WORK/run.json" --top-k 4 > "$WORK/inspect.log"
[ -s "$WORK/run.json" ] || fail "inspect wrote no dump"

# The dump's contract, checked rather than assumed: valid JSON, attention that is
# a causal distribution, and a last-layer lens that agrees with the model's own
# output. That last one is the non-circular check — at the final layer the lens IS
# the model's final computation, so disagreement means the lens is wired wrong.
python3 - "$WORK/run.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["schema"] == 3, f"unexpected schema {d['schema']}"
n = d["n_positions"]
assert n == len(d["tokens"]) > 0, "token count disagrees with n_positions"
assert len(d["residual_norms"]) == d["config"]["n_layer"], "residual_norms has wrong layer count"
for li, L in enumerate(d["attention"]["data"]):
    for hi, H in enumerate(L):
        assert len(H) == n, "attention is not n_positions square"
        for q, row in enumerate(H):
            assert abs(sum(row[:q+1]) - 1.0) <= 1e-3, f"attention row {q} does not sum to 1"
            assert all(v == 0 for v in row[q+1:]), f"attention row {q} attends to the future"
assert d["logit_lens"][-1]["top"][0]["id"] == d["final_top"][0]["id"], \
    "last-layer logit lens disagrees with the model's own top-1"

nl, nh = d["config"]["n_layer"], d["config"]["n_head"]

# Attribution: shaped per head, and aimed at the token the model actually
# predicted. A decomposition of some OTHER token's logit would look perfectly
# well-formed here, which is why the target is checked and not just the shape.
a = d["attribution"]
assert a["token"]["id"] == d["final_top"][0]["id"], \
    "attribution explains a different token than the model's top-1"
assert len(a["heads"]) == nl and all(len(r) == nh for r in a["heads"]), \
    "attribution head matrix is not n_layer x n_head"
assert len(a["mlps"]) == nl, "attribution mlp row is not n_layer long"

# Ablation: exhaustive, each component exactly once, and KL is non-negative by
# definition — a negative value means the two distributions were swapped.
ab = d["ablation"]
assert len(ab) == nl * (nh + 2), \
    f"ablation swept {len(ab)} components, expected {nl * (nh + 2)}"
assert len({(r["kind"], r["layer"], r["head"]) for r in ab}) == len(ab), \
    "ablation sweep repeats a component"
assert all(r["kl"] >= 0 for r in ab), "KL divergence cannot be negative"
assert {r["kind"] for r in ab} == {"head", "mlp", "attn"}, "ablation missed a component kind"
# Positional embeddings: square, symmetric, unit diagonal. wpe is LEARNED, so
# these are trained values -- but cosine similarity has those properties by
# construction, which makes them a real check on the emitter rather than on the
# model. A transposed or mis-strided read breaks symmetry immediately.
pe = d["pos_embed"]
assert len(pe["norms"]) == n, "pos_embed norms is not n_positions long"
assert len(pe["similarity"]) == n and all(len(r) == n for r in pe["similarity"]), \
    "pos_embed similarity is not n_positions square"
for i in range(n):
    assert abs(pe["similarity"][i][i] - 1.0) <= 1e-3, "a position is not similar to itself"
    for j in range(n):
        assert abs(pe["similarity"][i][j] - pe["similarity"][j][i]) <= 1e-3, \
            "cosine similarity is not symmetric"
        assert -1.001 <= pe["similarity"][i][j] <= 1.001, "cosine outside [-1,1]"
print("  dump contract ok")
PYEOF

# --ablate 0 must skip the sweep, not silently run it anyway: the flag is the
# only escape hatch on a model where L*(NH+2) forward passes is not free.
"$INSPECT" --checkpoint "$WORK/base.ckpt" --vocab "$WORK/base.vocab" \
           --prompt "alpha be" --out "$WORK/noabl.json" --ablate 0 > /dev/null
python3 - "$WORK/noabl.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["ablation"] == [], "--ablate 0 still ran the sweep"
assert d["attribution"]["heads"], "--ablate 0 should not disable attribution (it costs no forward)"
print("  --ablate 0 ok")
PYEOF

# A selection that would be enormous must be refused, not written.
if "$INSPECT" --checkpoint "$WORK/base.ckpt" --vocab "$WORK/base.vocab" \
              --prompt "alpha be" --out "$WORK/huge.json" --max-mb 0.000001 \
              > "$WORK/huge.log" 2>&1; then
  fail "inspect ignored its size ceiling"
fi
grep -q "max-mb" "$WORK/huge.log" || fail "size-ceiling error does not say how to fix it"

# ---------- ablation_stats: is a component important, or was it important once? ----
"$ABLSTATS" --checkpoint "$WORK/base.ckpt" --data "$WORK/base.val.bin" \
            --prompts 6 --seq 8 --seed 3 --top 99 > "$WORK/abl.log" 2>&1 \
  || fail "ablation_stats failed: $(tail -2 "$WORK/abl.log")"

# Determinism is what makes two runs comparable, so it is checked, not assumed.
"$ABLSTATS" --checkpoint "$WORK/base.ckpt" --data "$WORK/base.val.bin" \
            --prompts 6 --seq 8 --seed 3 --top 99 > "$WORK/abl2.log" 2>&1
diff -q "$WORK/abl.log" "$WORK/abl2.log" > /dev/null || fail "ablation_stats is not deterministic"

python3 - "$WORK/abl.log" <<'PYEOF'
import re, sys
txt = open(sys.argv[1]).read()
rows = re.findall(r"^\s+(L\d+ (?:H\d+|MLP|attn))\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+(\d+)%",
                  txt, re.M)
# 2 layers x (2 heads + MLP + attn block)
assert len(rows) == 2 * (2 + 2), f"expected 8 components, parsed {len(rows)}"
for name, mean, med, p90, mx, act in rows:
    mean, med, p90, mx, act = float(mean), float(med), float(p90), float(mx), int(act)
    # Quantiles must be ordered. A quantile index off by one, or sorting the wrong
    # array, breaks this immediately -- and would otherwise look like plausible
    # numbers in a plausible table.
    assert med <= p90 <= mx, f"{name}: quantiles out of order ({med}, {p90}, {mx})"
    assert mean >= 0 and med >= 0, f"{name}: KL cannot be negative"
    assert mean <= mx, f"{name}: mean {mean} exceeds max {mx}"
    assert 0 <= act <= 100, f"{name}: active fraction {act} out of range"
assert "active on >=90%" in txt, "summary line missing"
print("  ablation_stats contract ok")
PYEOF

# ---------- eval: the number is only meaningful next to a baseline ----------
"$EVAL" --checkpoint "$WORK/base.ckpt" --data "$WORK/base.val.bin" \
        --train "$WORK/base.train.bin" --batch 2 > "$WORK/eval.log" 2>&1 \
  || fail "eval failed: $(tail -2 "$WORK/eval.log")"

# Determinism is the property that makes eval usable for comparing checkpoints,
# so it is checked rather than asserted in a comment.
"$EVAL" --checkpoint "$WORK/base.ckpt" --data "$WORK/base.val.bin" \
        --train "$WORK/base.train.bin" --batch 2 > "$WORK/eval2.log" 2>&1
diff -q "$WORK/eval.log" "$WORK/eval2.log" > /dev/null \
  || fail "eval is not deterministic across runs"

python3 - "$WORK/eval.log" "$WORK/train.log" <<'PYEOF'
import math, re, sys
txt = open(sys.argv[1]).read()
def row(name):
    m = re.search(rf"^\s+{name}\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", txt, re.M)
    assert m, f"eval output has no {name!r} row:\n{txt}"
    return tuple(float(g) for g in m.groups())
V = int(re.search(r"V(\d+)", txt).group(1))
model_n, model_ppl, model_bits = row("model")
uni_n, _, _ = row("uniform baseline")
big_n, _, _ = row("bigram baseline")

# Exact, not empirical: uniform cross-entropy over V symbols IS ln(V). If this
# disagrees the tool is reading the wrong vocab size, which would silently
# rescale every comparison.
assert abs(uni_n - math.log(V)) < 1e-3, f"uniform {uni_n} != ln({V})={math.log(V)}"
# Internal consistency of the three columns.
assert abs(model_ppl - math.exp(model_n)) < 0.05, "perplexity is not exp(nats)"
assert abs(model_bits - model_n / math.log(2)) < 1e-2, "bits/char is not nats/ln2"
# Ordering: more context cannot be worse on a corpus with any structure.
assert big_n < uni_n, "bigram baseline is not better than uniform - counting is broken"
# The tail must be declared, never silently dropped.
assert "tokens scored" in txt, "eval does not report how many tokens it scored"

# External anchor. Every assertion above is SELF-CONSISTENT: perplexity and
# bits/char are both derived from the same nats figure, so a wrong nats satisfies
# all three together. A mutation that dropped the B*T weighting (making the loss
# 128x too small) passed every one of them.
#
# train's own eval is a genuinely independent implementation -- shuffled
# DataLoader batches rather than sequential windows -- so it cannot move in
# sympathy. The two sample different tokens and will not agree exactly; the point
# is that a scaling error shows up as orders of magnitude, not percent.
train_log = open(sys.argv[2]).read()
m = re.search(r"val loss\s+([\d.]+)", train_log)
assert m, "train log has no val loss to cross-check against"
train_val = float(m.group(1))
# +/-10%. The two agree to 0.03% here and 0.5% on the real corpus, so this is
# ~20x the observed spread -- loose enough not to flake on subset sampling, tight
# enough that an off-by-one in the target shift (which scores the model on the
# WRONG prediction) cannot hide inside it. A +/-50% band let exactly that through.
assert 0.9 * train_val < model_n < 1.1 * train_val, \
    f"eval reports {model_n:.4f} nats but train's own eval reports {train_val:.4f}"
print(f"  eval contract ok (eval {model_n:.4f} vs train's own eval {train_val:.4f} nats, "
      f"{100*abs(model_n-train_val)/train_val:.1f}% apart)")
PYEOF

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
