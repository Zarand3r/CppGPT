#!/usr/bin/env bash
# Run a curated mutation battery and report SURVIVORS.
#
# Written because prose rules did not prevent recurrence: three tests in this
# repo passed while the code under test did nothing (docs/engineering-lessons.md
# L19), all written by the same author who had just written the rule. A survivor
# is either a weak test or dead code; both are findings.
#
# Not wired into CI: a full battery rebuilds per mutation and takes minutes. Run
# it when touching the files below, and at every milestone gate.
#
# A mutation must be aimed at the test that COVERS it. On this suite's first run
# the activation-guard mutation was pointed at model_test rather than
# act_guard_test and reported a false SURVIVED -- which would have sent someone
# to fix a test that was never broken. A false survivor costs more than a missing
# mutation.
#
# Usage: tools/mutation_suite.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CONFIG="${CONFIG:-dev}"

# file <TAB> target <TAB> sed <TAB> description
read -r -d '' MUTATIONS <<'TSV' || true
src/bpe.cpp	//tests/unit:bpe_test	s|if (r < best) { best = r; at = k; }|if (r > best \&\& r != ~0U) { best = r; at = k; }|	bpe: merge highest rank first
src/bpe.cpp	//tests/unit:bpe_test	s|seed(0xA1, 0xAC);|seed(0xA1, 0xAB);|	bpe: byte-map seed range off by one
src/bpe.cpp	//tests/unit:bpe_test	s|seed(0xAE, 0xFF);|seed(0xAE, 0xFE);|	bpe: byte-map upper range off by one
src/bpe.cpp	//tests/unit:bpe_test	s|out.append(blob_.data() + a, b - a);|out.append(blob_.data() + a, b > a ? b - a - 1 : 0);|	bpe: decode drops a byte
src/model.cpp	//tests/unit:model_test	s|if (logits_at < 0) {|if (true) {|	model: logits_at ignored (correct but slow)
src/model.cpp	//tests/unit:act_guard_test	s|ASSERT_MSG(atot <= static_cast<std::size_t>(std::numeric_limits<int>::max()),|ASSERT_MSG(true \|\|(atot <= static_cast<std::size_t>(std::numeric_limits<int>::max())),|	model: activation INT_MAX guard removed
src/ops.cpp	//tests/unit:softmax_test	s|std::exp(inp\[i\] - maxv)|std::exp(inp[i])|	ops: softmax loses max-subtraction
src/ops.cpp	//tests/unit:matmul_test	s|for (; c < Cz; ++c) s += inp_bt\[c\] \* w_oc\[c\];|;|	ops: matmul drops the ragged tail
src/interpret.cpp	//tests/unit:interpret_test	s|(dot - (sum / C) \* u_sum)|(dot)|	interpret: attribution loses centering
TSV

total=0; survived=0
while IFS=$'\t' read -r f t e d; do
  [ -z "${f:-}" ] && continue
  total=$((total + 1))
  if ! tools/mutate.sh "$t" "$f" "$e" "$d"; then survived=$((survived + 1)); fi
done <<< "$MUTATIONS"

echo
echo "  $((total - survived))/$total caught, $survived SURVIVED"
[ "$survived" -eq 0 ] || echo "  a survivor is a weak test or dead code — both are findings"
exit $([ "$survived" -eq 0 ] && echo 0 || echo 1)
