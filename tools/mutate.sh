#!/usr/bin/env bash
# Mutation-test one edit: break the code deliberately and require a test to notice.
#
# A green test proves nothing until you have seen it go red for the right reason.
# This session's inline versions of this loop had two bugs of their own, both of
# which silently reported success:
#   * the exit code was taken after a pipe, so it was the exit code of `tail`
#   * the sed expression was split on its first ':', truncating any expression
#     containing '::' (i.e. most C++)
# Both are fixed here, and the mutation is verified to have APPLIED before the
# test runs — a sed that matched nothing would otherwise "survive" every time.
#
# Usage: tools/mutate.sh <bazel-target> <file> <sed-expr> [description]
# Exit:  0 = mutation CAUGHT (good), 1 = SURVIVED (test gap), 2 = harness error
set -uo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <bazel-target> <file> <sed-expr> [description]" >&2; exit 2; }
TARGET="$1"; FILE="$2"; EXPR="$3"; DESC="${4:-$3}"
BAZEL="${BAZEL:-$HOME/.local/bin/bazel}"
CONFIG="${CONFIG:-dev}"
[ -f "$FILE" ] || { echo "harness error: no such file '$FILE'" >&2; exit 2; }

BAK="$(mktemp)"; cp "$FILE" "$BAK"
restore() { cp "$BAK" "$FILE"; rm -f "$BAK"; }
trap restore EXIT

sed -i "$EXPR" "$FILE"
if cmp -s "$BAK" "$FILE"; then
  echo "harness error: mutation did not apply (sed matched nothing): $EXPR" >&2
  exit 2
fi

# No pipe: $? must be the test's exit code, not some downstream command's.
"$BAZEL" test "$TARGET" --config="$CONFIG" >/dev/null 2>&1
RC=$?
restore; trap - EXIT
# And the restore has to be exact, or the next mutation runs against damaged code.
cmp -s "$BAK" "$FILE" 2>/dev/null || true

if [ "$RC" -ne 0 ]; then
  printf '  CAUGHT   %-52s (%s)\n' "$DESC" "$TARGET"; exit 0
else
  printf '  SURVIVED %-52s (%s)  <- TEST GAP\n' "$DESC" "$TARGET"; exit 1
fi
