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
# Refuse to publish a page whose script does not parse. This is not theoretical:
# a merge resolution once left viewer.html with an unbalanced brace, and site/ was
# rebuilt and the service restarted from it. The API kept answering 200 because
# that is the C++ binary -- only the HTML was broken, which a health check cannot
# see. Verify the artifact, not the service.
if command -v node >/dev/null 2>&1; then
  python3 - "$ROOT/tools/viewer.html" <<'PYEOF' > "$OUT/.viewer.js"
import re, sys
h = open(sys.argv[1]).read()
m = re.search(r"<script>(.*)</script>", h, re.S)
sys.stdout.write(m.group(1) if m else "")
PYEOF
  node --check "$OUT/.viewer.js" || { echo "make-site: viewer.html script does not parse — refusing to publish" >&2; rm -f "$OUT/.viewer.js"; exit 1; }
  rm -f "$OUT/.viewer.js"
else
  echo "make-site: node not found — publishing WITHOUT the parse check" >&2
fi

cp "$ROOT/tools/viewer.html" "$OUT/index.html"
cp "$DUMP" "$OUT/run.json"
echo "site/ built from $DUMP — serve with: python3 -m http.server 8092 --bind 127.0.0.1 -d $OUT"
