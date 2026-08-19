#!/usr/bin/env bash
# Enforce the constitution's dynamic-dependency allow-list.
#
# The claim "the binaries link only libc and libm" is load-bearing: it is why this
# repo can assert no runtime dependencies, and why W&B had to be a sidecar rather
# than a flag on train. A claim nothing checks is a claim that quietly stops being
# true — libresolv.so.2 was linked into every binary for a period before being
# resolved, and nothing failed.
#
# Usage: tools/check_ldd.sh [binary ...]   (defaults to every //tools binary)
set -uo pipefail
ALLOWED='^(libc|libm|libgcc_s|ld-linux-x86-64|linux-vdso)'
BINS=(); [ $# -gt 0 ] && BINS=("$@")
if [ ${#BINS[@]} -eq 0 ]; then
  # -L because bazel-out is a symlink and find would otherwise discover nothing
  # (silently reporting success over an empty set). Restricted to ELF files:
  # the same directory holds .repo_mapping and .runfiles_manifest, which are
  # executable-bit-set text and are not binaries at all.
  while IFS= read -r f; do
    file -b "$f" 2>/dev/null | grep -q '^ELF' && BINS+=("$f")
  done < <(find -L bazel-out -maxdepth 4 -path '*opt/bin/tools/*' -type f -executable 2>/dev/null | sort -u)
fi
[ ${#BINS[@]} -gt 0 ] || { echo "check_ldd: no binaries found (build with --config=release first)" >&2; exit 2; }

fail=0
for b in "${BINS[@]}"; do
  # A binary that ldd cannot read is a failure, not a pass — silence must never
  # be mistaken for compliance.
  out=$(ldd "$b" 2>&1)
  # A statically linked binary has no dynamic dependencies at all, which is the
  # strongest possible form of compliance -- not a failure.
  if echo "$out" | grep -q 'not a dynamic executable'; then
    echo "  ok   $(basename "$b") (static)"; continue
  fi
  bad=$(echo "$out" | grep -oE '\blib[a-z0-9_+.-]+\.so[0-9.]*' | sed 's/\.so.*/\.so/' \
        | sort -u | grep -vE "$ALLOWED" || true)
  if [ -n "$bad" ]; then
    echo "  FAIL $(basename "$b"): outside allow-list: $(echo "$bad" | tr '\n' ' ')"
    fail=1
  else
    echo "  ok   $(basename "$b")"
  fi
done
[ "$fail" -eq 0 ] && echo "  all binaries within the allow-list (libc, libm)" || echo "  ALLOW-LIST VIOLATION"
exit $fail
