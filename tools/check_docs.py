#!/usr/bin/env python3
"""Every path a doc names must exist, or be listed here with a reason.

A doc describing code that no longer exists teaches a wrong mental model, which is
worse than no doc. That is not hypothetical here: `docs/M3_INFERENCE_PLAN.md`
described `tools/import_hf.cpp` and a Python conversion script for weeks after the
thing that shipped was `tools/convert_hf.cpp` reading safetensors directly in C++.

References are matched by BASENAME anywhere in the tree, because docs legitimately
say `model.hpp` for `include/cppgpt/model.hpp`. An earlier version of this check
compared literal paths, flagged 60 false positives, and would have been ignored
within a day.

Usage: check_docs.py            (exit 1 if any unexplained reference)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

# Intentional references to things that do not exist. Each needs a reason: an
# allowlist without reasons becomes a place to silence findings.
ALLOWED = {
    "device.hpp": "deleted on purpose (DECISIONS.md D7); named historically",
    "docs/ARCHITECTURE.md": "a listed deliverable, not yet written",
    "convert_hf_gpt2.py": "superseded design, inside the SUPERSEDED table in M3_INFERENCE_PLAN",
    "scripts/convert_hf_gpt2.py": "same",
    "tools/import_hf.cpp": "same",
    "IMPLEMENTATION_PLAN.md": "produced on demand by the implementation-plan skill",
    "docs/AGENT_HARNESS.md": "lives in the skills repo, not this one",
    # CLAUDE.md routes to reference files inside the eng-skills plugin. They are
    # real files, just not in this repository, and CLAUDE.md names them bare.
    "data_oriented_design.md": "eng-skills plugin reference",
    "cpp_api_style.md": "eng-skills plugin reference",
    "cpp_ownership_and_arenas.md": "eng-skills plugin reference",
    "cache_lines_and_alignment.md": "eng-skills plugin reference",
    "vtables_and_polymorphism.md": "eng-skills plugin reference",
    "templates_and_codegen.md": "eng-skills plugin reference",
    "memory_mapping.md": "eng-skills plugin reference",
}
# Third-party artifacts docs name to explain what we do or do not read.
EXTERNAL = {"pytorch_model.bin", "encoder.json", "tokenizer.json", "config.json",
            "vocab.json", "merges.txt", "model.safetensors", "tokenizer_config.json"}

DOCS = ["ROADMAP.md", "CLAUDE.md", "tools/README.md"]
PAT = re.compile(r"`(/?/?[A-Za-z0-9_./-]+\.(?:md|cpp|hpp|sh|py|bazel))`")


def main() -> int:
    tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split())
    bases: dict[str, list[str]] = {}
    for t in tracked:
        bases.setdefault(os.path.basename(t), []).append(t)

    docs = DOCS + sorted(d for d in tracked if d.startswith("docs/") and d.endswith(".md"))
    missing: list[tuple[str, str]] = []
    for d in docs:
        if not os.path.exists(d):
            continue
        for m in PAT.finditer(open(d).read()):
            ref = m.group(1)
            p = ref.lstrip("/")
            base = os.path.basename(p)
            if p in tracked or os.path.exists(p) or base in bases:
                continue
            if p in ALLOWED or base in ALLOWED or base in EXTERNAL:
                continue
            missing.append((d, ref))

    seen, out = set(), []
    for d, r in missing:
        if (d, r) in seen:
            continue
        seen.add((d, r))
        out.append(f"  {d}: {r}")

    print(f"  checked {len(docs)} documents")
    if out:
        print(f"  [FAIL] {len(out)} reference(s) name something that does not exist:")
        print("\n".join(out))
        print("  Fix the doc, or add the path to ALLOWED with a reason.")
        return 1
    print(f"  [PASS] every referenced path exists ({len(ALLOWED)} intentional exceptions, "
          f"{len(EXTERNAL)} third-party artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
