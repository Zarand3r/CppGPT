#!/usr/bin/env python3
"""Static + runtime checks for tools/viewer.html.

The viewer has no test target: it is a single self-contained page with no build
step. Two classes of defect have actually shipped in it, and both are checked
here rather than trusted to review:

  * A button rendered but never wired, because a scripted edit's anchor no longer
    matched and the replace silently did nothing. Caught by cross-referencing
    every $('#id') against every id=.
  * A panel that "rendered" while doing nothing, because the check inspected
    innerHTML and the elements were built with appendChild.

The ID check asserts its own extraction is non-empty first. An earlier version
used a shell regex that matched zero IDs and reported success — a check that
cannot fail is worse than no check, because it is counted as coverage.

Usage: check_viewer.py [--html tools/viewer.html] [--dump data/run.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HARNESS = r"""
const fs = require('fs');
const ctx = new Proxy({}, {get:(t,p)=> p in t ? t[p] : (()=>{})});
const els = {};
function mk(id){ if(els[id]) return els[id];
  els[id] = {id, innerHTML:'', textContent:'', hidden:false, value:'0', files:[],
             width:300, height:200, style:{}, className:'', title:'',
             classList:{add(){},remove(){}}, kids:[],
             appendChild(c){ this.kids.push(c); if (c && c.textContent) this.textContent += c.textContent; },
             drawn:false, getContext(){ this.drawn = true; return ctx; }};
  return els[id]; }
// createTextNode is part of the DOM the page uses; a stub missing it reports the
// panel as THROWING rather than as working, which is a false failure. The stub
// must model what the page calls, not a convenient subset.
global.document = { querySelector: s => mk(s.replace('#','')),
                    createElement: () => mk('_t'+Math.random()),
                    createTextNode: t => ({nodeValue: String(t), textContent: String(t)}),
                    documentElement: {} };
global.getComputedStyle = () => ({getPropertyValue: () => '#3b5bdb'});
global.location = {protocol:'http:', href:'http://x/cppgpt/'};
global.fetch = () => Promise.reject(new Error('stub'));
global.alert = () => {};
global.FileReader = class {};
eval(fs.readFileSync(process.argv[2],'utf8'));
show(JSON.parse(fs.readFileSync(process.argv[3],'utf8')));
const out = {};
for (const [k,v] of Object.entries(els))
  out[k] = {html: (v.innerHTML||'').length, text: (v.textContent||'').length,
            kids: v.kids.length, drawn: !!v.drawn};
// The residual-stream bar widths, in flow order. The panel CLAIMS width encodes
// the norm; without pulling them out, nothing checks that claim.
const bars = [];
(function walk(n){ if(!n||!n.kids) return;
  for (const c of n.kids){ if (c.className === 'streambar' && c.style && c.style.width)
    bars.push(parseInt(c.style.width)); walk(c); } })(els['arch']);
out.__streambars = bars;
console.log(JSON.stringify(out));
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="tools/viewer.html")
    ap.add_argument("--dump", default="data/run.json")
    ap.add_argument("--require-dump", action="store_true",
                    help="fail if the dump is missing instead of warning. In CI the panel check "
                         "is the point; a WARN that still exits 0 means it silently never ran.")
    a = ap.parse_args()
    html = open(a.html).read()
    fails = 0

    used = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", html))
    defined = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
    if not used or not defined:
        print(f"  [FAIL] extraction found {len(used)} refs / {len(defined)} ids — "
              "the check itself is broken, not the page")
        return 1
    missing = sorted(used - defined)
    print(f"  [{'FAIL' if missing else 'PASS'}] ids: {len(used)} referenced, "
          f"{len(defined)} defined" + (f", MISSING {missing}" if missing else ""))
    fails += bool(missing)

    m = re.search(r"<script>(.*)</script>", html, re.S)
    if not m:
        print("  [FAIL] no <script> block found")
        return 1
    with tempfile.TemporaryDirectory() as td:
        js = os.path.join(td, "v.js")
        open(js, "w").write(m.group(1))
        rc = subprocess.run(["node", "--check", js], capture_output=True)
        print(f"  [{'PASS' if rc.returncode == 0 else 'FAIL'}] javascript parses"
              + ("" if rc.returncode == 0 else f": {rc.stderr.decode()[:200]}"))
        fails += rc.returncode != 0

        if not os.path.exists(a.dump):
            lvl = "FAIL" if a.require_dump else "WARN"
            print(f"  [{lvl}] no dump at {a.dump} — panels not exercised")
            return 1 if (fails or a.require_dump) else 0
        h = os.path.join(td, "h.js")
        open(h, "w").write(HARNESS)
        r = subprocess.run(["node", h, js, a.dump], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [FAIL] panels threw: {r.stderr.strip().splitlines()[-1][:200]}")
            return 1
        els = json.loads(r.stdout.strip().splitlines()[-1])
        # "Rendered" means produced content by SOME means — innerHTML or children.
        # Counting only innerHTML reported canvas panels as empty.
        # A canvas produces no innerHTML and no children — it is drawn into. Counting
        # only markup listed every chart as inert, which is exactly the kind of
        # misleading green this file exists to prevent.
        bars = els.pop("__streambars", [])
        empty = sorted(k for k, v in els.items()
                       if not k.startswith("_t") and v["html"] == 0
                       and v["text"] == 0 and v["kids"] == 0 and not v["drawn"])
        live = len(els) - len(empty)
        print(f"  [{'PASS' if live else 'FAIL'}] panels rendered: {live} elements produced "
              f"content; inert: {empty if empty else 'none'}")
        fails += not live

        # The stream encoding, checked against the dump rather than trusted. Ties
        # are allowed: widths are integer pixels, so two close norms can round to
        # the same width. What must never happen is the ORDER disagreeing.
        dump = json.load(open(a.dump))
        mid = dump.get("residual_mid") or []
        end = dump.get("residual_norms") or []
        if bars and mid and end:
            p = dump["n_positions"] - 1
            want = []
            for l in range(len(end)):
                want.append(mid[l][p])
                want.append(end[l][p])
            ok_len = len(bars) == len(want)
            inversions = [i for i in range(1, min(len(bars), len(want)))
                          if (bars[i] - bars[i - 1]) * (want[i] - want[i - 1]) < 0]
            # Extremes must agree. Without this a CONSTANT width passes: every
            # delta is zero, so the product test above is never negative and the
            # check cannot tell "encodes the norm" from "encodes nothing".
            # Verified by mutation -- a fixed 12px bar passed the first version.
            extremes = (ok_len
                        and bars.index(max(bars)) == want.index(max(want))
                        and bars.index(min(bars)) == want.index(min(want)))
            good = ok_len and not inversions and extremes
            why = []
            if not ok_len: why.append("length mismatch")
            if inversions: why.append(f"inversions at {inversions}")
            if ok_len and not extremes: why.append("widest/narrowest bar is not the "
                                                   "largest/smallest norm")
            print(f"  [{'PASS' if good else 'FAIL'}] residual bar width tracks the norm: "
                  f"{len(bars)} bars vs {len(want)} writes"
                  + ("" if good else " — " + ", ".join(why)))
            fails += not good
        else:
            print("  [WARN] no residual bars found — the stream encoding is unchecked")
    print(f"\n  {'VIEWER OK' if not fails else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
