# Review audit — mechanical sweeps

Durable record of tree-wide sweeps for **mechanically detectable defect classes**. A lesson in
`docs/engineering-lessons.md` that is checkable but unchecked is half a lesson; when a lesson has a
greppable or scriptable signature, the sweep and its result are recorded here.

---

## 2026-08-03 — Magic constants vs their canonical values (L6)

**Trigger.** `kFnvOffset64` was found to be the textbook FNV-1a-64 basis with a digit dropped
(`1469598103934665603` vs `14695981039346656037`). Signature: *a constant whose comment asserts a
canonical identity the value does not have.*

**Scope.** Whole tree — `include/`, `src/`, `tests/`, `scripts/`. Every named mathematical constant
and every four-character-code magic, compared against its canonical value (and, for 4CCs, against
the ASCII its little-endian bytes actually spell).

| Constant | Location | Result |
|---|---|---|
| FNV-1a-64 offset basis | `include/cppgpt/checkpoint.hpp` | ❌ **`1469598103934665603`** — textbook value with the last digit dropped |
| FNV-1a-64 prime `1099511628211` | `src/checkpoint.cpp` | ✅ correct |
| GELU `√(2/π) = 0.7978845608028654` | `src/ops.cpp` | ✅ correct to float precision |
| GELU cubic `0.044715` | `src/ops.cpp` | ✅ correct |
| `π` (cosine LR) | `src/optimizer.cpp` | ✅ correct |
| `2π` (Box–Muller) `6.283185307179586` | `include/cppgpt/random.hpp` | ✅ correct |
| `2⁻²⁴` uniform scale `16777216.0` | `include/cppgpt/random.hpp` | ✅ correct |
| LayerNorm eps `1e-5` | `src/ops.cpp` | ✅ correct (matches HF `layer_norm_epsilon`) |
| `kCheckpointMagic = 0x54504B43` | `include/cppgpt/checkpoint.hpp` | ✅ LE bytes spell `CKPT`, as commented |
| `kMagic = 0x43475446` | `include/cppgpt/verify.hpp` | ❌ commented `"CGTF"`; LE bytes spell **`FTGC`** |
| `MAGIC = 0x43475446` | `scripts/gen_fixtures.py` | ❌ same wrong comment (both sides agree, so the fixture still round-trips) |

**Findings: 2 defect sites (3 locations).**

1. **`kFnvOffset64`** — functional (it is still a valid hash; corruption is still detected) but it is
   not the named standard, and the comment says it is. Fix requires a `kCheckpointVersion` bump since
   every existing checkpoint's checksum changes. **Deferred** — tracked at the point of use in
   `checkpoint.hpp`; bundling it with the next format change avoids a second gratuitous version bump.
2. **`0x43475446` commented `"CGTF"`** — comment-only error, no functional impact (writer and reader
   share the constant). **Fix now** — correct both comments to `FTGC`.

**Not swept (no signature):** semantic constants with no external canonical value (arena alignment,
default LR/betas, batch dims) — correctness for these is a design question, not a lookup.
