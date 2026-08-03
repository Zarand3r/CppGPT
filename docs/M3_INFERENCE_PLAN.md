# M3 — Feature-Complete GPT-2 + Pretrained Inference

**Status:** planned, not started. Supersedes the three-line M3 sketch in `ROADMAP.md`.
**Prerequisite:** M1 complete (parity gate met, ~1e-6). M2 perf work is *independent* of this
milestone and can proceed in parallel.

---

## 0. North star — the ultimate integration test

> **`tools/infer` loads the real OpenAI GPT-2 124M weights, tokenizes a text prompt with our own
> byte-level BPE, generates 50 tokens greedily, and emits the same 50 token ids — and the same
> decoded text — as HuggingFace.**

That single command exercises every component at once: BPE encode → weight load → embedding →
12 transformer blocks → tied classifier → sampling → BPE decode. Nothing else in this project
tests the whole stack against an external ground truth. **It is the definition of done for M3.**

Binary acceptance:

```sh
bazel run //tools:infer -- --weights data/gpt2_124M.ckpt --prompt "def fibonacci(n):" --n 50 --greedy
# stdout token ids == tests/fixtures/gpt2_hf_greedy.bin, all 50, exactly
```

Everything below exists to make that command trustworthy rather than coincidentally correct.

---

## 1. What is already verified (do not re-litigate)

Established empirically during planning — cited so no slice wastes effort re-deriving it.

| Fact | Evidence |
|---|---|
| cppgpt's param layout **is** GPT-2 124M | our `param_sizes()` computes **124,439,808**, equal to HF's `named_parameters()` sum |
| LayerNorm eps, final LN, residual order, MLP order, attention scale, causal mask, no head bias, unpadded V=50257 | each read from transformers 5.12 source and matched against `src/ops.cpp` / `src/model.cpp` |
| GELU is **tanh** (`gelu_new`), not erf | HF config `activation_function="gelu_new"`; measured `max\|act − tanh_gelu\| = 0.0` exactly |
| QKV packing is **Q‖K‖V**, head-major within each block | all 6 permutations tried; only `(0,1,2)` matches HF (3.4e-05 vs 63–238 for the rest) |
| A plain `.T` on the 4 Conv1D weights is **sufficient and complete** — no head interleave | end-to-end fp32 reimplementation from the flat block reproduced HF greedy output, **token-exact for 50 tokens** |
| The full 50,257 vocab is reconstructible from `vocab.bpe` alone | verified byte-for-byte against `encoder.json`; **no JSON parser needed in C++** |
| `merge_rank == token_id − 256` | 0 mismatches across all 50,000 merges |
| `transformers` == `tiktoken` for ordinary text | 0 disagreements over 6,072 varied strings (differences appear only for literal `<\|endoftext\|>`) |

**Consequence:** the model math is already canonical. M3 is an *integration and I/O* milestone, not a
numerics milestone. The risk has moved from "is the math right" to "is the data plumbed right".

---

## 2. The blocking finding — `generate()` cannot match HF

**`include/cppgpt/generate.hpp` uses a sliding fixed-size window.** After each token it drops the
oldest and appends the new one, so every surviving token's `wpe` row shifts by one every step
(`embedding_forward_cpu` indexes `wpe + t*C` with `t` = *buffer* index, not absolute position).
HF uses absolute `position_ids = arange(len)` over a growing context.

Measured with HF playing both roles, so this isolates the protocol from numerics:

```
sliding-window(T=16, left-pad): [783, 1363, 284, 262, 995, 338, 4387, ...]
growing-context               : [262, 3139, 286, 262, 4141, 2066,  11, ...]
identical: False   <-- diverges at the FIRST token
```

Left-padding a short prompt makes it worse: cppgpt has no attention mask, so pad tokens are fully
attended.

**Fix (verified equivalent): right-pad-and-read.** Allocate at `T = prompt_len + n_new`, place the
prompt at `[0, len)`, fill the tail with anything, forward the whole buffer, read logits at position
`len − 1`. Causality makes the tail inert — `attention_forward_cpu` only visits `t2 <= t`, and every
other op is per-position.

```
max|right_padded.logits[n-1] − unpadded.logits[n-1]| = 1.14e-04   (kernel-blocking noise only)
right-pad protocol == growing context, 50 greedy tokens: True
```

In cppgpt's naive per-position matmul this is *bit-identical*, not 1e-4.

**Do not silently change `generate()`'s semantics.** The sliding window is a legitimate mode for
unbounded generation past the context limit. Add `generate_absolute()` alongside it.

---

## 3. Invariants

1. **Positional identity.** A token at absolute position `p` is always embedded with `wpe[p]`.
   Any generation protocol that violates this is not GPT-2.
2. **Tokenizer round-trip on bytes.** `decode(encode(s)) == s` for *every* `std::string` — no UTF-8
   validity requirement (byte-level BPE guarantees this). The converse does **not** hold.
3. **`decode()` returns raw bytes.** Never validates, never substitutes `U+FFFD`. 344 of 50,256
   tokens decode to individually-invalid UTF-8 fragments; that is correct and must survive.
4. **One vocabulary source.** The C++ tokenizer, the fixture generator, and the converter all derive
   from the same pinned `vocab.bpe` (sha256 recorded). No second table.
5. **Unicode tables are pinned, not derived.** `\p{L}`/`\p{N}`/`\s` tables are extracted from the
   pinned oracle and committed; a test asserts the C++ tables equal the fixture tables.
6. **Weights are never committed.** The 497.8 MB payload is a locally generated artifact.
7. **Existing gates never loosen.** The M1 parity gate and all 25 tests stay green throughout.

## 4. Failure model

| Condition | Response |
|---|---|
| Missing/short/corrupt weight file | `Result` error at load — reuse `load_checkpoint`'s existing `ShapeMismatch`/`ChecksumMismatch`/`CorruptCheckpoint` |
| `vocab.bpe` missing or wrong sha256 | fail fast at tokenizer construction — a wrong vocab is silent gibberish, not a degraded mode |
| Token id ≥ vocab_size on decode | fail fast (`ASSERT`) — matches `CharTokenizer` |
| Byte sequence that no alternative matches | **impossible by construction** (alternation is total — verified over 3,000 random strings, 0 gaps); assert it anyway, since a table bug would manifest here |
| Heavy test run without the weight file present | **fail loudly, never skip silently** — a skipped parity test that reports green is the worst outcome |

---

## 5. Slices

Each slice is independently verifiable and lands as its own PR. Ordered by **risk retired per unit
of work**, not by dependency convenience.

```
S1 ──► S2 ──┐
            ├──► S5 (ultimate integration test)
S3 ─────────┘
S4 (KV cache) ──► S5'   S6 (inference memory) ──► M4
```

### S1 — Absolute-position generation protocol
*Retires the blocking divergence. Small, fully specified, no external deps.*

- Add `generate_absolute(model, prompt_ids, n_new, temp, top_k, gen)` using right-pad-and-read.
- `ASSERT(prompt_len + n_new <= model.seq_len())` — fail fast rather than silently sliding.
- Keep `generate()` unchanged; document when each applies.

**Gate:** with a randomly-initialized model, `generate_absolute` with `n_new=1` equals a plain
`forward` + `argmax` at position `len−1`; and for a model whose context never overflows,
generating `k` tokens one-at-a-time (re-forwarding a growing prefix) equals one `generate_absolute`
call of `k`. Both bit-exact.

### S2 — HF weight converter + forward parity at real scale
*Highest silent-wrongness risk (a square `attn.c_proj` hides a missing transpose). Needs no tokenizer.*

- `scripts/convert_hf_gpt2.py` (dev-only, torch venv) → **raw fp32 payload**, 497,759,232 bytes.
  Python does only name mapping + transposes.
  Mandatory self-checks: per-tensor shape/dtype/**C-contiguity** (catches a forgotten `.contiguous()`
  after `.T`), exact allow-list of the 16 source groups, assert leftover keys `== {"lm_head.weight"}`,
  `flat.size == 124_439_808`, all-finite, memmap re-read equality.
- `tools/import_hf.cpp` — reads the raw payload into the parameter arena and calls the **existing**
  `GPT2::save_checkpoint`. This is why Python does not write the container: it would duplicate the
  header + checksum format, and `kFnvOffset64` is deliberately **not** the textbook FNV-1a-64 basis,
  so a Python reimplementation reaching for the standard constant silently produces an unloadable file.
- Fixtures (generated with `attn_implementation="eager"` — the 5.12 default is `sdpa`, which differs
  by 6.9e-05 — and an explicit `model.eval()`, or dropout randomizes every reference logit).

**Gate:**
- `max|Δlogits| ≤ 2e-3` **and** `argmax` equal at every position. (The fp32 floor is ~1e-4 at
  `max|logit| ≈ 130`; cppgpt's naive sequential accumulator over C=768/3072 has worse error growth
  than torch's blocked reduction — budget 2–5×. An absolute tolerance alone is not a correctness
  statement; the argmax check is what protects S5.)
- Layer-bisect: relative agreement `max|a−b| / max(1, max|b|) ≤ 1e-5` after the embedding and after
  each of the 12 blocks. **Never use an absolute tolerance on hidden states** — the residual stream
  reaches `max|.| ≈ 3013` by layer 11 (GPT-2's outlier channel).

### S3 — Byte-level BPE tokenizer
*Largest slice by code volume, but failures are loud.*

- **Assets:** commit `vocab.bpe` (456,318 B) + its sha256. **Do not commit `encoder.json`** — the
  vocab is reconstructible: ids 0–255 from the byte map, 256–50255 as `lhs+rhs` of merge `i−256`,
  50256 = `<|endoftext|>`. Skips a JSON parser (with `\uXXXX` + surrogate-pair unescaping) entirely.
  Watch the file's shape: line 0 is the literal `#version: 0.2` header, and a trailing newline makes
  a naive split yield 50,002 elements.
- **`byte_to_unicode`:** 188 identity bytes (`0x21–0x7E`, `0xA1–0xAC`, `0xAE–0xFF`), the other 68
  (`0x00–0x20`, `0x7F–0xA0`, `0xAD`) map to `U+0100..U+0143` in ascending byte order. Max codepoint
  323 ⇒ `uint16_t b2u[256]` + reverse `int8_t u2b[324]`.
- **Pre-tokenizer:** hand-rolled. `std::regex` fails on **all** load-bearing constructs — no
  `\p{L}`/`\p{N}`, no UTF-8 awareness (it would match per-byte and split multibyte sequences),
  and unreliable lookahead. Six alternatives tried strictly in order, with exactly one backtracking
  site (`\s+(?!\S)`: match the run, then give back the last codepoint if a non-ws follows).
  Unicode tables as `constexpr` sorted ranges + binary search: `\p{L}` 684 ranges, `\p{N}` 146,
  `\s` 10 — **~6.7 KB total**, plus a 128-byte ASCII fast-path LUT.
- **Merges:** rank-ordered, lowest rank first, ranks unique (no ties). Represent symbols as `u32`
  ids throughout (every merged symbol is itself a vocab entry) keyed on a packed `u64` — no string
  allocation. Per-pre-token cache: a pre-token is unbounded (` ?\p{L}+` matches a 1 MB letter run)
  and the naive loop is O(n²).
- **Scope decision:** implement `encode_ordinary` semantics only. `<|endoftext|>` in input text
  tokenizes as ordinary bytes; `eot_id() == 50256` is a separate constant the generation loop appends
  explicitly. This matches `tiktoken.encode_ordinary`, avoids a prompt-injection surface, and dodges
  HF's special-token whitespace-resegmentation trap.

**Gate:** byte-exact vs tiktoken over ≥1,000 committed cases, **plus** every trap below pinned
individually so a failure names the bug rather than dumping a random string:

| Trap | Correct output |
|---|---|
| `DON'T` — contractions are lowercase-only | `[41173, 6, 51]`, not a `'T` match |
| `don’t` — only ASCII `U+0027` counts | `[9099, 447, 247, 83]` |
| `" 's"` — no "skip space then check contraction" | `[705, 82]`, not `[220, 338]` |
| `hello\n\nworld` vs `a \n\n` — same substring, different tokens | `198,198` vs `628` (the `(?!\S)` rule) |
| `x  \n  y` — run gives back exactly its last codepoint | `[87, 220, 220, 198, 220, 331]` |
| `a\xa0\xa0b` — NBSP **is** `\s`; BOM is **not** | `[64, 1849, 1849, 65]` |
| `strengthened` — rank-ordered, not greedy-longest | `[22853, 782, 8524, 276]`, not `[41402, 2945]` |
| `aaaa` / `!!!!!!` — overlap handling in the merge pass | `[24794]` / `[13896, 3228]` |
| 8 spaces — GPT-2 has **no** multi-space token | eight × `220` |

Property test: `decode(encode(s)) == s` on the byte corpus. **Do not** assert
`encode(decode(x)) == x` — measured **~6.5%** of random token sequences are not BPE-reachable and
legitimately re-encode differently. The correct invariant is `decode(encode(decode(x))) == decode(x)`.

Drift guard: assert the committed C++ Unicode tables equal the fixture tables. The `regex` module's
tables differ from Python 3.12's `unicodedata` by **9,568** codepoints in `L` — a codepoint that is a
letter in one version and unassigned in another moves between alternatives and changes token ids.

### S4 — KV cache
- `Storage` of `[n_layer, 2, max_seq, n_head, head_dim]`; write the new k/v slice, read the full
  prefix; crop the classifier to the last position.

**Gate:** KV-cache-on and KV-cache-off produce **identical** token ids (the ROADMAP's stated M3
criterion), and cache-on generation is measurably faster.

### S5 — The ultimate integration test + `tools/infer`
- Wire BPE + weights + `generate_absolute` into a CLI.
- The committed gate fixture is ~220 bytes (prompt ids + 50 expected ids).

**Gate:** token-exact for 50 greedy tokens vs HF.

> **Pick the gate prompt by measurement, not by taste.** Greedy exactness is a *discrete* decision,
> so what matters is the top1−top2 logit margin. Measured over 250 steps: min **0.00716**, p5 0.106,
> median 2.81 — against an fp32 logit error of ~1e-4…1e-3, that is only **7–70× headroom at the worst
> step**. 50-token exactness is achievable but **not robust for an arbitrary prompt**. Choose a prompt
> whose measured min margin > 0.05 (`"def fibonacci(n):"` measured **0.105**), and **store that margin
> in the fixture** so the test asserts `min_margin > 20 × observed_max_logit_error`. That turns a
> flaky discrete gate into a quantified one.

### S6 — Inference-only arenas *(not required for the gate; required for real use and M4)*
The ctor allocates `grad_store_` + `act_grad_store_` unconditionally:

| config | training-shaped | inference-only |
|---|---|---|
| 124M, B=1, T=1024 | **5.32 GB** | **2.61 GB** |
| 350M, B=1, T=1024 | **12.93 GB** | **6.40 GB** |

At the gate's `T ≈ 55` this is ~1.05 GB and irrelevant; at full context it is the difference between
comfortable and not. Logit cropping (S4) also removes `logits+probs` at `[B,T,V]` = 412 MB.

---

## 6. Fixture inventory (all committed; none needs torch at test time)

| Fixture | Size | Purpose |
|---|---|---|
| `vocab.bpe` + sha256 | 456 KB | the tokenizer itself |
| `gpt2_tokenizer.bin` (1,200 cases, raw UTF-8 + ids) | **272 KB** | byte-exactness gate |
| unicode range tables | ~7 KB | drift guard |
| `gpt2_hf_logits.bin` (5-token prompt, all positions) | **982 KB** | forward parity |
| `gpt2_hf_hidden.bin` (13 × 768, last position) | **39 KB** | layer-bisect — *this is what saves a day of debugging* |
| `gpt2_124M.manifest.txt` (16 × name/offset/count/fnv) | ~1 KB | convert-time structural check |
| `gpt2_hf_greedy.bin` (prompt + 50 ids + min margin) | **~220 B** | the M3 gate |
| **weights** | 497.8 MB | **NOT committed** — locally generated |

Fixtures use raw bytes, not escaped text — escaping is itself a bug surface, and the corpus must
exercise NUL, control chars, and lone `\r`. Exclude lone surrogates from the generator corpus: Python
`str` can hold them, UTF-8 cannot, so they would encode a `U+FFFD` the C++ side never sees.

---

## 7. Risk register

| # | Risk | Evidence / mitigation |
|---|---|---|
| R1 | **Generation protocol mismatch** | *Confirmed, blocking.* Fixed by S1; verified equivalent. |
| R2 | Missing transpose on square `attn.c_proj` (768,768) — **no shape assertion can catch it** | layer-bisect fixture (S2); plus a convert-time `assert not allclose(W, W.T)` |
| R3 | Pre-tokenizer divergence on an untested input class | 9 traps pinned individually; 3.3 M-string full-codepoint scan showed 0 disagreements for the reference impl |
| R4 | Unicode table version drift | tables pinned from the oracle + equality test |
| R5 | Greedy gate is flaky on a badly-chosen prompt | measure min margin; require > 0.05; store it in the fixture |
| R6 | `transformers` version skew (≤4.x exposes `attn.bias` buffers, 5.x does not) | explicit 16-name allow-list + "no unexpected leftover keys" assertion; pin the version |
| R7 | `lm_head.weight` double-counted (163,037,184 vs 124,439,808) | drop it (tied storage, `data_ptr` equal); `param_count` check fails loudly anyway |
| R8 | Heavy tests silently skipped when weights absent | fail loudly by policy |
| R9 | Streaming output splits multibyte codepoints | `tools/infer` buffers and flushes only complete UTF-8 sequences |

---

## 8. Decisions needed before S1

1. **`tiktoken` in the venv** — installed cleanly during research and is the oracle named in
   `PLAN.md`. Confirm we add it to `notebooks/requirements.txt` (dev-only; no runtime dep).
2. **Weight artifact location** — `data/` (gitignored) vs a documented cache path.
3. **Slice granularity for S3** — one PR, or split (byte map + tables) / (pre-tokenizer) /
   (merges + decode)? Recommend **split**: the pre-tokenizer is the risky half and deserves its own
   review.
4. **KV cache before or after the gate?** The ROADMAP lists it in M3, but the gate does not need it.
   Recommend **after** — get token-exactness first, then optimize with a
   "must not change any token" gate.

---

## 9. Out of scope

Top-p / repetition penalty, batched inference, GPT-2 medium/large/XL (M4), quantization (E2/E3),
flash attention (E1), and the M2 matmul/threading work — which is independent and can proceed
in parallel.
