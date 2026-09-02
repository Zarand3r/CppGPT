# A worked example: one prompt, panel by panel

What the viewer actually shows, on a real prompt, with the real numbers — and at each panel, what
follows from it and what does not. [`INTERPRETING.md`](INTERPRETING.md) owns the doctrine; this file
is the walkthrough. Every number here is reproducible with the command at the bottom.

**The prompt:** `"ROMEO:\nWhat is"` — 14 characters, so 14 positions.
**The donor:** `"JULIET:\nWhy is"` — also 14, because a donor replaces activations and must match shape.
**The model:** 4 layers, 4 heads, `n_embd` 128, character-level, 65-symbol vocabulary.

---

## 1. Logit lens — where the prediction forms

Reads the residual stream at each layer through the final layernorm and the tied unembedding: *what
would the model predict if it stopped here?*

| after layer | top-1 | p |
|---|---|---|
| 0 | `'s'` | 0.506 |
| 1 | `'s'` | 0.483 |
| 2 | `'h'` | 0.284 |
| 3 | `' '` | **0.859** |
| output | `' '` | 0.859 |

**What follows.** The model does not commit until the last layer. Through layer 1 it is still
completing `is` → `is` + `s`; at layer 2 it briefly favours `h`; only at layer 3 does it decide the
word has ended and a space follows.

**What does not.** That layers 0–2 are "wrong" or idle. The lens reads one projection — what the
unembedding makes of the stream *right now*. A layer can carry decisive information the lens cannot
see, because the unembedding does not read that subspace.

## 2. Which layer moved the prediction — step KL

The divergence each layer introduced, and how far remains to the output.

| layer | step | remaining |
|---|---|---|
| 0 | — | 3.890 |
| 1 | 0.156 | 2.936 |
| 2 | 1.322 | 1.499 |
| 3 | **1.499** | 0.000 |

**What follows.** Layers 2 and 3 do 94% of the work on this prompt (2.82 of 2.98 nats of movement).

**What does not.** That layer 0 does nothing. Its step KL is **undefined**, not zero — there is no
previous layer to difference against, and the panel prints `—` rather than a zero-height bar that
would read as "idle". Ablating L0 is in fact the most damaging edit in the model across a corpus
(M-21). It builds the representation the later layers read without yet moving the readout.

## 3. What pushed this prediction — direct logit attribution

Exact decomposition of the predicted token's logit. The parts sum to the logit, which is how the
panel is tested rather than eyeballed.

```
embed +0.069   bias +1.298
MLPs:  L0 +1.67   L1 +0.04   L2 +0.72   L3 +6.81
heads: L2H1 +0.907   L3H0 -0.667   L3H3 -0.421   L1H3 +0.407
```

**What follows.** `L3mlp` is overwhelmingly the largest single writer into the output direction.
Some heads push the prediction *down* — negative contributions are real, not noise.

**What does not.** That `L3mlp` is "the important component". This is the **direct** write with the
final layernorm frozen. A component acting entirely through later layers scores near zero here and
can still be decisive — see the next panel, where the ranking is different.

## 4. Ablate a component — and why the baseline is printed

Silence one component, re-run, measure how far the output moved (KL, nats).

| component | donor baseline | zero baseline |
|---|---|---|
| L2attn | **0.1952** | 1.0947 |
| L2H2 | 0.0615 | 0.1051 |
| L2mlp | 0.0499 | 0.5256 |
| L3mlp | 0.0335 | **2.1015** |

**What follows.** The choice of baseline reorders the answer. Under zero ablation `L3mlp` looks
second-most important; under a donor it is fourth and 63× smaller. The viewer therefore names the
baseline beside every number rather than presenting one as *the* ablation KL.

**What does not.** Either column as a fact about the model. Zero ablation asks *what if this were
destroyed* — off-distribution, the model has never seen it. A donor asks *what if it had processed a
different but similar input*, so it measures only what differs between the two prompts. Both prompts
here are `SPEAKER:\nQuestion-word is`, which is why the donor column is small across the board.

**Compare against the corpus before believing any of it.** On 128 windows `L0mlp` is the most
important component under both baselines — and on this single prompt it reads **0.0108** against a
corpus median of **6.79** (M-21). A 628× understatement, from a donor that looked reasonable.

## 5. Backup relationships — conditional co-ablation

How much each component's effect *grows* once another is silenced. Answers the question the ablation
table cannot: *why does this component look unimportant?*

| silence | and this grows | nats |
|---|---|---|
| L1H0 | L0mlp | **+5.18** |
| L0mlp | L1H0 | +4.28 |
| L0attn | L0mlp | +2.60 |
| L3mlp | L0attn | **−2.87** |

**What follows.** `L0mlp` is the model's universal backup — almost anything you silence makes it
matter more. Positive growth is a backup relationship: the second component was idle until the first
was removed. Negative growth is mediation: `L0attn` reached the output partly *through* `L3mlp`, so
breaking `L3mlp` makes `L0attn` matter less.

This is what explains M-17's long-standing anomaly. Ablating the L0 attention block does **22.9×** the
damage of ablating its four heads individually — because the heads cover for each other, and `L0mlp`
absorbs the block. Remove one head and three siblings plus the MLP compensate; remove the block and
the compensators go with the function.

**What does not.** Symmetry. `growth(i,j)` and `growth(j,i)` are different numbers, and both are
computed — a backup takes over for its primary, not the reverse.

## 6. Head summary — the weakest panel, and the honest reason

Per-head attention statistics on this prompt.

| head | entropy | mean distance | mass on position 0 |
|---|---|---|---|
| L2H0 | 0.36 | 1.14 | 0.27 |
| L2H2 | 0.53 | 1.07 | 0.20 |
| L2H1 | 0.59 | 1.15 | 0.15 |

**What follows.** L2's heads are sharply focused (low entropy) on roughly the previous character
(distance ≈ 1.1). That is the shape of a previous-token head.

**What does not.** That they *are* previous-token heads. These statistics characterise **this
prompt**, not the head. Calling something a previous-token head from one input is the standard error
in this field, and `head_stats` carries the warning in its own documentation.

---

## What the viewer does not yet show

The panels above answer **"which layer and head matters, and when"** well. They barely answer
**"what does this layer or head represent"** — and that gap is structural, not an oversight:

- **No QK/OV circuit panels** (`ROADMAP.md` A5). A head reads through its QK circuit and writes
  through its OV circuit, and both have closed forms over the vocabulary. At this model's 65 symbols
  those are 65×65 tables that fit on screen and are **prompt-independent** — a property of the head
  rather than of one input. This is the single highest-value missing panel and it needs zero forward
  passes.
- **No neuron views** (A6) — top `fch_gelu` activations per position, already in the arena.
- **No max-activating examples** (B3) — the corpus-wide version, which needs the artifact channel
  that does not exist yet.
- **No component card** (A7). Everything above lives in six separate panels; clicking a head and
  seeing its numbers together is the consolidation that would make the viewer answer the question.

So the honest summary of the current frontend: it is a good instrument for **causal importance** and a
weak one for **feature identity**.

## Reproduce

```sh
bazel run --config=release //tools:inspect -- \
  --checkpoint $PWD/data/shakespeare.ckpt --vocab $PWD/data/shakespeare.vocab \
  --prompt "ROMEO:
What is" --donor "JULIET:
Why is" --coax 1 --out /tmp/walk.json
xdg-open tools/viewer.html   # then load /tmp/walk.json
```

Numbers owned by [`measurements.md`](measurements.md) M-17, M-19, M-20, M-21.
