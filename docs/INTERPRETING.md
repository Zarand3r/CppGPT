# Reading the interpretability tools

What the viewer and `//tools:inspect` actually measure, what follows from it, and —
mostly — what does not. Every number cited here is owned by
[`docs/measurements.md`](measurements.md); this file owns the *reasoning*.

The short version: **the panels are honest about single forward passes and the
reader supplies the overreach.** Almost every mistake available here is the same
one — treating a measurement of one input as a property of the model.

---

## 1. The residual stream is a channel, not a workspace

It is tempting to ask "what does layer 2 represent". The question is malformed. In
the circuits framing (Elhage et al., *A Mathematical Framework for Transformer
Circuits*) the residual stream is a **communication channel**:

```
resid = embed + Σ(attention writes) + Σ(MLP writes)
```

Each component *reads* a subspace and *writes* a subspace. The answerable question
is **"what does this component write, and which downstream component reads it?"**

Two consequences for the panels:

- The **residual norm** measures magnitude, not content. A layer can add a large
  vector orthogonal to the output direction and change the prediction by nothing.
- The **logit lens** measures one projection — what the unembedding makes of the
  stream *right now*. A layer can carry decisive information the lens cannot see,
  because the unembedding does not read that subspace.

Layer 0 is exactly that case: its step KL is undefined and its heads' direct
effects are ~0.01, yet ablating its attention block is the single most damaging
edit in the model (M-17). It builds the representation everything downstream
reads without yet moving the readout.

## 2. What each panel supports

| panel | measures | supports | does **not** support |
|---|---|---|---|
| Logit lens | the unembedding's reading of the stream at each layer | "the prediction was formed by layer *k*" | "layer *k* represents X" |
| Step KL | KL between consecutive layers' lens distributions | "this layer moved the prediction" | anything at layer 0, where it is **undefined** — no previous layer to difference against |
| Residual stream | ‖resid‖ at the selected token | "this sublayer wrote a lot" | "this sublayer wrote something *useful*" |
| Attribution (DLA) | the component of a write landing in one token's output direction | "this head pushed the predicted token up/down" | total importance — see §3 |
| Ablation | KL between baseline and the output with a component silenced | "the output depends on this component *for this input*" | importance in general — see §4 |
| Attention panels | where information moved | hypotheses | conclusions. Attention is routing, not explanation (Jain & Wallace 2019; Wiegreffe & Pinter 2019) |
| Positional encoding | learned `wpe`, norms and cosine similarity | "positions near each other are encoded alike" | a claim about any particular prompt — `wpe` is prompt-independent |

## 3. Direct effect and total effect are different quantities

Attribution is the **direct** write into the output direction with the final
layernorm's scale frozen. Ablation is the **total** downstream effect. They can
disagree completely — see M-17, where one head has the largest direct effect and
a negligible total effect, and another the reverse.

A head with large ablation and small attribution is acting **through later
layers**. A head with large attribution and small ablation is being
**compensated**. Reading either number alone gives the opposite answer about which
head matters.

## 4. One prompt is not a claim about the model

This is the most important limitation and the easiest to forget, because the
architecture panel renders a single prompt as a confident map.

`//tools:ablation_stats` measures the same components across many prompts. The
result (M-16): the head the single-prompt map shows as most important has a
**median about 8× smaller** than the value shown. Only two components in this
model are both large and consistent.

`head_stats` carries the same warning in its own documentation: entropy, distance
and →first characterise **this prompt**, not the head. Calling something "a
previous-token head" from one input is the standard error in this field.

**Before believing any single-prompt number, check its median.**

## 4a. Every ablation number here is a *zero*-ablation number

`save_and_ablate` zeroes the weights that carry a component. That is why it needs
no forward hook — and it is also the only ablation it can express. The field
treats zero ablation (and mean ablation) as taking the model **off distribution
in an unprincipled manner**: the activation distribution the rest of the network
was trained against is destroyed, so the ablated model can come out either worse
or better than the component's true importance would imply, and **the sign of
that error is not known in advance**. The recommended default is *resample*
ablation — substitute the component's output from a randomly chosen other input,
so the replacement is drawn from the component's own empirical distribution.

So M-16 and M-17 are honest measurements of a well-defined intervention, but that
intervention is not the one the field would choose.

**That sweep has now been run, and the answer is that the baseline dominates.**
`//tools:inspect --donor` substitutes a component's activations from a second
prompt instead of zeroing its weights, and on the seed prompt **21 of 24
components change rank** (M-19). `attn L0` — §5's "single most damaging edit" —
falls from first to sixth, a 224× smaller KL.

Do not read that as "L0 does not matter". The two baselines ask different
questions:

| baseline | question | failure mode |
|---|---|---|
| zero | what if this component were **destroyed**? | off-distribution; the model has never seen it |
| resample (donor) | what if it had processed a **different but similar input**? | measures only what differs between the two prompts |

Pick a donor that differs in nothing important and every component looks
irrelevant. Pick one that differs in everything and the measurement is noise.
**The donor is part of the measurement**, which is why the dump records it and
the viewer prints it beside every number.

One more limit worth naming. The literature's recommended corruption is
*symmetric token replacement* — swap one semantically matched token, hold the
rest fixed. This model is character-level with a 65-symbol vocabulary and has no
semantic tokens to swap, so a donor prompt varies everything at once. What the
tool does is **resample ablation**, not the controlled single-feature contrast
the papers describe, and it is named that way rather than borrowing the stronger
term.

### Noising, not denoising

Both baselines here damage a clean run and measure the fallout — *noising*, which
asks whether a component is **necessary**. The mirror experiment, *denoising*,
starts from a corrupted run and restores one component to ask whether it is
**sufficient**. The two are not symmetric and can disagree; only the first is
implemented.

## 4b. A component can look unimportant *because* it was important

§5 records that ablation does not decompose. The mechanism has a name: when a
component is removed, a dormant backup can take over. The primary then measures
small (the damage was repaired) and the backup also measures small on the intact
model (it was silent). **Both look unimportant, and neither is** — which is a
better description of the 22.9× gap in §5 than "non-additivity".

The correction is to score components *conditionally*: ablate a primary set
first, then ask how much each remaining component's effect **grows**. With 24
components in this model all 576 ordered pairs are affordable, so the exhaustive
version is available here — the field uses approximations only because at real
scale it is not. `ROADMAP.md` M6-A2.

## 5. Ablation does not decompose

Silencing a whole attention block is not the sum of silencing its heads — in this
model the ratio ranges from **0.30× to 22.9×** across four layers (M-17). Remove
one head and the others compensate (the *hydra effect*, McGrath et al. 2023);
remove all of them and the model breaks.

So there is no additive theory that predicts block importance from head
importance, and any first-order approximation (including attribution patching)
should be expected to fail on joint ablation while tracking single-component
ablation. That is a testable prediction, not a caveat.

## 6. What this model can and cannot be asked

It is character-level, 4 layers, `n_embd` 128, at 1.54 nats — it beats a
well-smoothed 5-gram by 8.9% (M-11, M-13). Its regularities are character n-gram
statistics, capitalisation, and line structure; it learned speaker labels like
`CAMILLO:`. **Semantic-concept language is unwarranted at this scale regardless of
tooling.**

A worked example: asked to continue `romeo: the lover of ju`, it does not predict
`l`. That is correct behaviour, not a defect — lowercase `romeo` and `juliet`
occur **zero** times in the corpus, and after lowercase `ju` the training data
says `s` half the time. Given the casing it was trained on, it predicts `l` at
67%. `tools/corpus_stats.py` exists to settle this class of question from the data
before blaming the model.

## 7. What would be needed for genuine feature claims

Nothing here identifies *what a component detects*. That needs, roughly in order
of cost:

1. **Max-activating examples** over a corpus — `fch_gelu` `[L,B,T,4C]` is already
   in the activation arena, so this needs no new model.
2. **Corpus-wide attention statistics**, to turn "this head did X here" into a
   claim about the head.
3. **Induction-head probes** (Olsson et al. 2022) — repeated random sequences.
4. **Transcoders / SAEs**, which decompose superposition. Deliberately last: they
   exist to resolve more features than dimensions, and at `n_embd` 128 in a model
   this weak the payoff is the least certain on the list. For calibration,
   Anthropic's circuit tracing on a production model gave satisfying insight on
   roughly a quarter of tested prompts.

See `ROADMAP.md` → "Resuming in a fresh session" for the build order and reasoning.

## 8. Design choices in the viewer, and why

- **Cells are shaded by ablation, never by attention.** Weighting them by
  attention would invite exactly the misreading §2 warns about.
- **The flow is drawn downward.** Canonical diagrams (Vaswani Fig 1, Jalammar)
  run bottom-to-top; this runs top-to-bottom because the page scrolls that way and
  the prompt sits above it. A deliberate departure.
- **The residual stream is the spine.** Those canonical diagrams say little about
  it; the circuits framework treats it as the central object, and so does this
  tool.
- **Two residual adds per layer are drawn, not one.** A block writes after
  attention *and* after the MLP; drawing one bar per layer halves the structure.
- **Step KL at layer 0 is shown as `—`, not `0`.** It is undefined, and a
  zero-height bar read as "this layer does nothing" beside the darkest cells on
  the page.
