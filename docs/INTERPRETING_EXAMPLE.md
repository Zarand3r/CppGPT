# A walk through one prompt

The viewer, panel by panel, on a prompt small enough to check by hand.
[`INTERPRETING.md`](INTERPRETING.md) explains the reasoning; this is what it looks like in practice.

We give the model `"ROMEO:\nWhat is"` and a donor prompt, `"JULIET:\nWhy is"`, which supplies
replacement activations for the ablation panels. Both are 14 characters, because a donor stands in
for the real thing and has to be the same shape. The model is four layers, four heads, 128
dimensions, and it reads one character at a time from a 65-symbol alphabet.

## Where the prediction forms

The logit lens reads the residual stream at each layer through the unembedding and asks: if the model
stopped here, what would it say?

Layer 0 says `'s'` at p=0.51. Layer 1 still says `'s'`. Layer 2 changes its mind to `'h'`. Layer 3
says `' '` at p=0.86, and that is the answer.

So the model spends most of its depth still finishing the word — `is` plus another `s` — and only at
the last layer decides the word has ended and a space comes next. The step-KL panel agrees: layers 2
and 3 account for 94% of the total movement (1.32 and 1.50 nats).

What you should not read into this is that the early layers are idle. Layer 0's step KL is shown as
`—` rather than 0, because there is no previous layer to compare against and a zero-height bar would
be a lie. Across a corpus, silencing layer 0 does more damage than silencing anything else in the
model. It builds the representation the later layers read; it just has not moved the readout yet.

## What pushed it there

Attribution decomposes the predicted token's logit into the pieces that wrote it. The parts sum
exactly to the logit, which is how the panel is tested rather than eyeballed.

`L3mlp` contributes **+6.81**, dwarfing everything else. Among heads, `L2H1` adds +0.91 and `L3H0`
subtracts 0.67 — components can and do push the prediction *down*.

But this is the *direct* write into the output direction. A component that acts entirely through
later layers scores near zero here and can still be decisive, which is exactly what the next panel
shows.

## What breaks when you remove it

Silence one component, run the prompt again, measure how far the output moved:

| | donor baseline | zero baseline |
|---|---|---|
| L2attn | **0.1952** | 1.0947 |
| L2mlp | 0.0499 | 0.5256 |
| L3mlp | 0.0335 | **2.1015** |

The two columns disagree about almost everything. `L3mlp` — the biggest writer in the attribution
panel — is second-most important under zero ablation and fourth under a donor, sixty-three times
smaller. That is why the viewer prints the baseline next to every number instead of showing one and
calling it *the* ablation KL.

Neither column is a fact about the model. Zeroing a component asks what happens if it is destroyed,
which is a state the model has never seen. Substituting a donor asks what happens if it had read a
slightly different input, so it only measures what differs between the two prompts — and these two
prompts are both `SPEAKER:\nQuestion-word is`, which is why the donor column is small everywhere.

Before believing any of it, check the corpus. On this prompt `L0mlp` reads **0.0108**. Its median
over 128 windows is **6.79**. The single prompt understates it by 628× (M-21), and the donor that
produced that number looked entirely reasonable when it was chosen.

## Why a component can look unimportant

The backup panel silences one component and asks how much every *other* component's effect grows.

Run it on this prompt with the donor and the whole matrix comes back tiny — the largest entry is
**+0.064**. That is not a finding about the model, it is the collapse the previous section warned
about: two prompts this similar barely change anything, so there is nothing for the second component
to grow into. The panel is only worth reading here with the zero baseline, which is what the numbers
below use (drop `--donor` from the command at the bottom).

With that baseline, silencing `L1H0` makes `L0mlp` grow by **+5.18** nats, and silencing `L0attn`
makes it grow by +2.60. `L0mlp` is the model's universal understudy — remove almost anything and it
picks up the slack.

Negative entries mean the opposite. Silencing `L3mlp` makes `L0attn` matter **2.87 nats less**,
because L0's attention was reaching the output partly through L3's MLP. Break the channel and the
component upstream of it stops mattering.

This is what finally explained a number that sat unexplained in the measurements for weeks: ablating
L0's whole attention block does 22.9× the damage of ablating its four heads one at a time. The heads
cover for each other, and `L0mlp` covers for the block. Take one head away and three siblings plus an
MLP absorb it; take the block away and the absorbers leave with it.

## What each head is doing

Here the viewer runs out of road.

The head summary shows L2's heads are sharply focused (entropy 0.36–0.61) on roughly the previous
character (mean distance ≈ 1.1). That is the shape of a previous-token head, and it is the strongest
statement the current panels can make about what a head *is*.

It is also a statement about this prompt, not about the head. Characterising a head from one input is
the standard mistake in this field, and fourteen characters of Shakespeare is a thin basis for it.

The panels that would actually answer the question are not built:

- **QK/OV circuit tables** — a head reads through one matrix and writes through another, and both
  have closed forms over the vocabulary. At 65 symbols they are 65×65 tables that fit on a screen and
  describe the head itself, independent of any prompt. Zero forward passes. This is the biggest gap
  and the cheapest fix (`ROADMAP.md` A5).
- **Neuron views** — the top MLP activations per position, already sitting in the arena (A6).
- **Max-activating examples** — the corpus version, blocked on there being no way for an offline
  artifact to reach the viewer (B3).
- **A component card** — everything above lives in six separate panels. Clicking a head and seeing
  its numbers in one place is what would turn this from a set of instruments into an answer (A7).

So: the viewer is a good tool for working out **which** components matter and **when**, and a poor
one for working out **what** any of them represents.

## Reproduce

```sh
bazel run --config=release //tools:inspect -- \
  --checkpoint $PWD/data/shakespeare.ckpt --vocab $PWD/data/shakespeare.vocab \
  --prompt "ROMEO:
What is" --donor "JULIET:
Why is" --coax 1 --out /tmp/walk.json
```

Then open `tools/viewer.html` and load `/tmp/walk.json`. The backup numbers above come from the same
command with `--donor` removed, for the reason given in that section. Numbers are owned by
[`measurements.md`](measurements.md) M-17, M-19, M-20 and M-21.
