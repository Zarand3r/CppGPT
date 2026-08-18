# Experiments

Pre-registered training experiments: hypothesis, design, and the result that would
**falsify** it, written down *before* the run finishes. A prediction recorded after
seeing the number is not a prediction.

Measured numbers live in [`docs/measurements.md`](measurements.md); this file owns
the *reasoning*. Raw logs are `data/runs/<name>.{log,csv}` (gitignored — they are
large, machine-specific, and regenerable).

---

## E-1 · Is the model weak because of a bug, or because of training?

**Context.** `//tools:eval` (M-11) found the toy checkpoint scores 1.8199 nats and
**loses to a 5-gram counting table** (1.6860) by 7.9%. That is a bad result, and
the first question is whether it indicts the implementation or the training run.

### Bug hypothesis — rejected, on two independent pieces of evidence

**1. PyTorch parity.** `//tests/integration:parity_test` matches canonical GPT-2
on the same weights and inputs: forward logits 1.43e-06, gradients 2.38e-07, and
the loss after each of 10 AdamW steps 9.54e-07 — all against a 1e-5 gate. A defect
in the forward, the backward, or the optimizer cannot survive that.

**2. The overfit test.** A correct trainer must be able to memorise a small corpus.
On a 4 KB slice (52-symbol vocab, L4 H4 C128, ctx 64, batch 8, lr 3e-3, 600 steps
≈ 75 epochs):

```
step   1/600  loss 3.9508
step 600/600  loss 0.0211
```

Loss falls to **0.021**. The dataloader, tokenizer, LR schedule, gradient clipping
and AdamW all compose into something that learns to convergence. Parity alone
would not have shown this — it covers 10 steps on a fixture and cannot see a
dataloader or schedule defect. The two together cover the machinery end to end.

**Conclusion: there is no bug in the training machinery.** The model is weak
because of how it was trained.

### Why the training was insufficient

| | M-8 baseline | nanoGPT char-Shakespeare | gap |
|---|---|---|---|
| tokens seen | 1.84 M (1.8 epochs) | ~82 M (~82 epochs) | **45×** |
| distinct examples | 15,685 | ~1.0 M | **64×** |

The second row is a design consequence, not an accident of budget.
`DataLoader::open` sets `n_examples = (n_tokens - 1) / T` — **non-overlapping**
windows — so example *k* spans `[k·T, k·T+T]` and every token is only ever
predicted from one fixed alignment. Token 64 is *always* at window position 0,
with no context, forever.

That is the correct choice for GPT-2-scale pretraining, where the corpus is huge
and you make less than one pass. In this regime — small corpus, many epochs — it
throws away 64× of the available (context, target) pairs. nanoGPT's
char-Shakespeare script samples random offsets for exactly this reason; its
GPT-2 repro does not, for exactly the opposite one.

### Run A — controlled scale-up

Tests the budget hypothesis alone, changing **one variable** against M-8.

| | M-8 | Run A |
|---|---|---|
| steps | 900 | **9000** |
| tokens | 1.84 M | **18.4 M** (18.4 epochs) |
| everything else | L4 H4 C128, ctx 64, batch 32, lr 3e-3, default seed | identical |

```sh
bazel-out/k8-opt/bin/tools/train \
  --data data/shakespeare.train.bin --val data/shakespeare.val.bin \
  --layers 4 --heads 4 --embd 128 --ctx 64 --batch 32 \
  --steps 9000 --lr 3e-3 --eval-interval 250 \
  --ckpt data/runs/long-a.ckpt --log-csv data/runs/long-a.csv
```

It writes to a **new** checkpoint path: `data/shakespeare.ckpt` is what the live
viewer serves, and a half-trained overwrite would silently change every published
interpretability panel mid-run.

**Prediction, registered before the result.** Val loss ends **below 1.686 nats**,
the 5-gram baseline it currently loses to. Grounds: the M-8 curve was still
descending steeply at its end (1.86 → 1.81 over its last 150 steps), which is a
model still learning rather than one that has converged.

**Falsification.** If val loss plateaus **at or above 1.686**, budget is *not* the
sufficient explanation and the window-alignment defect (or something not yet
identified) dominates. In that case Run B is the next test, not more steps.

**Verification, so the result is trustworthy either way.** The final checkpoint is
scored with `//tools:eval` against the same n-gram ladder on the same validation
split — not with the training loop's own eval, which samples a drifting subset.
The two are independent implementations and agree to 0.03%, so a disagreement
between them is itself a signal.

### Run B — random-offset sampling (planned)

Same budget as Run A, sampling window starts uniformly at random rather than on a
T-aligned grid. Isolates the dataloader change. Run A must complete first:
changing budget and sampling together would leave neither attributable.
