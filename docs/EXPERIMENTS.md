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

### Run A — result: prediction confirmed, and a second problem exposed

**Prediction was: final val loss below 1.686 nats. Result: 1.6133** (scored by
`//tools:eval`, which agrees with the training loop's own 1.6151 to 0.1% — two
independent implementations). **Confirmed.** The budget hypothesis was right: the
model now **beats the 5-gram by 4.3%**, having lost to it by 7.9%.

| | M-8 (900 steps) | Run A (9000 steps) |
|---|---|---|
| val loss (eval) | 1.8199 | **1.6133** |
| bits/char | 2.626 | **2.328** |
| vs best n-gram | **loses** by 7.9% | **beats** by 4.3% |
| top-1 accuracy | 45.6% | **54.1%** |
| context used (within 2% of best) | 18 chars | **38 chars** |

The context curve moving from 18 to 38 characters is the substantive change: the
model is not merely fitting better, it is *using more of the sequence*.

**But the run overfit, and we shipped the wrong checkpoint.**

| | |
|---|---|
| best val | **1.5514 at step 6250** (12.8 epochs) |
| final val | 1.6151 at step 9000 |
| regression after the minimum | **+0.0637 nats** |
| train/val gap | 0.17 → **0.50 nats** |

Val loss bottomed at 12.8 epochs and rose monotonically-ish thereafter while train
loss kept falling to 1.1196 — textbook overfitting. Two consequences:

1. **The checkpoint we evaluated is not the best one we trained.** `--ckpt` saves
   the *latest* every 500 steps, overwriting, so the step-6250 model is gone. We
   reported 1.6133 for a model whose best form scored ~1.55. That is a tooling
   defect, not a modelling one, and it would silently corrupt every future
   comparison. Fixed by `--ckpt-best`.
2. **The sampling defect now binds.** Overfitting at ~13 epochs is exactly what
   15,685 distinct examples predicts. At 1.8 epochs (M-8) the model could not
   reach the memorisation regime, so this was invisible; with budget fixed, it is
   the limiting factor. Run B is now motivated by measurement rather than by
   theory.

**Revised understanding.** Both causes were real and they bind in sequence: budget
was the binding constraint up to ~13 epochs, and example diversity is the binding
constraint after it. Fixing only one would have left the other hidden.
