# Toy example — character-level Shakespeare

The whole product in four commands: tokenize a corpus, train a model, sample from it,
then fine-tune it onto a different corpus.

Everything here is char-level. There is no BPE and no pretrained weight loading —
that is the MVP scope on purpose (see `ROADMAP.md`). What the model *is* is a
canonical GPT-2 architecture, verified against PyTorch to ~1e-6 by
`//tests/integration:parity_test`. It is a real GPT-2; it is just small, and
trained on your own text.

## Run it

```sh
# one command, ~10 minutes on a modern desktop CPU
examples/shakespeare/run.sh
```

Or step by step, from the repo root:

```sh
# 1. corpus (1.1 MB of Shakespeare; downloads once)
python3 scripts/prepare_shakespeare.py data/shakespeare.txt

# 2. tokenize, holding out 10% for validation
bazel run //tools:prepare -- \
    "$PWD/data/shakespeare.txt" "$PWD/data/shakespeare" --val-frac 0.1
#   -> shakespeare.train.bin  shakespeare.val.bin  shakespeare.vocab   (vocab 65)

# 3. train
bazel run //tools:train --config=release -- \
    --data "$PWD/data/shakespeare.train.bin" --val "$PWD/data/shakespeare.val.bin" \
    --layers 4 --heads 4 --embd 128 --ctx 64 --batch 32 \
    --steps 900 --lr 3e-3 --eval-interval 150 \
    --ckpt "$PWD/data/shakespeare.ckpt"

# 4. sample
bazel run //tools:generate --config=release -- \
    --checkpoint "$PWD/data/shakespeare.ckpt" --vocab "$PWD/data/shakespeare.vocab" \
    --prompt $'ROMEO:\n' --n 400 --temperature 0.8 --top-k 20
```

**Always `--config=release`.** The default build is unoptimised and roughly an order
of magnitude slower; a run that should take minutes will take hours.

Note the `"$PWD"/…` prefixes: `bazel run` sets the working directory to the target's
runfiles tree, not the workspace root, so a bare relative path resolves somewhere you
did not mean.

## What to expect

Character-level models learn the *shape* of the text long before they learn English.
The progression is legible and worth watching:

| val loss | what the samples look like |
|---|---|
| ~4.2 (step 0) | uniform noise over 65 characters |
| ~2.5 | correct character frequencies; space-separated blobs |
| ~2.0 | word-shaped tokens, some real short words, line breaks in the right places |
| ~1.8 | mostly real words, `NAME:` speaker labels, dialogue structure |
| ~1.5 | fluent pseudo-Shakespeare — the nanoGPT reference point |

### Actual output from the command above

Measured on a Ryzen 9 9950X3D, `--config=release`, 900 steps = 1.84 M tokens in
**634 s** at 2908 tok/s, peak RSS 204 MB. Validation loss fell monotonically:

```
[eval] step 150  val loss 2.4214      [eval] step 600  val loss 1.9489
[eval] step 300  val loss 2.1499      [eval] step 750  val loss 1.8600
[eval] step 450  val loss 2.0290      [eval] step 900  val loss 1.8120
```

Sampling at `--temperature 0.8 --top-k 20 --seed 42`:

```
ROMEO:
This be he most; be the can the statine
The dording are your loth.

GLOUCESTER:
I shall monely will shall parlenty his the indeess
To the die.

First I speak:
I be many, let he be the
heave anshall the told he dread be with leave:
Which we to son the pursed an and in the boy
In now to set but wife, and be more to be uneyal,
Uno, thou not you his black'd of thee,
In will now say but a not brows! Go
```

Read that for structure rather than sense. In ten minutes the model has learned the
`NAME:` speaker convention, blank lines between speeches, verse-length line breaks,
a real character name it only ever saw in the corpus (`GLOUCESTER`), ordinary English
words, and Shakespearean forms — `thou`, `thee`, `dread`, and the elided `black'd`.
It has not learned to mean anything. That is the honest state of a 1.8-val-loss
character model, and it is the point of the table above: you can watch which of these
appears at which loss.

Getting to ~1.5 needs roughly 10× more tokens; the model and the pipeline do not
change, only `--steps`. The reasoning behind the build flags is in
`docs/DECISIONS.md` D1, and throughput baselines are in `docs/measurements.md`.

## Fine-tuning onto a second corpus

The one non-obvious constraint: **the second corpus must be tokenized with the first
corpus's vocabulary.** A fresh corpus produces a different character set, hence a
different `vocab_size`, hence a different `wte` shape — and the base checkpoint will
not load. That is what `--vocab` is for.

```sh
# tokenize the target corpus with the ORIGINAL vocabulary
bazel run //tools:prepare -- \
    "$PWD/data/target.txt" "$PWD/data/target" \
    --vocab "$PWD/data/shakespeare.vocab" --val-frac 0.1

# continue from the base weights, fresh optimizer, lower learning rate
bazel run //tools:train --config=release -- \
    --data "$PWD/data/target.train.bin" --val "$PWD/data/target.val.bin" \
    --vocab "$PWD/data/shakespeare.vocab" \
    --layers 4 --heads 4 --embd 128 --ctx 64 --batch 32 \
    --steps 300 --lr 3e-4 --eval-interval 100 \
    --init-from "$PWD/data/shakespeare.ckpt" --ckpt "$PWD/data/target.ckpt"
```

If the target corpus contains a character the base vocabulary lacks, `prepare` stops
and names every offending byte with its count. It will not silently drop or remap it —
that would train the model on text that differs from the file on disk.

`--init-from` loads **weights only** and resets the optimizer, which is what
fine-tuning wants; `--ckpt` on an existing file resumes a run and deliberately restores
the optimizer state instead. The difference and why it matters is `docs/DECISIONS.md` D3.

## Things worth knowing

- **`--top-k 1` is greedy and fully deterministic** — same checkpoint, same prompt,
  same output, every time. Useful for checking a change did nothing unexpected.
- **Any run reproduces from `--seed`.** Two runs at the same seed produce identical
  loss curves, bit for bit.
- **Resume is automatic.** Point `--ckpt` at an existing checkpoint and training picks
  up with the optimizer state intact.
- **The context is fixed at training time.** Generating past `--ctx` slides the window
  and restarts positions, the same way every fixed-context model does. Prompts shorter
  than the context are handled exactly.

## This example is also a test

`//tests/integration:e2e_pipeline_test` runs this same pipeline — prepare, train,
generate, fine-tune, plus three failure paths — at a scale that finishes in under a
second. It is the only test that runs the actual binaries, so it is what stops the CLI
wiring from rotting. It was mutation-checked: reintroducing a real historical bug (the
vocab-sidecar path rule) turns it red.
