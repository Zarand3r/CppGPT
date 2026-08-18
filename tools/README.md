# tools/

Two kinds of thing live here: the **binaries** that are the product, and the
**utility scripts** that verify and analyse them. The scripts exist because each
one was written inline several times during development, and in several cases the
inline version was wrong in a way that reported success.

## Binaries (Bazel targets, C++, libc/libm only)

| target | what it does |
|---|---|
| `//tools:prepare` | corpus → `uint16` token `.bin` + `.vocab` |
| `//tools:train` | train / resume / fine-tune, `--log-csv`, `--ckpt-best`, `--sample` |
| `//tools:generate` | sample from a checkpoint |
| `//tools:eval` | quality vs an n-gram ladder — nats, perplexity, bits/char, accuracy, calibration, context curve |
| `//tools:profile` | forward-pass latency and per-op breakdown |
| `//tools:bench` | matmul throughput (GFLOP/s) |
| `//tools:inspect` | one forward pass → JSON for the viewer (lens, attention, attribution, ablation, positional encoding) |

## Utility scripts (dev tooling; Python/bash, never linked into the binaries)

| script | what it answers | why it exists |
|---|---|---|
| `run_report.py` | how did this run go, and is it overfitting? | re-derived inline 4× with two different definitions of "overfitting" |
| `audit_run.py` | does this run show any bug signature? | "the loss went down" is not evidence training was correct |
| `corpus_stats.py` | what does the corpus say should follow this text? | settled whether the model was wrong about `ju` → `l` (it was not) |
| `mutate.sh` | does this test actually catch that defect? | the inline loop had two bugs, both silently reporting success |
| `check_viewer.py` | is the viewer wired up and does it render? | a button once shipped rendered but unwired |
| `wandb_log.py` | ship a run to W&B | keeps the network out of the training binary |
| `serve_viewer.py` | serve the viewer + run the model on submitted prompts | public-facing; see `docs/DECISIONS.md` D5 |
| `make-site.sh` | assemble `site/` from `viewer.html` + a dump | one source for the page |

### Notes that matter

`run_report.py` judges overfitting by the **trend and its standard error**, never
by `final > best` — for any noisy series the final point is above the minimum,
so that test is satisfied by pure noise. It reported a healthy run as overfitting
before this was fixed.

`mutate.sh` verifies the mutation **applied** before running the test. A `sed`
that matches nothing otherwise "survives" every time and looks like a test gap.

`check_viewer.py` asserts its own extraction is non-empty before judging the page,
and counts canvas drawing as content. Earlier versions of both checks passed while
measuring nothing.
