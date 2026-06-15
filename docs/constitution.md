# Constitution — cppgpt

The **deal-breaker behaviors**: if any of these is broken, the PR is reverted
without further reading. This file exists to beat the *gaming problem* — the
agent writes both the code and the tests, so it can satisfy the letter of a test
while missing the point. These promises are authored by the human and the
numerical oracle is **canonical GPT-2 in PyTorch**, which the agent does not
author and cannot narrow. The elves **Judge** checks this file every batch.

Deal-breakers only — not nice-to-haves, not implementation details, not exact
file names. Stated as behavior, abstract enough to survive refactoring.

---

## Flows
The compute paths that must hold end to end.

- **Forward parity.** For a fixed seed and fixed weights, every intermediate
  activation of the model matches the canonical-GPT-2 PyTorch reference to within
  the agreed forward tolerance. A change that makes a test pass by loosening that
  tolerance, skipping the fixture comparison, or comparing against a self-authored
  oracle instead of PyTorch is a deal-breaker.
- **Backward parity.** For the same fixture, every gradient matches the PyTorch
  reference (and an independent finite-difference check) to within the agreed
  backward tolerance. Hand-derived backward must be verified against a reference
  the agent did not write.
- **Training-loop fidelity.** Starting from identical initialization, the loss
  after the agreed number of AdamW steps matches the PyTorch reference to within
  the agreed tolerance. The 10-step "overall okay" parity is the gate on the
  whole training path.
- **Inference parity.** For a fixed prompt and fixed sampling seed, generation
  reproduces the reference (HuggingFace `transformers`) token-for-token over the
  agreed horizon. KV-cache-on and KV-cache-off must produce identical tokens.

## Business logic
The GPT-2 numerical and recipe rules that must stay correct. These are what make
the parity flows above achievable; weakening one silently breaks parity.

- **Canonical GPT-2 semantics, not nanoGPT shortcuts.** tanh-approximation GELU
  (`gelu_new`), bias on every Linear and LayerNorm, learned absolute position
  embeddings, unpadded vocab of 50257, pre-norm LayerNorm with the standard eps.
  Deviating to an erf GELU, `bias=False`, or a padded vocab to make something
  easier is a deal-breaker.
- **Weight tying is real.** The classifier head and the token embedding are the
  same parameter buffer; backward accumulates into that one region from both the
  embedding and classifier paths; the parameter is counted once. A second,
  untied copy is a deal-breaker.
- **AdamW two-group weight decay.** Decay applies to matmul/embedding weights;
  biases and LayerNorm gains/shifts are never decayed. Applying decay uniformly
  (or to none) is a deal-breaker — it silently diverges from the reference.

## Invariants
Must always be true regardless of what else changes.

- **Determinism by construction.** The same seed produces bit-identical logits
  across runs. Every stochastic operation takes an explicit generator; there is
  no global/default RNG anywhere in the model, optimizer, sampler, or dataloader.
- **No external runtime dependencies.** The shipped binaries link only the C
  runtime the toolchain provides (verified by `ldd` against the agreed
  allow-list). No third-party C++ library is introduced into the build graph —
  no BLAS, Eigen, JSON, logging, or tokenizer library. Build-time tooling
  (Bazel, the pinned LLVM toolchain, the Python oracle/data scripts) is not part
  of the shipped binary and does not count against this.
- **Failures fail fast, never silently.** An out-of-range token id, a NaN/Inf
  loss, or a mis-shaped / wrong-magic checkpoint aborts the operation with a
  precise error. Silently clamping, zero-filling, skipping, or degrading past a
  corrupt input is a deal-breaker.
- **No hidden allocation in the steady-state training step.** After warm-up, a
  training step allocates no heap memory outside the per-step arena reset. A
  change that adds per-step allocation to the hot path without justification is a
  deal-breaker.
- **Ownership is unambiguous.** Owning storage owns its buffer; views borrow and
  never free. No `shared_ptr` to raw tensor floats, no ambiguous ownership across
  an API boundary.
- **Verification is run, not promised.** Any change touching an op runs that op's
  fixture comparison against the PyTorch reference. A claim of correctness without
  the corresponding gate having actually run is a deal-breaker.
