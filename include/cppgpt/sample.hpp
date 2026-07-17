// cppgpt token sampling — pick the next token from a logits vector during
// generation. Sampling is not on the training hot path, so `sample` may allocate
// a small scratch buffer; `argmax` is allocation-free.
#pragma once

#include "cppgpt/random.hpp"

namespace cppgpt {

// Index of the maximum logit (ties resolved to the lowest index). V > 0.
[[nodiscard]] int argmax(const float* logits, int V) noexcept;

// Sample a token id from `logits` (length V > 0). `temperature` > 0 scales the
// logits before softmax (→ 0 approaches greedy/argmax). `top_k <= 0` disables
// top-k; otherwise only the `top_k` highest-logit tokens are eligible. Draws one
// uniform from `gen`.
[[nodiscard]] int sample(const float* logits, int V, float temperature, int top_k, Generator& gen);

}  // namespace cppgpt
