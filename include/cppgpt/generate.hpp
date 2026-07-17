// cppgpt autoregressive text generation.
//
// Sliding fixed-size window: at each step, forward the model's T-token context,
// sample the next token from the LAST position's logits, then slide the window
// (drop the oldest token, append the new one). No KV cache yet (that's M3), so
// every generated token costs a full forward over T — fine for toy/interactive
// generation, quadratic for long sequences.
#pragma once

#include <cstddef>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/sample.hpp"

namespace cppgpt {

// Generate `n_new` tokens continuing `context` (length == model.seq_len()). The
// model must be batch size 1. temperature/top_k are passed to sample() (top_k = 1
// gives greedy/argmax decoding). Returns the newly generated tokens.
[[nodiscard]] inline std::vector<int> generate(GPT2& model, const int* context, int n_new,
                                               float temperature, int top_k, Generator& gen) {
    ASSERT(model.batch() == 1);
    ASSERT(context != nullptr && n_new >= 0);
    const int T = model.seq_len();
    const int V = model.config().vocab_size;
    std::vector<int> window(context, context + T);
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(n_new));
    for (int i = 0; i < n_new; ++i) {
        model.forward(window.data(), nullptr);  // inference: logits only
        const float* last = model.acts().logits + static_cast<std::size_t>(T - 1) * V;
        const int tok = sample(last, V, temperature, top_k, gen);
        out.push_back(tok);
        for (int t = 0; t < T - 1; ++t)
            window[static_cast<std::size_t>(t)] = window[static_cast<std::size_t>(t + 1)];
        window[static_cast<std::size_t>(T - 1)] = tok;
    }
    return out;
}

}  // namespace cppgpt
