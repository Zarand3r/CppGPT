// cppgpt autoregressive text generation.
//
// Sliding fixed-size window: at each step, forward the model's T-token context,
// sample the next token from the LAST position's logits, then slide the window
// (drop the oldest token, append the new one). No KV cache yet (that's M3), so
// every generated token costs a full forward over T — fine for toy/interactive
// generation, quadratic for long sequences.
#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
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

// Generate `n_new` tokens continuing `prompt`, which may be SHORTER than the
// model's context (unlike generate() above, which requires exactly seq_len).
//
// Each token is embedded at its true absolute position: the prompt sits at
// [0, len), the rest of the buffer is padded with token 0, and the logits are
// read at position len-1. Causality makes the padded tail inert —
// attention_forward only visits t2 <= t, and every other op is per-position — so
// the padded forward is bit-identical to an unpadded one of length len. (Padding
// with 0 rather than "anything": embedding_forward asserts 0 <= token < V at
// every position, including the tail.)
//
// Once the context fills, the window slides and positions restart at 0, which is
// the standard way to generate past a fixed context (nanoGPT crops identically).
// So a prompt shorter than seq_len is exact, and generation longer than seq_len
// degrades the same way every fixed-context model does.
[[nodiscard]] inline std::vector<int> generate_absolute(GPT2& model, std::span<const int> prompt,
                                                        int n_new, float temperature, int top_k,
                                                        Generator& gen) {
    ASSERT(model.batch() == 1);
    ASSERT(!prompt.empty() && n_new >= 0);
    const int T = model.seq_len();
    const int V = model.config().vocab_size;
    ASSERT(static_cast<int>(prompt.size()) <= T);

    std::vector<int> window(prompt.begin(), prompt.end());
    std::vector<int> buf(static_cast<std::size_t>(T), 0);
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(n_new));

    for (int i = 0; i < n_new; ++i) {
        const auto len = static_cast<int>(window.size());
        std::fill(buf.begin(), buf.end(), 0);
        std::copy(window.begin(), window.end(), buf.begin());
        // Only position len-1 is ever read, and at V=50257 the full classifier is
        // ~34% of the forward. Computing one row is exact, not approximate.
        model.forward(buf.data(), nullptr, /*logits_at=*/len - 1);
        const float* last = model.acts().logits + static_cast<std::size_t>(len - 1) * V;
        const int tok = sample(last, V, temperature, top_k, gen);
        out.push_back(tok);
        if (len < T) {
            window.push_back(tok);
        } else {
            window.erase(window.begin());
            window.push_back(tok);
        }
    }
    return out;
}

}  // namespace cppgpt
