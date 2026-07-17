// cppgpt fixture verification: load a canonical-GPT-2 PyTorch parity fixture and
// compare tensors element-wise. The fixture is produced by scripts/gen_fixtures.py
// (run in the torch venv) and committed, so the C++ gate runs under plain
// `bazel test` with no torch dependency.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/model.hpp"

namespace cppgpt {

// A loaded gpt2_parity.bin: config, inputs, initial parameters, and the PyTorch
// reference outputs (logits + gradients at the initial weights, and the loss after
// each of `n_steps` AdamW steps). Parameters/gradients are in cppgpt .bin order.
struct ParityFixture {
    Config cfg{};
    int B = 0, T = 0, n_steps = 0;
    std::size_t n_params = 0;
    std::vector<int> tokens;    // [B*T]
    std::vector<int> targets;   // [B*T]
    std::vector<float> params;  // [n_params]  initial weights
    std::vector<float> logits;  // [B*T*V]     forward output at init
    std::vector<float> grads;   // [n_params]  gradients at init
    std::vector<float> losses;  // [n_steps]   AdamW loss trajectory
};

// Load a fixture; aborts (fail-fast) on a missing file, bad magic/version, or a
// truncated read.
[[nodiscard]] inline ParityFixture load_parity_fixture(const std::string& path) {
    constexpr std::uint32_t kMagic = 0x43475446u;  // "CGTF"
    std::FILE* f = std::fopen(path.c_str(), "rb");
    ASSERT_MSG(f != nullptr, "parity fixture: cannot open file");
    const auto rd = [&](void* p, std::size_t n) {
        ASSERT_MSG(std::fread(p, 1, n, f) == n, "parity fixture: truncated read");
    };
    std::uint32_t magic = 0, version = 0;
    rd(&magic, 4);
    rd(&version, 4);
    ASSERT_MSG(magic == kMagic && version == 1u, "parity fixture: bad magic/version");
    int h[9];
    rd(h, sizeof(h));
    ParityFixture fx;
    fx.cfg.max_seq_len = h[0];
    fx.cfg.vocab_size = h[1];
    fx.cfg.n_layer = h[2];
    fx.cfg.n_head = h[3];
    fx.cfg.n_embd = h[4];
    fx.B = h[5];
    fx.T = h[6];
    fx.n_steps = h[7];
    fx.n_params = static_cast<std::size_t>(h[8]);
    const std::size_t bt = static_cast<std::size_t>(fx.B) * static_cast<std::size_t>(fx.T);
    const std::size_t nlog = bt * static_cast<std::size_t>(fx.cfg.vocab_size);
    fx.tokens.resize(bt);
    fx.targets.resize(bt);
    fx.params.resize(fx.n_params);
    fx.logits.resize(nlog);
    fx.grads.resize(fx.n_params);
    fx.losses.resize(static_cast<std::size_t>(fx.n_steps));
    rd(fx.tokens.data(), bt * sizeof(int));
    rd(fx.targets.data(), bt * sizeof(int));
    rd(fx.params.data(), fx.n_params * sizeof(float));
    rd(fx.logits.data(), nlog * sizeof(float));
    rd(fx.grads.data(), fx.n_params * sizeof(float));
    rd(fx.losses.data(), static_cast<std::size_t>(fx.n_steps) * sizeof(float));
    std::fclose(f);
    return fx;
}

// Max absolute difference over n elements.
[[nodiscard]] inline float max_abs_diff(const float* a, const float* b, std::size_t n) {
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace cppgpt
