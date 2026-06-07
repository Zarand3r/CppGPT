// cppgpt RNG: a single explicit generator. There is no global RNG and no
// default constructor — every stochastic op takes a Generator& so runs are
// reproducible from a seed (PLAN invariant 5). For independent streams,
// construct separate Generators from derived seeds rather than sharing one.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>

#include "cppgpt/core.hpp"

namespace cppgpt {

class Generator {
public:
    explicit Generator(std::uint64_t seed) noexcept : engine_(seed) {}

    // Raw 64-bit draw.
    std::uint64_t next_u64() noexcept { return engine_(); }

    // Uniform float in [0, 1): 24 high bits scaled by 2^-24 (full float mantissa).
    float uniform() noexcept {
        const auto bits = static_cast<std::uint32_t>(engine_() >> 40);
        return static_cast<float>(bits) * (1.0f / 16777216.0f);
    }

    // Uniform integer in [low, high] inclusive. Modulo bias is negligible at the
    // index/vocab sizes we draw for (shuffles, sampling) and accepted in v1.
    std::int64_t uniform_int(std::int64_t low, std::int64_t high) noexcept {
        ASSERT(low <= high);
        const auto span = static_cast<std::uint64_t>(high - low) + 1U;
        return low + static_cast<std::int64_t>(engine_() % span);
    }

    // Standard normal N(0, 1). Box-Muller, caching the second variate.
    float normal() noexcept {
        if (has_cached_) {
            has_cached_ = false;
            return cached_;
        }
        float u1 = 0.0f;
        do {
            u1 = uniform();
        } while (u1 <= 0.0f);  // avoid log(0)
        const float u2 = uniform();
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float theta = 6.283185307179586f * u2;  // 2*pi
        cached_ = radius * std::sin(theta);
        has_cached_ = true;
        return radius * std::cos(theta);
    }

private:
    std::mt19937_64 engine_;
    float cached_ = 0.0f;
    bool has_cached_ = false;
    // NOTE: exact engine-state serialization for checkpoint resume (M2) will use
    // the standard operator<</>> on std::mt19937_64; not needed before then.
};

}  // namespace cppgpt
