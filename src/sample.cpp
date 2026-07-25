#include "cppgpt/sample.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "cppgpt/core.hpp"

namespace cppgpt {

int argmax(const float* logits, int V) noexcept {
    ASSERT(logits != nullptr && V > 0);
    int best = 0;
    for (int i = 1; i < V; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

int sample(const float* logits, int V, float temperature, int top_k, Generator& gen) {
    ASSERT(logits != nullptr && V > 0 && temperature > 0.0f);
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();

    // top_k == 1 is greedy decoding. It cannot go through the threshold path
    // below: that keeps every logit >= the k-th largest, so a TIED maximum would
    // leave several tokens eligible and softmax would spread over them. argmax
    // breaks ties by lowest index, which is what greedy decoding must mean.
    if (top_k == 1) return argmax(logits, V);

    // top-k threshold: only logits >= the k-th largest are eligible.
    float thresh = kNegInf;
    if (top_k > 0 && top_k < V) {
        std::vector<float> tmp(logits, logits + V);
        std::nth_element(tmp.begin(), tmp.begin() + (V - top_k), tmp.end());
        thresh = tmp[static_cast<std::size_t>(V - top_k)];
    }

    // Softmax over the eligible logits/temperature (max-subtraction for stability).
    const float inv_t = 1.0f / temperature;
    float maxv = kNegInf;
    for (int i = 0; i < V; ++i)
        if (logits[i] >= thresh) maxv = std::fmax(maxv, logits[i] * inv_t);
    std::vector<float> probs(static_cast<std::size_t>(V), 0.0f);
    double sum = 0.0;
    for (int i = 0; i < V; ++i) {
        if (logits[i] >= thresh) {
            const float e = std::exp(logits[i] * inv_t - maxv);
            probs[static_cast<std::size_t>(i)] = e;
            sum += e;
        }
    }

    // A non-finite or empty distribution means the numerics upstream already
    // broke (diverged LR, corrupt weights): every NaN comparison above is false,
    // so nothing would be eligible and the walk below would silently return
    // token 0 forever. That is semantic corruption — fail fast, never paper over it.
    ASSERT_MSG(sum > 0.0 && std::isfinite(sum), "sample: logits are not a finite distribution");

    // Draw u in [0, sum); walk the cumulative distribution over eligible tokens.
    const double u = static_cast<double>(gen.uniform()) * sum;
    double cum = 0.0;
    int last_eligible = 0;
    for (int i = 0; i < V; ++i) {
        const float p = probs[static_cast<std::size_t>(i)];
        if (p <= 0.0f) continue;
        last_eligible = i;
        cum += p;
        if (u < cum) return i;
    }
    return last_eligible;  // fp fallback: u ≈ sum
}

}  // namespace cppgpt
