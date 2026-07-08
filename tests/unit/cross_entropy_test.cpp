// cross_entropy_forward / cross_entropy_backward (softmax-fused).
//
// forward: losses[b,t] = −log(probs[b,t,target]). backward: the gradient of the
// MEAN cross-entropy w.r.t. the LOGITS, fusing the softmax backward —
// dlogits = (probs − onehot)/(B·T). Pinned by: uniform probs ⇒ loss = log(V);
// exact (probs−onehot)/(B·T) with the per-row gradient summing to zero; and a
// finite-difference gradcheck through the full logits→softmax→CE→mean chain
// (which verifies the softmax/CE fusion is the true gradient).
#include "cppgpt/ops.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "cppgpt/random.hpp"
#include "tests/check.hpp"
#include "tests/numeric.hpp"

using namespace cppgpt;
using cppgpt::test::grad_check;
using cppgpt::test::rel_close;

int main() {
    // ---- forward: uniform probs ⇒ loss = log(V) ----
    {
        const int B = 1, T = 1, V = 4;
        std::vector<float> probs(static_cast<std::size_t>(V), 1.0f / static_cast<float>(V));
        std::vector<int> targets = {2};
        std::vector<float> losses(1);
        cross_entropy_forward(losses.data(), probs.data(), targets.data(), B, T, V);
        CHECK(rel_close(losses[0], std::log(static_cast<double>(V)), 1e-5));
    }

    // ---- backward: (probs − onehot)/(B·T); per-row gradient sums to zero ----
    {
        const int B = 1, T = 1, V = 4;  // B·T = 1
        std::vector<float> probs = {0.1f, 0.2f, 0.3f, 0.4f};
        std::vector<int> targets = {1};
        std::vector<float> dlogits(static_cast<std::size_t>(V), 0.0f);
        cross_entropy_backward(dlogits.data(), probs.data(), targets.data(), B, T, V);
        CHECK(rel_close(dlogits[0], 0.1, 1e-5));
        CHECK(rel_close(dlogits[1], 0.2 - 1.0, 1e-5));  // the target channel
        CHECK(rel_close(dlogits[2], 0.3, 1e-5));
        CHECK(rel_close(dlogits[3], 0.4, 1e-5));
        double s = 0.0;
        for (float x : dlogits) s += x;
        CHECK(std::fabs(s) < 1e-6);  // softmax gradient sums to zero
    }

    // ---- fused gradcheck: logits → softmax → CE → mean ----
    {
        const int B = 2, T = 2, V = 5;
        const std::size_t BT = static_cast<std::size_t>(B * T);
        const std::size_t n = static_cast<std::size_t>(B * T * V);
        const std::size_t vz = static_cast<std::size_t>(V);
        Generator gen(0xCE10ULL);
        std::vector<float> logits(n), probs(n), losses(BT), dlogits(n, 0.0f);
        std::vector<int> targets(BT);
        for (auto& x : logits) x = gen.normal();
        for (auto& t : targets) t = static_cast<int>(gen.uniform_int(0, V - 1));

        for (std::size_t bt = 0; bt < BT; ++bt)
            softmax_forward(probs.data() + bt * vz, logits.data() + bt * vz, V);
        cross_entropy_backward(dlogits.data(), probs.data(), targets.data(), B, T, V);

        auto loss = [&]() {
            for (std::size_t bt = 0; bt < BT; ++bt)
                softmax_forward(probs.data() + bt * vz, logits.data() + bt * vz, V);
            cross_entropy_forward(losses.data(), probs.data(), targets.data(), B, T, V);
            double tot = 0.0;
            for (std::size_t bt = 0; bt < BT; ++bt) tot += losses[bt];
            return tot / static_cast<double>(BT);
        };
        CHECK(grad_check(loss, logits.data(), dlogits.data(), n) < 2e-3);
    }

    return cppgpt::test::summary();
}
