// attention_forward / attention_backward: causal multi-head self-attention.
//
// `inp` is [B,T,3C] (concatenated q,k,v; head h uses columns [h·hs,(h+1)·hs) with
// hs=C/NH). Forward pinned by: T=1 ⇒ out equals the value vector (softmax of a
// single score is 1), and causality (out[t] cannot depend on tokens > t).
// Backward (the most error-prone op) pinned by a finite-difference gradient
// check on dinp.
#include "cppgpt/ops.hpp"

#include <cstddef>
#include <vector>

#include "cppgpt/random.hpp"
#include "tests/check.hpp"
#include "tests/numeric.hpp"

using namespace cppgpt;
using cppgpt::test::dot;
using cppgpt::test::grad_check;

int main() {
    // ---- T=1: attention output equals the value vector ----
    {
        const int B = 1, T = 1, C = 4, NH = 2;
        std::vector<float> inp(static_cast<std::size_t>(3 * C)), out(static_cast<std::size_t>(C)),
            preatt(static_cast<std::size_t>(B * NH * T * T)),
            att(static_cast<std::size_t>(B * NH * T * T));
        Generator gen(0x7A77ULL);
        for (auto& x : inp) x = gen.normal();
        attention_forward(out.data(), preatt.data(), att.data(), inp.data(), B, T, C, NH,
                          Device::CPU);
        for (int i = 0; i < C; ++i)
            CHECK(out[static_cast<std::size_t>(i)] == inp[static_cast<std::size_t>(2 * C + i)]);
    }

    // ---- causality: out[t] is independent of tokens > t ----
    {
        const int B = 1, T = 3, C = 4, NH = 2;
        const std::size_t n_in = static_cast<std::size_t>(B * T * 3 * C);
        const std::size_t n_out = static_cast<std::size_t>(B * T * C);
        const std::size_t n_att = static_cast<std::size_t>(B * NH * T * T);
        Generator gen(0xCA05ULL);
        std::vector<float> inp(n_in), out(n_out), preatt(n_att), att(n_att);
        for (auto& x : inp) x = gen.normal();
        attention_forward(out.data(), preatt.data(), att.data(), inp.data(), B, T, C, NH,
                          Device::CPU);

        // Perturb token 2's whole qkv; out at t=0,1 must be bitwise unchanged.
        std::vector<float> inp2 = inp;
        for (int j = 0; j < 3 * C; ++j) inp2[static_cast<std::size_t>(2 * 3 * C + j)] += 1.0f;
        std::vector<float> out2(n_out), pa(n_att), at(n_att);
        attention_forward(out2.data(), pa.data(), at.data(), inp2.data(), B, T, C, NH, Device::CPU);

        bool causal = true;
        for (int t = 0; t < 2; ++t)
            for (int i = 0; i < C; ++i)
                causal = causal && (out2[static_cast<std::size_t>(t * C + i)] ==
                                    out[static_cast<std::size_t>(t * C + i)]);
        CHECK(causal);
        bool changed = false;  // sanity: t=2 did move
        for (int i = 0; i < C; ++i)
            changed = changed || (out2[static_cast<std::size_t>(2 * C + i)] !=
                                  out[static_cast<std::size_t>(2 * C + i)]);
        CHECK(changed);
    }

    // ---- backward: finite-difference gradient check on dinp ----
    {
        const int B = 2, T = 3, C = 4, NH = 2;
        const std::size_t n_in = static_cast<std::size_t>(B * T * 3 * C);
        const std::size_t n_out = static_cast<std::size_t>(B * T * C);
        const std::size_t n_att = static_cast<std::size_t>(B * NH * T * T);
        Generator gen(0xA77BULL);
        std::vector<float> inp(n_in), dout(n_out), out(n_out), preatt(n_att), att(n_att);
        for (auto& x : inp) x = gen.normal();
        for (auto& x : dout) x = gen.normal();

        attention_forward(out.data(), preatt.data(), att.data(), inp.data(), B, T, C, NH,
                          Device::CPU);
        std::vector<float> dinp(n_in, 0.0f), datt(n_att), dpreatt(n_att);
        attention_backward(dinp.data(), datt.data(), dpreatt.data(), dout.data(), inp.data(),
                           att.data(), B, T, C, NH, Device::CPU);

        auto loss = [&]() {
            attention_forward(out.data(), preatt.data(), att.data(), inp.data(), B, T, C, NH,
                              Device::CPU);
            return dot(dout.data(), out.data(), n_out);
        };
        CHECK(grad_check(loss, inp.data(), dinp.data(), n_in) < 2e-2);
    }

    return cppgpt::test::summary();
}
