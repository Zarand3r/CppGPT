// embedding_forward / embedding_backward: token + learned-position embedding.
//   out[b,t,:] = wte[tokens[b,t], :] + wpe[t, :]
// Linear in the embedding tables, so exact fixtures suffice (no gradcheck). The
// backward is a scatter-add into dwte/dwpe; the key case is a repeated token,
// whose row must accumulate contributions from every position that used it.
#include "cppgpt/ops.hpp"

#include <vector>

#include "tests/check.hpp"

using namespace cppgpt;

int main() {
    const int B = 1, T = 2, C = 2, V = 3;

    // wte[V,C]: token rows; wpe[T,C]: position rows.
    std::vector<float> wte = {10, 11, 20, 21, 30, 31};
    std::vector<float> wpe = {100, 101, 200, 201};

    // ---- forward ----
    {
        std::vector<int> tokens = {2, 0};  // pos0 -> token2, pos1 -> token0
        std::vector<float> out(static_cast<std::size_t>(B * T * C));
        embedding_forward(out.data(), tokens.data(), wte.data(), wpe.data(), B, T, C, V, Device::CPU);
        // out[0] = wte[2] + wpe[0] = [130,132]; out[1] = wte[0] + wpe[1] = [210,212]
        CHECK(out[0] == 130.0f && out[1] == 132.0f);
        CHECK(out[2] == 210.0f && out[3] == 212.0f);
    }

    // ---- backward: scatter-add, with a repeated token to test accumulation ----
    {
        std::vector<int> tokens = {1, 1};        // both positions use token 1
        std::vector<float> dout = {1, 2, 3, 4};  // dout[pos0]=[1,2], dout[pos1]=[3,4]
        std::vector<float> dwte(static_cast<std::size_t>(V * C), 0.0f);
        std::vector<float> dwpe(static_cast<std::size_t>(T * C), 0.0f);
        embedding_backward(dwte.data(), dwpe.data(), tokens.data(), dout.data(), B, T, C, V,
                           Device::CPU);
        // token 1 row (offset C=2) accumulates both positions: [1+3, 2+4] = [4,6].
        CHECK(dwte[2] == 4.0f && dwte[3] == 6.0f);
        CHECK(dwte[0] == 0.0f && dwte[1] == 0.0f && dwte[4] == 0.0f && dwte[5] == 0.0f);
        // position rows each get their own dout.
        CHECK(dwpe[0] == 1.0f && dwpe[1] == 2.0f && dwpe[2] == 3.0f && dwpe[3] == 4.0f);

        // accumulation: a second backward doubles the gradients.
        embedding_backward(dwte.data(), dwpe.data(), tokens.data(), dout.data(), B, T, C, V,
                           Device::CPU);
        CHECK(dwte[2] == 8.0f && dwte[3] == 12.0f);
        CHECK(dwpe[0] == 2.0f && dwpe[3] == 8.0f);
    }

    return cppgpt::test::summary();
}
