// residual_forward / residual_backward (out = a + b). Linear, so exact checks.
#include "cppgpt/ops.hpp"

#include <vector>

#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

int main() {
    const int N = 16;
    Generator gen(0x5151ULL);
    std::vector<float> a(N), b(N), out(N);
    for (auto& x : a) x = gen.normal();
    for (auto& x : b) x = gen.normal();

    // forward: out == a + b exactly.
    residual_forward(out.data(), a.data(), b.data(), N);
    bool fwd_ok = true;
    for (int i = 0; i < N; ++i) fwd_ok = fwd_ok && (out[i] == a[i] + b[i]);
    CHECK(fwd_ok);

    // in-place: out == a is allowed.
    std::vector<float> a2 = a;
    residual_forward(a2.data(), a2.data(), b.data(), N);
    bool inplace_ok = true;
    for (int i = 0; i < N; ++i) inplace_ok = inplace_ok && (a2[i] == a[i] + b[i]);
    CHECK(inplace_ok);

    // backward: both grads receive dout; accumulates.
    std::vector<float> dout(N), da(N, 0.0f), db(N, 0.0f);
    for (auto& x : dout) x = gen.normal();
    residual_backward(da.data(), db.data(), dout.data(), N);
    bool bwd_ok = true;
    for (int i = 0; i < N; ++i) bwd_ok = bwd_ok && (da[i] == dout[i]) && (db[i] == dout[i]);
    CHECK(bwd_ok);

    // accumulation: a second backward doubles both.
    residual_backward(da.data(), db.data(), dout.data(), N);
    bool accum_ok = true;
    for (int i = 0; i < N; ++i)
        accum_ok = accum_ok && (da[i] == 2.0f * dout[i]) && (db[i] == 2.0f * dout[i]);
    CHECK(accum_ok);

    return cppgpt::test::summary();
}
