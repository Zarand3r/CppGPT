// bench — microbenchmark for the matmul hot path (single-thread GFLOP/s).
//
// matmul is ~90% of transformer FLOPs, so it is the number the M2 perf gate
// tracks (>= 30 GFLOP/s single-thread). This establishes the baseline the
// cache-blocked / threaded matmul must beat, measured honestly: aligned buffers
// (like the model's arenas), warmup, best-of-N over repetitions (peak = least
// noise), and a printed checksum so the compiler cannot elide the work.
//
// Usage: bench [reps]   (reps per shape, default 20)
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "cppgpt/ops.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/storage.hpp"

namespace {
using namespace cppgpt;
using Clock = std::chrono::steady_clock;

struct Shape {
    const char* name;
    int BT;  // rows = batch * seq_len
    int C;   // input channels
    int OC;  // output channels
};

// GPT-2 124M projection shapes (C = 768), at BT = 1024 tokens.
constexpr Shape kShapes[] = {
    {"attn_qkv  ", 1024, 768, 2304},   // qkv:   C   -> 3C
    {"attn_proj ", 1024, 768, 768},    // proj:  C   -> C
    {"mlp_fc    ", 1024, 768, 3072},   // fc:    C   -> 4C
    {"mlp_proj  ", 1024, 3072, 768},   // fcproj: 4C -> C
};

void bench_shape(const Shape& s, int reps, Generator& gen, double& sink) {
    const std::size_t rows = static_cast<std::size_t>(s.BT);
    const std::size_t C = static_cast<std::size_t>(s.C);
    const std::size_t OC = static_cast<std::size_t>(s.OC);

    Storage buf(rows * C + OC * C + OC + rows * OC);  // 64B-aligned, one arena
    float* inp = buf.alloc(rows * C);
    float* weight = buf.alloc(OC * C);
    float* bias = buf.alloc(OC);
    float* out = buf.alloc(rows * OC);
    for (std::size_t i = 0; i < rows * C; ++i) inp[i] = gen.normal();
    for (std::size_t i = 0; i < OC * C; ++i) weight[i] = gen.normal() * 0.02f;
    for (std::size_t i = 0; i < OC; ++i) bias[i] = 0.0f;

    matmul_forward(out, inp, weight, bias, 1, s.BT, s.C, s.OC);  // warmup (also fills caches)

    const double flops = 2.0 * static_cast<double>(rows) * static_cast<double>(C) *
                         static_cast<double>(OC);  // multiply-add = 2 FLOP
    double best = 0.0, sum = 0.0;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        matmul_forward(out, inp, weight, bias, 1, s.BT, s.C, s.OC);
        const auto t1 = Clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        const double gflops = flops / sec / 1e9;
        if (gflops > best) best = gflops;
        sum += gflops;
        sink += out[static_cast<std::size_t>(r) % (rows * OC)];  // defeat dead-code elimination
    }
    std::printf("  %s  BT=%d C=%4d OC=%4d  %6.2f GFLOP  best %6.2f  avg %6.2f  GFLOP/s\n", s.name,
                s.BT, s.C, s.OC, flops / 1e9, best, sum / reps);
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = (argc > 1) ? std::atoi(argv[1]) : 20;
    std::printf("matmul_forward single-thread (%d reps/shape):\n", reps);
    Generator gen(1234ULL);
    double sink = 0.0;
    for (const Shape& s : kShapes) bench_shape(s, reps, gen, sink);
    std::printf("(checksum %.3e)\n", sink);
    return 0;
}
