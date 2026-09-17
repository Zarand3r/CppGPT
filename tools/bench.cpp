// bench — microbenchmark for the matmul hot path (single-thread GFLOP/s).
//
// matmul is ~90% of transformer FLOPs, so it is the number the M2 perf gate
// tracks (>= 30 GFLOP/s single-thread). This establishes the baseline the
// cache-blocked / threaded matmul must beat, measured honestly: aligned buffers
// (like the model's arenas), warmup, best-of-N over repetitions (peak = least
// noise), and a printed checksum so the compiler cannot elide the work.
//
// Two extra modes exist to PRICE a staging decision rather than argue it (see
// docs/THREADING_PLAN.md):
//
//   --blocked    times a CANDIDATE kernel that reuses each loaded weight vector
//                across R input rows, against the shipping one. The candidate
//                lives here and not in the library on purpose: it is a
//                measurement of a change not yet made, and it must not be
//                mistaken for the code the parity gate covers. Bit-identity
//                against matmul_forward is checked BEFORE any timing is printed,
//                because a faster wrong kernel is not a result (L7).
//
//                READ THE R=2 AND R=4 COLUMNS WITH SUSPICION. Whether clang
//                vectorises the R-loop is a cost-model decision that depends on
//                the translation unit, not on the algorithm: in this TU R=2 and
//                R=4 scalarise (2 packed ops against ~100 scalar ones) while the
//                identical source in a standalone file vectorises and runs 2x
//                faster. R=8 vectorises in both. A low number here is therefore
//                evidence about codegen, not about register blocking --
//                disassemble before concluding otherwise (docs/measurements.md
//                M-30, and D1's own note to the same effect).
//
//   --footprint  sweeps the weight-matrix size at fixed FLOPs, to answer whether
//                this kernel is bound by cache capacity / memory bandwidth. It
//                exists because a design doc asserted that it was, and a claim
//                that a resource binds is a measurement, not a deduction (L13).
//
// Usage: bench [reps] [--blocked] [--footprint]   (reps per shape, default 20)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// --- candidate: reuse each weight row across R input rows --------------------
//
// NOT the shipping kernel. `matmul_forward` walks one input row at a time, so
// every weight row is re-loaded for every input row. Holding R input rows in
// flight reuses each loaded weight vector R times.
//
// It is BIT-IDENTICAL to the shipping kernel by construction, not by tolerance:
// each output element still sums over c in the same order, into the same eight
// lanes, combined the same way. Only the order in which output elements are
// VISITED changes, and that is not a floating-point property. This is why the
// candidate needs no new parity argument, unlike D8 which did change the order.
template <int R>
void matmul_blocked(float* out, const float* inp, const float* weight, const float* bias, int BT,
                    int C, int OC) noexcept {
    const std::size_t Cz = static_cast<std::size_t>(C);
    const std::size_t OCz = static_cast<std::size_t>(OC);
    const std::size_t rows = static_cast<std::size_t>(BT);
    constexpr int kLanes = 8;  // same lane count as matmul_forward_cpu (D8)
    std::size_t bt = 0;
    for (; bt + R <= rows; bt += R) {
        for (std::size_t oc = 0; oc < OCz; ++oc) {
            const float* w_oc = weight + oc * Cz;
            float acc[R][kLanes] = {};
            std::size_t c = 0;
            for (; c + kLanes <= Cz; c += kLanes)
                for (int r = 0; r < R; ++r)
                    for (int j = 0; j < kLanes; ++j)
                        acc[r][j] += inp[(bt + static_cast<std::size_t>(r)) * Cz + c + j] *
                                     w_oc[c + j];
            for (int r = 0; r < R; ++r) {
                const std::size_t row = bt + static_cast<std::size_t>(r);
                float sum = 0.0f;
                for (int j = 0; j < kLanes; ++j) sum += acc[r][j];
                for (std::size_t cc = c; cc < Cz; ++cc) sum += inp[row * Cz + cc] * w_oc[cc];
                out[row * OCz + oc] = bias != nullptr ? bias[oc] + sum : sum;
            }
        }
    }
    // Ragged tail: fewer than R rows left. Delegate so the tail cannot drift.
    if (bt < rows)
        matmul_forward(out + bt * OCz, inp + bt * Cz, weight, bias, 1,
                       static_cast<int>(rows - bt), C, OC);
}

template <class F>
double time_best(F kernel, int reps, double flops, const float* out, std::size_t n, double& sink) {
    kernel();  // warmup
    double best = 0.0;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        kernel();
        const auto t1 = Clock::now();
        const double g = flops / std::chrono::duration<double>(t1 - t0).count() / 1e9;
        if (g > best) best = g;
        sink += out[static_cast<std::size_t>(r) % n];
    }
    return best;
}

// Max |candidate - shipping| over the whole output. Must be exactly 0.
double max_abs_diff(const float* a, const float* b, std::size_t n) noexcept {
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        worst = (d < 0 ? -d : d) > worst ? (d < 0 ? -d : d) : worst;
    }
    return worst;
}

// Returns false if any shape failed bit-identity, so main() can exit non-zero.
[[nodiscard]] bool bench_blocked(int reps, Generator& gen, double& sink) {
    std::printf("\nmatmul_forward vs R-row blocked candidate (GFLOP/s, best of %d):\n", reps);
    std::printf("  %-11s %9s %9s %9s %9s   %s\n", "shape", "shipping", "R=2", "R=4", "R=8",
                "max|diff| vs shipping (must be 0)");
    bool all_identical = true;
    for (const Shape& s : kShapes) {
        const std::size_t rows = static_cast<std::size_t>(s.BT);
        const std::size_t C = static_cast<std::size_t>(s.C);
        const std::size_t OC = static_cast<std::size_t>(s.OC);
        const std::size_t n_out = rows * OC;

        Storage buf(rows * C + OC * C + OC + 2 * n_out);
        float* inp = buf.alloc(rows * C);
        float* weight = buf.alloc(OC * C);
        float* bias = buf.alloc(OC);
        float* out = buf.alloc(n_out);
        float* ref = buf.alloc(n_out);
        for (std::size_t i = 0; i < rows * C; ++i) inp[i] = gen.normal();
        for (std::size_t i = 0; i < OC * C; ++i) weight[i] = gen.normal() * 0.02f;
        for (std::size_t i = 0; i < OC; ++i) bias[i] = gen.normal() * 0.01f;

        // Correctness BEFORE timing, at every R that is reported.
        matmul_forward(ref, inp, weight, bias, 1, s.BT, s.C, s.OC);
        double worst = 0.0;
        matmul_blocked<2>(out, inp, weight, bias, s.BT, s.C, s.OC);
        worst = std::fmax(worst, max_abs_diff(out, ref, n_out));
        matmul_blocked<4>(out, inp, weight, bias, s.BT, s.C, s.OC);
        worst = std::fmax(worst, max_abs_diff(out, ref, n_out));
        matmul_blocked<8>(out, inp, weight, bias, s.BT, s.C, s.OC);
        worst = std::fmax(worst, max_abs_diff(out, ref, n_out));
        if (worst != 0.0) all_identical = false;

        const double flops = 2.0 * static_cast<double>(rows) * static_cast<double>(C) *
                             static_cast<double>(OC);
        const double b0 = time_best(
            [&] { matmul_forward(out, inp, weight, bias, 1, s.BT, s.C, s.OC); }, reps, flops, out,
            n_out, sink);
        const double b2 = time_best([&] { matmul_blocked<2>(out, inp, weight, bias, s.BT, s.C, s.OC); },
                                    reps, flops, out, n_out, sink);
        const double b4 = time_best([&] { matmul_blocked<4>(out, inp, weight, bias, s.BT, s.C, s.OC); },
                                    reps, flops, out, n_out, sink);
        const double b8 = time_best([&] { matmul_blocked<8>(out, inp, weight, bias, s.BT, s.C, s.OC); },
                                    reps, flops, out, n_out, sink);
        std::printf("  %-11s %9.2f %9.2f %9.2f %9.2f   %.2e%s\n", s.name, b0, b2, b4, b8, worst,
                    worst == 0.0 ? "" : "   <-- NOT BIT-IDENTICAL");
    }
    if (!all_identical)
        std::printf("  FAIL: the candidate is not bit-identical; its timings mean nothing.\n");
    std::printf(
        "  note: R=2/R=4 scalarise in this TU (clang cost model, not the algorithm); R=8 is the\n"
        "        configuration that vectorises in every TU tried. See docs/measurements.md M-30.\n");
    return all_identical;
}

// Is this kernel bound by cache capacity / memory bandwidth? Hold the FLOPs
// fixed and vary only how large the weight matrix is, by trading OC against BT.
// A kernel bound on weight footprint falls off as the matrix leaves each level.
void bench_footprint(int reps, Generator& gen, double& sink) {
    std::printf("\nweight-footprint sweep at fixed FLOPs (C=768, BT*OC held constant):\n");
    std::printf("  %6s %7s %11s %9s\n", "OC", "BT", "W bytes", "GFLOP/s");
    constexpr int C = 768;
    constexpr long long kWork = 1024LL * 3072;  // BT * OC
    for (const int OC : {16, 64, 256, 1024, 3072, 12288}) {
        const int BT = static_cast<int>(kWork / OC);
        const std::size_t rows = static_cast<std::size_t>(BT);
        const std::size_t Cz = static_cast<std::size_t>(C);
        const std::size_t OCz = static_cast<std::size_t>(OC);
        Storage buf(rows * Cz + OCz * Cz + rows * OCz);
        float* inp = buf.alloc(rows * Cz);
        float* weight = buf.alloc(OCz * Cz);
        float* out = buf.alloc(rows * OCz);
        for (std::size_t i = 0; i < rows * Cz; ++i) inp[i] = gen.normal();
        for (std::size_t i = 0; i < OCz * Cz; ++i) weight[i] = gen.normal() * 0.02f;
        const double flops = 2.0 * static_cast<double>(rows) * C * static_cast<double>(OC);
        const double best = time_best(
            [&] { matmul_forward(out, inp, weight, nullptr, 1, BT, C, OC); }, reps, flops, out,
            rows * OCz, sink);
        std::printf("  %6d %7d %8.1f KiB %9.2f\n", OC, BT,
                    static_cast<double>(OCz) * C * 4.0 / 1024.0, best);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 20;
    bool blocked = false;
    bool footprint = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--blocked") == 0) {
            blocked = true;
        } else if (std::strcmp(argv[i], "--footprint") == 0) {
            footprint = true;
        } else {
            reps = std::atoi(argv[i]);
        }
    }
    if (reps <= 0) {
        std::fprintf(stderr, "bench: reps must be positive\n");
        return 2;
    }
    std::printf("matmul_forward single-thread (%d reps/shape):\n", reps);
    Generator gen(1234ULL);
    double sink = 0.0;
    for (const Shape& s : kShapes) bench_shape(s, reps, gen, sink);
    bool ok = true;
    if (blocked) ok = bench_blocked(reps, gen, sink);
    if (footprint) bench_footprint(reps, gen, sink);
    std::printf("(checksum %.3e)\n", sink);
    return ok ? 0 : 1;
}
