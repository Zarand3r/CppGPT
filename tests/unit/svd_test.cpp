// Singular value decomposition of a head's circuit tables (M7-1).
//
// WHY THIS TEST EXISTS BEFORE THE CODE. An SVD is unusually easy to get
// plausibly wrong: a routine that returns *a* valid decomposition of the wrong
// matrix, or an orthonormal basis with the singular values misordered, produces
// output that looks entirely reasonable and is useless. Two gates make that
// impossible to fake:
//
//   RECONSTRUCTION   u * diag(s) * vt must equal the input. Nothing that
//                    decomposes the wrong thing survives this.
//   KNOWN ANSWER     a matrix built from chosen singular values under known
//                    rotations must give those values back. Reconstruction alone
//                    passes for a routine that returns (I, a_as_diagonal, I) on a
//                    matrix that happens to be diagonal, and for other identities
//                    that are true but not a decomposition into singular values.
//
// Both are needed. Neither is sufficient alone.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <vector>

#include "cppgpt/interp/interpret.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

namespace {

// max |a - u diag(s) vt| over the whole matrix.
double reconstruction_error(const float* a, const float* u, const float* s, const float* vt,
                            int n) {
    double worst = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k)
                acc += static_cast<double>(u[static_cast<std::size_t>(i) * n + k]) *
                       static_cast<double>(s[k]) *
                       static_cast<double>(vt[static_cast<std::size_t>(k) * n + j]);
            worst = std::fmax(worst, std::fabs(acc - static_cast<double>(
                                                         a[static_cast<std::size_t>(i) * n + j])));
        }
    return worst;
}

// max |M^T M - I| for a matrix whose COLUMNS should be orthonormal.
double column_orthonormality_error(const float* m, int n) {
    double worst = 0.0;
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q) {
            double acc = 0.0;
            for (int i = 0; i < n; ++i)
                acc += static_cast<double>(m[static_cast<std::size_t>(i) * n + p]) *
                       static_cast<double>(m[static_cast<std::size_t>(i) * n + q]);
            worst = std::fmax(worst, std::fabs(acc - (p == q ? 1.0 : 0.0)));
        }
    return worst;
}

// max |M M^T - I| for a matrix whose ROWS should be orthonormal (vt).
double row_orthonormality_error(const float* m, int n) {
    double worst = 0.0;
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q) {
            double acc = 0.0;
            for (int j = 0; j < n; ++j)
                acc += static_cast<double>(m[static_cast<std::size_t>(p) * n + j]) *
                       static_cast<double>(m[static_cast<std::size_t>(q) * n + j]);
            worst = std::fmax(worst, std::fabs(acc - (p == q ? 1.0 : 0.0)));
        }
    return worst;
}

}  // namespace

int main() {
    Generator g(20260906ULL);

    // ---- RECONSTRUCTION on a random matrix ----
    {
        const int n = 12;
        std::vector<float> a(static_cast<std::size_t>(n) * n);
        for (auto& x : a) x = static_cast<float>(g.normal() * 2.0);

        std::vector<float> u(a.size()), s(static_cast<std::size_t>(n)), vt(a.size());
        svd_square(a.data(), n, u.data(), s.data(), vt.data());

        CHECK(reconstruction_error(a.data(), u.data(), s.data(), vt.data(), n) < 1e-4);
        CHECK(column_orthonormality_error(u.data(), n) < 1e-4);
        CHECK(row_orthonormality_error(vt.data(), n) < 1e-4);

        bool ordered = true, nonneg = true;
        for (int i = 0; i < n; ++i) nonneg = nonneg && s[static_cast<std::size_t>(i)] >= 0.0f;
        for (int i = 1; i < n; ++i)
            ordered = ordered && s[static_cast<std::size_t>(i - 1)] >= s[static_cast<std::size_t>(i)];
        CHECK(ordered);
        CHECK(nonneg);
    }

    // ---- KNOWN ANSWER: build a matrix with chosen singular values ----
    // A = R1 * diag(w) * R2, with R1 and R2 plane rotations. The singular values
    // of A are exactly w, whatever the rotations are. Reconstruction alone cannot
    // catch a routine that returns the right product with the wrong spectrum.
    {
        const int n = 4;
        const double want[4] = {7.0, 3.0, 2.0, 0.5};
        std::vector<float> d(static_cast<std::size_t>(n) * n, 0.0f);
        for (int i = 0; i < n; ++i)
            d[static_cast<std::size_t>(i) * n + i] = static_cast<float>(want[i]);

        const auto rotate_rows = [&](std::vector<float>& m, int p, int q, double th) {
            const double c = std::cos(th), sn = std::sin(th);
            for (int j = 0; j < n; ++j) {
                const double a1 = m[static_cast<std::size_t>(p) * n + j];
                const double b1 = m[static_cast<std::size_t>(q) * n + j];
                m[static_cast<std::size_t>(p) * n + j] = static_cast<float>(c * a1 - sn * b1);
                m[static_cast<std::size_t>(q) * n + j] = static_cast<float>(sn * a1 + c * b1);
            }
        };
        const auto rotate_cols = [&](std::vector<float>& m, int p, int q, double th) {
            const double c = std::cos(th), sn = std::sin(th);
            for (int i = 0; i < n; ++i) {
                const double a1 = m[static_cast<std::size_t>(i) * n + p];
                const double b1 = m[static_cast<std::size_t>(i) * n + q];
                m[static_cast<std::size_t>(i) * n + p] = static_cast<float>(c * a1 - sn * b1);
                m[static_cast<std::size_t>(i) * n + q] = static_cast<float>(sn * a1 + c * b1);
            }
        };
        rotate_rows(d, 0, 2, 0.7);
        rotate_rows(d, 1, 3, -1.1);
        rotate_cols(d, 0, 1, 0.4);
        rotate_cols(d, 2, 3, 1.3);

        std::vector<float> u(d.size()), s(static_cast<std::size_t>(n)), vt(d.size());
        svd_square(d.data(), n, u.data(), s.data(), vt.data());

        bool matched = true;
        for (int i = 0; i < n; ++i)
            matched = matched && std::fabs(static_cast<double>(s[static_cast<std::size_t>(i)]) -
                                           want[i]) < 1e-3;
        CHECK(matched);
        CHECK(reconstruction_error(d.data(), u.data(), s.data(), vt.data(), n) < 1e-4);
    }

    // ---- DEGENERATE INPUTS terminate and stay well-formed ----
    //
    // The sizes here are the point. An adversarial review found svd_square
    // ABORTING THE PROCESS on exactly-rank-deficient input from n=6 upward, and
    // this block passed only because it was pinned to n=5 -- below the threshold
    // where the defect appears. A degenerate-input test that runs at one small
    // size is not a degenerate-input test. Every size below is checked.
    for (const int n : {5, 6, 8, 16, 32, 65}) {
        std::vector<float> z(static_cast<std::size_t>(n) * n, 0.0f);
        std::vector<float> u(z.size()), s(static_cast<std::size_t>(n)), vt(z.size());
        svd_square(z.data(), n, u.data(), s.data(), vt.data());
        bool all_zero = true;
        for (int i = 0; i < n; ++i) all_zero = all_zero && s[static_cast<std::size_t>(i)] == 0.0f;
        CHECK(all_zero);
        // Even for the zero matrix the bases must be valid, or downstream code
        // reading a "direction" gets uninitialised memory.
        CHECK(column_orthonormality_error(u.data(), n) < 1e-4);
        CHECK(row_orthonormality_error(vt.data(), n) < 1e-4);

        // Rank 1: one nonzero singular value, the rest zero.
        std::vector<float> r1(static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                r1[static_cast<std::size_t>(i) * n + j] = static_cast<float>((i + 1) * (j + 1));
        svd_square(r1.data(), n, u.data(), s.data(), vt.data());
        CHECK(s[0] > 1.0f);
        bool rest_zero = true;
        for (int i = 1; i < n; ++i)
            rest_zero = rest_zero && s[static_cast<std::size_t>(i)] < 1e-3f;
        CHECK(rest_zero);
        CHECK(reconstruction_error(r1.data(), u.data(), s.data(), vt.data(), n) < 1e-3);

        // An exactly-rank-1 matrix of identical entries: the case that aborted.
        // Its answer is known exactly -- s = {n, 0, 0, ...}.
        std::vector<float> ones(static_cast<std::size_t>(n) * n, 1.0f);
        svd_square(ones.data(), n, u.data(), s.data(), vt.data());
        CHECK(std::fabs(static_cast<double>(s[0]) - n) < 1e-3);
        bool ones_rest_zero = true;
        for (int i = 1; i < n; ++i)
            ones_rest_zero = ones_rest_zero && s[static_cast<std::size_t>(i)] < 1e-3f;
        CHECK(ones_rest_zero);
        CHECK(reconstruction_error(ones.data(), u.data(), s.data(), vt.data(), n) < 1e-3);
    }

    // ---- THE INPUT IS NOT MODIFIED ----
    // The signature takes a const pointer; this is the check that it means it,
    // since a routine working in place through a cast would still compile.
    {
        const int n = 6;
        std::vector<float> a(static_cast<std::size_t>(n) * n);
        for (auto& x : a) x = static_cast<float>(g.normal());
        const std::vector<float> before = a;
        std::vector<float> u(a.size()), s(static_cast<std::size_t>(n)), vt(a.size());
        svd_square(a.data(), n, u.data(), s.data(), vt.data());
        bool unchanged = true;
        for (std::size_t i = 0; i < a.size(); ++i) unchanged = unchanged && (a[i] == before[i]);
        CHECK(unchanged);
    }

    // ---- ON A REAL HEAD: purity, and the circuit's rank bound ----
    {
        Config cfg{};
        cfg.max_seq_len = 8;
        cfg.vocab_size = 11;
        cfg.n_layer = 2;
        cfg.n_head = 2;
        cfg.n_embd = 16;
        const int V = cfg.vocab_size;
        GPT2 m(cfg, 1, 5);
        Generator g2(31337ULL);
        m.init_weights(g2);

        const std::size_t nn = circuit_floats(cfg);
        std::vector<float> u(nn), s(static_cast<std::size_t>(V)), vt(nn);
        std::vector<float> u2(nn), s2(static_cast<std::size_t>(V)), vt2(nn);

        svd_circuit(m, 1, 0, CircuitKind::Ov, u.data(), s.data(), vt.data());

        // A forward in between must not move it: the decomposition describes the
        // head, and that is the entire claim the panel makes.
        std::vector<int> tok(5);
        for (auto& x : tok) x = static_cast<int>(g2.uniform_int(0, V - 1));
        m.forward(tok.data(), nullptr);
        svd_circuit(m, 1, 0, CircuitKind::Ov, u2.data(), s2.data(), vt2.data());
        bool same = true;
        for (int i = 0; i < V; ++i)
            same = same && (s[static_cast<std::size_t>(i)] == s2[static_cast<std::size_t>(i)]);
        CHECK(same);

        // The OV circuit factors through a head of size hs, so its rank cannot
        // exceed hs however large the vocabulary is. Column-centring removes one
        // more, so the bound is hs. A routine returning V nonzero singular values
        // is decomposing something other than this circuit.
        const int hs = cfg.n_embd / cfg.n_head;
        int above = 0;
        for (int i = 0; i < V; ++i)
            if (s[static_cast<std::size_t>(i)] > 1e-3f) ++above;
        CHECK(above <= hs);
        CHECK(above > 0);  // else the bound is satisfied by an all-zero spectrum
    }

    // ---- ORIENTATION: which singular axis is the source side ----
    //
    // Adversarial review found the viewer's "reads" and "writes" columns
    // swapped, which inverted the published example in M-23. Nothing in the test
    // suite could catch it, because both sides are valid orthonormal bases and
    // reconstruction holds either way.
    //
    // The invariant that settles it: ov_circuit COLUMN-centres its table, so
    // every column sums to zero over sources. A[:,j] summing to zero forces each
    // LEFT singular vector to be zero-sum, and leaves the right ones unconstrained.
    // So u's columns index the source axis -- what the head reads -- and vt's
    // rows index the target axis. This is a property of the data, not of the
    // implementation, so it cannot be satisfied by relabelling.
    {
        Config cfg{};
        cfg.max_seq_len = 8;
        cfg.vocab_size = 11;
        cfg.n_layer = 2;
        cfg.n_head = 2;
        cfg.n_embd = 16;
        const int V = cfg.vocab_size;
        GPT2 m(cfg, 1, 5);
        Generator g4(909ULL);
        m.init_weights(g4);

        std::vector<float> u(circuit_floats(cfg)), s(static_cast<std::size_t>(V)),
            vt(circuit_floats(cfg));
        svd_circuit(m, 0, 0, CircuitKind::Ov, u.data(), s.data(), vt.data());

        double worst_left = 0.0, best_right = 0.0;
        for (int k = 0; k < V; ++k) {
            if (s[static_cast<std::size_t>(k)] < 1e-4f) continue;  // null space is arbitrary
            double su_ = 0.0, sv_ = 0.0;
            for (int i = 0; i < V; ++i) {
                su_ += static_cast<double>(u[static_cast<std::size_t>(i) * V + k]);
                sv_ += static_cast<double>(vt[static_cast<std::size_t>(k) * V + i]);
            }
            worst_left = std::fmax(worst_left, std::fabs(su_));
            best_right = std::fmax(best_right, std::fabs(sv_));
        }
        CHECK(worst_left < 1e-4);   // left vectors are zero-sum: the source axis
        CHECK(best_right > 1e-3);   // right ones are not, or the check is vacuous
    }

    // ---- NON-FINITE INPUT is refused, not silently propagated ----
    //
    // Audit finding: a single +Inf entry made svd_square return NORMALLY with
    // NaN in u and a non-finite singular value -- garbage indistinguishable from
    // a result. The repo's rule is that invalid input fails fast and says why,
    // and a decomposition of a matrix containing infinity is not defined.
    {
        const int n = 6;
        std::vector<float> a(static_cast<std::size_t>(n) * n, 1.0f);
        std::vector<float> u(a.size()), s(static_cast<std::size_t>(n)), vt(a.size());
        a[3] = std::numeric_limits<float>::infinity();
        CHECK_DIES_WITH(svd_square(a.data(), n, u.data(), s.data(), vt.data()), "not finite");
        a[3] = std::numeric_limits<float>::quiet_NaN();
        CHECK_DIES_WITH(svd_square(a.data(), n, u.data(), s.data(), vt.data()), "not finite");
        // Control: the same matrix without the bad entry must still work, or the
        // death checks above prove only that something in this block aborts.
        a[3] = 1.0f;
        svd_square(a.data(), n, u.data(), s.data(), vt.data());
        CHECK(std::fabs(static_cast<double>(s[0]) - n) < 1e-3);
    }

    // ---- range checks ----
    {
        Config cfg{};
        cfg.max_seq_len = 8;
        cfg.vocab_size = 11;
        cfg.n_layer = 2;
        cfg.n_head = 2;
        cfg.n_embd = 16;
        GPT2 m(cfg, 1, 5);
        Generator g3(5ULL);
        m.init_weights(g3);
        std::vector<float> u(circuit_floats(cfg)), s(11), vt(circuit_floats(cfg));
        CHECK_DIES_WITH(svd_circuit(m, 2, 0, CircuitKind::Ov, u.data(), s.data(), vt.data()),
                        "layer out of range");
        CHECK_DIES_WITH(svd_circuit(m, 0, 2, CircuitKind::Qk, u.data(), s.data(), vt.data()),
                        "head out of range");
    }

    return cppgpt::test::summary();
}
