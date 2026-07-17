// DataLoader: mmap'd uint16 batching. Verifies the next-token shift, full-epoch
// coverage (every example exactly once), determinism from a seed, epoch wrap,
// and the error paths — using a temp .bin written with identity tokens so that
// token value == absolute position, which makes every offset checkable.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/dataloader.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

namespace {

// A writable path inside the test sandbox (bazel sets TEST_TMPDIR).
std::string tmp_path(const char* name) {
    const char* dir = std::getenv("TEST_TMPDIR");
    return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

// Write `n` identity tokens (0,1,2,...,n-1) as little-endian uint16.
void write_identity_bin(const std::string& path, std::size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t v = static_cast<std::uint16_t>(i);
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
}

}  // namespace

int main() {
    const std::string path = tmp_path("ident100.bin");
    write_identity_bin(path, 100);
    const int B = 2, T = 4;

    // ---- open + geometry ----
    {
        auto r = DataLoader::open(path.c_str(), B, T, 42ULL);
        CHECK(r.has_value());
        DataLoader& dl = *r;
        CHECK(dl.num_tokens() == 100);
        CHECK(dl.num_examples() == (100 - 1) / T);  // 24
        CHECK(dl.batches_per_epoch() == dl.num_examples() / static_cast<std::size_t>(B));  // 12
    }

    // ---- next-token shift + valid example offsets ----
    {
        auto r = DataLoader::open(path.c_str(), B, T, 7ULL);
        CHECK(r.has_value());
        DataLoader& dl = *r;
        dl.next_batch();
        bool shift_ok = true, offset_ok = true, contig_ok = true;
        for (int b = 0; b < B; ++b) {
            const int* in = dl.inputs() + b * T;
            const int* tg = dl.targets() + b * T;
            offset_ok = offset_ok && (in[0] % T == 0);  // example starts on a T boundary
            for (int t = 0; t < T; ++t) {
                shift_ok = shift_ok && (tg[t] == in[t] + 1);       // identity tokens: shift by one
                if (t > 0) contig_ok = contig_ok && (in[t] == in[t - 1] + 1);  // contiguous window
            }
        }
        CHECK(shift_ok);
        CHECK(offset_ok);
        CHECK(contig_ok);
    }

    // ---- full-epoch coverage: every example index appears exactly once ----
    {
        auto r = DataLoader::open(path.c_str(), B, T, 123ULL);
        CHECK(r.has_value());
        DataLoader& dl = *r;
        const std::size_t n_ex = dl.num_examples();
        std::vector<int> seen(n_ex, 0);
        for (std::size_t batch = 0; batch < dl.batches_per_epoch(); ++batch) {
            dl.next_batch();
            for (int b = 0; b < B; ++b) {
                const int ex = dl.inputs()[b * T] / T;  // recover example index from offset
                CHECK(ex >= 0 && static_cast<std::size_t>(ex) < n_ex);
                seen[static_cast<std::size_t>(ex)]++;
            }
        }
        bool each_once = true;
        for (std::size_t i = 0; i < n_ex; ++i) each_once = each_once && (seen[i] == 1);
        CHECK(each_once);  // 12 batches * B=2 == 24 examples, the whole set, once each
    }

    // ---- determinism: same seed ⇒ identical batch stream ----
    {
        auto ra = DataLoader::open(path.c_str(), B, T, 999ULL);
        auto rb = DataLoader::open(path.c_str(), B, T, 999ULL);
        CHECK(ra.has_value() && rb.has_value());
        DataLoader& a = *ra;
        DataLoader& b = *rb;
        bool identical = true;
        for (int step = 0; step < 30; ++step) {  // spans multiple epochs (bpe=12)
            a.next_batch();
            b.next_batch();
            for (int i = 0; i < B * T; ++i)
                identical = identical && (a.inputs()[i] == b.inputs()[i]) &&
                            (a.targets()[i] == b.targets()[i]);
        }
        CHECK(identical);
    }

    // ---- epoch wrap: batches past one epoch stay valid (shift holds) ----
    {
        auto r = DataLoader::open(path.c_str(), B, T, 5ULL);
        CHECK(r.has_value());
        DataLoader& dl = *r;
        bool valid = true;
        for (std::size_t step = 0; step < dl.batches_per_epoch() * 3 + 1; ++step) {
            dl.next_batch();
            for (int i = 0; i < B * T; ++i)
                valid = valid && (dl.targets()[i] == dl.inputs()[i] + 1) &&
                        (dl.inputs()[i] >= 0 && dl.inputs()[i] < 100);
        }
        CHECK(valid);
    }

    // ---- move preserves the mapping (no double-munmap, batching still works) ----
    {
        auto r = DataLoader::open(path.c_str(), B, T, 1ULL);
        CHECK(r.has_value());
        DataLoader moved = std::move(*r);
        moved.next_batch();
        bool ok = true;
        for (int i = 0; i < B * T; ++i) ok = ok && (moved.targets()[i] == moved.inputs()[i] + 1);
        CHECK(ok);
    }

    // ---- error paths ----
    {
        auto miss = DataLoader::open(tmp_path("does_not_exist.bin").c_str(), B, T, 0ULL);
        CHECK(!miss.has_value() && miss.error() == ErrorCode::IoError);

        const std::string odd = tmp_path("odd.bin");
        {
            std::ofstream f(odd, std::ios::binary | std::ios::trunc);
            const char junk[3] = {1, 2, 3};  // 3 bytes: not a whole number of uint16
            f.write(junk, sizeof(junk));
        }
        auto bad = DataLoader::open(odd.c_str(), B, T, 0ULL);
        CHECK(!bad.has_value() && bad.error() == ErrorCode::ParseError);

        const std::string tiny = tmp_path("tiny.bin");
        write_identity_bin(tiny, 4);  // 4 tokens: too few for B=2 examples at T=4
        auto small = DataLoader::open(tiny.c_str(), B, T, 0ULL);
        CHECK(!small.has_value() && small.error() == ErrorCode::OutOfRange);
    }

    return cppgpt::test::summary();
}
