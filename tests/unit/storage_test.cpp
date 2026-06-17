// Storage: alignment, bump/reset lifecycle, non-aliasing, move.
#include "cppgpt/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "tests/check.hpp"

using namespace cppgpt;

namespace {
bool aligned(const void* p, std::size_t a) {
    return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}
}  // namespace

int main() {
    // 32 floats = 128 bytes; already a multiple of kAlign (64).
    Storage s(32);
    CHECK(s.device() == Device::CPU);
    CHECK(s.capacity_bytes() == 128);
    CHECK(s.used_bytes() == 0);

    // First slice is cache-line aligned and advances the head exactly.
    float* a = s.alloc(4);
    CHECK(aligned(a, Storage::kAlign));
    CHECK(s.used_bytes() == 4 * sizeof(float));
    for (int i = 0; i < 4; ++i) a[i] = static_cast<float>(i) + 0.5f;
    CHECK(a[0] == 0.5f && a[3] == 3.5f);

    // Second slice rounds the head up to kAlign: aligned and non-overlapping.
    float* b = s.alloc(4);
    CHECK(aligned(b, Storage::kAlign));
    CHECK(b == a + Storage::kAlign / sizeof(float));  // 64 bytes apart
    b[0] = 99.0f;
    CHECK(a[0] == 0.5f);  // no aliasing into the first slice

    // reset() reuses storage from the top.
    s.reset();
    CHECK(s.used_bytes() == 0);
    float* c = s.alloc(4);
    CHECK(c == a);

    // A smaller power-of-two alignment is honored.
    s.reset();
    float* d = s.alloc(1, 16);
    CHECK(aligned(d, 16));

    // Move leaves the source empty and the target fully usable.
    Storage moved = std::move(s);
    CHECK(moved.capacity_bytes() == 128);
    CHECK(s.capacity_bytes() == 0);  // NOLINT(bugprone-use-after-move) — checking moved-from
    float* e = moved.alloc(8);
    CHECK(aligned(e, Storage::kAlign));

    return cppgpt::test::summary();
}
