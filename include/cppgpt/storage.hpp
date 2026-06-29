// cppgpt Storage: an owning, device-tagged, aligned bump arena.
//
// One up-front aligned allocation; alloc() hands out cache-line-aligned slices
// by bumping a head offset; reset() drops everything (head -> 0) between phases
// (e.g. per training step). This is the single memory substrate for params,
// grads, and activations. It exists so the steady-state hot path performs NO
// per-step heap allocation (PLAN invariant 1), and so the planned GPU phase can
// swap host memory for device memory behind the same interface (PLAN device
// seam, invariant 8).
//
// Ownership: Storage owns its buffer uniquely (move-only, no copy). Callers
// borrow the returned float* for the lifetime of the Storage (or until reset());
// they never free it. No shared ownership of tensor memory.
#pragma once

#include <cstddef>
#include <cstring>
#include <new>

#include "cppgpt/core.hpp"
#include "cppgpt/device.hpp"

namespace cppgpt {

namespace detail {
constexpr bool is_pow2(std::size_t x) noexcept { return x != 0 && (x & (x - 1)) == 0; }
constexpr std::size_t round_up(std::size_t x, std::size_t a) noexcept {
    return (x + a - 1) & ~(a - 1);
}
}  // namespace detail

class Storage {
public:
    static constexpr std::size_t kAlign = 64;  // base + sub-alloc alignment (cache line)

    // Empty arena (capacity 0); for members initialized later via move-assignment.
    Storage() noexcept = default;

    // Reserve capacity_floats of fp32 storage, rounded up to kAlign bytes.
    // Allocation failure is fatal: there is no degraded path without the arena,
    // so OOM aborts with a message (nothrow new + fail-fast) rather than raising
    // an exception — the ctor is therefore noexcept.
    explicit Storage(std::size_t capacity_floats, Device dev = Device::CPU) noexcept
        : dev_(dev), capacity_(detail::round_up(capacity_floats * sizeof(float), kAlign)) {
        if (capacity_ != 0) {
            base_ = static_cast<std::byte*>(
                ::operator new(capacity_, std::align_val_t{kAlign}, std::nothrow));
            ASSERT_MSG(base_ != nullptr, "Storage: out of memory");
        }
    }

    ~Storage() { ::operator delete(base_, std::align_val_t{kAlign}); }

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    Storage(Storage&& o) noexcept
        : base_(o.base_), dev_(o.dev_), capacity_(o.capacity_), head_(o.head_) {
        o.base_ = nullptr;
        o.capacity_ = 0;
        o.head_ = 0;
    }
    Storage& operator=(Storage&& o) noexcept {
        if (this != &o) {
            ::operator delete(base_, std::align_val_t{kAlign});
            base_ = o.base_;
            dev_ = o.dev_;
            capacity_ = o.capacity_;
            head_ = o.head_;
            o.base_ = nullptr;
            o.capacity_ = 0;
            o.head_ = 0;
        }
        return *this;
    }

    // Bump-allocate n_floats, aligned to `align` (power of two, <= kAlign).
    // Exceeding capacity is an invariant violation (a sizing bug), not an
    // expected failure: fail fast.
    [[nodiscard]] float* alloc(std::size_t n_floats, std::size_t align = kAlign) {
        ASSERT(detail::is_pow2(align) && align <= kAlign);
        head_ = detail::round_up(head_, align);
        const std::size_t need = n_floats * sizeof(float);
        ASSERT_MSG(head_ + need <= capacity_, "Storage::alloc exceeds capacity");
        float* p = reinterpret_cast<float*>(base_ + head_);
        head_ += need;
        return p;
    }

    // Like alloc(), but zero-initializes the slice. Use for accumulator buffers
    // (e.g. gradients written with +=) that must start at zero each phase. alloc()
    // deliberately returns uninitialized memory — there is no hidden memset on the
    // hot path; callers that need zeros ask for them here, explicitly.
    [[nodiscard]] float* alloc_zeroed(std::size_t n_floats, std::size_t align = kAlign) {
        float* p = alloc(n_floats, align);
        std::memset(p, 0, n_floats * sizeof(float));
        return p;
    }

    void reset() noexcept { head_ = 0; }

    // Zero the allocated region (e.g. gradient arenas before a backward pass).
    void zero() noexcept {
        if (base_ != nullptr) std::memset(base_, 0, head_);
    }

    [[nodiscard]] Device device() const noexcept { return dev_; }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return head_; }

private:
    std::byte* base_ = nullptr;
    Device dev_ = Device::CPU;
    std::size_t capacity_ = 0;  // bytes
    std::size_t head_ = 0;      // bytes, bump offset
};

}  // namespace cppgpt
