// cppgpt DataLoader: streams (B, T) token batches from an mmap'd uint16 .bin.
//
// The training token file is a flat little-endian uint16 array (the nanoGPT /
// llm.c convention). We map it read-only (MAP_PRIVATE | PROT_READ) so tokens
// come straight from the kernel page cache — zero-copy, on-demand paged — and
// convert uint16 -> int per batch into small owned buffers the model borrows.
//
// Examples are non-overlapping length-T windows: example k spans tokens
// [k*T, k*T + T]; inputs = tokens[k*T .. k*T+T-1], targets = the next-token
// shift tokens[k*T+1 .. k*T+T]. Each epoch shuffles the example order
// (Fisher-Yates over an owned index permutation) and yields floor(n_examples/B)
// full batches; a trailing partial batch is dropped and a fresh shuffle begins.
//
// Ownership: the loader owns the mapping (munmap on destruction; move-only) and
// the two int batch buffers. inputs()/targets() borrow those buffers and stay
// valid only until the next next_batch(). The loader owns its own Generator so
// shuffling is reproducible from `seed` and independent of model/sampling RNG.
//
// Assumptions: the backing file is on a local filesystem and is NOT truncated
// during the loader's lifetime (single-process training) — a shrink would SIGBUS
// on the next page fault. Token ids are trusted to be < vocab_size; an
// out-of-range id is caught fail-fast downstream by the embedding lookup, not
// re-validated here (the mapped file is our own prepared data, not adversarial).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/random.hpp"

namespace cppgpt {

// Write token ids as the flat little-endian uint16 .bin that DataLoader reads
// (the inverse of its read path). Every id must be in [0, 65536); an id outside
// that range is OutOfRange. Returns IoError on write failure. Used by
// tools/prepare to serialize a tokenized corpus.
[[nodiscard]] Result<void> write_token_bin(const char* path, std::span<const int> ids) noexcept;

class DataLoader {
public:
    // Map `path` and prepare epoch-shuffled (B, T) batching seeded by `seed`.
    // Returns an error (never throws) if the file cannot be opened/mapped, is not
    // a whole number of uint16 tokens, or is too small to form one B-wide batch.
    [[nodiscard]] static Result<DataLoader> open(const char* path, int B, int T,
                                                 std::uint64_t seed) noexcept;

    // Advance to the next batch, filling inputs()/targets() ([B*T] each). At an
    // epoch boundary (fewer than B examples left) it reshuffles and restarts.
    // Reads mapped pages only — no heap allocation, never throws.
    void next_batch() noexcept;

    // Borrowed views into the current batch; valid until the next next_batch().
    [[nodiscard]] const int* inputs() const noexcept { return inputs_.data(); }
    [[nodiscard]] const int* targets() const noexcept { return targets_.data(); }

    [[nodiscard]] int batch() const noexcept { return B_; }
    [[nodiscard]] int seq_len() const noexcept { return T_; }
    [[nodiscard]] std::size_t num_tokens() const noexcept { return n_tokens_; }
    [[nodiscard]] std::size_t num_examples() const noexcept { return order_.size(); }
    [[nodiscard]] std::size_t batches_per_epoch() const noexcept {
        return order_.size() / static_cast<std::size_t>(B_);
    }

    ~DataLoader();
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;
    DataLoader(DataLoader&& o) noexcept;
    DataLoader& operator=(DataLoader&& o) noexcept;

private:
    DataLoader() noexcept = default;  // built only by open()
    void shuffle() noexcept;
    void release() noexcept;  // munmap if mapped (idempotent)

    const std::uint16_t* tokens_ = nullptr;  // mmap base, typed (borrowed page cache)
    void* map_base_ = nullptr;               // munmap target (== tokens_), nullptr if unmapped
    std::size_t map_size_ = 0;               // bytes, for munmap
    std::size_t n_tokens_ = 0;
    int B_ = 0;
    int T_ = 0;
    std::vector<std::size_t> order_;  // shuffled example indices, [0, n_examples)
    std::size_t cursor_ = 0;          // next position within order_
    std::vector<int> inputs_;         // [B*T], reused each batch
    std::vector<int> targets_;        // [B*T], reused each batch
    Generator gen_{0};                // owns the shuffle RNG (reseeded in open())
};

}  // namespace cppgpt
