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
#include <string_view>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/random.hpp"

namespace cppgpt {

// Write token ids as the flat little-endian uint16 .bin that DataLoader reads
// (the inverse of its read path). Every id must be in [0, 65536); an id outside
// that range is OutOfRange. Returns IoError on write failure. Used by
// tools/prepare to serialize a tokenized corpus.
[[nodiscard]] Result<void> write_token_bin(const char* path, std::span<const int> ids) noexcept;

// Write `bytes` to `path` atomically: temp file, fsync, rename, fsync the
// directory, unlink the temp on any error. Same protocol as write_token_bin and
// the checkpoint writer. Exposed because tools/prepare's .vocab sidecar needs it:
// it carries the id->byte mapping every downstream tool derives meaning from, and
// like the token files it has no length and no checksum, so a torn write is
// undetectable -- a size-matched but semantically wrong vocab trains and decodes
// confidently against the wrong alphabet.
[[nodiscard]] Result<void> write_file_atomic(const char* path, std::string_view bytes) noexcept;

// How window start offsets are chosen.
//
//   Windows — T-aligned, non-overlapping, epoch-shuffled. Example k spans
//             [k*T, k*T+T], so a corpus yields (n_tokens-1)/T distinct examples
//             and every token is only ever predicted from ONE alignment. Correct
//             for GPT-2-scale pretraining, where the corpus is huge and you make
//             less than one pass over it.
//   Random  — start offsets drawn uniformly, with replacement. A corpus yields
//             n_tokens-T distinct windows instead, so each (context, target) pair
//             is seen at every alignment. This is what a small corpus with many
//             epochs needs: at T=64 over 1M tokens it is 15,685 examples versus
//             ~1.0M, and measurably the difference between a model that
//             generalises and one that memorises (docs/EXPERIMENTS.md E-1).
//
// Random has no epoch: it samples with replacement, so batches_per_epoch() is
// nominal under it and num_examples() reports distinct window starts.
enum class Sampling { Windows, Random };

class DataLoader {
public:
    // Map `path` and prepare (B, T) batching seeded by `seed`.
    // Returns an error (never throws) if the file cannot be opened/mapped, is not
    // a whole number of uint16 tokens, or is too small to form one B-wide batch.
    [[nodiscard]] static Result<DataLoader> open(const char* path, int B, int T,
                                                 std::uint64_t seed,
                                                 Sampling sampling = Sampling::Windows) noexcept;

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
    [[nodiscard]] Sampling sampling() const noexcept { return sampling_; }
    // Distinct windows the corpus can yield. Under Random every token position
    // can start one, which is the whole point of the mode.
    [[nodiscard]] std::size_t num_examples() const noexcept {
        if (sampling_ != Sampling::Random) return order_.size();
        const auto T = static_cast<std::size_t>(T_);
        return n_tokens_ > T ? n_tokens_ - T : 0;
    }
    [[nodiscard]] std::size_t batches_per_epoch() const noexcept {
        return B_ > 0 ? num_examples() / static_cast<std::size_t>(B_) : 0;
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
    Sampling sampling_ = Sampling::Windows;
    Generator gen_{0};                // owns the shuffle/offset RNG (reseeded in open())
};

}  // namespace cppgpt
