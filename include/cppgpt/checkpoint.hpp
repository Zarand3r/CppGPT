// cppgpt checkpoint format: a versioned header + raw fp32 payload.
//
// The on-disk layout is a fixed 64-byte little-endian header (magic, version,
// Config, optimizer step, flags, payload checksum) followed by the tensors as
// raw contiguous fp32 in .bin order:
//
//   [CheckpointHeader]  (64 bytes)
//   f32 params[param_count]                 // the weights (all 16 tensors)
//   f32 m[param_count], f32 v[param_count]  // AdamW moments, iff HAS_MOMENTS
//
// No container format (safetensors/GGUF/HDF5/pickle): the tensor layout is fixed
// and known, so a struct header + raw fp32 is the entire codec — dependency-free,
// mmap-able, byte-comparable against a PyTorch dump. Little-endian x86 only
// (hermetic toolchain), matching the parity fixture and token .bin files.
//
// This header owns only the format + non-I/O helpers (checksum) and two I/O
// primitives (atomic write, whole-file read). GPT2::save_checkpoint /
// load_checkpoint orchestrate them over the model's arenas.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cppgpt/core.hpp"

namespace cppgpt {

inline constexpr std::uint32_t kCheckpointMagic = 0x54504B43;  // 'CKPT' little-endian
inline constexpr std::uint32_t kCheckpointVersion = 2;
inline constexpr std::uint32_t kCkptHasMoments = 1u << 0;  // flags: m/v present in payload
inline constexpr std::uint32_t kCkptKnownFlags = kCkptHasMoments;  // any other bit => corrupt

// 64-byte fixed header. Standard-layout, no padding (static_assert below). The
// checksum covers the HEADER (with the checksum field itself zeroed) AND the
// payload, so fields the loader trusts but cannot range-check on their own —
// adam_step, flags — cannot be silently corrupted. (v1 checksummed only the
// payload; v2 exists because widening that coverage changes the bytes.)
struct CheckpointHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::int32_t max_seq_len;  // Config, validated against the target model on load
    std::int32_t vocab_size;
    std::int32_t n_layer;
    std::int32_t n_head;
    std::int32_t n_embd;
    std::int32_t adam_step;  // AdamW step counter (bias correction)
    std::uint32_t flags;     // kCkptHasMoments, ...
    std::uint32_t reserved0;
    std::uint64_t param_count;  // must equal the model's param_count()
    std::uint64_t checksum;     // FNV-1a-64 over the payload
    std::uint64_t reserved1;
};
static_assert(sizeof(CheckpointHeader) == 64, "checkpoint header must be exactly 64 bytes");

inline constexpr std::uint64_t kFnvOffset64 = 1469598103934665603ULL;

// FNV-1a-64, chainable across regions: seed with kFnvOffset64, then fold each
// region in order. Used to detect on-disk corruption/truncation (not crypto).
[[nodiscard]] std::uint64_t fnv1a_64(std::uint64_t hash, const void* data, std::size_t n) noexcept;

// One payload region to write.
struct ByteSpan {
    const void* data;
    std::size_t size;  // bytes
};

// Atomically write `header` followed by `sections` (in order) to `path`:
// write to `path`.tmp, fsync, then rename over `path` — so a crash leaves either
// the old file or the new one, never a torn one. Removes the temp file on any
// error. `header.checksum` must already be set by the caller.
[[nodiscard]] Result<void> atomic_write(const char* path, const CheckpointHeader& header,
                                        const ByteSpan* sections, std::size_t n_sections) noexcept;

// Compute the checksum a checkpoint must carry: the header with its `checksum`
// field zeroed, then each payload section in order.
[[nodiscard]] std::uint64_t checkpoint_checksum(const CheckpointHeader& header,
                                                const ByteSpan* sections,
                                                std::size_t n_sections) noexcept;

// Streaming reader for a checkpoint file, so the header can be validated BEFORE
// any payload-sized allocation. Reading the whole file up front would let a
// bogus (or simply wrong) file dictate an unbounded allocation inside noexcept
// code — i.e. std::terminate on bad_alloc. open() reads and checks only the
// 64-byte header; the caller then checks the Config/param_count against its own
// model, and only then calls read_payload() with a buffer it has sized itself.
// Owns the fd (closed on destruction); move-only.
class CheckpointFile {
public:
    // Opens `path` and reads the header. Fails with IoError (open/read/stat),
    // CorruptCheckpoint (short file, bad magic, unknown flag bits, negative
    // adam_step) or VersionMismatch.
    [[nodiscard]] static Result<CheckpointFile> open(const char* path) noexcept;

    [[nodiscard]] const CheckpointHeader& header() const noexcept { return header_; }

    // Bytes after the 64-byte header, as reported by the filesystem.
    [[nodiscard]] std::uint64_t payload_bytes() const noexcept { return payload_bytes_; }

    // Read exactly `n` payload bytes into `dst`. CorruptCheckpoint on short read.
    [[nodiscard]] Result<void> read_payload(void* dst, std::size_t n) noexcept;

    ~CheckpointFile();
    CheckpointFile(const CheckpointFile&) = delete;
    CheckpointFile& operator=(const CheckpointFile&) = delete;
    CheckpointFile(CheckpointFile&& o) noexcept;
    CheckpointFile& operator=(CheckpointFile&& o) noexcept;

private:
    CheckpointFile() noexcept = default;
    void close_fd() noexcept;

    int fd_ = -1;
    CheckpointHeader header_{};
    std::uint64_t payload_bytes_ = 0;
};

}  // namespace cppgpt
