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
inline constexpr std::uint32_t kCheckpointVersion = 1;
inline constexpr std::uint32_t kCkptHasMoments = 1u << 0;  // flags: m/v present in payload

// 64-byte fixed header. Standard-layout, no padding (static_assert below). The
// checksum covers the PAYLOAD only; the header's own integrity is gated by the
// magic/version/Config checks plus an exact-file-size check.
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

// Read the whole file into memory (checkpoints are read transactionally: the
// caller validates header + checksum before touching any live arena). Returns
// IoError on open/read failure.
[[nodiscard]] Result<std::vector<std::byte>> read_file(const char* path) noexcept;

}  // namespace cppgpt
