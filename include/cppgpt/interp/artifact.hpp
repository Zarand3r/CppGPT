// Corpus artifacts: what an offline pass produces and `inspect` merges in.
//
// See docs/DECISIONS.md D11 for why this is binary rather than JSON: there is no
// JSON parser in this repo, convert_hf only manages a flat header, and a general
// parser is hundreds of lines of new surface for a format nobody reads by hand.
// The checkpoint format already solves this and is tested, so this reuses its
// shape -- magic, version, checksum, atomic write, Result on load.
//
// THE IDENTITY CHECK IS THE POINT. An artifact built from a DIFFERENT training
// run is not obviously wrong when rendered: the neuron indices are valid, the
// contexts are real text, every panel populates. It simply describes another
// model. Silent staleness is the failure this channel exists to prevent, so the
// producing checkpoint's checksum is recorded and compared, and a mismatch is
// refused rather than warned about.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "cppgpt/core.hpp"

namespace cppgpt {

inline constexpr std::uint32_t kArtifactMagic = 0x49505043;  // 'CPPI' little-endian
inline constexpr std::uint32_t kArtifactVersion = 1;

// What a payload contains. The envelope is shared; the payload layout belongs to
// whichever producer owns the kind.
enum class ArtifactKind : std::uint32_t {
    NeuronTopK = 1,  // per-MLP-neuron top-activating contexts over a corpus
};

// Fixed 40-byte little-endian header, then the payload.
struct ArtifactHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t kind;
    std::uint32_t reserved;              // zero; a nonzero value is corrupt
    std::uint64_t checkpoint_checksum;   // identity of the model it describes
    std::uint64_t payload_bytes;
    std::uint64_t checksum;              // over the header (this field zeroed) + payload
};
static_assert(sizeof(ArtifactHeader) == 40, "ArtifactHeader must be exactly 40 bytes on disk");

// ---------------------------------------------------------------------------
// NeuronTopK payload
// ---------------------------------------------------------------------------
//
// For every MLP neuron, the contexts from a corpus that activated it most. This
// is the classic first question about a neuron -- "what does it fire on" -- and
// it needs no new model, because `fch_gelu` is already in the activation arena.
//
// Layout, all little-endian, immediately after the envelope header:
//
//   uint32 n_neurons     L * 4C
//   uint32 top_k         entries kept per neuron
//   uint32 ctx_len       characters of context stored per entry
//   uint32 reserved      zero
//   then n_neurons * top_k records, neuron-major, each entry ranked descending:
//     float32 activation
//     int32   focus        index within ctx of the position that fired
//     int32   ctx[ctx_len] token ids, right-aligned so focus is the last real one
//
// A neuron that never activated has entries with activation 0 and focus -1;
// they are kept rather than omitted so a record's offset is arithmetic rather
// than a lookup, and so "this neuron is dead" is representable.
struct NeuronTopKHeader {
    std::uint32_t n_neurons;
    std::uint32_t top_k;
    std::uint32_t ctx_len;
    std::uint32_t reserved;
};
static_assert(sizeof(NeuronTopKHeader) == 16, "NeuronTopKHeader must be 16 bytes on disk");

// Bytes one entry occupies: activation + focus + ctx_len token ids.
[[nodiscard]] inline std::size_t neuron_entry_bytes(std::uint32_t ctx_len) noexcept {
    return sizeof(float) + sizeof(std::int32_t) +
           static_cast<std::size_t>(ctx_len) * sizeof(std::int32_t);
}

// Write atomically (tmp + rename + fsync), like every other cross-tool file here.
[[nodiscard]] Result<void> write_artifact(const char* path, ArtifactKind kind,
                                          std::uint64_t checkpoint_checksum,
                                          std::string_view payload) noexcept;

// Read and validate. Returns the payload, or:
//   IoError           — cannot open or read
//   CorruptCheckpoint — short file, bad magic, nonzero reserved, or bad size
//   VersionMismatch   — a version this build does not know
//   ShapeMismatch     — the artifact describes a DIFFERENT checkpoint, or another kind
//   ChecksumMismatch  — the bytes do not hash to what the header claims
//
// `live_checkpoint_checksum` is the checksum of the model currently loaded. Pass
// it always: an artifact that does not match is the failure mode this format
// exists to catch, and there is no legitimate reason to skip the check.
[[nodiscard]] Result<std::string> read_artifact(const char* path, ArtifactKind want,
                                                std::uint64_t live_checkpoint_checksum) noexcept;

}  // namespace cppgpt
