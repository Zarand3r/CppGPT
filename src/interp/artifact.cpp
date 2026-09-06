#include "cppgpt/interp/artifact.hpp"

#include <cstring>
#include <fstream>
#include <ios>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/dataloader.hpp"

namespace cppgpt {
namespace {

// Checksum over the header with its own checksum field zeroed, then the payload.
// Same construction as checkpoint_checksum, for the same reason: a checksum that
// does not cover the header lets a field be edited without detection.
std::uint64_t artifact_checksum(const ArtifactHeader& h, std::string_view payload) noexcept {
    ArtifactHeader copy = h;
    copy.checksum = 0;
    std::uint64_t hash = fnv1a_64(kFnvOffset64, &copy, sizeof(copy));
    return fnv1a_64(hash, payload.data(), payload.size());
}

}  // namespace

Result<void> write_artifact(const char* path, ArtifactKind kind,
                            std::uint64_t checkpoint_checksum, std::string_view payload) noexcept {
    ASSERT(path != nullptr);
    ArtifactHeader h{};
    h.magic = kArtifactMagic;
    h.version = kArtifactVersion;
    h.kind = static_cast<std::uint32_t>(kind);
    h.reserved = 0;
    h.checkpoint_checksum = checkpoint_checksum;
    h.payload_bytes = payload.size();
    h.checksum = artifact_checksum(h, payload);

    std::string bytes;
    bytes.resize(sizeof(h) + payload.size());
    std::memcpy(bytes.data(), &h, sizeof(h));
    if (!payload.empty()) std::memcpy(bytes.data() + sizeof(h), payload.data(), payload.size());
    return write_file_atomic(path, bytes);
}

Result<std::string> read_artifact(const char* path, ArtifactKind want,
                                  std::uint64_t live_checkpoint_checksum) noexcept {
    ASSERT(path != nullptr);
    std::ifstream f(path, std::ios::binary);
    if (!f) return err(ErrorCode::IoError);

    ArtifactHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return err(ErrorCode::CorruptCheckpoint);

    // Order matters: identify the format before trusting any field in it, and
    // check the version before checking anything a later version might redefine.
    if (h.magic != kArtifactMagic) return err(ErrorCode::CorruptCheckpoint);
    if (h.version != kArtifactVersion) return err(ErrorCode::VersionMismatch);
    if (h.reserved != 0) return err(ErrorCode::CorruptCheckpoint);
    if (h.kind != static_cast<std::uint32_t>(want)) return err(ErrorCode::ShapeMismatch);

    // The identity check this format exists for. An artifact from another run
    // renders perfectly and describes a different model, so a mismatch is
    // refused rather than warned about.
    if (h.checkpoint_checksum != live_checkpoint_checksum) return err(ErrorCode::ShapeMismatch);

    // payload_bytes comes from the file, so it is untrusted until the checksum
    // passes. Bound it against the actual file size before allocating (L3).
    f.seekg(0, std::ios::end);
    const auto file_bytes = static_cast<std::uint64_t>(f.tellg());
    if (file_bytes < sizeof(h) || h.payload_bytes != file_bytes - sizeof(h))
        return err(ErrorCode::CorruptCheckpoint);

    std::string payload;
    payload.resize(static_cast<std::size_t>(h.payload_bytes));
    f.seekg(static_cast<std::streamoff>(sizeof(h)), std::ios::beg);
    if (h.payload_bytes != 0 &&
        !f.read(payload.data(), static_cast<std::streamsize>(h.payload_bytes)))
        return err(ErrorCode::CorruptCheckpoint);

    if (artifact_checksum(h, payload) != h.checksum) return err(ErrorCode::ChecksumMismatch);
    return payload;
}

}  // namespace cppgpt
