#include "cppgpt/checkpoint.hpp"

#include <fcntl.h>     // open, O_*
#include <sys/stat.h>  // fstat
#include <unistd.h>    // write, read, fsync, close, unlink, rename

#include <cstring>  // memcpy
#include <string>

namespace cppgpt {
namespace {

// Write exactly n bytes, looping over short writes. Returns false on error.
[[nodiscard]] bool write_all(int fd, const void* data, std::size_t n) noexcept {
    const auto* p = static_cast<const std::byte*>(data);
    while (n > 0) {
        const ssize_t w = ::write(fd, p, n);
        if (w < 0) return false;  // EINTR is not retried: our writes are large and rare
        p += w;
        n -= static_cast<std::size_t>(w);
    }
    return true;
}

// Read exactly n bytes; false on error OR short read (truncated file).
[[nodiscard]] bool read_all(int fd, void* data, std::size_t n) noexcept {
    auto* p = static_cast<std::byte*>(data);
    while (n > 0) {
        const ssize_t r = ::read(fd, p, n);
        if (r <= 0) return false;  // r==0 => unexpected EOF (truncated)
        p += r;
        n -= static_cast<std::size_t>(r);
    }
    return true;
}

}  // namespace

std::uint64_t fnv1a_64(std::uint64_t hash, const void* data, std::size_t n) noexcept {
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        hash ^= p[i];
        hash *= kPrime;
    }
    return hash;
}

Result<void> atomic_write(const char* path, const CheckpointHeader& header, const ByteSpan* sections,
                          std::size_t n_sections) noexcept {
    ASSERT(path != nullptr && (sections != nullptr || n_sections == 0));
    const std::string tmp = std::string(path) + ".tmp";

    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return err(ErrorCode::IoError);

    bool ok = write_all(fd, &header, sizeof(header));
    for (std::size_t i = 0; ok && i < n_sections; ++i)
        ok = write_all(fd, sections[i].data, sections[i].size);
    // Durability: the bytes must hit disk before the rename makes them visible.
    if (ok) ok = (::fsync(fd) == 0);
    // Check close(): deferred ENOSPC/EIO surface here, and the twin writer in
    // dataloader.cpp already checks it. Ignoring it silently dropped write errors
    // in the MORE important of the two.
    if (::close(fd) != 0) ok = false;

    if (!ok || ::rename(tmp.c_str(), path) != 0) {
        ::unlink(tmp.c_str());  // never leave a partial temp behind
        return err(ErrorCode::IoError);
    }

    // fsync the containing directory too: without it the rename itself is not
    // durable on ext4/XFS, so a crash could leave NEITHER the old nor the new
    // file — which is exactly what the atomicity claim rules out. Best-effort:
    // the data is already safe on disk, so a failure here is not worth failing
    // an otherwise-complete save.
    const std::size_t slash = std::string(path).find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".") : std::string(path).substr(0, slash);
    const int dfd = ::open(dir.empty() ? "/" : dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) {
        (void)::fsync(dfd);
        ::close(dfd);
    }
    return {};
}

std::uint64_t checkpoint_checksum(const CheckpointHeader& header, const ByteSpan* sections,
                                  std::size_t n_sections) noexcept {
    CheckpointHeader h = header;
    h.checksum = 0;  // the field cannot cover itself
    std::uint64_t sum = fnv1a_64(kFnvOffset64, &h, sizeof(h));
    for (std::size_t i = 0; i < n_sections; ++i) sum = fnv1a_64(sum, sections[i].data, sections[i].size);
    return sum;
}

Result<CheckpointFile> CheckpointFile::open(const char* path) noexcept {
    ASSERT(path != nullptr);
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) return err(ErrorCode::IoError);

    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
        ::close(fd);
        return err(ErrorCode::IoError);
    }
    const auto size = static_cast<std::uint64_t>(st.st_size);
    if (size < sizeof(CheckpointHeader)) {
        ::close(fd);
        return err(ErrorCode::CorruptCheckpoint);
    }

    CheckpointFile f;
    if (!read_all(fd, &f.header_, sizeof(f.header_))) {
        ::close(fd);
        return err(ErrorCode::IoError);
    }
    // Cheap, self-contained header checks. Anything depending on the caller's
    // model (Config, param_count) is deliberately left to the caller.
    if (f.header_.magic != kCheckpointMagic) {
        ::close(fd);
        return err(ErrorCode::CorruptCheckpoint);
    }
    // A RANGE, not equality: v2 files wrote zeros where v3 keeps tokenizer_kind
    // and vocab_fingerprint, and zero already means "char tokenizer, unknown
    // fingerprint". Rejecting them would invalidate every existing checkpoint to
    // record something they already say.
    if (f.header_.version < kCheckpointMinVersion || f.header_.version > kCheckpointVersion) {
        ::close(fd);
        return err(ErrorCode::VersionMismatch);
    }
    // adam_step feeds bias correction (and is incremented); a negative or absurd
    // value would overflow or divide by zero later. Unknown flag bits mean the
    // file was written by something we do not understand.
    if (f.header_.adam_step < 0 || (f.header_.flags & ~kCkptKnownFlags) != 0) {
        ::close(fd);
        return err(ErrorCode::CorruptCheckpoint);
    }
    f.fd_ = fd;
    f.payload_bytes_ = size - sizeof(CheckpointHeader);
    return f;
}

Result<void> CheckpointFile::read_payload(void* dst, std::size_t n) noexcept {
    ASSERT(fd_ >= 0 && (dst != nullptr || n == 0));
    if (n == 0) return {};
    if (!read_all(fd_, dst, n)) return err(ErrorCode::CorruptCheckpoint);
    return {};
}

void CheckpointFile::close_fd() noexcept {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

CheckpointFile::~CheckpointFile() { close_fd(); }

CheckpointFile::CheckpointFile(CheckpointFile&& o) noexcept
    : fd_(o.fd_), header_(o.header_), payload_bytes_(o.payload_bytes_) {
    o.fd_ = -1;  // the moved-from file must not close our fd
    o.payload_bytes_ = 0;
}

CheckpointFile& CheckpointFile::operator=(CheckpointFile&& o) noexcept {
    if (this != &o) {
        close_fd();
        fd_ = o.fd_;
        header_ = o.header_;
        payload_bytes_ = o.payload_bytes_;
        o.fd_ = -1;
        o.payload_bytes_ = 0;
    }
    return *this;
}

}  // namespace cppgpt
