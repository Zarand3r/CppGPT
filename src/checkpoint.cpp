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
    ::close(fd);

    if (!ok || ::rename(tmp.c_str(), path) != 0) {
        ::unlink(tmp.c_str());  // never leave a partial temp behind
        return err(ErrorCode::IoError);
    }
    return {};
}

Result<std::vector<std::byte>> read_file(const char* path) noexcept {
    ASSERT(path != nullptr);
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) return err(ErrorCode::IoError);

    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 0) {
        ::close(fd);
        return err(ErrorCode::IoError);
    }
    std::vector<std::byte> buf(static_cast<std::size_t>(st.st_size));
    const bool ok = buf.empty() || read_all(fd, buf.data(), buf.size());
    ::close(fd);
    if (!ok) return err(ErrorCode::IoError);
    return buf;
}

}  // namespace cppgpt
