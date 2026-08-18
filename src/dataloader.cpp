#include "cppgpt/dataloader.hpp"

#include <fcntl.h>     // open, O_RDONLY
#include <sys/mman.h>  // mmap, munmap, PROT_*, MAP_*
#include <sys/stat.h>  // fstat, struct stat
#include <unistd.h>    // close

#include <cstddef>
#include <string>
#include <utility>  // swap, move
#include <vector>

namespace cppgpt {

namespace {

// tmp -> write -> fsync -> rename -> fsync(dir); unlink the temp on any failure.
[[nodiscard]] Result<void> write_atomic(const char* path, const void* data,
                                        std::size_t nbytes) noexcept {
    const std::string tmp = std::string(path) + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return err(ErrorCode::IoError);
    const auto* p = static_cast<const std::byte*>(data);
    std::size_t n = nbytes;
    bool ok = true;
    while (ok && n > 0) {
        const ssize_t w = ::write(fd, p, n);
        if (w <= 0) {
            ok = false;  // w == 0 would otherwise spin forever
        } else {
            p += w;
            n -= static_cast<std::size_t>(w);
        }
    }
    if (ok) ok = (::fsync(fd) == 0);
    if (::close(fd) != 0) ok = false;
    if (!ok || ::rename(tmp.c_str(), path) != 0) {
        ::unlink(tmp.c_str());
        return err(ErrorCode::IoError);
    }
    const std::string ps(path);
    const std::size_t slash = ps.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".") : ps.substr(0, slash);
    const int dfd = ::open(dir.empty() ? "/" : dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) {
        (void)::fsync(dfd);
        ::close(dfd);
    }
    return {};
}

}  // namespace

Result<void> write_file_atomic(const char* path, std::string_view bytes) noexcept {
    ASSERT(path != nullptr);
    return write_atomic(path, bytes.data(), bytes.size());
}

Result<void> write_token_bin(const char* path, std::span<const int> ids) noexcept {
    ASSERT(path != nullptr);
    // Pack to little-endian uint16 in one buffer, then a single write.
    std::vector<std::uint16_t> u16(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const int v = ids[i];
        if (v < 0 || v > 0xFFFF) return err(ErrorCode::OutOfRange);
        u16[i] = static_cast<std::uint16_t>(v);
    }
    return write_atomic(path, u16.data(), u16.size() * sizeof(std::uint16_t));
}

Result<DataLoader> DataLoader::open(const char* path, int B, int T, std::uint64_t seed,
                                    Sampling sampling) noexcept {
    ASSERT(path != nullptr && B >= 1 && T >= 1);

    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) return err(ErrorCode::IoError);

    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return err(ErrorCode::IoError);
    }
    const std::size_t size = static_cast<std::size_t>(st.st_size);
    if (size % sizeof(std::uint16_t) != 0) {  // not a whole number of tokens
        ::close(fd);
        return err(ErrorCode::ParseError);
    }
    const std::size_t n_tokens = size / sizeof(std::uint16_t);
    // Need T+1 tokens for one example (T inputs + their next-token shift), and at
    // least B examples so every epoch yields at least one full batch.
    const std::size_t n_examples =
        n_tokens > static_cast<std::size_t>(T) ? (n_tokens - 1) / static_cast<std::size_t>(T) : 0;
    if (n_examples < static_cast<std::size_t>(B)) {
        ::close(fd);
        return err(ErrorCode::OutOfRange);
    }

    void* base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // the mapping holds its own reference; the fd is no longer needed
    if (base == MAP_FAILED) return err(ErrorCode::IoError);

    DataLoader dl;
    dl.map_base_ = base;
    dl.map_size_ = size;
    dl.tokens_ = static_cast<const std::uint16_t*>(base);
    dl.n_tokens_ = n_tokens;
    dl.B_ = B;
    dl.T_ = T;
    dl.gen_ = Generator(seed);
    dl.sampling_ = sampling;
    // Random draws a fresh offset per example, so the permutation is dead weight
    // — and at 1M tokens it would be a 1M-element vector serving nothing.
    if (sampling != Sampling::Random) {
        dl.order_.resize(n_examples);
        for (std::size_t i = 0; i < n_examples; ++i) dl.order_[i] = i;
    }
    dl.inputs_.assign(static_cast<std::size_t>(B) * static_cast<std::size_t>(T), 0);
    dl.targets_.assign(static_cast<std::size_t>(B) * static_cast<std::size_t>(T), 0);
    if (sampling != Sampling::Random) dl.shuffle();
    dl.cursor_ = 0;
    return dl;
}

void DataLoader::next_batch() noexcept {
    const std::size_t B = static_cast<std::size_t>(B_);
    const std::size_t T = static_cast<std::size_t>(T_);
    if (sampling_ == Sampling::Random) {
        // Uniform start offsets, with replacement. `hi` is inclusive and chosen so
        // start + T + 1 <= n_tokens_: the example needs T inputs plus the one
        // extra token its last target reads.
        const std::size_t hi = n_tokens_ - T - 1;
        for (std::size_t b = 0; b < B; ++b) {
            const auto start = static_cast<std::size_t>(
                gen_.uniform_int(0, static_cast<std::int64_t>(hi)));
            int* in = inputs_.data() + b * T;
            int* tg = targets_.data() + b * T;
            for (std::size_t t = 0; t < T; ++t) {
                in[t] = static_cast<int>(tokens_[start + t]);
                tg[t] = static_cast<int>(tokens_[start + t + 1]);
            }
        }
        return;
    }
    if (cursor_ + B > order_.size()) {  // not enough examples left for a full batch
        shuffle();
        cursor_ = 0;
    }
    for (std::size_t b = 0; b < B; ++b) {
        const std::size_t start = order_[cursor_ + b] * T;  // token offset of this example
        int* in = inputs_.data() + b * T;
        int* tg = targets_.data() + b * T;
        for (std::size_t t = 0; t < T; ++t) {
            in[t] = static_cast<int>(tokens_[start + t]);
            tg[t] = static_cast<int>(tokens_[start + t + 1]);
        }
    }
    cursor_ += B;
}

void DataLoader::shuffle() noexcept {
    // Fisher-Yates: an unbiased permutation of the example order in place.
    for (std::size_t i = order_.size(); i > 1; --i) {
        const std::size_t j =
            static_cast<std::size_t>(gen_.uniform_int(0, static_cast<std::int64_t>(i - 1)));
        std::swap(order_[i - 1], order_[j]);
    }
}

void DataLoader::release() noexcept {
    if (map_base_ != nullptr) {
        ::munmap(map_base_, map_size_);
        map_base_ = nullptr;
        tokens_ = nullptr;
        map_size_ = 0;
    }
}

DataLoader::~DataLoader() { release(); }

DataLoader::DataLoader(DataLoader&& o) noexcept
    : tokens_(o.tokens_),
      map_base_(o.map_base_),
      map_size_(o.map_size_),
      n_tokens_(o.n_tokens_),
      B_(o.B_),
      T_(o.T_),
      order_(std::move(o.order_)),
      cursor_(o.cursor_),
      inputs_(std::move(o.inputs_)),
      targets_(std::move(o.targets_)),
      sampling_(o.sampling_),
      gen_(o.gen_) {
    // Clear ALL state, not just the mapping: leaving B_/T_/n_tokens_ intact
    // would make the moved-from loader's accessors report a live-looking config
    // while order_/inputs_ are empty.
    o.map_base_ = nullptr;  // the moved-from loader must not munmap our mapping
    o.tokens_ = nullptr;
    o.map_size_ = 0;
    o.n_tokens_ = 0;
    o.B_ = 0;
    o.T_ = 0;
    o.cursor_ = 0;
    o.sampling_ = Sampling::Windows;
}

DataLoader& DataLoader::operator=(DataLoader&& o) noexcept {
    if (this != &o) {
        release();
        tokens_ = o.tokens_;
        map_base_ = o.map_base_;
        map_size_ = o.map_size_;
        n_tokens_ = o.n_tokens_;
        B_ = o.B_;
        T_ = o.T_;
        order_ = std::move(o.order_);
        cursor_ = o.cursor_;
        sampling_ = o.sampling_;
        inputs_ = std::move(o.inputs_);
        targets_ = std::move(o.targets_);
        gen_ = o.gen_;
        o.map_base_ = nullptr;
        o.tokens_ = nullptr;
        o.map_size_ = 0;
        o.n_tokens_ = 0;
        o.B_ = 0;
        o.T_ = 0;
        o.cursor_ = 0;
        o.sampling_ = Sampling::Windows;
    }
    return *this;
}

}  // namespace cppgpt
