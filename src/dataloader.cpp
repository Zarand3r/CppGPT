#include "cppgpt/dataloader.hpp"

#include <fcntl.h>     // open, O_RDONLY
#include <sys/mman.h>  // mmap, munmap, PROT_*, MAP_*
#include <sys/stat.h>  // fstat, struct stat
#include <unistd.h>    // close

#include <utility>  // swap, move

namespace cppgpt {

Result<DataLoader> DataLoader::open(const char* path, int B, int T, std::uint64_t seed) noexcept {
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
    dl.order_.resize(n_examples);
    for (std::size_t i = 0; i < n_examples; ++i) dl.order_[i] = i;
    dl.inputs_.assign(static_cast<std::size_t>(B) * static_cast<std::size_t>(T), 0);
    dl.targets_.assign(static_cast<std::size_t>(B) * static_cast<std::size_t>(T), 0);
    dl.shuffle();
    dl.cursor_ = 0;
    return dl;
}

void DataLoader::next_batch() noexcept {
    const std::size_t B = static_cast<std::size_t>(B_);
    const std::size_t T = static_cast<std::size_t>(T_);
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
      gen_(o.gen_) {
    o.map_base_ = nullptr;  // the moved-from loader must not munmap our mapping
    o.tokens_ = nullptr;
    o.map_size_ = 0;
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
        inputs_ = std::move(o.inputs_);
        targets_ = std::move(o.targets_);
        gen_ = o.gen_;
        o.map_base_ = nullptr;
        o.tokens_ = nullptr;
        o.map_size_ = 0;
    }
    return *this;
}

}  // namespace cppgpt
