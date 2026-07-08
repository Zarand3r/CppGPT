#include "cppgpt/tokenizer.hpp"

#include <cstddef>

#include "cppgpt/core.hpp"

namespace cppgpt {

CharTokenizer::CharTokenizer(std::string_view corpus) {
    stoi_.fill(-1);
    bool present[256] = {};
    for (unsigned char c : corpus) present[c] = true;
    // Assign ids in ascending byte order so the mapping is deterministic.
    for (int b = 0; b < 256; ++b) {
        if (present[b]) {
            stoi_[static_cast<std::size_t>(b)] = vocab_size_;
            itos_.push_back(static_cast<char>(b));
            ++vocab_size_;
        }
    }
    ASSERT_MSG(vocab_size_ > 0, "CharTokenizer: empty corpus");
}

std::vector<int> CharTokenizer::encode(std::string_view text) const {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) {
        const int id = stoi_[c];
        ASSERT_MSG(id >= 0, "CharTokenizer::encode: byte not in vocab");
        ids.push_back(id);
    }
    return ids;
}

std::string CharTokenizer::decode(std::span<const int> ids) const {
    std::string out;
    out.reserve(ids.size());
    for (int id : ids) {
        ASSERT(id >= 0 && id < vocab_size_);
        out.push_back(itos_[static_cast<std::size_t>(id)]);
    }
    return out;
}

}  // namespace cppgpt
