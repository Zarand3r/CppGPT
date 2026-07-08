// cppgpt character-level tokenizer.
//
// The vocabulary is the set of distinct bytes in a training corpus, assigned ids
// 0..V-1 in ascending byte order — so the same corpus always yields the same ids
// (a reproducible mapping, no hidden state). Char-level is the v1 tokenizer; the
// byte-level BPE tokenizer is M3. Encoding a byte that is absent from the vocab is
// an invariant violation (fail fast): the tokenizer is built from the very text it
// is expected to encode.
#pragma once

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cppgpt {

class CharTokenizer {
public:
    // Build the vocabulary from the distinct bytes of `corpus` (non-empty).
    explicit CharTokenizer(std::string_view corpus);

    // Map text to token ids / ids back to text. encode() aborts on a byte outside
    // the vocab; decode() aborts on an id outside [0, vocab_size).
    [[nodiscard]] std::vector<int> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(std::span<const int> ids) const;

    [[nodiscard]] int vocab_size() const noexcept { return vocab_size_; }

private:
    std::array<int, 256> stoi_{};  // byte -> id, or -1 if absent from the vocab
    std::string itos_;             // id -> byte (length vocab_size_)
    int vocab_size_ = 0;
};

}  // namespace cppgpt
