// Byte-level BPE, GPT-2's tokenizer.
//
// The tokenizer is not preprocessing: `wte` row i is the trained vector for
// token i, so the token<->id mapping is half the model's contract. Feeding
// character ids into GPT-2's weights indexes unrelated rows and produces fluent
// nonsense. This is why loading real GPT-2 weights requires this file.
//
// Reads HuggingFace's `vocab.json` + `merges.txt` directly. No conversion step
// and no JSON library: `merges.txt` is plain text, and `vocab.json` is a FLAT
// map of string to int, which needs a targeted parser rather than a general one.
//
// Data layout, because encode() is the hot path:
//   * Merges are applied in ID space, never string space. `merges.txt` names
//     pairs by string, but that is resolved once at load into
//     (id_a, id_b) -> (rank, id_ab). Encoding then never hashes a string.
//   * That table is open-addressed over two dense arrays — no per-entry node,
//     no pointer chasing, one cache miss per probe.
//   * decode() is a memcpy: each id's REAL bytes are precomputed at load into
//     one blob with an offsets array, so there is no 50k-element vector<string>
//     and no byte-map arithmetic at call time.
//
// Pre-tokenization implements GPT-2's regex by hand:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// std::regex cannot express \p{L}/\p{N}, and they apply to the original UTF-8
// rather than to the byte-mapped alphabet, so this is not reducible to a table
// lookup. ASCII is handled exactly; a non-ASCII byte is reported as an error
// rather than silently mis-split (see `encode`).
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cppgpt/core.hpp"

namespace cppgpt {

class BpeTokenizer {
public:
    // Load from HuggingFace's `vocab.json` and `merges.txt`. Returns ParseError
    // if either is malformed, or if a merge names a token absent from the vocab
    // (which would leave the merge table silently incomplete).
    [[nodiscard]] static Result<BpeTokenizer> open(const char* vocab_json,
                                                   const char* merges_txt) noexcept;

    // Text -> token ids. `ok` is set false if the text contains a byte this
    // pre-tokenizer cannot split exactly (any byte >= 0x80); the ids produced up
    // to that point are not meaningful. Failing loudly is deliberate: a
    // mis-split produces plausible tokens and a wrong prompt.
    [[nodiscard]] std::vector<int> encode(std::string_view text, bool* ok = nullptr) const;

    // Token ids -> the exact original bytes. decode(encode(x)) == x.
    [[nodiscard]] std::string decode(std::span<const int> ids) const;

    [[nodiscard]] int vocab_size() const noexcept { return vocab_size_; }
    [[nodiscard]] int eot_id() const noexcept { return eot_id_; }
    // FNV-1a-64 over the two source files: the checkpoint records this so a
    // mismatched vocabulary fails loudly. `vocab_size == vocab_size` is vacuous
    // for BPE — 50257 == 50257 says nothing about WHICH 50257.
    [[nodiscard]] std::uint64_t fingerprint() const noexcept { return fingerprint_; }

private:
    BpeTokenizer() = default;
    [[nodiscard]] std::uint32_t rank_of(std::uint32_t a, std::uint32_t b) const noexcept;
    [[nodiscard]] std::uint32_t merged_of(std::uint32_t a, std::uint32_t b) const noexcept;

    std::vector<char> blob_;            // decoded bytes for every id, concatenated
    std::vector<std::uint32_t> off_;    // [vocab_size + 1] offsets into blob_
    std::vector<std::uint64_t> key_;    // open-addressed: (a << 32) | b, kEmpty when free
    std::vector<std::uint32_t> rank_;   // merge rank (line number in merges.txt)
    std::vector<std::uint32_t> merged_; // resulting token id
    std::uint64_t hmask_ = 0;           // capacity - 1, capacity a power of two
    std::int32_t byte_id_[256]{};       // byte -> id of the alphabet char it maps to
    int vocab_size_ = 0;
    int eot_id_ = -1;
    std::uint64_t fingerprint_ = 0;
};

}  // namespace cppgpt
