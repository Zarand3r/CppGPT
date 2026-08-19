#include "cppgpt/bpe.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "cppgpt/checkpoint.hpp"  // fnv1a_64

namespace cppgpt {
namespace {

constexpr std::uint64_t kEmpty = ~0ULL;  // no id pair can be (0xFFFFFFFF, 0xFFFFFFFF)

std::uint64_t mix64(std::uint64_t x) noexcept {  // splitmix64 finalizer
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// GPT-2's bytes_to_unicode: every byte maps to a PRINTABLE codepoint, so no
// whitespace or control byte ever appears literally in the vocabulary. The three
// seeded ranges are the printable ones; everything else is assigned 256+n in
// ascending byte order.
std::array<int, 256> byte_to_codepoint() noexcept {
    std::array<int, 256> cp{};
    std::array<bool, 256> seeded{};
    const auto seed = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) {
            cp[static_cast<std::size_t>(b)] = b;
            seeded[static_cast<std::size_t>(b)] = true;
        }
    };
    seed('!', '~');
    seed(0xA1, 0xAC);
    seed(0xAE, 0xFF);
    int n = 0;
    for (int b = 0; b < 256; ++b)
        if (!seeded[static_cast<std::size_t>(b)]) cp[static_cast<std::size_t>(b)] = 256 + n++;
    return cp;
}

void utf8_append(std::string& s, int cp) {
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

std::string read_file(const char* path, bool* ok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        *ok = false;
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    *ok = true;
    return ss.str();
}

// Targeted parser for vocab.json: a FLAT {"token": id, ...} object. Not a
// general JSON parser -- there is no nesting to handle, and a general one would
// be both larger and a wider surface on a file the user downloaded.
bool parse_vocab(std::string_view s, std::unordered_map<std::string, int>& out) {
    std::size_t i = 0;
    const auto skip_ws = [&] { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; };
    const auto parse_string = [&](std::string& dst) -> bool {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        while (i < s.size() && s[i] != '"') {
            if (s[i] != '\\') { dst += s[i++]; continue; }
            if (++i >= s.size()) return false;
            const char e = s[i++];
            switch (e) {
                case '"': dst += '"'; break;
                case '\\': dst += '\\'; break;
                case '/': dst += '/'; break;
                case 'b': dst += '\b'; break;
                case 'f': dst += '\f'; break;
                case 'n': dst += '\n'; break;
                case 'r': dst += '\r'; break;
                case 't': dst += '\t'; break;
                case 'u': {
                    if (i + 4 > s.size()) return false;
                    int cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        const char c = s[i + static_cast<std::size_t>(k)];
                        const int d = (c >= '0' && c <= '9')   ? c - '0'
                                      : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                                      : (c >= 'A' && c <= 'F') ? c - 'A' + 10
                                                               : -1;
                        if (d < 0) return false;
                        cp = cp * 16 + d;
                    }
                    i += 4;
                    utf8_append(dst, cp);
                    break;
                }
                default: return false;
            }
        }
        if (i >= s.size()) return false;
        ++i;  // closing quote
        return true;
    };

    skip_ws();
    if (i >= s.size() || s[i] != '{') return false;
    ++i;
    skip_ws();
    if (i < s.size() && s[i] == '}') return true;
    while (i < s.size()) {
        skip_ws();
        std::string key;
        if (!parse_string(key)) return false;
        skip_ws();
        if (i >= s.size() || s[i] != ':') return false;
        ++i;
        skip_ws();
        bool neg = false;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) neg = (s[i++] == '-');
        if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
        long long v = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = v * 10 + (s[i++] - '0');
        out.emplace(std::move(key), static_cast<int>(neg ? -v : v));
        skip_ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') return true;
        return false;
    }
    return false;
}

}  // namespace

std::uint32_t BpeTokenizer::rank_of(std::uint32_t a, std::uint32_t b) const noexcept {
    const std::uint64_t k = (static_cast<std::uint64_t>(a) << 32) | b;
    std::uint64_t h = mix64(k) & hmask_;
    while (key_[h] != kEmpty) {
        if (key_[h] == k) return rank_[h];
        h = (h + 1) & hmask_;
    }
    return ~0U;
}

std::uint32_t BpeTokenizer::merged_of(std::uint32_t a, std::uint32_t b) const noexcept {
    const std::uint64_t k = (static_cast<std::uint64_t>(a) << 32) | b;
    std::uint64_t h = mix64(k) & hmask_;
    while (key_[h] != kEmpty) {
        if (key_[h] == k) return merged_[h];
        h = (h + 1) & hmask_;
    }
    return ~0U;
}

Result<BpeTokenizer> BpeTokenizer::open(const char* vocab_json, const char* merges_txt) noexcept {
    ASSERT(vocab_json != nullptr && merges_txt != nullptr);
    bool ok = false;
    const std::string vs = read_file(vocab_json, &ok);
    if (!ok) return err(ErrorCode::IoError);
    const std::string ms = read_file(merges_txt, &ok);
    if (!ok) return err(ErrorCode::IoError);

    std::unordered_map<std::string, int> vocab;
    vocab.reserve(60000);
    if (!parse_vocab(vs, vocab)) return err(ErrorCode::ParseError);
    if (vocab.empty()) return err(ErrorCode::ParseError);

    BpeTokenizer t;
    t.vocab_size_ = 0;
    for (const auto& [k, v] : vocab) t.vocab_size_ = std::max(t.vocab_size_, v + 1);
    if (static_cast<std::size_t>(t.vocab_size_) != vocab.size()) return err(ErrorCode::ParseError);

    // Reverse byte map: a token's stored form is in mapped-codepoint space, but
    // decode() must hand back REAL bytes. Resolve that once, here, so decode is
    // a memcpy rather than per-call codepoint arithmetic.
    const std::array<int, 256> cp = byte_to_codepoint();
    std::unordered_map<std::string, char> cp_to_byte;
    cp_to_byte.reserve(512);
    for (int b = 0; b < 256; ++b) {
        std::string u;
        utf8_append(u, cp[static_cast<std::size_t>(b)]);
        cp_to_byte.emplace(u, static_cast<char>(b));
        const auto it = vocab.find(u);
        t.byte_id_[b] = (it == vocab.end()) ? -1 : it->second;
    }

    std::vector<std::string> id_to_tok(static_cast<std::size_t>(t.vocab_size_));
    for (const auto& [k, v] : vocab) {
        if (v < 0 || v >= t.vocab_size_) return err(ErrorCode::ParseError);
        id_to_tok[static_cast<std::size_t>(v)] = k;
        if (k == "<|endoftext|>") t.eot_id_ = v;
    }

    t.off_.resize(static_cast<std::size_t>(t.vocab_size_) + 1, 0);
    t.blob_.reserve(static_cast<std::size_t>(t.vocab_size_) * 6);
    for (int id = 0; id < t.vocab_size_; ++id) {
        const std::string& tok = id_to_tok[static_cast<std::size_t>(id)];
        // Walk the token's codepoints and map each back to its byte. A special
        // token like <|endoftext|> has no byte preimage; it decodes to itself.
        std::size_t p = 0;
        bool mapped = true;
        std::string bytes;
        while (p < tok.size()) {
            std::size_t len = 1;
            const auto c = static_cast<unsigned char>(tok[p]);
            if (c >= 0xE0) len = 3;
            else if (c >= 0xC0) len = 2;
            if (p + len > tok.size()) { mapped = false; break; }
            const auto it = cp_to_byte.find(tok.substr(p, len));
            if (it == cp_to_byte.end()) { mapped = false; break; }
            bytes += it->second;
            p += len;
        }
        const std::string& put = mapped ? bytes : tok;
        t.blob_.insert(t.blob_.end(), put.begin(), put.end());
        t.off_[static_cast<std::size_t>(id) + 1] = static_cast<std::uint32_t>(t.blob_.size());
    }

    // merges.txt: line 1 is "#version: ...". Every other non-empty line is a
    // merge, INCLUDING lines beginning with '#' -- eight of them are merges whose
    // first token is the '#' character. Filtering by prefix silently drops them.
    std::vector<std::pair<std::uint64_t, std::uint32_t>> pairs;  // packed key -> rank
    std::vector<std::uint32_t> merged_ids;
    {
        std::size_t pos = 0;
        const auto next_line = [&](std::string_view& out_line) {
            if (pos >= ms.size()) return false;
            const std::size_t e = ms.find('\n', pos);
            out_line = std::string_view(ms).substr(pos, (e == std::string::npos ? ms.size() : e) - pos);
            pos = (e == std::string::npos) ? ms.size() : e + 1;
            return true;
        };
        std::string_view line;
        if (!next_line(line)) return err(ErrorCode::ParseError);  // version line
        std::uint32_t rank = 0;
        while (next_line(line)) {
            if (line.empty()) continue;
            const std::size_t sp = line.find(' ');
            if (sp == std::string_view::npos) return err(ErrorCode::ParseError);
            const std::string a(line.substr(0, sp));
            const std::string b(line.substr(sp + 1));
            const auto ia = vocab.find(a);
            const auto ib = vocab.find(b);
            const auto iab = vocab.find(a + b);
            // A merge naming a token the vocab lacks means the two files do not
            // belong together; proceeding would leave a silently partial table.
            if (ia == vocab.end() || ib == vocab.end() || iab == vocab.end())
                return err(ErrorCode::ParseError);
            pairs.emplace_back((static_cast<std::uint64_t>(ia->second) << 32) |
                                   static_cast<std::uint32_t>(ib->second),
                               rank);
            merged_ids.push_back(static_cast<std::uint32_t>(iab->second));
            ++rank;
        }
    }
    if (pairs.empty()) return err(ErrorCode::ParseError);

    std::size_t cap = 1;
    while (cap < pairs.size() * 2) cap <<= 1;
    t.key_.assign(cap, kEmpty);
    t.rank_.assign(cap, 0);
    t.merged_.assign(cap, 0);
    t.hmask_ = cap - 1;
    for (std::size_t n = 0; n < pairs.size(); ++n) {
        std::uint64_t h = mix64(pairs[n].first) & t.hmask_;
        while (t.key_[h] != kEmpty) h = (h + 1) & t.hmask_;
        t.key_[h] = pairs[n].first;
        t.rank_[h] = pairs[n].second;
        t.merged_[h] = merged_ids[n];
    }

    t.fingerprint_ = fnv1a_64(fnv1a_64(kFnvOffset64, vs.data(), vs.size()), ms.data(), ms.size());
    return t;
}

std::vector<int> BpeTokenizer::encode(std::string_view text, bool* ok) const {
    if (ok != nullptr) *ok = true;
    std::vector<int> out;
    out.reserve(text.size() / 3 + 4);

    const auto is_letter = [](unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    const auto is_digit = [](unsigned char c) { return c >= '0' && c <= '9'; };
    const auto is_space = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };

    std::vector<std::uint32_t> sym;  // reused across pieces: no per-word allocation
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        const auto c0 = static_cast<unsigned char>(text[i]);
        if (c0 >= 0x80) {  // non-ASCII: this pre-tokenizer cannot split it exactly
            if (ok != nullptr) *ok = false;
            return out;
        }
        std::size_t j = i;

        // 's | 't | 're | 've | 'm | 'll | 'd  -- lowercase only, as in the regex
        if (c0 == '\'' && i + 1 < n) {
            static const char* kSuf[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            for (const char* suf : kSuf) {
                const std::size_t L = std::strlen(suf);
                if (text.compare(i + 1, L, suf) == 0) { j = i + 1 + L; break; }
            }
        }
        if (j == i) {
            // ` ?\p{L}+` then ` ?\p{N}+` then ` ?[^\s\p{L}\p{N}]+`
            const std::size_t s0 = (c0 == ' ' && i + 1 < n) ? i + 1 : i;
            const auto c1 = static_cast<unsigned char>(s0 < n ? text[s0] : 0);
            if (s0 < n && is_letter(c1)) {
                j = s0;
                while (j < n && is_letter(static_cast<unsigned char>(text[j]))) ++j;
            } else if (s0 < n && is_digit(c1)) {
                j = s0;
                while (j < n && is_digit(static_cast<unsigned char>(text[j]))) ++j;
            } else if (s0 < n && !is_space(c1) && c1 < 0x80) {
                j = s0;
                while (j < n && !is_space(static_cast<unsigned char>(text[j])) &&
                       !is_letter(static_cast<unsigned char>(text[j])) &&
                       !is_digit(static_cast<unsigned char>(text[j])) &&
                       static_cast<unsigned char>(text[j]) < 0x80)
                    ++j;
            }
        }
        if (j == i && is_space(c0)) {
            // `\s+(?!\S)`: the run, minus its last character when a non-space
            // follows -- that last space belongs to the next word's ` ?` prefix.
            std::size_t e = i;
            while (e < n && is_space(static_cast<unsigned char>(text[e]))) ++e;
            j = (e == n) ? e : (e - 1 > i ? e - 1 : i);
            if (j == i) j = e;  // `\s+`: a single space with a non-space after it
        }
        if (j <= i) j = i + 1;  // never stall

        // Piece -> byte ids -> merge to fixpoint, always taking the lowest rank.
        sym.clear();
        for (std::size_t k = i; k < j; ++k) {
            const int id = byte_id_[static_cast<unsigned char>(text[k])];
            if (id < 0) { if (ok != nullptr) *ok = false; return out; }
            sym.push_back(static_cast<std::uint32_t>(id));
        }
        while (sym.size() >= 2) {
            std::uint32_t best = ~0U;
            std::size_t at = 0;
            for (std::size_t k = 0; k + 1 < sym.size(); ++k) {
                const std::uint32_t r = rank_of(sym[k], sym[k + 1]);
                if (r < best) { best = r; at = k; }
            }
            if (best == ~0U) break;
            sym[at] = merged_of(sym[at], sym[at + 1]);
            sym.erase(sym.begin() + static_cast<std::ptrdiff_t>(at) + 1);
        }
        for (const std::uint32_t s : sym) out.push_back(static_cast<int>(s));
        i = j;
    }
    return out;
}

std::string BpeTokenizer::decode(std::span<const int> ids) const {
    std::string out;
    std::size_t total = 0;
    for (const int id : ids)
        if (id >= 0 && id < vocab_size_)
            total += off_[static_cast<std::size_t>(id) + 1] - off_[static_cast<std::size_t>(id)];
    out.reserve(total);
    for (const int id : ids) {
        ASSERT_MSG(id >= 0 && id < vocab_size_, "BpeTokenizer::decode: id out of range");
        const std::size_t a = off_[static_cast<std::size_t>(id)];
        const std::size_t b = off_[static_cast<std::size_t>(id) + 1];
        out.append(blob_.data() + a, b - a);
    }
    return out;
}

}  // namespace cppgpt
