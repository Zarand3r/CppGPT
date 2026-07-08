// CharTokenizer: deterministic char-level vocab with exact round-trip.
#include "cppgpt/tokenizer.hpp"

#include <string>
#include <string_view>
#include <vector>

#include "tests/check.hpp"

using namespace cppgpt;

int main() {
    // ---- vocab: distinct bytes, ids in ascending byte order ----
    {
        CharTokenizer tok("ba");           // distinct bytes {'a','b'}, 'a' < 'b'
        CHECK(tok.vocab_size() == 2);
        CHECK(tok.encode("ba") == (std::vector<int>{1, 0}));  // 'a'->0, 'b'->1
        CHECK(tok.encode("ab") == (std::vector<int>{0, 1}));
    }

    // ---- round-trip: decode(encode(text)) == text over the corpus alphabet ----
    {
        const std::string corpus = "hello, world!\nThe quick brown fox 0123456789.";
        CharTokenizer tok(corpus);
        const std::vector<int> ids = tok.encode(corpus);
        CHECK(ids.size() == corpus.size());
        CHECK(tok.decode(ids) == corpus);

        // A substring drawn from the same alphabet also round-trips.
        const std::string_view sub = "brown fox";
        CHECK(tok.decode(tok.encode(sub)) == sub);
    }

    // ---- determinism: same corpus builds the identical mapping ----
    {
        const std::string corpus = "the rain in spain";
        CharTokenizer a(corpus), b(corpus);
        CHECK(a.vocab_size() == b.vocab_size());
        CHECK(a.encode(corpus) == b.encode(corpus));
    }

    // ---- vocab size equals the number of distinct bytes ----
    {
        CharTokenizer tok("aaaabbbccd");  // {a,b,c,d}
        CHECK(tok.vocab_size() == 4);
        // ids are contiguous [0, V): every distinct byte maps back exactly.
        const std::vector<int> ids = tok.encode("abcd");
        CHECK(ids == (std::vector<int>{0, 1, 2, 3}));
    }

    return cppgpt::test::summary();
}
