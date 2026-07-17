// The prepare -> train data contract: CharTokenizer + write_token_bin produce a
// uint16 .bin that DataLoader reads back as the same token stream, and the
// persisted vocab reconstructs the identical tokenizer. This is what keeps
// tools/prepare and tools/train agreeing without a second tokenizer.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/dataloader.hpp"
#include "cppgpt/tokenizer.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

namespace {
std::string tmp_path(const char* name) {
    const char* dir = std::getenv("TEST_TMPDIR");
    return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}
}  // namespace

int main() {
    const std::string corpus =
        "the quick brown fox jumps over the lazy dog.\n"
        "pack my box with five dozen liquor jugs.\n";
    CharTokenizer tok(corpus);
    const std::vector<int> ids = tok.encode(corpus);
    const std::string path = tmp_path("prep.bin");

    // ---- write_token_bin serializes exactly the ids as little-endian uint16 ----
    {
        auto r = write_token_bin(path.c_str(), ids);
        CHECK(r.has_value());
    }
    {
        std::ifstream f(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
        CHECK(bytes.size() == ids.size() * sizeof(std::uint16_t));
        bool same = true;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            std::uint16_t v = 0;
            std::memcpy(&v, bytes.data() + i * sizeof(std::uint16_t), sizeof(v));  // LE host
            same = same && (static_cast<int>(v) == ids[i]);
        }
        CHECK(same);
    }

    // ---- persisted vocab reconstructs the identical tokenizer ----
    {
        CharTokenizer tok2(tok.vocab());
        CHECK(tok2.vocab_size() == tok.vocab_size());
        CHECK(tok2.encode(corpus) == ids);  // same ids => same mapping
    }

    // ---- DataLoader reads the .bin back as the same stream ----
    {
        const int B = 2, T = 4, V = tok.vocab_size();
        auto r = DataLoader::open(path.c_str(), B, T, 42ULL);
        CHECK(r.has_value());
        DataLoader& dl = *r;
        CHECK(dl.num_tokens() == ids.size());
        dl.next_batch();
        bool ok = true;
        for (int b = 0; b < B; ++b) {
            const int* in = dl.inputs() + b * T;
            const int* tg = dl.targets() + b * T;
            for (int t = 0; t < T; ++t) {
                ok = ok && (in[t] >= 0 && in[t] < V) && (tg[t] >= 0 && tg[t] < V);
                if (t < T - 1) ok = ok && (tg[t] == in[t + 1]);  // contiguous next-token shift
            }
            std::vector<int> row(in, in + T);
            CHECK(!tok.decode(row).empty());  // decodes without aborting (all ids valid)
        }
        CHECK(ok);
    }

    // ---- write_token_bin rejects an out-of-uint16 id ----
    {
        const std::vector<int> bad{1, 2, 70000, 3};  // 70000 > 0xFFFF
        auto r = write_token_bin(tmp_path("bad.bin").c_str(), bad);
        CHECK(!r.has_value() && r.error() == ErrorCode::OutOfRange);
    }

    return cppgpt::test::summary();
}
