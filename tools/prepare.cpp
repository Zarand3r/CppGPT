// prepare — tokenize a text corpus into the training data format.
//
// Reads a UTF-8/ASCII text file, builds the char-level vocabulary, and writes two
// files that tools/train consumes via the mmap DataLoader:
//   <out.bin>    the tokens as a flat little-endian uint16 array
//   <out.vocab>  the vocabulary (id->byte string) so the trainer can size the
//                model and reload the exact tokenizer without the original text
//
// Keeping tokenization in C++ (one CharTokenizer) makes the .vocab authoritative
// — there is no second, drifting tokenizer to keep in parity. This is the M2
// analogue of nanoGPT's prepare.py; the BPE tokenizer is M3.
//
// Usage: prepare <corpus.txt> <out.bin> [out.vocab]
//   out.vocab defaults to <out.bin>.vocab
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/dataloader.hpp"
#include "cppgpt/tokenizer.hpp"

namespace {
using namespace cppgpt;

std::string read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "prepare: cannot open '%s'\n", path);
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: prepare <corpus.txt> <out.bin> [out.vocab]\n");
        return 2;
    }
    const char* corpus_path = argv[1];
    const char* bin_path = argv[2];
    const std::string vocab_path = (argc > 3) ? argv[3] : std::string(bin_path) + ".vocab";

    const std::string corpus = read_file(corpus_path);
    if (corpus.empty()) {
        std::fprintf(stderr, "prepare: corpus is empty\n");
        return 1;
    }

    CharTokenizer tok(corpus);
    const std::vector<int> ids = tok.encode(corpus);

    if (const auto r = write_token_bin(bin_path, ids); !r) {
        std::fprintf(stderr, "prepare: writing '%s' failed: %s\n", bin_path, describe(r.error()));
        return 1;
    }

    // The vocabulary is just the distinct corpus bytes (ascending); persisting it
    // lets train reconstruct the identical CharTokenizer via CharTokenizer(vocab).
    {
        std::ofstream vf(vocab_path, std::ios::binary | std::ios::trunc);
        const std::string_view v = tok.vocab();
        vf.write(v.data(), static_cast<std::streamsize>(v.size()));
        if (!vf) {
            std::fprintf(stderr, "prepare: writing '%s' failed\n", vocab_path.c_str());
            return 1;
        }
    }

    std::printf("prepare: %zu chars -> %zu tokens, vocab %d\n  %s\n  %s\n", corpus.size(),
                ids.size(), tok.vocab_size(), bin_path, vocab_path.c_str());
    return 0;
}
