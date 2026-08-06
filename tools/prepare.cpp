// prepare — tokenize a text corpus into the training data format.
//
// Reads a UTF-8/ASCII text file and writes, from an output *stem*:
//   <stem>.train.bin   tokens as a flat little-endian uint16 array
//   <stem>.val.bin     the held-out tail (only when --val-frac > 0)
//   <stem>.vocab       the vocabulary (id->byte string)
//
// Tokenization lives only here, in the C++ CharTokenizer, so the .vocab is
// authoritative and there is no second tokenizer to drift out of parity.
//
// --vocab <existing.vocab> reuses a previous run's vocabulary instead of building
// a new one. This is what makes fine-tuning possible: a fresh corpus yields a
// different character set, hence a different vocab_size, hence ShapeMismatch when
// loading the base checkpoint. A byte absent from the reused vocabulary is a hard
// error naming the byte — never a silent drop or remap.
//
// Usage: prepare <corpus.txt> <stem> [--val-frac F] [--vocab <existing.vocab>]
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/dataloader.hpp"
#include "cppgpt/tokenizer.hpp"
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "prepare: cannot open '%s'\n", path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void write_text(const std::string& path, std::string_view body) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
    f.close();  // close before checking: a small write may not have flushed yet
    if (!f) {
        std::fprintf(stderr, "prepare: writing '%s' failed\n", path.c_str());
        std::exit(1);
    }
}

// Report every byte of `corpus` absent from `vocab`, then exit. Reporting all of
// them (with counts) beats aborting on the first: the user needs to know whether
// this is one stray character or a fundamentally different alphabet.
[[noreturn]] void report_unknown_bytes(std::string_view corpus, std::string_view vocab,
                                       const std::string& vocab_path) {
    bool in_vocab[256] = {};
    for (unsigned char c : vocab) in_vocab[c] = true;
    std::map<unsigned char, std::size_t> missing;
    for (unsigned char c : corpus)
        if (!in_vocab[c]) ++missing[c];
    std::fprintf(stderr,
                 "prepare: corpus contains %zu byte value(s) absent from '%s'.\n"
                 "  The vocabulary cannot grow without changing vocab_size, which would make the\n"
                 "  corpus incompatible with any checkpoint trained on that vocabulary.\n",
                 missing.size(), vocab_path.c_str());
    for (const auto& [byte, count] : missing) {
        if (byte >= 0x20 && byte < 0x7F)
            std::fprintf(stderr, "    '%c'  (0x%02X) x%zu\n", byte, byte, count);
        else
            std::fprintf(stderr, "    0x%02X x%zu\n", byte, count);
    }
    std::fprintf(stderr, "  Fix: clean the corpus, or omit --vocab to build a fresh vocabulary\n"
                         "  (which then needs a from-scratch model, not a fine-tune).\n");
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    // Line-buffer stdout. Redirected to a file or a pipe it is block-buffered by
    // default, so a long run emits nothing until 4 KB accumulates: progress is
    // invisible to `tail -f` and a working run is indistinguishable from a hung
    // one. Found by running a real training job, not by any test.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(argc, argv, {"val-frac", "vocab"});
    
    if (args.positional().size() != 2) {
        std::fprintf(stderr,
                     "usage: prepare <corpus.txt> <stem> [--val-frac F] [--vocab <existing.vocab>]\n"
                     "  writes <stem>.train.bin, <stem>.val.bin (if --val-frac > 0), <stem>.vocab\n");
        return 2;
    }
    const std::string corpus_path = args.positional()[0];
    const std::string stem = args.positional()[1];
    const float val_frac = args.real("val-frac", 0.0f);
    const std::string reuse_vocab = std::string(args.str("vocab", ""));

    if (val_frac < 0.0f || val_frac >= 1.0f) {
        std::fprintf(stderr, "prepare: --val-frac must be in [0, 1), got %g\n",
                     static_cast<double>(val_frac));
        return 2;
    }

    const std::string corpus = read_file(corpus_path);
    if (corpus.empty()) {
        std::fprintf(stderr, "prepare: corpus is empty\n");
        return 1;
    }

    // Either reuse a vocabulary (fine-tuning) or derive one from this corpus.
    std::string vocab;
    if (!reuse_vocab.empty()) {
        vocab = read_file(reuse_vocab);
        if (vocab.empty()) {
            std::fprintf(stderr, "prepare: vocab '%s' is empty\n", reuse_vocab.c_str());
            return 1;
        }
        bool in_vocab[256] = {};
        for (unsigned char c : vocab) in_vocab[c] = true;
        for (unsigned char c : corpus)
            if (!in_vocab[c]) report_unknown_bytes(corpus, vocab, reuse_vocab);
    } else {
        vocab = std::string(CharTokenizer(corpus).vocab());
    }

    const CharTokenizer tok(vocab);
    const std::vector<int> ids = tok.encode(corpus);

    // Contiguous tail split: the val set must not be interleaved with train, or a
    // model that memorizes n-grams scores well on it for the wrong reason.
    const std::size_t n_val =
        static_cast<std::size_t>(static_cast<double>(ids.size()) * static_cast<double>(val_frac));
    const std::size_t n_train = ids.size() - n_val;

    const std::string train_path = stem + ".train.bin";
    const std::string vocab_path = stem + ".vocab";
    if (const auto r = write_token_bin(train_path.c_str(),
                                       std::span<const int>(ids.data(), n_train));
        !r) {
        std::fprintf(stderr, "prepare: writing '%s' failed: %s\n", train_path.c_str(),
                     describe(r.error()));
        return 1;
    }
    if (n_val > 0) {
        const std::string val_path = stem + ".val.bin";
        if (const auto r = write_token_bin(val_path.c_str(),
                                           std::span<const int>(ids.data() + n_train, n_val));
            !r) {
            std::fprintf(stderr, "prepare: writing '%s' failed: %s\n", val_path.c_str(),
                         describe(r.error()));
            return 1;
        }
    }
    write_text(vocab_path, vocab);

    std::printf("prepare: %zu chars -> %zu tokens (train %zu, val %zu), vocab %d\n", corpus.size(),
                ids.size(), n_train, n_val, tok.vocab_size());
    std::printf("  %s\n", train_path.c_str());
    if (n_val > 0) std::printf("  %s.val.bin\n", stem.c_str());
    std::printf("  %s\n", vocab_path.c_str());
    return 0;
}
