// generate — sample text from a checkpoint you trained.
//
// Reads the model's architecture out of the checkpoint header, rebuilds the
// model, loads the weights, tokenizes the prompt with the run's vocabulary, and
// samples autoregressively.
//
// The header peek is load-bearing, not a convenience: GPT2's constructor needs a
// Config *before* load_checkpoint can validate one against it, so without reading
// the header first you would have to retype the exact architecture on the command
// line and any mismatch would be a ShapeMismatch.
//
// Usage:
//   generate --checkpoint <file.ckpt> --vocab <file.vocab> --prompt "text"
//            [--n K] [--temperature F] [--top-k K] [--seed N]
//
// --top-k 1 is greedy decoding (tie-stable: sample() routes it to argmax).
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <optional>
#include <string>
#include <vector>

#include "cppgpt/bpe.hpp"
#include "cppgpt/checkpoint.hpp"
#include "cppgpt/core.hpp"
#include "cppgpt/generate.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/tokenizer.hpp"
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;

std::string read_file(const std::string& path, const char* what) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "generate: cannot open %s '%s'\n", what, path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    // Line-buffer stdout. Redirected to a file or a pipe it is block-buffered by
    // default, so a long run emits nothing until 4 KB accumulates: progress is
    // invisible to `tail -f` and a working run is indistinguishable from a hung
    // one. Found by running a real training job, not by any test.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const cli::Args args(
        argc, argv, {"checkpoint", "vocab", "prompt", "n", "temperature", "top-k", "seed", "merges"});

    const std::string ckpt(args.str("checkpoint", ""));
    const std::string vocab_path(args.str("vocab", ""));
    const std::string prompt(args.str("prompt", ""));
    if (ckpt.empty() || vocab_path.empty() || prompt.empty()) {
        std::fprintf(stderr,
                     "usage: generate --checkpoint <file.ckpt> --vocab <file.vocab> "
                     "--prompt \"text\"\n"
                     "                [--merges merges.txt] [--n K] [--temperature F]\n"
                     "                [--top-k K] [--seed N]\n"
                     "  --merges is required for a BPE checkpoint (--vocab is then vocab.json)\n"
                     "  --top-k 1 gives greedy (deterministic) decoding\n");
        return 2;
    }
    const int n_new = args.integer("n", 200);
    const float temperature = args.real("temperature", 0.8f);
    const int top_k = args.integer("top-k", 40);
    const auto seed = static_cast<std::uint64_t>(args.integer("seed", 1337));

    // Architecture comes from the checkpoint, not the command line.
    auto peek = CheckpointFile::open(ckpt.c_str());
    if (!peek) {
        std::fprintf(stderr, "generate: cannot read checkpoint '%s': %s\n", ckpt.c_str(),
                     describe(peek.error()));
        return 1;
    }
    const CheckpointHeader& h = peek->header();
    Config cfg{};
    cfg.max_seq_len = h.max_seq_len;
    cfg.vocab_size = h.vocab_size;
    cfg.n_layer = h.n_layer;
    cfg.n_head = h.n_head;
    cfg.n_embd = h.n_embd;

    // Which tokenizer the weights expect is recorded IN the checkpoint (v3+).
    // Guessing from vocab size would be vacuous for BPE -- 50257 == 50257 says
    // nothing about which 50257 -- so the header decides and the fingerprint
    // confirms.
    const bool is_bpe = (h.tokenizer_kind == kTokenizerGpt2Bpe);
    const std::string merges_path(args.str("merges", ""));
    if (is_bpe && merges_path.empty()) {
        std::fprintf(stderr,
                     "generate: this checkpoint uses GPT-2 BPE; pass --merges merges.txt\n"
                     "  (and --vocab vocab.json)\n");
        return 1;
    }

    std::optional<CharTokenizer> ctok;
    std::optional<BpeTokenizer> btok;
    if (is_bpe) {
        auto t = BpeTokenizer::open(vocab_path.c_str(), merges_path.c_str());
        if (!t) {
            std::fprintf(stderr, "generate: cannot load BPE tokenizer: %s\n", describe(t.error()));
            return 1;
        }
        btok.emplace(std::move(*t));
        if (h.vocab_fingerprint != 0 && btok->fingerprint() != h.vocab_fingerprint) {
            std::fprintf(stderr,
                         "generate: tokenizer fingerprint %016llx does not match the checkpoint's "
                         "%016llx.\n  These files were not built together — decoding would be "
                         "gibberish.\n",
                         static_cast<unsigned long long>(btok->fingerprint()),
                         static_cast<unsigned long long>(h.vocab_fingerprint));
            return 1;
        }
    } else {
        ctok.emplace(read_file(vocab_path, "vocab"));
    }
    const int tok_vocab = is_bpe ? btok->vocab_size() : ctok->vocab_size();
    if (tok_vocab != cfg.vocab_size) {
        std::fprintf(stderr,
                     "generate: vocab '%s' has %d symbols but the checkpoint was trained with %d.\n"
                     "  This is the wrong vocabulary for this model — decoding would be gibberish.\n",
                     vocab_path.c_str(), tok_vocab, cfg.vocab_size);
        return 1;
    }

    GPT2 model(cfg, /*B=*/1, cfg.max_seq_len);
    if (const auto r = model.load_checkpoint(ckpt.c_str()); !r) {
        std::fprintf(stderr, "generate: cannot load '%s': %s\n", ckpt.c_str(), describe(r.error()));
        return 1;
    }

    std::vector<int> prompt_ids;
    if (is_bpe) {
        bool ok = false;
        prompt_ids = btok->encode(prompt, &ok);
        if (!ok) {
            std::fprintf(stderr,
                         "generate: this prompt contains bytes the BPE pre-tokenizer cannot split "
                         "exactly (non-ASCII).\n  Refusing rather than mis-tokenising it.\n");
            return 1;
        }
    } else {
        prompt_ids = ctok->encode(prompt);
    }
    if (static_cast<int>(prompt_ids.size()) > cfg.max_seq_len) {
        std::fprintf(stderr, "generate: prompt is %zu tokens but the context is %d\n",
                     prompt_ids.size(), cfg.max_seq_len);
        return 1;
    }

    std::fprintf(stderr, "generate: L%d H%d C%d ctx %d vocab %d | %d tokens, temp %.2f, top-k %d\n",
                 cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.max_seq_len, cfg.vocab_size, n_new,
                 static_cast<double>(temperature), top_k);

    Generator gen(seed);
    const std::vector<int> out = generate_absolute(model, prompt_ids, n_new, temperature, top_k, gen);

    std::printf("%s%s\n", prompt.c_str(),
                (is_bpe ? btok->decode(out) : ctok->decode(out)).c_str());
    return 0;
}
