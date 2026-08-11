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
#include <string>
#include <vector>

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
        argc, argv, {"checkpoint", "vocab", "prompt", "n", "temperature", "top-k", "seed"});

    const std::string ckpt(args.str("checkpoint", ""));
    const std::string vocab_path(args.str("vocab", ""));
    const std::string prompt(args.str("prompt", ""));
    if (ckpt.empty() || vocab_path.empty() || prompt.empty()) {
        std::fprintf(stderr,
                     "usage: generate --checkpoint <file.ckpt> --vocab <file.vocab> "
                     "--prompt \"text\"\n"
                     "                [--n K] [--temperature F] [--top-k K] [--seed N]\n"
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

    const std::string vocab = read_file(vocab_path, "vocab");
    const CharTokenizer tok(vocab);
    if (tok.vocab_size() != cfg.vocab_size) {
        std::fprintf(stderr,
                     "generate: vocab '%s' has %d symbols but the checkpoint was trained with %d.\n"
                     "  This is the wrong vocabulary for this model — decoding would be gibberish.\n",
                     vocab_path.c_str(), tok.vocab_size(), cfg.vocab_size);
        return 1;
    }

    GPT2 model(cfg, /*B=*/1, cfg.max_seq_len);
    if (const auto r = model.load_checkpoint(ckpt.c_str()); !r) {
        std::fprintf(stderr, "generate: cannot load '%s': %s\n", ckpt.c_str(), describe(r.error()));
        return 1;
    }

    const std::vector<int> prompt_ids = tok.encode(prompt);
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

    std::printf("%s%s\n", prompt.c_str(), tok.decode(out).c_str());
    return 0;
}
