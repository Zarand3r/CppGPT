// generate — train a baby GPT-2 on a corpus, then sample text from it.
//
// Demonstrates the full toy loop: tokenize → train (forward/backward/AdamW) →
// autoregressively generate and decode. Char-level, single process (no checkpoint
// loading yet — that's M2). The production sampler (nucleus, repetition penalty)
// and a KV cache (M3) are out of scope; this is the minimal train-and-sample demo.
//
// Usage: generate [corpus.txt] [train_steps] [n_generate]
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cppgpt/generate.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/optimizer.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/tokenizer.hpp"

namespace {
using namespace cppgpt;

constexpr const char* kDefaultCorpus =
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die-to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n";

std::string read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "generate: cannot open '%s'\n", path);
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string corpus = (argc > 1) ? read_file(argv[1]) : std::string(kDefaultCorpus);
    const int steps = (argc > 2) ? std::atoi(argv[2]) : 400;
    const int n_gen = (argc > 3) ? std::atoi(argv[3]) : 256;

    CharTokenizer tok(corpus);
    const std::vector<int> data = tok.encode(corpus);

    Config cfg{};
    cfg.max_seq_len = 64;
    cfg.vocab_size = tok.vocab_size();
    cfg.n_layer = 3;
    cfg.n_head = 4;
    cfg.n_embd = 64;
    const int T = 32;
    if (data.size() <= static_cast<std::size_t>(T) + 1) {
        std::fprintf(stderr, "generate: corpus too short for context T=%d\n", T);
        return 1;
    }

    Generator gen(1337ULL);
    GPT2 model(cfg, /*B=*/1, T);
    model.init_weights(gen);

    // Train on random windows (single-sequence batches).
    std::printf("training: %zu chars, vocab %d, %d steps ...\n", corpus.size(), cfg.vocab_size,
                steps);
    const AdamW opt{.lr = 1e-3f};
    std::vector<int> inp(static_cast<std::size_t>(T)), tgt(static_cast<std::size_t>(T));
    const std::int64_t max_start = static_cast<std::int64_t>(data.size()) - T - 1;
    for (int step = 1; step <= steps; ++step) {
        const std::int64_t s = gen.uniform_int(0, max_start);
        for (int t = 0; t < T; ++t) {
            inp[static_cast<std::size_t>(t)] = data[static_cast<std::size_t>(s + t)];
            tgt[static_cast<std::size_t>(t)] = data[static_cast<std::size_t>(s + t + 1)];
        }
        model.forward(inp.data(), tgt.data());
        if (step % 50 == 0 || step == 1)
            std::printf("  step %4d/%d  loss %.4f\n", step, steps,
                        static_cast<double>(model.mean_loss()));
        model.zero_grads();
        model.backward(inp.data(), tgt.data());
        model.update(opt);
    }

    // Seed with the first T tokens of the corpus, then sample.
    std::vector<int> seed(data.begin(), data.begin() + T);
    const std::vector<int> out = generate(model, seed.data(), n_gen, /*temperature=*/0.8f,
                                          /*top_k=*/20, gen);

    std::printf("\n--- sample (seed + %d generated) ---\n%s", n_gen, tok.decode(seed).c_str());
    std::printf("%s\n", tok.decode(out).c_str());
    return 0;
}
