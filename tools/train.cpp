// train — the first end-to-end cppgpt training demo.
//
// Trains a baby GPT-2 (char-level) on a text corpus for a handful of steps,
// printing the loss each step. This is the minimal runnable loop over the already
// verified forward / backward / AdamW core: tokenize → sample random windows →
// forward → backward → AdamW. The production data pipeline (mmap uint16, shuffled
// epochs), LR schedule, gradient clipping, checkpointing, and sampling are M2 —
// deliberately out of scope here.
//
// Usage: train [corpus.txt] [steps]
//   corpus.txt  path to a UTF-8/ASCII text file (default: a small built-in corpus)
//   steps       number of optimizer steps (default: 30)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "cppgpt/tokenizer.hpp"

namespace {
using namespace cppgpt;

// A small built-in corpus so the trainer runs with zero external files.
constexpr const char* kDefaultCorpus =
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die-to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep, perchance to dream-ay, there's the rub:\n"
    "For in that sleep of death what dreams may come,\n"
    "When we have shuffled off this mortal coil,\n"
    "Must give us pause-there's the respect\n"
    "That makes calamity of so long life.\n";

std::string read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "train: cannot open corpus file '%s'\n", path);
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Fill inputs/targets [B*T] with random contiguous windows of `data`; targets are
// the next-token shift of inputs. Requires data.size() > T (checked by caller).
void sample_batch(const std::vector<int>& data, int B, int T, Generator& gen,
                  std::vector<int>& inputs, std::vector<int>& targets) {
    // 64-bit start index so a large corpus never truncates through int.
    const std::int64_t max_start = static_cast<std::int64_t>(data.size()) - T - 1;
    for (int b = 0; b < B; ++b) {
        const std::int64_t s = gen.uniform_int(0, max_start);
        for (int t = 0; t < T; ++t) {
            const std::size_t o = static_cast<std::size_t>(b) * T + t;
            inputs[o] = data[static_cast<std::size_t>(s + t)];
            targets[o] = data[static_cast<std::size_t>(s + t + 1)];
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string corpus = (argc > 1) ? read_file(argv[1]) : std::string(kDefaultCorpus);
    const int steps = (argc > 2) ? std::atoi(argv[2]) : 30;

    if (corpus.empty()) {
        std::fprintf(stderr, "train: corpus is empty\n");
        return 1;
    }
    CharTokenizer tok(corpus);
    const std::vector<int> data = tok.encode(corpus);

    // Baby config (tanh GELU, bias, fp32 — canonical GPT-2, just small).
    Config cfg{};
    cfg.max_seq_len = 64;
    cfg.vocab_size = tok.vocab_size();
    cfg.n_layer = 3;
    cfg.n_head = 4;
    cfg.n_embd = 64;
    const int B = 4, T = 32;

    if (data.size() <= static_cast<std::size_t>(T) + 1) {
        std::fprintf(stderr, "train: corpus too short (%zu tokens) for context T=%d\n", data.size(),
                     T);
        return 1;
    }

    std::printf("train: %zu chars, vocab %d, model L%d H%d C%d, B%d T%d, %d steps\n", corpus.size(),
                cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_embd, B, T, steps);

    Generator gen(1337ULL);
    GPT2 model(cfg, B, T);
    model.init_weights(gen);

    std::vector<int> inputs(static_cast<std::size_t>(B) * T);
    std::vector<int> targets(static_cast<std::size_t>(B) * T);
    const float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.95f, eps = 1e-8f, weight_decay = 0.0f;

    for (int step = 1; step <= steps; ++step) {
        sample_batch(data, B, T, gen, inputs, targets);
        model.forward(inputs.data(), targets.data(), B, T);
        const float loss = model.mean_loss();
        std::printf("step %3d/%d  loss %.4f\n", step, steps, static_cast<double>(loss));
        model.zero_grads();
        model.backward(inputs.data(), targets.data(), B, T);
        model.update(lr, beta1, beta2, eps, weight_decay);
    }

    std::printf("train: done.\n");
    return 0;
}
