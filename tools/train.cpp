// train — the first end-to-end cppgpt training demo.
//
// Trains a baby GPT-2 (char-level) on a text corpus, printing loss/lr/grad-norm
// each step. Runnable loop over the verified forward / backward / AdamW core:
// tokenize → sample random windows → forward → backward → clip → AdamW, on a
// cosine LR schedule. If a checkpoint path is given it resumes from it (when the
// file exists) and saves periodically. The mmap uint16 dataloader is wired via
// tools/prepare + a follow-up; here batches are sampled from the in-memory corpus.
//
// Usage: train [corpus.txt] [steps] [checkpoint.ckpt]
//   corpus.txt       path to a UTF-8/ASCII text file (default: a small built-in corpus)
//   steps            number of optimizer steps (default: 30)
//   checkpoint.ckpt  optional: resume from this file if it exists; save to it every
//                    50 steps and at the end
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/optimizer.hpp"
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
    // An empty first arg ("") selects the built-in corpus, so the later positional
    // args (steps, checkpoint) stay reachable without a real corpus file.
    const bool have_corpus = (argc > 1) && (argv[1][0] != '\0');
    const std::string corpus = have_corpus ? read_file(argv[1]) : std::string(kDefaultCorpus);
    const int steps = (argc > 2) ? std::atoi(argv[2]) : 30;
    const char* ckpt = (argc > 3) ? argv[3] : nullptr;

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

    // Resume from `ckpt` if it exists; a corrupt/mismatched file is fatal (we do
    // not silently start fresh over a real but unusable checkpoint).
    if (ckpt != nullptr) {
        std::ifstream probe(ckpt, std::ios::binary);
        if (probe.good()) {
            probe.close();
            const auto r = model.load_checkpoint(ckpt);
            if (!r) {
                std::fprintf(stderr, "train: cannot resume from '%s': %s\n", ckpt,
                             describe(r.error()));
                return 1;
            }
            std::printf("train: resumed from %s\n", ckpt);
        }
    }

    std::vector<int> inputs(static_cast<std::size_t>(B) * T);
    std::vector<int> targets(static_cast<std::size_t>(B) * T);

    // Cosine LR with linear warmup + global grad-norm clipping (canonical betas
    // 0.9/0.95, eps 1e-8). opt.lr is overwritten each step by the schedule.
    AdamW opt{};
    const float max_lr = 1e-3f, min_lr = 1e-4f, grad_clip = 1.0f;
    const int warmup = std::max(0, std::min(steps / 10, steps - 1));

    for (int step = 1; step <= steps; ++step) {
        sample_batch(data, B, T, gen, inputs, targets);
        model.forward(inputs.data(), targets.data());
        const float loss = model.mean_loss();
        model.zero_grads();
        model.backward(inputs.data(), targets.data());
        const float gnorm = model.clip_grad_norm(grad_clip);
        opt.lr = cosine_lr(step - 1, max_lr, min_lr, warmup, steps);  // 0-based step
        model.update(opt);
        std::printf("step %3d/%d  loss %.4f  lr %.2e  |g| %.3f\n", step, steps,
                    static_cast<double>(loss), static_cast<double>(opt.lr),
                    static_cast<double>(gnorm));

        if (ckpt != nullptr && (step % 50 == 0 || step == steps)) {
            const auto r = model.save_checkpoint(ckpt);
            if (!r)
                std::fprintf(stderr, "train: checkpoint save failed: %s\n", describe(r.error()));
            else
                std::printf("  [checkpoint saved: %s]\n", ckpt);
        }
    }

    std::printf("train: done.\n");
    return 0;
}
