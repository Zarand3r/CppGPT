// train — the first end-to-end cppgpt training demo.
//
// Trains a baby GPT-2 (char-level), printing loss/lr/grad-norm each step: forward
// → backward → clip → AdamW on a cosine LR schedule. Two data sources, selected by
// the first arg's extension: a prepared *.bin token file streamed via the mmap
// DataLoader (its .vocab sidecar gives vocab_size), or a text corpus tokenized in
// memory. If a checkpoint path is given it resumes from it (when present) and
// saves periodically.
//
// Usage: train [corpus.txt | tokens.bin] [steps] [checkpoint.ckpt]
//   1st arg          *.bin -> mmap DataLoader (needs <arg>.vocab from tools/prepare);
//                    otherwise a text file ("" / omitted uses a small built-in corpus)
//   steps            number of optimizer steps (default: 30)
//   checkpoint.ckpt  optional: resume from this file if it exists; save every 50
//                    steps and at the end
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "cppgpt/core.hpp"
#include "cppgpt/dataloader.hpp"
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
    // First arg: a prepared token file (*.bin, streamed via the mmap DataLoader)
    // or a text corpus (tokenized in memory). "" / omitted selects the built-in
    // corpus so the later positional args (steps, checkpoint) stay reachable.
    const char* data_arg = (argc > 1 && argv[1][0] != '\0') ? argv[1] : nullptr;
    const int steps = (argc > 2) ? std::atoi(argv[2]) : 30;
    const char* ckpt = (argc > 3) ? argv[3] : nullptr;
    const bool from_bin = data_arg != nullptr && std::string_view(data_arg).ends_with(".bin");

    // Baby config (tanh GELU, bias, fp32 — canonical GPT-2, just small).
    Config cfg{};
    cfg.max_seq_len = 64;
    cfg.n_layer = 3;
    cfg.n_head = 4;
    cfg.n_embd = 64;
    const int B = 4, T = 32;

    // Data source: prepared .bin (mmap) fills `loader`; a text corpus fills `data`
    // and the reusable `inputs`/`targets` scratch. Exactly one is used below.
    Generator gen(1337ULL);
    std::optional<DataLoader> loader;
    std::vector<int> data, inputs, targets;

    if (from_bin) {
        // Vocab from the .vocab sidecar (tools/prepare wrote it): reconstruct the
        // tokenizer to recover vocab_size — never inferred by scanning the tokens.
        const std::string vpath = std::string(data_arg) + ".vocab";
        std::ifstream vf(vpath, std::ios::binary);
        if (!vf) {
            std::fprintf(stderr, "train: cannot open vocab '%s' (run tools/prepare first)\n",
                         vpath.c_str());
            return 1;
        }
        std::ostringstream vss;
        vss << vf.rdbuf();
        const std::string vocab = vss.str();
        if (vocab.empty()) {
            std::fprintf(stderr, "train: vocab '%s' is empty\n", vpath.c_str());
            return 1;
        }
        cfg.vocab_size = CharTokenizer(vocab).vocab_size();
        auto r = DataLoader::open(data_arg, B, T, 1337ULL);
        if (!r) {
            std::fprintf(stderr, "train: cannot open '%s': %s\n", data_arg, describe(r.error()));
            return 1;
        }
        loader.emplace(std::move(*r));
        std::printf("train: %zu tokens (mmap), vocab %d, model L%d H%d C%d, B%d T%d, %d steps\n",
                    loader->num_tokens(), cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_embd, B, T,
                    steps);
    } else {
        const std::string corpus = data_arg ? read_file(data_arg) : std::string(kDefaultCorpus);
        if (corpus.empty()) {
            std::fprintf(stderr, "train: corpus is empty\n");
            return 1;
        }
        CharTokenizer tok(corpus);
        data = tok.encode(corpus);
        cfg.vocab_size = tok.vocab_size();
        if (data.size() <= static_cast<std::size_t>(T) + 1) {
            std::fprintf(stderr, "train: corpus too short (%zu tokens) for context T=%d\n",
                         data.size(), T);
            return 1;
        }
        inputs.assign(static_cast<std::size_t>(B) * T, 0);
        targets.assign(static_cast<std::size_t>(B) * T, 0);
        std::printf("train: %zu chars, vocab %d, model L%d H%d C%d, B%d T%d, %d steps\n",
                    corpus.size(), cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_embd, B, T, steps);
    }

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

    // Cosine LR with linear warmup + global grad-norm clipping (canonical betas
    // 0.9/0.95, eps 1e-8). opt.lr is overwritten each step by the schedule.
    AdamW opt{};
    const float max_lr = 1e-3f, min_lr = 1e-4f, grad_clip = 1.0f;
    const int warmup = std::max(0, std::min(steps / 10, steps - 1));

    for (int step = 1; step <= steps; ++step) {
        // Next batch from whichever source is active.
        const int* in;
        const int* tgt;
        if (from_bin) {
            loader->next_batch();
            in = loader->inputs();
            tgt = loader->targets();
        } else {
            sample_batch(data, B, T, gen, inputs, targets);
            in = inputs.data();
            tgt = targets.data();
        }
        model.forward(in, tgt);
        const float loss = model.mean_loss();
        model.zero_grads();
        model.backward(in, tgt);
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
