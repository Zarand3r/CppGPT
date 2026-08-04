// train — train a small canonical GPT-2 on your own text.
//
// forward → backward → clip → AdamW on a cosine LR schedule with linear warmup,
// reading mmap'd uint16 tokens from tools/prepare. Periodically evaluates a
// held-out split: without a val number you cannot tell learning from memorizing,
// which is the whole reason --val exists.
//
// Usage:
//   train --data <tokens.bin|corpus.txt> [--val <tokens.bin>] [--vocab <file>]
//         [--layers N --heads N --embd N --ctx N --batch N]
//         [--steps N --lr F --min-lr F --warmup N --clip F --seed N]
//         [--eval-interval N --eval-batches N] [--ckpt <file>]
//
// --data ending in .bin is streamed via the mmap DataLoader (needs its .vocab
// sidecar); anything else is treated as a text corpus and tokenized in memory.
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
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
#include "tools/cli.hpp"

namespace {
using namespace cppgpt;
using Clock = std::chrono::steady_clock;

std::string read_file(const std::string& path, const char* what) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "train: cannot open %s '%s'\n", what, path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// tools/prepare emits <stem>.train.bin / <stem>.val.bin / <stem>.vocab, so the
// sidecar is found by stripping the data suffix, never by appending to it.
std::string vocab_path_for(const std::string& data_path) {
    std::string stem = data_path;
    for (const char* suffix : {".train.bin", ".val.bin", ".bin"}) {
        const std::size_t n = std::string(suffix).size();
        if (stem.size() > n && stem.compare(stem.size() - n, n, suffix) == 0) {
            stem.resize(stem.size() - n);
            break;
        }
    }
    return stem + ".vocab";
}

double peak_rss_mb() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_maxrss) / 1024.0;  // ru_maxrss is KiB on Linux
}

// Mean loss over `batches` validation batches. No backward, so the gradient
// arenas are untouched; it does overwrite the activation arena, which is fine
// because the next training step recomputes it.
float evaluate(GPT2& model, DataLoader& val, int batches) {
    double sum = 0.0;
    for (int i = 0; i < batches; ++i) {
        val.next_batch();
        model.forward(val.inputs(), val.targets());
        sum += static_cast<double>(model.mean_loss());
    }
    return static_cast<float>(sum / std::max(1, batches));
}

}  // namespace

int main(int argc, char** argv) {
    const cli::Args args(argc, argv,
                         {"data", "val", "vocab", "ckpt", "layers", "heads", "embd", "ctx", "batch",
                          "steps", "lr", "min-lr", "warmup", "clip", "seed", "eval-interval",
                          "eval-batches", "init-from"});

    const std::string data_path(args.str("data", ""));
    if (data_path.empty()) {
        std::fprintf(
            stderr,
            "usage: train --data <tokens.bin|corpus.txt> [--val <tokens.bin>] [--vocab <file>]\n"
            "             [--layers N --heads N --embd N --ctx N --batch N]\n"
            "             [--steps N --lr F --seed N] [--eval-interval N] [--ckpt <file>]\n");
        return 2;
    }
    const std::string val_path(args.str("val", ""));
    const std::string ckpt(args.str("ckpt", ""));
    const int steps = args.integer("steps", 100);
    const int B = args.integer("batch", 4);
    const int T = args.integer("ctx", 32);
    const float max_lr = args.real("lr", 1e-3f);
    const float min_lr = args.real("min-lr", max_lr * 0.1f);
    const float grad_clip = args.real("clip", 1.0f);
    const auto seed = static_cast<std::uint64_t>(args.integer("seed", 1337));
    const int eval_interval = args.integer("eval-interval", 0);
    const int eval_batches = args.integer("eval-batches", 20);
    const bool from_bin = std::string_view(data_path).ends_with(".bin");

    Config cfg{};
    cfg.n_layer = args.integer("layers", 3);
    cfg.n_head = args.integer("heads", 4);
    cfg.n_embd = args.integer("embd", 64);
    cfg.max_seq_len = T;

    Generator gen(seed);
    std::optional<DataLoader> loader, val_loader;
    std::vector<int> data, inputs, targets;

    if (from_bin) {
        const std::string vpath =
            args.has("vocab") ? std::string(args.str("vocab", "")) : vocab_path_for(data_path);
        const std::string vocab = read_file(vpath, "vocab");
        if (vocab.empty()) {
            std::fprintf(stderr, "train: vocab '%s' is empty\n", vpath.c_str());
            return 1;
        }
        cfg.vocab_size = CharTokenizer(vocab).vocab_size();
        auto r = DataLoader::open(data_path.c_str(), B, T, seed);
        if (!r) {
            std::fprintf(stderr, "train: cannot open '%s': %s\n", data_path.c_str(),
                         describe(r.error()));
            return 1;
        }
        loader.emplace(std::move(*r));
        if (!val_path.empty()) {
            auto v = DataLoader::open(val_path.c_str(), B, T, seed ^ 0x5DEECE66DULL);
            if (!v) {
                std::fprintf(stderr, "train: cannot open val '%s': %s\n", val_path.c_str(),
                             describe(v.error()));
                return 1;
            }
            val_loader.emplace(std::move(*v));
        }
    } else {
        const std::string corpus = read_file(data_path, "corpus");
        const CharTokenizer tok(corpus);
        data = tok.encode(corpus);
        cfg.vocab_size = tok.vocab_size();
        if (data.size() <= static_cast<std::size_t>(T) + 1) {
            std::fprintf(stderr, "train: corpus too short (%zu tokens) for --ctx %d\n", data.size(),
                         T);
            return 1;
        }
        inputs.assign(static_cast<std::size_t>(B) * T, 0);
        targets.assign(static_cast<std::size_t>(B) * T, 0);
    }

    GPT2 model(cfg, B, T);
    model.init_weights(gen);

    // --init-from starts a NEW run from a base model's weights: moments and step
    // reset, schedule restarts. Distinct from --ckpt resume below, which continues
    // an interrupted run and deliberately restores the optimizer state.
    const std::string init_from(args.str("init-from", ""));
    if (!init_from.empty()) {
        if (const auto r =
                model.load_checkpoint(init_from.c_str(), GPT2::LoadMode::WeightsOnly);
            !r) {
            std::fprintf(stderr, "train: cannot init from '%s': %s\n", init_from.c_str(),
                         describe(r.error()));
            return 1;
        }
        std::printf("train: initialized from %s (weights only; fresh optimizer)\n",
                    init_from.c_str());
    }

    // Resume if the checkpoint already exists. A corrupt or mismatched file is
    // fatal: silently starting fresh over a real but unusable checkpoint would
    // discard a run the user believes is continuing.
    if (!ckpt.empty() && init_from.empty()) {
        std::ifstream probe(ckpt, std::ios::binary);
        if (probe.good()) {
            probe.close();
            if (const auto r = model.load_checkpoint(ckpt.c_str()); !r) {
                std::fprintf(stderr, "train: cannot resume from '%s': %s\n", ckpt.c_str(),
                             describe(r.error()));
                return 1;
            }
            std::printf("train: resumed from %s\n", ckpt.c_str());
        }
    }

    std::printf("train: %zu tokens, vocab %d | L%d H%d C%d | B%d T%d | %d steps, lr %.2e\n",
                from_bin ? loader->num_tokens() : data.size(), cfg.vocab_size, cfg.n_layer,
                cfg.n_head, cfg.n_embd, B, T, steps, static_cast<double>(max_lr));
    if (val_loader) std::printf("train: val split %zu tokens\n", val_loader->num_tokens());

    AdamW opt{};
    const int warmup = args.integer("warmup", std::max(0, std::min(steps / 10, steps - 1)));
    const auto t_start = Clock::now();
    std::int64_t tokens_seen = 0;

    for (int step = 1; step <= steps; ++step) {
        const int* in;
        const int* tgt;
        if (from_bin) {
            loader->next_batch();
            in = loader->inputs();
            tgt = loader->targets();
        } else {
            const auto max_start = static_cast<std::int64_t>(data.size()) - T - 1;
            for (int b = 0; b < B; ++b) {
                const std::int64_t s = gen.uniform_int(0, max_start);
                for (int t = 0; t < T; ++t) {
                    const std::size_t o = static_cast<std::size_t>(b) * T + t;
                    inputs[o] = data[static_cast<std::size_t>(s + t)];
                    targets[o] = data[static_cast<std::size_t>(s + t + 1)];
                }
            }
            in = inputs.data();
            tgt = targets.data();
        }

        model.forward(in, tgt);
        const float loss = model.mean_loss();
        model.zero_grads();
        model.backward(in, tgt);
        const float gnorm = model.clip_grad_norm(grad_clip);
        opt.lr = cosine_lr(step - 1, max_lr, min_lr, warmup, steps);
        model.update(opt);
        tokens_seen += static_cast<std::int64_t>(B) * T;

        const double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if (step % 10 == 0 || step == 1 || step == steps)
            std::printf("step %5d/%d  loss %.4f  lr %.2e  |g| %.3f  %.0f tok/s  %.1fs\n", step,
                        steps, static_cast<double>(loss), static_cast<double>(opt.lr),
                        static_cast<double>(gnorm),
                        static_cast<double>(tokens_seen) / std::max(1e-9, elapsed), elapsed);

        if (val_loader && eval_interval > 0 && (step % eval_interval == 0 || step == steps)) {
            const float vl = evaluate(model, *val_loader, eval_batches);
            std::printf("  [eval] step %d  val loss %.4f  peak RSS %.0f MB\n", step,
                        static_cast<double>(vl), peak_rss_mb());
        }

        if (!ckpt.empty() && (step % 500 == 0 || step == steps)) {
            if (const auto r = model.save_checkpoint(ckpt.c_str()); !r)
                std::fprintf(stderr, "train: checkpoint save failed: %s\n", describe(r.error()));
            else
                std::printf("  [checkpoint] %s\n", ckpt.c_str());
        }
    }

    const double total = std::chrono::duration<double>(Clock::now() - t_start).count();
    std::printf("train: done — %lld tokens in %.1fs (%.0f tok/s), peak RSS %.0f MB\n",
                static_cast<long long>(tokens_seen), total,
                static_cast<double>(tokens_seen) / std::max(1e-9, total), peak_rss_mb());
    return 0;
}
