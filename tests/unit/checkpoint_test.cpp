// Checkpoint save/load: round-trip weight fidelity, trajectory-exact resume
// (moments + step restored), atomic write (no temp left, exact size), and the
// validation error paths (magic/version/shape/checksum/truncation/missing).
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "cppgpt/checkpoint.hpp"
#include "cppgpt/core.hpp"
#include "cppgpt/model.hpp"
#include "cppgpt/random.hpp"
#include "tests/check.hpp"

using namespace cppgpt;

namespace {

std::string tmp_path(const char* name) {
    const char* dir = std::getenv("TEST_TMPDIR");
    return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

Config baby() {
    Config c{};
    c.max_seq_len = 8;
    c.vocab_size = 5;
    c.n_layer = 2;
    c.n_head = 2;
    c.n_embd = 8;
    return c;
}

// One training step on a fixed batch (builds up optimizer moments deterministically).
void step(GPT2& m, const std::vector<int>& in, const std::vector<int>& tg) {
    m.forward(in.data(), tg.data());
    m.zero_grads();
    m.backward(in.data(), tg.data());
    IGNORE(m.clip_grad_norm(1.0f));
    m.update(AdamW{.lr = 1e-2f});
}

bool params_equal(const GPT2& a, const GPT2& b) {
    if (a.param_count() != b.param_count()) return false;
    const float* pa = a.params().wte;
    const float* pb = b.params().wte;
    for (std::size_t i = 0; i < a.param_count(); ++i)
        if (pa[i] != pb[i]) return false;  // bitwise: save/load must be lossless
    return true;
}

std::vector<char> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void dump(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

int main() {
    const Config cfg = baby();
    const int B = 1, T = 4, V = cfg.vocab_size;
    const std::vector<int> in{1, 2, 3, 4}, tg{2, 3, 4, 0};
    const std::string path = tmp_path("baby.ckpt");

    // ---- train A, save, and snapshot its weights at the save point ----
    Generator ga(1234ULL);
    GPT2 A(cfg, B, T);
    A.init_weights(ga);
    for (int i = 0; i < 5; ++i) step(A, in, tg);
    {
        auto r = A.save_checkpoint(path.c_str());
        CHECK(r.has_value());
    }
    const std::vector<float> saved(A.params().wte, A.params().wte + A.param_count());

    // ---- atomic write: temp gone, file exactly header + (params + m + v) ----
    {
        std::ifstream tmp(path + ".tmp", std::ios::binary);
        CHECK(!tmp.good());  // no leftover temp file
        const std::vector<char> bytes = slurp(path);
        const std::size_t expected =
            sizeof(CheckpointHeader) + 3 * A.param_count() * sizeof(float);  // moments present
        CHECK(bytes.size() == expected);
    }

    // ---- load into a fresh, differently-initialized model: weights restored ----
    Generator gb(9999ULL);
    GPT2 Bm(cfg, B, T);
    Bm.init_weights(gb);
    {
        auto r = Bm.load_checkpoint(path.c_str());
        CHECK(r.has_value());
    }
    bool restored = true;
    for (std::size_t i = 0; i < Bm.param_count(); ++i) restored = restored && (Bm.params().wte[i] == saved[i]);
    CHECK(restored);

    // ---- trajectory-exact resume: one more identical step matches on both ----
    step(A, in, tg);   // A continues from the save point
    step(Bm, in, tg);  // Bm resumes from the checkpoint
    CHECK(params_equal(A, Bm));  // identical => moments AND adam_step were restored

    // ---- ShapeMismatch: load into a model with a different architecture ----
    {
        Config other = cfg;
        other.n_layer = 3;  // different param_count
        Generator gc(7ULL);
        GPT2 C(other, B, T);
        C.init_weights(gc);
        auto r = C.load_checkpoint(path.c_str());
        CHECK(!r.has_value() && r.error() == ErrorCode::ShapeMismatch);
    }

    // ---- VersionMismatch: bump the header version byte ----
    {
        std::vector<char> bytes = slurp(path);
        bytes[4] = 2;  // version is the second u32 (offset 4); 1 -> 2
        const std::string p = tmp_path("badver.ckpt");
        dump(p, bytes);
        GPT2 M(cfg, B, T);
        Generator g(1ULL);
        M.init_weights(g);
        auto r = M.load_checkpoint(p.c_str());
        CHECK(!r.has_value() && r.error() == ErrorCode::VersionMismatch);
    }

    // ---- CorruptCheckpoint: clobber the magic ----
    {
        std::vector<char> bytes = slurp(path);
        bytes[0] ^= 0xFF;  // magic is the first u32
        const std::string p = tmp_path("badmagic.ckpt");
        dump(p, bytes);
        GPT2 M(cfg, B, T);
        Generator g(1ULL);
        M.init_weights(g);
        auto r = M.load_checkpoint(p.c_str());
        CHECK(!r.has_value() && r.error() == ErrorCode::CorruptCheckpoint);
    }

    // ---- ChecksumMismatch: flip a payload byte (valid header, bad data) ----
    {
        std::vector<char> bytes = slurp(path);
        bytes[sizeof(CheckpointHeader) + 3] ^= 0x01;  // first param bytes
        const std::string p = tmp_path("badsum.ckpt");
        dump(p, bytes);
        GPT2 M(cfg, B, T);
        Generator g(1ULL);
        M.init_weights(g);
        auto r = M.load_checkpoint(p.c_str());
        CHECK(!r.has_value() && r.error() == ErrorCode::ChecksumMismatch);
    }

    // ---- CorruptCheckpoint: truncated payload ----
    {
        std::vector<char> bytes = slurp(path);
        bytes.resize(sizeof(CheckpointHeader) + 8);  // header + a sliver of payload
        const std::string p = tmp_path("trunc.ckpt");
        dump(p, bytes);
        GPT2 M(cfg, B, T);
        Generator g(1ULL);
        M.init_weights(g);
        auto r = M.load_checkpoint(p.c_str());
        CHECK(!r.has_value() && r.error() == ErrorCode::CorruptCheckpoint);
    }

    // ---- IoError: missing file ----
    {
        GPT2 M(cfg, B, T);
        Generator g(1ULL);
        M.init_weights(g);
        auto r = M.load_checkpoint(tmp_path("nope.ckpt").c_str());
        CHECK(!r.has_value() && r.error() == ErrorCode::IoError);
    }

    // ---- params-only checkpoint (no optimizer run) loads with a warning ----
    {
        Generator gf(55ULL);
        GPT2 F(cfg, B, T);
        F.init_weights(gf);  // never call update() => no moments
        const std::string p = tmp_path("weights_only.ckpt");
        CHECK(F.save_checkpoint(p.c_str()).has_value());
        const std::vector<char> bytes = slurp(p);
        CHECK(bytes.size() == sizeof(CheckpointHeader) + F.param_count() * sizeof(float));  // no m/v
        GPT2 G(cfg, B, T);
        Generator gg(66ULL);
        G.init_weights(gg);
        CHECK(G.load_checkpoint(p.c_str()).has_value());
    }

    (void)V;
    return cppgpt::test::summary();
}
