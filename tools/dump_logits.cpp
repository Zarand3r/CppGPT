// dump_logits — run one forward at BPE scale and write [T, V] logits.
//
// Exists for the GPT-2 parity gate (scripts/check_gpt2_parity.py), which needs
// our raw logits to compare against a fp64 reference. Kept as a tool rather than
// folded into a test because the weights it needs are 475 MB and cannot be
// committed.
#include <cstdio>
#include <fstream>
#include <vector>
#include "cppgpt/bpe.hpp"
#include "cppgpt/checkpoint.hpp"
#include "cppgpt/model.hpp"
using namespace cppgpt;
int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: <ckpt> <vocab> <merges> <prompt> <out>\n"); return 2; }
    const char* ck = argv[1]; const char* vj = argv[2]; const char* mt = argv[3];
    const char* prompt = argv[4]; const char* out = argv[5];
    auto peek = CheckpointFile::open(ck); if (!peek) { std::fprintf(stderr,"open fail\n"); return 1; }
    const CheckpointHeader& h = peek->header();
    std::fprintf(stderr, "  ckpt v%u tokenizer_kind=%u fingerprint=%016llx\n",
                 h.version, h.tokenizer_kind, (unsigned long long)h.vocab_fingerprint);
    Config cfg{h.max_seq_len, h.vocab_size, h.n_layer, h.n_head, h.n_embd};
    auto t = BpeTokenizer::open(vj, mt); if (!t) { std::fprintf(stderr,"tok fail\n"); return 1; }
    if (t->fingerprint() != h.vocab_fingerprint) { std::fprintf(stderr,"  FINGERPRINT MISMATCH\n"); return 1; }
    bool ok=false; std::vector<int> ids = t->encode(prompt, &ok);
    if (!ok || ids.empty()) { std::fprintf(stderr,"encode fail\n"); return 1; }
    const int T = (int)ids.size();
    GPT2 m(cfg, 1, T);
    if (const auto r = m.load_checkpoint(ck, GPT2::LoadMode::WeightsOnly); !r) { std::fprintf(stderr,"load fail\n"); return 1; }
    m.forward(ids.data(), nullptr);
    std::ofstream o(out, std::ios::binary);
    const int nid = T; o.write((const char*)&nid, 4); o.write((const char*)ids.data(), 4L*T);
    const int V = cfg.vocab_size; o.write((const char*)&V, 4);
    o.write((const char*)m.acts().logits, (std::streamsize)sizeof(float)*T*V);
    std::fprintf(stderr, "  %d tokens, wrote [%d,%d] logits\n", T, T, V);
    return 0;
}
