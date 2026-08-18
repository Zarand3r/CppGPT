// The activation arena must be guarded against INT_MAX like the parameter arena.
//
// Every op narrows its element count to int (`static_cast<int>(BTC)`), so an
// arena past INT_MAX wraps silently and the forward pass computes garbage over a
// negative length. The parameter arena was guarded from the start; the
// activation arena — which is the LARGER of the two at any realistic B and T —
// was not.
//
// Both directions are checked. A guard that rejected everything would satisfy
// the death test alone, so the legal config is the one that proves the bound is
// placed correctly rather than merely present.
#include "cppgpt/model.hpp"

#include "tests/check.hpp"

using namespace cppgpt;

int main() {
    Config big{};
    big.max_seq_len = 4096;
    big.vocab_size = 50257;
    big.n_layer = 12;
    big.n_head = 12;
    big.n_embd = 768;
    // B=64 at T=4096 puts the activation arena far past INT_MAX floats.
    // Matched on the MESSAGE, not merely on death: without the guard this config
    // dies in the allocator instead, and a bare CHECK_DIES passes either way.
    // Mutation testing caught exactly that.
    CHECK_DIES_WITH(GPT2 m(big, 64, 4096), "activation count exceeds INT_MAX");

    // ...and a config that fits must still construct, or the guard is simply
    // rejecting everything.
    Config ok{};
    ok.max_seq_len = 64;
    ok.vocab_size = 65;
    ok.n_layer = 4;
    ok.n_head = 4;
    ok.n_embd = 128;
    GPT2 m(ok, 1, 64);
    CHECK(m.param_count() > 0);
    CHECK(m.batch() == 1 && m.seq_len() == 64);
    return cppgpt::test::summary();
}
