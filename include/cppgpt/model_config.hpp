// cppgpt model geometry. Split out of model.hpp so the tensor tables
// (tensors.hpp) can describe the layout without depending on the model class.
#pragma once

namespace cppgpt {

struct Config {
    int max_seq_len;  // maximum context length (rows of wpe)
    int vocab_size;   // V
    int n_layer;      // L
    int n_head;       // NH (must divide n_embd)
    int n_embd;       // C
};

}  // namespace cppgpt
