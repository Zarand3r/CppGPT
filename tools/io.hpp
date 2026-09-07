// Corpus reading shared by the tools that sweep one.
//
// `read_tokens` existed as a private copy in tools/ablation_stats.cpp, and
// tools/neuron_stats.cpp would have been a third implementation of the same
// twenty lines. ROADMAP.md logs ~92 lines of tool duplication of which one copy
// had ALREADY diverged, so a third copy is the pattern rather than an instance.
//
// Returns a Result rather than calling exit(): a library-shaped helper that
// terminates the process cannot be used by a test, and this one is.
#pragma once

#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "cppgpt/core.hpp"

namespace cppgpt::toolio {

// A .bin corpus: a whole number of little-endian uint16 token ids.
//   IoError    — cannot open
//   ParseError — the size is not a whole number of tokens, or the file is empty
[[nodiscard]] inline Result<std::vector<std::uint16_t>> read_tokens(const char* path) noexcept {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return err(ErrorCode::IoError);
    const std::streamoff bytes = f.tellg();
    if (bytes <= 0 || bytes % 2 != 0) return err(ErrorCode::ParseError);

    std::vector<std::uint16_t> t(static_cast<std::size_t>(bytes) / 2);
    f.seekg(0, std::ios::beg);
    if (!f.read(reinterpret_cast<char*>(t.data()), bytes)) return err(ErrorCode::IoError);
    return t;
}

// A whole file as bytes.
//   IoError — cannot open or read
//
// ROADMAP logs four private copies of this across tools/. This is the shared
// one; the copies are migrated as each tool is next touched rather than in one
// sweep, so a behaviour change lands with a reviewer who is already reading that
// file.
[[nodiscard]] inline Result<std::string> read_file(const char* path) noexcept {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return err(ErrorCode::IoError);
    const std::streamoff bytes = f.tellg();
    if (bytes < 0) return err(ErrorCode::IoError);
    std::string s(static_cast<std::size_t>(bytes), '\0');
    f.seekg(0, std::ios::beg);
    if (bytes > 0 && !f.read(s.data(), bytes)) return err(ErrorCode::IoError);
    return s;
}

}  // namespace cppgpt::toolio
