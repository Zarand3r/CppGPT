// Public library version. Exists so the skeleton library exports a real symbol
// (not header-only) and tools/tests have something concrete to link against.
#pragma once

namespace cppgpt {

struct Version {
    int major;
    int minor;
    int patch;
};

// Compiled in src/version.cpp — forces a real link step through the toolchain.
[[nodiscard]] Version version() noexcept;
[[nodiscard]] const char* version_string() noexcept;

}  // namespace cppgpt
