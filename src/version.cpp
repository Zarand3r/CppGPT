#include "cppgpt/version.hpp"

namespace cppgpt {

Version version() noexcept { return Version{0, 0, 0}; }

const char* version_string() noexcept { return "cppgpt 0.0.0"; }

}  // namespace cppgpt
