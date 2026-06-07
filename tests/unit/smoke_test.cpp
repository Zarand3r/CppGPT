// Skeleton smoke test: proves the cppgpt library links and the std-only test
// harness reports pass/fail correctly. Replaced by real op tests at M0/M1.
#include "cppgpt/version.hpp"
#include "tests/check.hpp"

int main() {
    const cppgpt::Version v = cppgpt::version();
    CHECK(v.major == 0);
    CHECK(v.minor == 0);
    CHECK(v.patch == 0);
    CHECK(cppgpt::version_string() != nullptr);
    return cppgpt::test::summary();
}
