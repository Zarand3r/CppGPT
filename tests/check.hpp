// Minimal std-only test harness. No third-party test framework: a Bazel cc_test
// passes when its binary exits 0, so a test is just `main` that runs CHECKs and
// returns cppgpt::test::summary(). Enough for fixture asserts and gradchecks;
// grows only if a real need appears.
#pragma once

#include <cstdio>

namespace cppgpt::test {

inline int g_checks = 0;
inline int g_failures = 0;

// Returns a process exit code: 0 = all passed, 1 = at least one failure.
[[nodiscard]] inline int summary() {
    std::printf("[ %s ] %d checks, %d failure%s\n", g_failures == 0 ? "PASS" : "FAIL", g_checks,
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}

}  // namespace cppgpt::test

#define CHECK(cond)                                                                  \
    do {                                                                             \
        ++::cppgpt::test::g_checks;                                                  \
        if (!(cond)) {                                                               \
            ++::cppgpt::test::g_failures;                                            \
            std::fprintf(stderr, "  CHECK failed: %s  (%s:%d)\n", #cond, __FILE__,   \
                         __LINE__);                                                  \
        }                                                                            \
    } while (0)
