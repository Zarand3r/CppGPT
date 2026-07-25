// Minimal std-only test harness. No third-party test framework: a Bazel cc_test
// passes when its binary exits 0, so a test is just `main` that runs CHECKs and
// returns cppgpt::test::summary(). Enough for fixture asserts and gradchecks;
// grows only if a real need appears.
#pragma once

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <utility>

namespace cppgpt::test {

inline int g_checks = 0;
inline int g_failures = 0;

// Run `fn` in a forked child and report whether it failed fast (aborted, or was
// killed by a signal, or exited non-zero). The codebase asserts aggressively on
// invariant violations, so "this input must abort rather than return a wrong
// answer" is a real behavior worth testing. The child's stderr is silenced —
// the abort message is expected output, not a failure.
template <class F>
[[nodiscard]] inline bool dies(F&& fn) {
    std::fflush(nullptr);
    const pid_t pid = ::fork();
    if (pid == 0) {
        (void)std::freopen("/dev/null", "w", stderr);
        fn();
        ::_exit(0);  // returned normally => did NOT fail fast
    }
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

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

// Assert that `stmt` fails fast (aborts) instead of returning a wrong answer.
#define CHECK_DIES(stmt) CHECK(::cppgpt::test::dies([&] { stmt; }))
