// Minimal std-only test harness. No third-party test framework: a Bazel cc_test
// passes when its binary exits 0, so a test is just `main` that runs CHECKs and
// returns cppgpt::test::summary(). Enough for fixture asserts and gradchecks;
// grows only if a real need appears.
#pragma once

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <string>
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
    if (pid < 0) return false;  // never fall through to waitpid(-1, ...) and reap a stranger
    if (pid == 0) {
        (void)std::freopen("/dev/null", "w", stderr);
        fn();
        ::_exit(0);  // returned normally => did NOT fail fast
    }
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

// Like dies(), but also requires the failure message to contain `needle`.
//
// A bare death test passes for ANY failure, including one that happens before the
// code under test is even reached. That is not hypothetical: a test for the
// activation-arena INT_MAX guard passed with the guard deleted, because the
// oversized config then died in the allocator instead — the assertion under test
// never ran, and mutation testing was what exposed it. Matching the message is
// what makes a death test a test of a SPECIFIC invariant.
template <class F>
[[nodiscard]] inline bool dies_with(F&& fn, const char* needle) {
    std::fflush(nullptr);
    int fds[2];
    if (::pipe(fds) != 0) return false;
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(fds[0]);
        ::close(fds[1]);
        return false;  // never fall through to waitpid(-1, ...) and reap a stranger
    }
    if (pid == 0) {
        ::close(fds[0]);
        (void)::dup2(fds[1], 2);  // stderr -> pipe, so the abort message is capturable
        ::close(fds[1]);
        fn();
        ::_exit(0);  // returned normally => did NOT fail fast
    }
    ::close(fds[1]);
    std::string out;
    char buf[512];
    ssize_t n;
    while ((n = ::read(fds[0], buf, sizeof(buf))) > 0)
        out.append(buf, static_cast<std::size_t>(n));
    ::close(fds[0]);
    int status = 0;
    (void)::waitpid(pid, &status, 0);
    const bool died = !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    return died && out.find(needle) != std::string::npos;
}

// Returns a process exit code: 0 = all passed, 1 = at least one failure.
[[nodiscard]] inline int summary() {
    // A file that ran zero checks is a FAILURE, not a pass. Deleting every CHECK
    // from a test used to print "[ PASS ] 0 checks" and exit 0 — the cheapest
    // possible way for a refactor to silently drop coverage.
    const bool ok = g_failures == 0 && g_checks > 0;
    if (g_checks == 0) std::fprintf(stderr, "  no checks ran — a test must assert something\n");
    std::printf("[ %s ] %d checks, %d failure%s\n", ok ? "PASS" : "FAIL", g_checks, g_failures,
                g_failures == 1 ? "" : "s");
    return ok ? 0 : 1;
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

// Assert that `stmt` fails fast AND says why — see dies_with on why the message
// matters. Prefer this over CHECK_DIES whenever another failure could plausibly
// occur first.
#define CHECK_DIES_WITH(stmt, needle) CHECK(::cppgpt::test::dies_with([&] { stmt; }, needle))
