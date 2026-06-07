#include "cppgpt/log.hpp"

#include <source_location>
#include <string>
#include <utility>
#include <vector>

#include "tests/check.hpp"

namespace {

std::vector<std::pair<cppgpt::LogLevel, std::string>> g_captured;

void capture_sink(cppgpt::LogLevel level, const std::source_location& /*loc*/,
                  std::string_view message) noexcept {
    g_captured.emplace_back(level, std::string(message));
}

// Side effect to prove arguments are not evaluated on the disabled path.
int g_arg_calls = 0;
int counted() {
    ++g_arg_calls;
    return 1;
}

}  // namespace

int main() {
    using cppgpt::LogLevel;
    const auto prev_sink = cppgpt::log::set_sink(&capture_sink);
    const auto prev_level = cppgpt::log::set_min_level(LogLevel::Trace);

    // levels + formatting
    g_captured.clear();
    LOG_INFO("hello {}", 42);
    LOG_WARNING("warn {}", 7);
    LOG_ERROR("err");
    CHECK(g_captured.size() == 3);
    CHECK(g_captured[0].first == LogLevel::Info);
    CHECK(g_captured[0].second == "hello 42");
    CHECK(g_captured[1].first == LogLevel::Warning);
    CHECK(g_captured[2].second == "err");

    // threshold suppresses below the active level
    g_captured.clear();
    cppgpt::log::set_min_level(LogLevel::Warning);
    LOG_INFO("suppressed");
    LOG_WARNING("kept");
    CHECK(g_captured.size() == 1);
    CHECK(g_captured[0].second == "kept");

    // disabled path does not evaluate arguments
    g_arg_calls = 0;
    cppgpt::log::set_min_level(LogLevel::Error);
    LOG_INFO("x {}", counted());
    CHECK(g_arg_calls == 0);
    cppgpt::log::set_min_level(LogLevel::Info);
    LOG_INFO("x {}", counted());
    CHECK(g_arg_calls == 1);

    // LOG_EVERY_N: 1st call + every 3rd (i = 0, 3, 6 over 7 calls)
    g_captured.clear();
    for (int i = 0; i < 7; ++i) {
        LOG_EVERY_N(Info, 3, "i={}", i);
    }
    CHECK(g_captured.size() == 3);
    CHECK(g_captured[0].second == "i=0");
    CHECK(g_captured[1].second == "i=3");
    CHECK(g_captured[2].second == "i=6");

    cppgpt::log::set_min_level(prev_level);
    cppgpt::log::set_sink(prev_sink);
    return cppgpt::test::summary();
}
