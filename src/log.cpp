#include "cppgpt/log.hpp"

#include <cstdio>
#include <mutex>

namespace {

// constinit: constant-initialized, so logging is safe during other static init
// (no static-initialization-order fiasco). The mutex guards only the sink call;
// the level lives in an atomic and is read lock-free.
constinit std::mutex g_mu;

char tag(cppgpt::LogLevel level) noexcept {
    switch (level) {
        case cppgpt::LogLevel::Trace:   return 'T';
        case cppgpt::LogLevel::Debug:   return 'D';
        case cppgpt::LogLevel::Info:    return 'I';
        case cppgpt::LogLevel::Warning: return 'W';
        case cppgpt::LogLevel::Error:   return 'E';
        case cppgpt::LogLevel::Fatal:   return 'F';
    }
    return '?';
}

const char* base_name(const char* path) noexcept {
    const char* base = path;
    for (const char* p = path; *p != '\0'; ++p) {
        if (*p == '/') base = p + 1;
    }
    return base;
}

void default_sink(cppgpt::LogLevel level, const std::source_location& loc,
                  std::string_view message) noexcept {
    std::fprintf(stderr, "%c %s:%u] %.*s\n", tag(level), base_name(loc.file_name()),
                 static_cast<unsigned>(loc.line()), static_cast<int>(message.size()),
                 message.data());
}

constinit cppgpt::log::Sink g_sink = &default_sink;

}  // namespace

namespace cppgpt::log {

namespace detail {

constinit std::atomic<LogLevel> g_min_level{LogLevel::Info};

void emit(LogLevel level, const std::source_location& loc, std::string_view message) noexcept {
    const std::scoped_lock lock(g_mu);
    g_sink(level, loc, message);
}

}  // namespace detail

LogLevel set_min_level(LogLevel level) noexcept {
    return detail::g_min_level.exchange(level, std::memory_order_relaxed);
}

Sink set_sink(Sink sink) noexcept {
    const std::scoped_lock lock(g_mu);
    Sink previous = g_sink;
    g_sink = sink;
    return previous;
}

}  // namespace cppgpt::log
