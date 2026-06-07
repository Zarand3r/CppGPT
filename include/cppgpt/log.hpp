// cppgpt logging: leveled, std-only, std::format-based.
//
//   LOG_INFO("loss {} at step {}", loss, step);
//   LOG_WARNING("retrying: {}", reason);
//   LOG_ERROR("load failed: {}", path);
//   LOG_FATAL("unrecoverable: {}", what);          // logs, then abort()
//   LOG_EVERY_N(Info, 100, "step {}", step);       // 1st call + every 100th
//
// Severity tokens (Trace/Debug/Info/Warning/Error/Fatal) are bare identifiers
// pasted onto cppgpt::LogLevel. The disabled path is a single relaxed atomic
// load plus a branch: the format string and its arguments are not evaluated
// below the active level. Logging is process-global by design (one level, one
// sink); tests swap the sink to capture output.
#pragma once

#include <atomic>
#include <format>
#include <source_location>
#include <string_view>
#include <utility>

namespace cppgpt {

enum class LogLevel : int { Trace = 0, Debug, Info, Warning, Error, Fatal };

namespace log {

// A sink receives a decided record. Plain function pointer: no allocation, no
// captured state. The default writes to stderr; set_sink swaps it (e.g. tests).
using Sink = void (*)(LogLevel level, const std::source_location& loc,
                      std::string_view message) noexcept;

namespace detail {
// Defined in log.cpp. Declared here so the level check inlines at the call site.
extern std::atomic<LogLevel> g_min_level;
}  // namespace detail

[[nodiscard]] inline bool enabled(LogLevel level) noexcept {
    return level >= detail::g_min_level.load(std::memory_order_relaxed);
}

// Configuration. Each returns the previous value so callers can save/restore.
LogLevel set_min_level(LogLevel level) noexcept;
Sink set_sink(Sink sink) noexcept;

namespace detail {

// Mutex-serialized emit through the active sink; defined in log.cpp.
void emit(LogLevel level, const std::source_location& loc, std::string_view message) noexcept;

// Format then emit. Called only once the level is known enabled. noexcept: a
// formatting/allocation failure degrades to a marker rather than throwing into
// a (typically noexcept) caller.
template <class... Args>
void dispatch(LogLevel level, const std::source_location& loc, std::format_string<Args...> fmt,
              Args&&... args) noexcept {
    try {
        emit(level, loc, std::format(fmt, std::forward<Args>(args)...));
    } catch (...) {
        emit(level, loc, "<log format failed>");
    }
}

}  // namespace detail
}  // namespace log
}  // namespace cppgpt

#define CPPGPT_LOG_SEVERITY(sev) ::cppgpt::LogLevel::sev

#define CPPGPT_LOG_AT(sev, ...)                                                        \
    do {                                                                              \
        if (::cppgpt::log::enabled(CPPGPT_LOG_SEVERITY(sev)))                         \
            ::cppgpt::log::detail::dispatch(CPPGPT_LOG_SEVERITY(sev),                 \
                                            ::std::source_location::current(),        \
                                            __VA_ARGS__);                             \
    } while (0)

#define LOG_TRACE(...) CPPGPT_LOG_AT(Trace, __VA_ARGS__)
#define LOG_DEBUG(...) CPPGPT_LOG_AT(Debug, __VA_ARGS__)
#define LOG_INFO(...) CPPGPT_LOG_AT(Info, __VA_ARGS__)
#define LOG_WARNING(...) CPPGPT_LOG_AT(Warning, __VA_ARGS__)
#define LOG_ERROR(...) CPPGPT_LOG_AT(Error, __VA_ARGS__)

// Logs at Fatal then aborts (unconditional — a fatal condition is not gated by
// the level threshold).
#define LOG_FATAL(...)                                                                \
    do {                                                                             \
        ::cppgpt::log::detail::dispatch(::cppgpt::LogLevel::Fatal,                    \
                                        ::std::source_location::current(),            \
                                        __VA_ARGS__);                                 \
        ::std::abort();                                                              \
    } while (0)

// Throttled: log on the 1st call and every Nth after, per call site. The counter
// is a per-site, block-scoped static atomic.
#define LOG_EVERY_N(sev, n, ...)                                                      \
    do {                                                                             \
        static ::std::atomic<::std::uint64_t> cppgpt_log_counter{0};                 \
        if (::cppgpt::log::enabled(CPPGPT_LOG_SEVERITY(sev))) {                       \
            const ::std::uint64_t cppgpt_log_i =                                      \
                cppgpt_log_counter.fetch_add(1, ::std::memory_order_relaxed);         \
            if (cppgpt_log_i % static_cast<::std::uint64_t>(n) == 0)                  \
                ::cppgpt::log::detail::dispatch(CPPGPT_LOG_SEVERITY(sev),             \
                                                ::std::source_location::current(),    \
                                                __VA_ARGS__);                         \
        }                                                                            \
    } while (0)
