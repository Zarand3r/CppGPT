// cppgpt core: explicit, non-throwing failure handling.
//
// Two orthogonal axes:
//   * Invariant violation (a programmer bug) -> ASSERT / DCHECK / MUST /
//     UNREACHABLE -> fail fast (abort). Always on (except DCHECK, debug-only).
//   * Expected failure where the reason matters -> Result<T> = std::expected<T,
//     ErrorCode>, propagated with TRY / ASSIGN_OR_RETURN / RETURN_IF_ERROR.
//     For simple yes/no, prefer a [[nodiscard]] bool.
//
// Nothing here throws. Result stays small and trivially copyable (so it is
// returned in registers, not via a hidden sret pointer) because the carried
// error is a 16-bit ErrorCode. Rich human-readable context is built at the
// log/abort site from the code + std::source_location, never stored in Result.
//
// The public macros are intentionally unprefixed (ASSERT, TRY, ...). Only the
// two internal token-paste helpers keep a CPPGPT_ guard so bare CAT/UNIQ do not
// leak into the global macro namespace.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <source_location>
#include <utility>

namespace cppgpt {

// Cheap, trivially-copyable failure reason. Kept 16-bit so Result<T, ErrorCode>
// stays enregisterable. Ok is the success sentinel and is never stored inside an
// error alternative.
enum class ErrorCode : std::uint16_t {
    Ok = 0,
    IoError,
    ParseError,
    ShapeMismatch,
    VersionMismatch,
    CorruptCheckpoint,
    ChecksumMismatch,
    TokenizerError,
    OutOfRange,
    NanOrInf,
    OutOfMemory,
    Unimplemented,
};

// Static description. constexpr; no allocation. Exhaustive switch (no default)
// so adding an ErrorCode without a description is a -Werror build break.
[[nodiscard]] constexpr const char* describe(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::Ok:                return "ok";
        case ErrorCode::IoError:           return "I/O error";
        case ErrorCode::ParseError:        return "parse error";
        case ErrorCode::ShapeMismatch:     return "shape mismatch";
        case ErrorCode::VersionMismatch:   return "version mismatch";
        case ErrorCode::CorruptCheckpoint: return "corrupt checkpoint";
        case ErrorCode::ChecksumMismatch:  return "checksum mismatch";
        case ErrorCode::TokenizerError:    return "tokenizer error";
        case ErrorCode::OutOfRange:        return "out of range";
        case ErrorCode::NanOrInf:          return "NaN or Inf";
        case ErrorCode::OutOfMemory:       return "out of memory";
        case ErrorCode::Unimplemented:     return "unimplemented";
    }
    return "unknown error";
}

// Value-or-reason. std::expected is conditionally trivially copyable, so for a
// trivially-copyable T this is trivially copyable and register-returned.
template <class T, class E = ErrorCode>
using Result = std::expected<T, E>;

// Build the error alternative: `return cppgpt::err(ErrorCode::ParseError);`.
[[nodiscard]] inline std::unexpected<ErrorCode> err(ErrorCode code) noexcept {
    return std::unexpected(code);
}

namespace detail {

[[noreturn]] inline void fail(const char* what, const char* detail_str,
                              std::source_location loc = std::source_location::current()) noexcept {
    std::fprintf(stderr, "cppgpt FATAL: %s: %s\n  at %s:%u in %s\n", what, detail_str,
                 loc.file_name(), static_cast<unsigned>(loc.line()), loc.function_name());
    std::fflush(stderr);
    std::abort();
}

inline void warn(const char* what, ErrorCode code,
                 std::source_location loc = std::source_location::current()) noexcept {
    std::fprintf(stderr, "cppgpt WARN: %s: %s\n  at %s:%u\n", what, describe(code), loc.file_name(),
                 static_cast<unsigned>(loc.line()));
    std::fflush(stderr);
}

}  // namespace detail
}  // namespace cppgpt

// --- internal token plumbing: a unique temporary name per macro expansion ---
#define CPPGPT_CAT_(a, b) a##b
#define CPPGPT_CAT(a, b) CPPGPT_CAT_(a, b)
#define CPPGPT_UNIQ(base) CPPGPT_CAT(base, __LINE__)

// --- invariant checks (fail fast; never throw) ---

// Always on, even in release. For invariant violations / programmer bugs.
#define ASSERT(cond)                                                   \
    do {                                                               \
        if (!(cond)) [[unlikely]]                                      \
            ::cppgpt::detail::fail("CHECK failed", #cond);             \
    } while (0)

// Always on; abort with a static message string.
#define ASSERT_MSG(cond, msg)                                          \
    do {                                                               \
        if (!(cond)) [[unlikely]]                                      \
            ::cppgpt::detail::fail((msg), #cond);                      \
    } while (0)

// Debug-only (compiled out when NDEBUG is defined, i.e. --config=release).
// For hot per-element checks; op-ENTRY shape checks should use ASSERT.
#ifdef NDEBUG
#  define DCHECK(cond) ((void)0)
#else
#  define DCHECK(cond) ASSERT(cond)
#endif

#define UNREACHABLE() ::cppgpt::detail::fail("unreachable", "")

// Deliberately discard a [[nodiscard]] result (rare; documents intent).
#define IGNORE(expr) (static_cast<void>(expr))

// --- Result<T, ErrorCode> propagation (statement forms; pure ISO C++) ---

// Propagate the error of a value-less Result (e.g. Result<void>).
#define RETURN_IF_ERROR(expr)                                          \
    do {                                                               \
        auto&& CPPGPT_UNIQ(_r) = (expr);                               \
        if (!CPPGPT_UNIQ(_r)) [[unlikely]]                             \
            return ::std::unexpected(CPPGPT_UNIQ(_r).error());         \
    } while (0)

// Bind `decl` to the value, or return the error. `decl` is declared in the
// enclosing scope, e.g. ASSIGN_OR_RETURN(uint32_t magic, read_u32(f));
#define ASSIGN_OR_RETURN(decl, expr)                                   \
    auto&& CPPGPT_UNIQ(_r) = (expr);                                   \
    if (!CPPGPT_UNIQ(_r)) [[unlikely]]                                 \
        return ::std::unexpected(CPPGPT_UNIQ(_r).error());             \
    decl = ::std::move(*CPPGPT_UNIQ(_r))

// --- Result<T, ErrorCode> propagation (expression forms) ---
// These use a GNU statement-expression so they can be unwrapped inline inside a
// larger expression (Rust's `?`). __extension__ suppresses the -Wpedantic
// warning for the construct, contained to these macros. Not for Result<void>.

#define TRY(expr)                                                      \
    __extension__({                                                    \
        auto&& CPPGPT_UNIQ(_r) = (expr);                               \
        if (!CPPGPT_UNIQ(_r)) [[unlikely]]                             \
            return ::std::unexpected(CPPGPT_UNIQ(_r).error());         \
        ::std::move(*CPPGPT_UNIQ(_r));                                 \
    })

// Unwrap, or evaluate to `fallback` (no return).
#define TRY_OR(expr, fallback)                                         \
    __extension__({                                                    \
        auto&& CPPGPT_UNIQ(_r) = (expr);                               \
        CPPGPT_UNIQ(_r) ? ::std::move(*CPPGPT_UNIQ(_r)) : (fallback);  \
    })

// Unwrap, or abort. The non-throwing analogue of "throw_if_not_ok".
#define MUST(expr)                                                     \
    __extension__({                                                    \
        auto&& CPPGPT_UNIQ(_r) = (expr);                               \
        if (!CPPGPT_UNIQ(_r)) [[unlikely]]                             \
            ::cppgpt::detail::fail("MUST failed",                      \
                                   ::cppgpt::describe(CPPGPT_UNIQ(_r).error())); \
        ::std::move(*CPPGPT_UNIQ(_r));                                 \
    })

// Loop body: bind `decl` to the value, or log and `continue` to the next
// iteration. Observable (logs the reason) — never a silent skip.
#define TRY_OR_CONTINUE(decl, expr)                                    \
    auto&& CPPGPT_UNIQ(_r) = (expr);                                   \
    if (!CPPGPT_UNIQ(_r)) {                                            \
        ::cppgpt::detail::warn("skipping (TRY_OR_CONTINUE)", CPPGPT_UNIQ(_r).error()); \
        continue;                                                      \
    }                                                                  \
    decl = ::std::move(*CPPGPT_UNIQ(_r))
