#include "cppgpt/core.hpp"

#include <type_traits>

#include "tests/check.hpp"

using cppgpt::ErrorCode;
using cppgpt::Result;

// Enregisterability is the whole point — encode it as compile-time invariants.
//
// The relevant ABI notion is "trivial for the purposes of calls" (Itanium C++
// ABI): a type is passed/returned in registers iff its copy ctor, move ctor, and
// destructor are trivial. ASSIGNMENT triviality is irrelevant to calls. So we do
// NOT use std::is_trivially_copyable here: std::expected has a non-trivial
// copy-assignment (it must switch the active union member), making it
// not-trivially-copyable BY DESIGN — yet it is still trivially constructible +
// destructible, hence still register-passed. A <=16-byte such type returns in
// RAX:RDX. (Ground truth confirmed separately via objdump.)
template <class T>
inline constexpr bool register_passable_v =
    std::is_trivially_copy_constructible_v<T> && std::is_trivially_move_constructible_v<T> &&
    std::is_trivially_destructible_v<T> && sizeof(T) <= 16;

static_assert(std::is_trivially_copyable_v<ErrorCode>);
static_assert(sizeof(ErrorCode) == 2);
static_assert(register_passable_v<Result<int>>);
static_assert(register_passable_v<Result<float>>);
static_assert(register_passable_v<Result<float*>>);
static_assert(register_passable_v<Result<void>>);

namespace {

Result<int> parse_positive(int x) {
    if (x < 0) return cppgpt::err(ErrorCode::OutOfRange);
    return x;
}

// ASSIGN_OR_RETURN: statement-form propagation.
Result<int> sum_two_positives(int a, int b) {
    ASSIGN_OR_RETURN(int x, parse_positive(a));
    ASSIGN_OR_RETURN(int y, parse_positive(b));
    return x + y;
}

// TRY: expression-form propagation, unwrapped inline inside a larger expression.
Result<int> product_two_positives(int a, int b) {
    return TRY(parse_positive(a)) * TRY(parse_positive(b));
}

// RETURN_IF_ERROR: value-less propagation.
Result<void> require_positive(int x) {
    RETURN_IF_ERROR(parse_positive(x));
    return {};
}

}  // namespace

int main() {
    // success path
    {
        const Result<int> r = sum_two_positives(2, 3);
        CHECK(r.has_value());
        CHECK(*r == 5);
    }
    // error propagates with the right code
    {
        const Result<int> r = sum_two_positives(2, -1);
        CHECK(!r.has_value());
        CHECK(r.error() == ErrorCode::OutOfRange);
    }
    // TRY inline composition
    {
        const Result<int> r = product_two_positives(4, 5);
        CHECK(r.has_value());
        CHECK(*r == 20);
        CHECK(!product_two_positives(-1, 5).has_value());
    }
    // TRY_OR fallback (no return)
    {
        const int v = TRY_OR(parse_positive(-7), -1);
        CHECK(v == -1);
        const int w = TRY_OR(parse_positive(7), -1);
        CHECK(w == 7);
    }
    // RETURN_IF_ERROR
    {
        CHECK(require_positive(1).has_value());
        CHECK(!require_positive(-1).has_value());
    }
    // describe is total and non-null
    {
        CHECK(cppgpt::describe(ErrorCode::Ok)[0] != '\0');
        CHECK(cppgpt::describe(ErrorCode::ParseError)[0] != '\0');
    }
    return cppgpt::test::summary();
}
