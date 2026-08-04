// tools/cli.hpp: argv parsing. The load-bearing case is that a mistyped flag is a
// hard error — a silently-ignored --lr would train at the wrong rate and look fine.
#include "tools/cli.hpp"

#include <string>
#include <string_view>
#include <vector>

#include "tests/check.hpp"

using namespace cppgpt;

namespace {
cli::Args make(std::vector<const char*> v, std::initializer_list<std::string_view> val,
               std::initializer_list<std::string_view> boolean = {}) {
    static std::vector<std::vector<char>> store;  // argv must be mutable char*
    store.clear();
    static std::vector<char*> argv;
    argv.clear();
    for (const char* s : v) {
        store.emplace_back(s, s + std::string(s).size() + 1);
        argv.push_back(store.back().data());
    }
    return cli::Args(static_cast<int>(argv.size()), argv.data(), val, boolean);
}
}  // namespace

int main() {
    // --flag value, --flag=value, and bare --flag all parse
    {
        const auto a = make({"prog", "--lr", "0.5", "--steps=7", "--greedy", "pos1", "pos2"},
                         {"lr", "steps"}, {"greedy"});
        CHECK(a.str("lr", "") == "0.5");
        CHECK(a.integer("steps", 0) == 7);
        CHECK(a.has("greedy"));
        CHECK(a.positional().size() == 2 && a.positional()[0] == "pos1");
    }
    // defaults apply when absent
    {
        const auto a = make({"prog"}, {"steps", "lr", "nope"});
        CHECK(a.integer("steps", 42) == 42);
        CHECK(a.real("lr", 1.5f) == 1.5f);
        CHECK(!a.has("nope"));
        CHECK(a.str("nope", "dflt") == "dflt");
    }
    // negative and float values are not mistaken for flags
    {
        const auto a = make({"prog", "--lr", "3e-4", "--n", "-1"}, {"lr", "n"});
        CHECK(a.real("lr", 0.0f) > 2.9e-4f && a.real("lr", 0.0f) < 3.1e-4f);
        CHECK(a.integer("n", 0) == -1);
    }
    // an unknown flag must abort, not be ignored
    {
        CHECK_DIES(make({"prog", "--stpes", "10"}, {"steps", "lr"}));  // typo
    }
    // a known flag set passes cleanly
    {
        const auto a = make({"prog", "--steps", "10"}, {"steps", "lr"});
        CHECK(a.integer("steps", 0) == 10);
    }
    // a malformed number is an error, not a silent 0
    {
        const auto a = make({"prog", "--steps", "12abc"}, {"steps"});
        CHECK_DIES((void)a.integer("steps", 0));
    }
    return cppgpt::test::summary();
}
