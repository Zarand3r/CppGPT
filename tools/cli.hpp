// Minimal argv parser shared by the cppgpt CLI tools.
//
// Supports `--flag value`, `--flag=value`, bare `--flag`, and positionals. It
// exists because three tools now take flags and each hand-rolling argv walking
// would drift; it deliberately does nothing else (no subcommands, no short
// options, no config files).
//
// The schema is passed to the constructor, which is load-bearing twice over:
//   * a mistyped flag is rejected immediately — a quietly-ignored --lr would train
//     at the wrong rate and look perfectly fine, exactly the silent failure this
//     repo forbids; and
//   * value-taking and boolean flags must be distinguished up front, or a bare
//     `--greedy prompt.txt` would swallow the positional as --greedy's value.
// Making the schema a constructor argument means neither can be forgotten.
#pragma once

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cppgpt::cli {

class Args {
public:
    // `value_flags` take an argument (--lr 3e-4 or --lr=3e-4); `bool_flags` never do.
    // Anything else on the command line is a hard error.
    Args(int argc, char** argv, std::initializer_list<std::string_view> value_flags,
         std::initializer_list<std::string_view> bool_flags = {}) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) {  // positional
                pos_.push_back(std::move(a));
                continue;
            }
            const std::size_t eq = a.find('=');
            const std::string name = a.substr(2, eq == std::string::npos ? eq : eq - 2);
            const bool takes_value = contains(value_flags, name);
            if (!takes_value && !contains(bool_flags, name)) unknown(name, value_flags, bool_flags);
            if (eq != std::string::npos) {
                if (!takes_value) {
                    std::fprintf(stderr, "error: --%s takes no value\n", name.c_str());
                    std::exit(2);
                }
                flags_.emplace_back(name, a.substr(eq + 1));
            } else if (takes_value) {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "error: --%s expects a value\n", name.c_str());
                    std::exit(2);
                }
                flags_.emplace_back(name, argv[++i]);
            } else {
                flags_.emplace_back(name, "");  // boolean, present
            }
        }
    }

    [[nodiscard]] bool has(std::string_view name) const { return find(name) != nullptr; }

    [[nodiscard]] std::string_view str(std::string_view name, std::string_view def) const {
        const std::string* v = find(name);
        return v != nullptr && !v->empty() ? std::string_view(*v) : def;
    }

    [[nodiscard]] int integer(std::string_view name, int def) const {
        const std::string* v = find(name);
        if (v == nullptr || v->empty()) return def;
        int out = 0;
        const auto* first = v->data();
        const auto r = std::from_chars(first, first + v->size(), out);
        if (r.ec != std::errc{} || r.ptr != first + v->size()) fail(name, *v, "an integer");
        return out;
    }

    [[nodiscard]] float real(std::string_view name, float def) const {
        const std::string* v = find(name);
        if (v == nullptr || v->empty()) return def;
        // std::from_chars for float is not available in every libc++ we target;
        // strtof plus an explicit full-consumption check is equivalent here.
        char* end = nullptr;
        const float out = std::strtof(v->c_str(), &end);
        if (end != v->c_str() + v->size()) fail(name, *v, "a number");
        return out;
    }

    [[nodiscard]] const std::vector<std::string>& positional() const { return pos_; }

private:
    static bool contains(std::initializer_list<std::string_view> set, const std::string& n) {
        for (std::string_view s : set)
            if (s == n) return true;
        return false;
    }
    [[noreturn]] static void unknown(const std::string& name,
                                     std::initializer_list<std::string_view> value_flags,
                                     std::initializer_list<std::string_view> bool_flags) {
        std::fprintf(stderr, "error: unknown flag --%s\nknown flags:", name.c_str());
        for (auto set : {value_flags, bool_flags})
            for (std::string_view k : set)
                std::fprintf(stderr, " --%.*s", static_cast<int>(k.size()), k.data());
        std::fprintf(stderr, "\n");
        std::exit(2);
    }
    [[nodiscard]] const std::string* find(std::string_view name) const {
        for (const auto& [n, v] : flags_)
            if (n == name) return &v;
        return nullptr;
    }
    [[noreturn]] static void fail(std::string_view name, const std::string& got,
                                  const char* want) {
        std::fprintf(stderr, "error: --%.*s expects %s, got '%s'\n", static_cast<int>(name.size()),
                     name.data(), want, got.c_str());
        std::exit(2);
    }

    std::vector<std::pair<std::string, std::string>> flags_;
    std::vector<std::string> pos_;
};

}  // namespace cppgpt::cli
