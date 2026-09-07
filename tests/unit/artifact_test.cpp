// The corpus-artifact envelope (M7-3, IMPLEMENTATION_PLAN P3).
//
// P3 says an artifact records the checkpoint it came from and that loading a
// mismatched one fails with a named error. That is the whole reason this format
// exists: an artifact from a different training run renders perfectly. Its
// neuron indices are in range, its contexts are real text, every panel fills.
// Nothing about the rendered page says it is describing another model.
//
// So the checks here are about REFUSAL, and each names the specific corruption
// it injects. A single "it rejects bad input" test would pass for a loader that
// rejects everything.
#include "cppgpt/interp/artifact.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "tests/check.hpp"

using namespace cppgpt;

namespace {

std::string tmp_path(const char* name) {
    const char* dir = std::getenv("TEST_TMPDIR");
    return std::string(dir != nullptr ? dir : "/tmp") + "/" + name;
}

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void spit(const std::string& p, const std::string& b) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(b.data(), static_cast<std::streamsize>(b.size()));
}

}  // namespace

int main() {
    const std::uint64_t kCkpt = 0xABCDEF0123456789ULL;
    const std::string payload = "neuron payload bytes, arbitrary content";
    const std::string good = tmp_path("good.art");

    // ---- round trip ----
    {
        CHECK(write_artifact(good.c_str(), ArtifactKind::NeuronTopK, kCkpt, payload).has_value());
        const auto r = read_artifact(good.c_str(), ArtifactKind::NeuronTopK, kCkpt);
        CHECK(r.has_value());
        CHECK(r.has_value() && *r == payload);
    }

    // ---- THE identity check: a different checkpoint is refused ----
    {
        const auto r = read_artifact(good.c_str(), ArtifactKind::NeuronTopK, kCkpt + 1);
        CHECK(!r.has_value());
        CHECK(!r.has_value() && r.error() == ErrorCode::ShapeMismatch);
    }

    // ---- the wrong kind is refused, not reinterpreted ----
    {
        const auto r = read_artifact(good.c_str(), static_cast<ArtifactKind>(99), kCkpt);
        CHECK(!r.has_value());
        CHECK(!r.has_value() && r.error() == ErrorCode::ShapeMismatch);
    }

    // ---- a missing file is an IoError, not a crash or an empty payload ----
    {
        const auto r = read_artifact(tmp_path("absent.art").c_str(), ArtifactKind::NeuronTopK, kCkpt);
        CHECK(!r.has_value());
        CHECK(!r.has_value() && r.error() == ErrorCode::IoError);
    }

    // ---- each corruption is refused, and named separately ----
    {
        const std::string raw = slurp(good);
        CHECK(raw.size() == 40 + payload.size());

        auto corrupt = [&](std::size_t off, char v) {
            std::string b = raw;
            b[off] ^= v;
            const std::string p = tmp_path("bad.art");
            spit(p, b);
            return read_artifact(p.c_str(), ArtifactKind::NeuronTopK, kCkpt);
        };

        const auto bad_magic = corrupt(0, 0x7f);
        CHECK(!bad_magic.has_value() && bad_magic.error() == ErrorCode::CorruptCheckpoint);

        const auto bad_ver = corrupt(4, 0x7f);
        CHECK(!bad_ver.has_value() && bad_ver.error() == ErrorCode::VersionMismatch);

        const auto bad_reserved = corrupt(12, 0x01);
        CHECK(!bad_reserved.has_value() && bad_reserved.error() == ErrorCode::CorruptCheckpoint);

        // A flipped PAYLOAD byte leaves every header field valid, so only the
        // checksum can catch it. This is the check that makes the others
        // meaningful rather than a set of magic-number comparisons.
        const auto bad_payload = corrupt(45, 0x01);
        CHECK(!bad_payload.has_value() && bad_payload.error() == ErrorCode::ChecksumMismatch);

        // Truncation: the header claims more payload than the file holds.
        std::string trunc = raw.substr(0, raw.size() - 5);
        const std::string tp = tmp_path("trunc.art");
        spit(tp, trunc);
        const auto r = read_artifact(tp.c_str(), ArtifactKind::NeuronTopK, kCkpt);
        CHECK(!r.has_value() && r.error() == ErrorCode::CorruptCheckpoint);
    }

    // ---- a header claiming a huge payload is refused BEFORE allocating ----
    //
    // engineering-lessons L3: never size an allocation from an unvalidated file
    // header. Removing the size check still failed the other tests -- the
    // checksum caught the truncation by another route -- so those tests did not
    // cover this guard's actual purpose. A header claiming 2^60 bytes would
    // reach resize() and terminate before any checksum ran.
    {
        std::string raw = slurp(good);
        const std::uint64_t huge = 1ULL << 60;
        std::memcpy(raw.data() + 24, &huge, sizeof(huge));  // payload_bytes field
        const std::string p = tmp_path("huge.art");
        spit(p, raw);
        const auto r = read_artifact(p.c_str(), ArtifactKind::NeuronTopK, kCkpt);
        CHECK(!r.has_value());
        CHECK(!r.has_value() && r.error() == ErrorCode::CorruptCheckpoint);
    }

    // ---- an empty payload is legal, and distinguishable from a missing one ----
    {
        const std::string e = tmp_path("empty.art");
        CHECK(write_artifact(e.c_str(), ArtifactKind::NeuronTopK, kCkpt, "").has_value());
        const auto r = read_artifact(e.c_str(), ArtifactKind::NeuronTopK, kCkpt);
        CHECK(r.has_value() && r->empty());
    }

    return cppgpt::test::summary();
}
