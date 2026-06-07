"""Smoke py_test: proves the hermetic Python toolchain runs tests via Bazel.

Run: bazel test //scripts:env_test

Plain assert-based (no third-party test framework), mirroring the std-only C++
harness. unittest is stdlib and integrates cleanly with Bazel's py_test.
"""

import sys
import unittest


class ToolchainTest(unittest.TestCase):
    def test_python_version(self) -> None:
        # Pinned in MODULE.bazel; guard against an accidental downgrade.
        self.assertGreaterEqual(sys.version_info[:2], (3, 12))

    def test_stdlib_available(self) -> None:
        import json
        import pathlib

        self.assertEqual(json.loads('{"ok": true}'), {"ok": True})
        self.assertTrue(pathlib.Path(__file__).exists())


if __name__ == "__main__":
    unittest.main()
