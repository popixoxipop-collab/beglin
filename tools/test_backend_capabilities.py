#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import backend_capabilities as cap


class CapabilityTests(unittest.TestCase):
    @patch.object(cap, "_run")
    def test_cpu_only_binary_marks_gpu_unsupported(self, run):
        run.return_value = (
            "a" * 64 + "  /tmp/bin\n"
            "123\nBOB.local\narm64\n"
            "/tmp/bin:\n/usr/lib/libSystem.B.dylib\n"
        )
        got = cap.collect("bob", "/tmp/bin")
        self.assertEqual(got["worker"]["binary_sha256"], "a" * 64)
        self.assertFalse(got["backends"]["mlx_metal"]["compiled"])
        self.assertEqual(
            got["backends"]["mlx_metal"]["qng64_widths"]["7"],
            "UNSUPPORTED_BINARY",
        )

    @patch.object(cap, "_run")
    def test_mlx_symbol_marks_gpu_implemented_unverified(self, run):
        run.return_value = (
            "b" * 64 + "  /tmp/bin\n"
            "456\nBOB.local\narm64\n"
            "/tmp/bin:\n"
            "0000000000000000 T _mlx_gpu_available\n"
        )
        got = cap.collect("bob", "/tmp/bin")
        self.assertTrue(got["backends"]["mlx_metal"]["compiled"])
        self.assertEqual(
            got["backends"]["mlx_metal"]["qng64_widths"]["6"],
            "IMPLEMENTED_UNVERIFIED",
        )
        self.assertEqual(
            got["backends"]["mlx_metal"]["qng64_widths"]["7"],
            "IMPLEMENTED_UNVERIFIED",
        )
        self.assertEqual(len(got["worker_identity_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
