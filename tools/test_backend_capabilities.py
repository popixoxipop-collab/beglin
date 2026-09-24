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
        self.assertFalse(got["backends"]["mlx_metal"]["runtime_control"]["compiled"])
        self.assertEqual(
            got["backends"]["mlx_metal"]["runtime_control"]["status"], "UNSUPPORTED_BINARY"
        )

    @patch.object(cap, "_run")
    def test_full_control_symbol_set_marks_runtime_control_unverified(self, run):
        symbols = "".join(
            f"0000000000000000 T _{symbol}\n" for symbol in cap.GPU_CONTROL_SYMBOLS
        )
        run.return_value = (
            "b" * 64 + "  /tmp/bin\n"
            "456\nBOB.local\narm64\n"
            "/tmp/bin:\n"
            + symbols
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
        self.assertTrue(got["backends"]["mlx_metal"]["runtime_control"]["compiled"])
        self.assertEqual(
            got["backends"]["mlx_metal"]["runtime_control"]["status"],
            "IMPLEMENTED_UNVERIFIED",
        )
        self.assertEqual(len(got["worker_identity_sha256"]), 64)

    @patch.object(cap, "_run")
    def test_partial_control_symbol_set_is_incomplete_binary(self, run):
        run.return_value = (
            "c" * 64 + "  /tmp/bin\n"
            "789\nXOX.local\narm64\n"
            "/tmp/bin:\n"
            "0000000000000000 T _mlx_gpu_available\n"
            "0000000000000000 T _mlx_gpu_binding_kind\n"
        )
        got = cap.collect("xox", "/tmp/bin")
        self.assertTrue(got["backends"]["mlx_metal"]["compiled"])
        self.assertFalse(got["backends"]["mlx_metal"]["runtime_control"]["compiled"])
        self.assertEqual(
            got["backends"]["mlx_metal"]["runtime_control"]["status"], "INCOMPLETE_BINARY"
        )


if __name__ == "__main__":
    unittest.main()
