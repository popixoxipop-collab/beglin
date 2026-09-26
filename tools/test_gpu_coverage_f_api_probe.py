#!/usr/bin/env python3
from pathlib import Path
import tempfile
import types
import unittest

import gpu_coverage_f_api_probe as p


def one(a, b=2):
    return a + b


class ProbeTests(unittest.TestCase):
    def test_probe_module_records_signature_without_calling(self):
        called = {"n": 0}
        def dangerous(x, *, y=None):
            called["n"] += 1
            raise AssertionError("must not be called")
        m = types.ModuleType("fake")
        m.dangerous = dangerous
        got = p.probe_module(m, ("dangerous",))
        self.assertTrue(got["ok"])
        self.assertEqual(called["n"], 0)
        self.assertIn("x", got["callables"]["dangerous"]["signature"])
        self.assertIsNone(got["file_sha256"])

    def test_missing_callable_fails_module(self):
        m = types.ModuleType("fake")
        got = p.probe_module(m, ("missing",))
        self.assertFalse(got["ok"])
        self.assertEqual(got["missing"], ["missing"])

    def test_contract_contains_required_g5_g6_functions(self):
        self.assertIn("request_regression_rollback", p.CONTRACT["gpu_observer_control"])
        self.assertIn("complete_regression_rollback", p.CONTRACT["gpu_observer_control"])
        self.assertIn("build_restart_canary", p.CONTRACT["gpu_restart_canary"])
        self.assertIn("evaluate_restart_canary", p.CONTRACT["gpu_restart_canary"])
        self.assertIn("run_ab_preflight", p.CONTRACT["gpu_isolated_preflight"])

    def test_atomic_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "probe.json"
            p.atomic_json(out, {"ok": True})
            self.assertIn('"ok": true', out.read_text())


if __name__ == "__main__":
    unittest.main()
