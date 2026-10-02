#!/usr/bin/env python3
from pathlib import Path
import tempfile
import unittest

import production_serving_supervisor_persistent as ps


class BuildStatusTests(unittest.TestCase):
    def test_missing_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ps.PersistentSupervisorError):
                ps.read_build_identity(Path(td)/"missing.txt")


class RouteTests(unittest.TestCase):
    def _build(self):
        return {
            "source_commit":"a"*40,
            "binary_sha256":"b"*64,
        }

    def test_candidate_route_is_exact_reviewed_precision(self):
        route=ps.persistent_route(candidate=True,build=self._build())
        self.assertEqual(route["route_id"],"beglin-candidate-shared-up-l3-n6-persistent-v1")
        self.assertEqual(route["role"],"shared_up_proj")
        self.assertEqual(route["layer"],3)
        self.assertEqual(route["n"],6)
        self.assertEqual(route["policy_hash"],ps.CANDIDATE_POLICY_HASH)

    def test_baseline_route_is_base_policy(self):
        route=ps.persistent_route(candidate=False,build=self._build())
        self.assertEqual(route["route_id"],"beglin-baseline-q4g64-persistent-v1")
        self.assertIsNone(route["n"])
        self.assertEqual(route["policy_hash"],ps.BASELINE_POLICY_HASH)


class SourceContractTests(unittest.TestCase):
    def test_supervisor_uses_two_persistent_workers(self):
        src=Path(ps.__file__).read_text()
        self.assertIn('"baseline"',src)
        self.assertIn('"candidate"',src)
        self.assertIn("candidate_pid_reused",src)
        self.assertIn("persistent_worker_reused",src)

    def test_no_external_bind(self):
        src=Path(ps.__file__).read_text()
        self.assertIn('host!="127.0.0.1"',src)
        self.assertIn('"external_network_exposed":False',src)


if __name__=="__main__":
    unittest.main()
