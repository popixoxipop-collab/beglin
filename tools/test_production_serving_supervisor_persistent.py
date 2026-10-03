#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps


class ResultParserTests(unittest.TestCase):
    def test_valid_result_roundtrip(self):
        text = (
            "BEGLIN_GPU_PERSISTENT_RESULT_V1 req-a 2 1 12.5\n"
            "REQ 0 3 10 11 12\n"
            "REQ 1 2 20 21\n"
            "END\n"
        )
        got = ps.parse_persistent_result(text, request_id="req-a", expected_requests=2)
        self.assertTrue(got["finite_logits"])
        self.assertEqual(got["engine_wall_ms"], 12.5)
        self.assertEqual(got["responses"], [[10,11,12],[20,21]])

    def test_wrong_request_id_fails(self):
        text = (
            "BEGLIN_GPU_PERSISTENT_RESULT_V1 req-b 1 1 1.0\n"
            "REQ 0 1 7\nEND\n"
        )
        with self.assertRaises(ps.PersistentSupervisorError):
            ps.parse_persistent_result(text, request_id="req-a", expected_requests=1)

    def test_incomplete_result_fails(self):
        text = "BEGLIN_GPU_PERSISTENT_RESULT_V1 req-a 1 1 1.0\nEND\n"
        with self.assertRaises(ps.PersistentSupervisorError):
            ps.parse_persistent_result(text, request_id="req-a", expected_requests=1)


class NearTieTelemetryTests(unittest.TestCase):
    def test_delta_parser_returns_only_new_valid_events(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "neartie.jsonl"
            path.write_text(
                json.dumps({
                    "kind": "event", "req": 9, "pos": 1,
                    "predicted_token": 99, "competing_token": 98,
                    "margin": 0.5, "batch_size": 1,
                }) + "\n"
            )
            offset = path.stat().st_size
            with path.open("a") as handle:
                handle.write("{not-json}\n")
                handle.write(json.dumps({"kind": "attribution", "req": 0}) + "\n")
                handle.write(json.dumps({
                    "kind": "event", "req": 0, "pos": 9,
                    "predicted_token": 372, "competing_token": 1,
                    "margin": 0.002424, "batch_size": 4,
                }) + "\n")
            self.assertEqual(
                ps._read_neartie_events_since(path, offset),
                [{
                    "req": 0, "pos": 9, "predicted_token": 372,
                    "competing_token": 1, "margin": 0.002424,
                    "batch_size": 4,
                }],
            )

    def test_worker_env_enables_telemetry_but_not_correction(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ps.PersistentRouteWorker(
                route=base.candidate_route(), root=Path(td)
            )
            env = worker._env()
            self.assertEqual(env["QWEN_MOE_NEARTIE_LOG"], "1")
            self.assertEqual(env["QWEN_MOE_NEARTIE_CORRECT"], "0")
            self.assertEqual(
                float(env["QWEN_MOE_NEARTIE_THRESHOLD"]),
                ps.NEARTIE_TELEMETRY_THRESHOLD,
            )
            self.assertEqual(
                env["QWEN_MOE_NEARTIE_EVENTS_LOG"],
                str(worker.neartie_path),
            )

    def test_worker_health_exposes_telemetry_contract(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ps.PersistentRouteWorker(
                route=base.candidate_route(), root=Path(td)
            )
            got = worker.health()["near_tie_telemetry"]
            self.assertTrue(got["enabled"])
            self.assertEqual(got["threshold"], 0.02)


class WorkerPoolContractTests(unittest.TestCase):
    def test_exact_two_routes_are_registered(self):
        pool = ps.PersistentWorkerPool()
        self.assertEqual(
            set(pool.workers),
            {
                base.baseline_route()["route_id"],
                base.candidate_route()["route_id"],
            },
        )

    def test_baseline_and_candidate_policy_hashes_are_distinct(self):
        pool = ps.PersistentWorkerPool()
        b = pool.workers[base.baseline_route()["route_id"]]
        c = pool.workers[base.candidate_route()["route_id"]]
        self.assertEqual(b.route["policy_hash"], base.BASELINE_POLICY_HASH)
        self.assertEqual(c.route["policy_hash"], base.CANDIDATE_POLICY_HASH)
        self.assertNotEqual(b.route["policy_hash"], c.route["policy_hash"])

    def test_unknown_route_is_fail_closed(self):
        pool = ps.PersistentWorkerPool()
        bad = base.baseline_route()
        bad["route_id"] = "unknown-route"
        with self.assertRaises(ps.PersistentSupervisorError):
            pool.get(bad)

    def test_route_policy_drift_is_fail_closed(self):
        pool = ps.PersistentWorkerPool()
        bad = base.candidate_route()
        bad["policy_hash"] = base.BASELINE_POLICY_HASH
        with self.assertRaises(ps.PersistentSupervisorError):
            pool.get(bad)


class ArtifactContractTests(unittest.TestCase):
    def test_expected_hardware_identity_is_pinned(self):
        self.assertEqual(
            ps.EXPECTED_PERSISTENT_BINARY_SHA,
            "3be6d59b77e554f5abee86851f4d901b7e2d9750ae6257a1538eebb64693d59e",
        )
        self.assertEqual(
            ps.EXPECTED_PERSISTENT_SOURCE_SHA,
            "a26f9a8ab93493a1aad8da643d00c1d998aa60cd",
        )

    def test_launchd_process_type_is_interactive(self):
        self.assertEqual(ps.LAUNCHD_PROCESS_TYPE, "Interactive")

    def test_executor_has_shutdown_contract_for_base_server(self):
        self.assertTrue(callable(getattr(ps.PersistentEngineExecutor, "shutdown", None)))

    def test_request_limits_inherit_reviewed_supervisor(self):
        self.assertEqual(base.MAX_BATCH_REQUESTS, 12)
        self.assertEqual(base.MAX_NEW_TOKENS, 256)

    def test_reference_mapping_matches_certified_routes(self):
        self.assertEqual(ps.PersistentWorkerPool._reference_token(base.baseline_route()), 3268)
        self.assertEqual(ps.PersistentWorkerPool._reference_token(base.candidate_route()), 1224)


if __name__ == "__main__":
    unittest.main()
