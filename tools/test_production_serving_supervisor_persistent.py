#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
import unittest.mock
from unittest.mock import patch

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


class RuntimeControlImportTests(unittest.TestCase):
    def test_runtime_control_is_loaded_from_supervisor_tools_tree(self):
        self.assertEqual(
            Path(ps._runtime_tools_path()).resolve(),
            Path(ps.__file__).resolve().parent,
        )
        grc = ps._load_gpu_runtime_control()
        self.assertEqual(
            Path(grc.__file__).resolve().parent,
            Path(ps.__file__).resolve().parent,
        )
        self.assertTrue(callable(grc.prepare_rebind))
        self.assertTrue(callable(grc.verify_terminal_ack))

    def test_precision_closed_loop_is_loaded_from_supervisor_tools_tree(self):
        pcl = ps._load_precision_closed_loop()
        self.assertEqual(
            Path(pcl.__file__).resolve().parent,
            Path(ps.__file__).resolve().parent,
        )
        self.assertTrue(callable(pcl.PrecisionClosedLoopEngine))

    def test_precision_epoch_scheduler_is_loaded_from_supervisor_tools_tree(self):
        pes = ps._load_precision_epoch_scheduler()
        self.assertEqual(
            Path(pes.__file__).resolve().parent,
            Path(ps.__file__).resolve().parent,
        )
        self.assertTrue(callable(pes.PrecisionEpochScheduler))

    def test_policy_hash_uses_local_precision_context(self):
        got = ps._policy_hash([
            {"role": "shared_down_proj", "layer": 26, "n": 5},
            {"role": "shared_up_proj", "layer": 3, "n": 6},
        ])
        self.assertEqual(len(got), 64)


class PrecisionEpochWorkerTests(unittest.TestCase):
    def test_worker_uses_reentrant_admission_lock_for_scheduler_bridge(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ps.PersistentRouteWorker(
                route=base.candidate_route(), root=Path(td)
            )
            acquired = worker.lock.acquire(blocking=False)
            self.assertTrue(acquired)
            try:
                self.assertTrue(worker.lock.acquire(blocking=False))
                worker.lock.release()
            finally:
                worker.lock.release()

    def test_closed_loop_requires_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ps.PersistentRouteWorker(
                route=base.candidate_route(), root=Path(td)
            )
            with self.assertRaises(ps.PersistentSupervisorError):
                worker.submit_with_closed_loop_precision(
                    [([1], 1)], signal={}, admission_id="unconfigured"
                )

    def test_closed_loop_bridge_binds_decision_to_scheduler_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ps.PersistentRouteWorker(
                route=base.candidate_route(), root=Path(td)
            )
            policy = [{"role": "shared_up_proj", "layer": 3, "n": 6}]
            ack = {
                "schema": "gpu-precision-applied-v1",
                "status": "PROMOTION_APPLIED",
                "backend": "mlx_metal",
                "correction_mode": "off",
                "weight_epoch": 1,
                "changed_targets": 1,
                "snapshot_count": 1,
                "txn_id": None,
                "expected_epoch": None,
                "expected_n": None,
                "expected_policy_hash": None,
                "target_role": None,
                "target_layer": None,
                "active_policy": policy,
            }
            worker.ack_path.parent.mkdir(parents=True, exist_ok=True)
            worker.ack_path.write_text(json.dumps(ack))
            ph = ps._policy_hash(policy)
            engine = unittest.mock.Mock()
            engine.decide.return_value = {
                "schema": "beglin-precision-closed-loop-decision-v1",
                "status": "READY_FOR_SCHEDULER",
                "evidence_snapshot_sha256": "a" * 64,
                "allocation_sha256": "b" * 64,
                "selection_sha256": "c" * 64,
                "current_policy_hash": ph,
                "selected_policy_hash": ph,
                "selected_policy": policy,
                "changes": [],
                "combined_policy_evidence": None,
                "signal": {"active_triggers": []},
            }
            scheduler = unittest.mock.Mock()
            scheduler.run.return_value = {
                "finite_logits": True,
                "responses": [[1]],
                "precision_epoch": {
                    "before_policy_hash": ph,
                    "after_policy_hash": ph,
                },
            }
            worker.precision_closed_loop_engine = engine
            worker.precision_epoch_scheduler = scheduler
            got = worker.submit_with_closed_loop_precision(
                [([1], 1)], signal={}, admission_id="bridge"
            )
            self.assertTrue(got["finite_logits"])
            self.assertEqual(
                got["precision_closed_loop"]["selected_policy_hash"], ph
            )
            engine.decide.assert_called_once()
            scheduler.run.assert_called_once()

    def test_submit_with_precision_policy_delegates_and_refreshes_ack(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ps.PersistentRouteWorker(
                route=base.candidate_route(), root=Path(td)
            )
            policy = [{"role": "shared_up_proj", "layer": 3, "n": 6}]
            fake = unittest.mock.Mock()
            fake.run.return_value = {
                "finite_logits": True,
                "responses": [[1224]],
                "precision_epoch": {"transitioned": False},
            }
            worker.precision_epoch_scheduler = fake
            ack = {
                "schema": "gpu-precision-applied-v1",
                "status": "PROMOTION_APPLIED",
                "backend": "mlx_metal",
                "correction_mode": "off",
                "weight_epoch": 1,
                "changed_targets": 1,
                "snapshot_count": 1,
                "txn_id": None,
                "expected_epoch": None,
                "expected_n": None,
                "expected_policy_hash": None,
                "target_role": None,
                "target_layer": None,
                "active_policy": policy,
            }
            worker.ack_path.parent.mkdir(parents=True, exist_ok=True)
            worker.ack_path.write_text(json.dumps(ack))
            got = worker.submit_with_precision_policy(
                [([1], 1)],
                target_policy=policy,
                admission_id="unit-admission",
            )
            self.assertTrue(got["finite_logits"])
            fake.run.assert_called_once()
            self.assertEqual(worker.ack["weight_epoch"], 1)


class AdaptiveTwoPassTests(unittest.TestCase):
    def test_trigger_indices_are_request_scoped_and_thresholded(self):
        events = [
            {"req": 0, "margin": 0.009},
            {"req": 1, "margin": 0.010714},
            {"req": 2, "margin": 0.0201},
            {"req": 1, "margin": 0.001},
            {"req": 99, "margin": 0.0},
            {"req": "bad", "margin": 0.0},
        ]
        self.assertEqual(ps.ADAPTIVE_L26_MARGIN_MAX, 0.02)
        self.assertEqual(ps._adaptive_trigger_indices(events, 3), [0, 1])

    def test_adaptive_pool_is_strictly_opt_in(self):
        normal = ps.PersistentWorkerPool()
        adaptive = ps.PersistentWorkerPool(adaptive_l26=True)
        rid = base.candidate_route()["route_id"]
        self.assertIsInstance(normal.workers[rid], ps.PersistentRouteWorker)
        self.assertNotIsInstance(normal.workers[rid], ps.AdaptivePersistentRouteWorker)
        self.assertIsInstance(adaptive.workers[rid], ps.AdaptivePersistentRouteWorker)
        self.assertFalse(normal.health()["adaptive_l26_enabled"])
        self.assertTrue(adaptive.health()["adaptive_l26_enabled"])

    def test_adaptive_startup_policy_is_candidate_plus_l26_base(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ps.AdaptivePersistentRouteWorker(
                route=base.candidate_route(), root=Path(td)
            )
            self.assertEqual(
                worker.startup_policy,
                [
                    {"role": "shared_up_proj", "layer": 3, "n": 6},
                    {"role": "shared_down_proj", "layer": 26, "n": 5},
                ],
            )
            self.assertEqual(
                worker.expected_runtime_policy_hash(),
                ps._policy_hash(worker.startup_policy),
            )
            self.assertNotEqual(
                worker.expected_runtime_policy_hash(),
                worker.route["policy_hash"],
            )
            self.assertEqual(
                worker.expected_runtime_policy_hash(),
                ps.ADAPTIVE_L26_STARTUP_POLICY_SHA256,
            )
            self.assertEqual(
                ps.ADAPTIVE_L26_EVIDENCE_SHA256,
                "de976ab12283673a8cf97638be9cf0b5c8f7ab3df2a0db87d4be96d9e67f6075",
            )
            self.assertEqual(
                ps.ADAPTIVE_L26_REBIND_EVIDENCE_SHA256,
                ps.ADAPTIVE_L26_EVIDENCE_SHA256,
            )
            worker._prepare_files()
            self.assertEqual(
                worker.promotion_path.read_text().splitlines(),
                ["shared_up_proj 3 6", "shared_down_proj 26 5"],
            )

    def test_triggered_request_only_is_recovered_at_n6(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ps.AdaptivePersistentRouteWorker(
                route=base.candidate_route(), root=Path(td)
            )
            first = {
                "finite_logits": True,
                "engine_wall_ms": 10.0,
                "roundtrip_ms": 11.0,
                "responses": [[100], [200]],
                "neartie_events": [
                    {
                        "req": 1, "pos": 9, "predicted_token": 372,
                        "competing_token": 1, "margin": 0.002424,
                        "batch_size": 2,
                    }
                ],
            }
            recovery = {
                "finite_logits": True,
                "engine_wall_ms": 4.0,
                "roundtrip_ms": 5.0,
                "responses": [[999]],
                "neartie_events": [],
            }
            with patch.object(
                ps.PersistentRouteWorker, "submit",
                side_effect=[first, recovery],
            ), patch.object(
                worker, "_current_l26_n", return_value=5
            ), patch.object(
                worker, "_prepare_rebind"
            ) as prepare, patch.object(
                worker, "_verify_rebind", return_value={}
            ) as verify:
                got = worker.submit([([1], 2), ([2], 2)])

            self.assertEqual(got["responses"], [[100], [999]])
            self.assertEqual(got["adaptive_precision"]["action"], "RECOVERY_N6")
            self.assertEqual(
                got["adaptive_precision"]["trigger_request_indices"], [1]
            )
            self.assertEqual(got["engine_wall_ms"], 14.0)
            self.assertEqual(got["roundtrip_ms"], 16.0)
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.kwargs["expected_n"], 5)
            self.assertEqual(prepare.call_args.kwargs["target_n"], 6)
            verify.assert_called_once()
            self.assertEqual(verify.call_args.kwargs["target_n"], 6)

    def test_next_request_restores_n5_before_base_pass(self):
        with tempfile.TemporaryDirectory() as td:
            worker = ps.AdaptivePersistentRouteWorker(
                route=base.candidate_route(), root=Path(td)
            )
            base_result = {
                "finite_logits": True,
                "engine_wall_ms": 3.0,
                "roundtrip_ms": 4.0,
                "responses": [[123]],
                "neartie_events": [],
            }
            with patch.object(
                ps.PersistentRouteWorker, "submit", return_value=base_result
            ), patch.object(
                worker, "_current_l26_n", return_value=6
            ), patch.object(
                worker, "_prepare_rebind"
            ) as prepare, patch.object(
                worker, "_verify_rebind", return_value={}
            ) as verify:
                got = worker.submit([([1], 2)])

            self.assertEqual(got["responses"], [[123]])
            self.assertEqual(got["adaptive_precision"]["action"], "BASE_N5")
            self.assertTrue(got["adaptive_precision"]["restored_from_n6"])
            prepare.assert_called_once()
            self.assertEqual(prepare.call_args.kwargs["expected_n"], 6)
            self.assertEqual(prepare.call_args.kwargs["target_n"], 5)
            verify.assert_called_once()
            self.assertEqual(verify.call_args.kwargs["target_n"], 5)


class AdaptivePrewarmTests(unittest.TestCase):
    def test_prewarm_uses_adaptive_recovery_not_raw_base(self):
        pool = ps.PersistentWorkerPool(adaptive_l26=True)
        worker = pool.workers[base.candidate_route()["route_id"]]
        fake = {
            "finite_logits": True,
            "engine_wall_ms": 7.0,
            "roundtrip_ms": 8.0,
            "responses": [[0] * 8 + [1224, 0] for _ in range(12)],
            "neartie_events": [],
            "adaptive_precision": {
                "enabled": True,
                "action": "RECOVERY_N6",
                "trigger_request_indices": list(range(12)),
            },
        }
        with patch.object(
            base, "_read_first_certified_prompt", return_value=[1, 2, 3]
        ), patch.object(
            worker, "submit", return_value=fake
        ) as submit, patch.object(
            worker, "submit_base"
        ) as submit_base, patch.object(
            worker, "health", return_value={"pid": 123}
        ):
            got = pool._prewarm(worker)
        self.assertEqual(got["reference_hits"], 12)
        submit.assert_called_once()
        submit_base.assert_not_called()


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


class AdaptiveAcceptanceEvidenceTests(unittest.TestCase):
    @staticmethod
    def _evidence():
        row = {
            "pid": 123,
            "finite_logits": True,
            "token8": 1224,
            "action": "RECOVERY_N6",
            "trigger_request_indices": [0],
            "restored_from_n6": True,
            "worker_left_at_n": 6,
            "base_events": [{"margin": 0.010715}],
        }
        return {
            "schema": "beglin-adaptive-isolated-final/1",
            "status": "PASS",
            "source_head": ps.ADAPTIVE_L26_ACCEPTANCE_SOURCE,
            "binary_sha256": ps.EXPECTED_PERSISTENT_BINARY_SHA,
            "adaptive_evidence_sha256": ps.ADAPTIVE_L26_EVIDENCE_SHA256,
            "adaptive_startup_policy_sha256": ps.ADAPTIVE_L26_STARTUP_POLICY_SHA256,
            "production_route_touched": False,
            "requests": [dict(row, restored_from_n6=False), row],
        }

    def test_adaptive_acceptance_file_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "acceptance.json"
            path.write_text(json.dumps(self._evidence(), indent=2, sort_keys=True) + "\n")
            digest = base._sha256_file(path)
            with patch.object(ps, "ADAPTIVE_L26_ACCEPTANCE_EVIDENCE", path), patch.object(
                ps, "ADAPTIVE_L26_ACCEPTANCE_SHA256", digest
            ):
                got = ps.verify_adaptive_l26_acceptance()
            self.assertEqual(got["sha256"], digest)
            self.assertEqual(got["request_count"], 2)
            self.assertEqual(got["worker_pid"], 123)

    def test_adaptive_acceptance_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "acceptance.json"
            path.write_text(json.dumps(self._evidence(), sort_keys=True))
            digest = base._sha256_file(path)
            path.write_text(path.read_text() + " ")
            with patch.object(ps, "ADAPTIVE_L26_ACCEPTANCE_EVIDENCE", path), patch.object(
                ps, "ADAPTIVE_L26_ACCEPTANCE_SHA256", digest
            ):
                with self.assertRaises(ps.PersistentSupervisorError):
                    ps.verify_adaptive_l26_acceptance()


class ArtifactContractTests(unittest.TestCase):
    def test_expected_hardware_identity_is_pinned(self):
        self.assertEqual(
            ps.EXPECTED_PERSISTENT_BINARY_SHA,
            "8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9",
        )
        self.assertEqual(
            ps.EXPECTED_PERSISTENT_SOURCE_SHA,
            "8b46cdd38f5b39dcaeaa9d1c5dfbd9642601a30a",
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
