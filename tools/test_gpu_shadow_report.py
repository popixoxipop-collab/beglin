#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import gpu_shadow_report as gr


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def base_launch(**overrides):
    value = {
        "schema": "gpu-shadow-xox-launch-v1",
        "status": "COMPLETE",
        "launch_id": "launch-1",
        "pid": 4242,
        "returncode": 0,
        "started_at": "2026-09-25T13:00:00+00:00",
        "finished_at": "2026-09-25T13:00:05+00:00",
        "production_write_allowed": False,
    }
    value.update(overrides)
    return value


def base_cycle(**overrides):
    value = {
        "schema": "gpu-shadow-pipeline-v1",
        "status": "SHADOW_CYCLE_COMPLETE",
        "launch_id": "launch-1",
        "cycle_id": "cycle-1",
        "shadow_run_id": "run-1",
        "selected_candidate_id": "candidate-1",
        "ready_count": 1,
        "runtime_identity_sha256": "a" * 64,
        "runtime_identity": {
            "binary_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
        },
        "production_write_allowed": False,
        "started_at": "2026-09-25T13:00:01+00:00",
        "finished_at": "2026-09-25T13:00:04+00:00",
    }
    value.update(overrides)
    return value


class GpuShadowReportTests(unittest.TestCase):
    def setup_root(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        write_json(root / "launcher_status.json", base_launch())
        write_json(root / "discovery.json", {
            "schema": "gpu-shadow-discovery-v1",
            "candidate_count": 1,
            "ready_count": 1,
        })
        write_json(root / "candidate_history.json", {
            "schema": "gpu-shadow-history-v1",
            "candidates": {
                "candidate-1": {
                    "fingerprint": "f" * 64,
                    "runtime_identity_sha256": "a" * 64,
                    "shadow_status": "SHADOW_ADMITTED",
                    "reusable_terminal": True,
                    "attempt_count": 1,
                    "launch_id": "launch-1",
                    "cycle_id": "cycle-1",
                    "shadow_run_id": "run-1",
                }
            },
        })
        return td, root

    def test_no_ready_is_not_run_not_gpu_pass(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "last_cycle.json", base_cycle(
            status="NO_READY_CANDIDATE",
            cycle_id="cycle-empty",
            shadow_run_id=None,
            selected_candidate_id=None,
            ready_count=0,
        ))
        report = gr.build_report(root, pid_alive_fn=lambda pid: False)
        self.assertEqual(report["cycle"]["pipeline_state"], "not_run")
        self.assertEqual(report["gpu_validation"]["state"], "not_run")
        self.assertEqual(report["evidence_level"], "not_run")
        self.assertEqual(report["cycle"]["dedupe_reason"], "no_ready_candidate")

    def test_no_new_ready_is_dedupe_not_gpu_pass(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "last_cycle.json", base_cycle(
            status="NO_NEW_READY_CANDIDATE",
            cycle_id="cycle-dedupe",
            shadow_run_id=None,
            selected_candidate_id=None,
            already_observed_candidate_ids=["candidate-1"],
            dedupe_reason="reusable_terminal_same_candidate_and_runtime",
        ))
        report = gr.build_report(root, pid_alive_fn=lambda pid: False)
        self.assertEqual(report["source_freshness"], "current")
        self.assertEqual(report["cycle"]["pipeline_state"], "not_run")
        self.assertEqual(report["gpu_validation"]["state"], "not_run")
        self.assertEqual(
            report["cycle"]["dedupe_reason"],
            "reusable_terminal_same_candidate_and_runtime",
        )
        fp = report["identity"]["candidate_fingerprints"]["candidate-1"]
        self.assertEqual(fp["validation_context_fingerprint"], "a" * 64)

    def test_running_launcher_does_not_claim_previous_cycle(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "launcher_status.json", base_launch(
            status="RUNNING",
            launch_id="launch-new",
            finished_at=None,
        ))
        write_json(root / "last_cycle.json", base_cycle(launch_id="launch-old"))
        report = gr.build_report(root, pid_alive_fn=lambda pid: True)
        self.assertEqual(report["source_freshness"], "stale_previous_cycle")
        self.assertEqual(report["cycle"]["relation_to_launcher"], "previous")
        self.assertIn("PREVIOUS_CYCLE_NOT_CURRENT_LAUNCH", report["warnings"])
        self.assertNotEqual(report["evidence_level"], "real_gpu")

    def test_explicit_g4_g6_pass_is_real_gpu_only_for_current_cycle(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        cycle = base_cycle()
        write_json(root / "last_cycle.json", cycle)
        write_json(
            root / "executions/runs/run-1/shadow_result.json",
            {
                "schema": "gpu-shadow-run-v1",
                "run_id": "run-1",
                "shadow_status": "SHADOW_ADMITTED",
                "production_write_allowed": False,
                "child_payload": {
                    "final_status": "ADMITTED",
                    "g4_status": "PASS",
                    "g6_status": "CANARY_PASS",
                },
            },
        )
        report = gr.build_report(root, pid_alive_fn=lambda pid: False)
        self.assertEqual(report["gpu_validation"]["state"], "passed")
        self.assertEqual(report["gpu_validation"]["g4_status"], "PASS")
        self.assertEqual(report["gpu_validation"]["g6_status"], "CANARY_PASS")
        self.assertEqual(report["evidence_level"], "real_gpu")

    def test_exit_zero_without_explicit_stage_status_is_unknown(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "last_cycle.json", base_cycle())
        write_json(
            root / "executions/runs/run-1/shadow_result.json",
            {
                "schema": "gpu-shadow-run-v1",
                "run_id": "run-1",
                "shadow_status": "SHADOW_ADMITTED",
                "child_returncode": 0,
                "production_write_allowed": False,
                "child_payload": {"final_status": "ADMITTED"},
            },
        )
        report = gr.build_report(root, pid_alive_fn=lambda pid: False)
        self.assertEqual(report["gpu_validation"]["state"], "unknown")
        self.assertIn("GPU_STAGE_STATUS_NOT_EXPLICIT", report["warnings"])
        self.assertNotEqual(report["evidence_level"], "real_gpu")

    def test_explicit_g4_reject_is_failed(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "last_cycle.json", base_cycle())
        write_json(
            root / "executions/runs/run-1/shadow_result.json",
            {
                "run_id": "run-1",
                "shadow_status": "SHADOW_REJECTED",
                "production_write_allowed": False,
                "child_payload": {
                    "final_status": "REJECTED_AT_G4",
                    "g4_status": "REJECTED_AT_G4",
                },
            },
        )
        report = gr.build_report(root, pid_alive_fn=lambda pid: False)
        self.assertEqual(report["gpu_validation"]["state"], "failed")
        self.assertEqual(report["evidence_level"], "real_gpu")

    def test_legacy_history_is_not_context_verified(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "candidate_history.json", {
            "schema": "gpu-shadow-history-v1",
            "candidates": {
                "candidate-1": {
                    "fingerprint": "f" * 64,
                    "shadow_status": "SHADOW_ADMITTED",
                }
            },
        })
        write_json(root / "last_cycle.json", base_cycle(
            status="NO_NEW_READY_CANDIDATE",
            shadow_run_id=None,
            selected_candidate_id=None,
            already_observed_candidate_ids=["candidate-1"],
        ))
        report = gr.build_report(root, pid_alive_fn=lambda pid: False)
        self.assertIn("LEGACY_CONTEXT_UNVERIFIED", report["warnings"])


    def test_manual_review_is_not_run(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "last_cycle.json", base_cycle(
            status="MANUAL_REVIEW_REQUIRED",
            shadow_run_id=None,
            selected_candidate_id=None,
            manual_review_candidate_ids=["candidate-1"],
            dedupe_reason="retry_budget_exhausted",
        ))
        report = gr.build_report(root, pid_alive_fn=lambda pid: False)
        self.assertEqual(report["gpu_validation"]["state"], "not_run")
        self.assertEqual(report["evidence_level"], "not_run")
        self.assertEqual(
            report["cycle"]["dedupe_reason"],
            "retry_budget_exhausted",
        )

    def test_running_dead_pid_is_stale_warning(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "launcher_status.json", base_launch(
            status="RUNNING",
            finished_at=None,
        ))
        write_json(root / "last_cycle.json", base_cycle(
            status="NO_READY_CANDIDATE",
            shadow_run_id=None,
            selected_candidate_id=None,
            ready_count=0,
        ))
        report = gr.build_report(root, pid_alive_fn=lambda pid: False)
        self.assertIn(
            "LAUNCHER_STATE_STALE_PROCESS_NOT_ALIVE",
            report["warnings"],
        )
        self.assertNotEqual(report["evidence_level"], "real_gpu")

    def test_rollback_required_stage_is_failed_not_passed(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "last_cycle.json", base_cycle())
        write_json(
            root / "executions/runs/run-1/shadow_result.json",
            {
                "run_id": "run-1",
                "shadow_status": "SHADOW_ROLLBACK_OR_REGRESSION",
                "production_write_allowed": False,
                "child_payload": {
                    "final_status": "ROLLBACK_REQUIRED",
                    "g4_status": "PASS",
                    "g6_status": "ROLLBACK_REQUIRED",
                },
            },
        )
        report = gr.build_report(root, pid_alive_fn=lambda pid: False)
        self.assertEqual(report["gpu_validation"]["state"], "failed")
        self.assertEqual(
            report["gpu_validation"]["g6_status"],
            "ROLLBACK_REQUIRED",
        )
        self.assertEqual(report["evidence_level"], "real_gpu")

    def test_secret_sentinels_are_not_copied_to_report(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        sentinel = "TOP_SECRET_SENTINEL_ABC"
        launch = base_launch(secret=sentinel, credential_value=sentinel)
        write_json(root / "launcher_status.json", launch)
        write_json(root / "last_cycle.json", base_cycle(
            status="NO_READY_CANDIDATE",
            shadow_run_id=None,
            selected_candidate_id=None,
            ready_count=0,
            secret=sentinel,
        ))
        text = json.dumps(gr.build_report(root, pid_alive_fn=lambda pid: False))
        self.assertNotIn(sentinel, text)

    def test_reporter_is_read_only_for_observed_files(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        write_json(root / "last_cycle.json", base_cycle(
            status="NO_READY_CANDIDATE",
            shadow_run_id=None,
            selected_candidate_id=None,
            ready_count=0,
        ))
        observed = [
            root / "launcher_status.json",
            root / "last_cycle.json",
            root / "candidate_history.json",
            root / "discovery.json",
        ]
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in observed
        }
        gr.build_report(root, pid_alive_fn=lambda pid: False)
        after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in observed
        }
        self.assertEqual(before, after)

    def test_report_output_inside_shadow_root_is_rejected(self):
        td, root = self.setup_root()
        self.addCleanup(td.cleanup)
        with self.assertRaises(gr.ShadowReportError):
            gr._validate_output(str(root / "report.json"), root)


if __name__ == "__main__":
    unittest.main()
