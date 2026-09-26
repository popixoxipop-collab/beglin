#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import agent_f_gpu_coverage_xox as f


class AgentFCoverageTests(unittest.TestCase):
    def test_candidate_classification(self):
        self.assertEqual(
            f.classify_candidate(
                returncode=0,
                requested_policy_applied=True,
                request_count=10,
                corrected_hits=10,
            ),
            "GOOD",
        )
        self.assertEqual(
            f.classify_candidate(
                returncode=0,
                requested_policy_applied=True,
                request_count=10,
                corrected_hits=0,
            ),
            "BAD",
        )
        self.assertEqual(
            f.classify_candidate(
                returncode=0,
                requested_policy_applied=True,
                request_count=9,
                corrected_hits=9,
            ),
            "INCONCLUSIVE",
        )
        self.assertEqual(
            f.classify_candidate(
                returncode=1,
                requested_policy_applied=True,
                request_count=10,
                corrected_hits=10,
            ),
            "UNSUPPORTED_OR_RUNTIME_ERROR",
        )
        self.assertEqual(
            f.classify_candidate(
                returncode=0,
                requested_policy_applied=False,
                request_count=10,
                corrected_hits=10,
            ),
            "UNSUPPORTED_OR_RUNTIME_ERROR",
        )

    def test_choose_bad_candidate_is_bounded_and_ordered(self):
        rows = [
            {"n": 5, "classification": "UNSUPPORTED_OR_RUNTIME_ERROR"},
            {"n": 6, "classification": "BAD"},
            {"n": 7, "classification": "GOOD"},
        ]
        self.assertEqual(f.choose_bad_candidate(rows)["n"], 6)
        self.assertIsNone(
            f.choose_bad_candidate(
                [
                    {"n": 5, "classification": "INCONCLUSIVE"},
                    {"n": 6, "classification": "GOOD"},
                ]
            )
        )

    def test_make_manifest_repeats_only_first_verified_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "prompt.i32"
            raw.write_bytes(b"\0" * 40)
            source = root / "source.txt"
            source.write_text(
                f"{raw} 1\n"
                f"{root / 'not-used.i32'} 7\n"
            )
            dest = root / "out" / "manifest.txt"
            meta = f._make_manifest(source, dest, 10)
            rows = dest.read_text().splitlines()
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row == f"{raw} 1" for row in rows))
            self.assertEqual(meta["requests"], 10)
            self.assertEqual(meta["raw_token_file"], str(raw))

    def test_manifest_repeat_budget_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "prompt.i32"
            raw.write_bytes(b"\0" * 4)
            source = root / "source.txt"
            source.write_text(f"{raw} 1\n")
            with self.assertRaises(f.CoverageError):
                f._make_manifest(source, root / "bad0.txt", 0)
            with self.assertRaises(f.CoverageError):
                f._make_manifest(source, root / "bad101.txt", 101)

    def test_token_counts_matches_absolute_event_contract(self):
        target = {
            "orig_token": 8713,
            "corrected_token": 4794,
            "pos": 14,
            "prompt_len": 15,
        }
        reqs = {
            0: [8713],
            1: [4794],
            2: [4794],
        }
        got = f._token_counts(reqs, target)
        self.assertEqual(got["gen_idx"], 0)
        self.assertEqual(got["eligible_requests"], 3)
        self.assertEqual(got["orig_hits"], 1)
        self.assertEqual(got["corrected_hits"], 2)
        self.assertEqual(got["distinct_tokens"], [4794, 8713])

    def test_checkpoint_evidence_requires_verified_two_pass_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = root / "checkpoint_identity.json"
            payload = {
                "status": "VERIFIED",
                "checkpoint_sha256": "a" * 64,
                "verification_passes": [
                    {"checkpoint_sha256": "a" * 64},
                    {"checkpoint_sha256": "a" * 64},
                ],
            }
            evidence.write_text(json.dumps(payload, sort_keys=True))
            result_sha = f._sha256_file(evidence)
            got = f._read_checkpoint_evidence(
                evidence,
                expected_checkpoint_sha="a" * 64,
                expected_result_sha=result_sha,
            )
            self.assertEqual(got["status"], "VERIFIED")

            payload["verification_passes"][1]["checkpoint_sha256"] = "b" * 64
            evidence.write_text(json.dumps(payload, sort_keys=True))
            with self.assertRaises(f.CoverageError):
                f._read_checkpoint_evidence(
                    evidence,
                    expected_checkpoint_sha="a" * 64,
                    expected_result_sha=f._sha256_file(evidence),
                )

    def test_minimal_env_does_not_forward_credentials(self):
        with patch.dict(
            f.os.environ,
            {
                "PATH": "/bin",
                "HOME": "/home/test",
                "QWEN_SUPABASE_KEY": "do-not-forward",
                "SECRET_OTHER": "do-not-forward",
            },
            clear=True,
        ):
            got = f._minimal_env()
        self.assertEqual(got["PATH"], "/bin")
        self.assertNotIn("QWEN_SUPABASE_KEY", got)
        self.assertNotIn("SECRET_OTHER", got)

    def test_launch_refuses_duplicate_running_worker(self):
        with patch.object(
            f,
            "_load_status",
            return_value={"status": "RUNNING", "pid": 123},
        ), patch.object(f, "_pid_alive", return_value=True):
            with self.assertRaises(f.CoverageError):
                f.launch()

    def test_terminal_status_exposes_sanitized_matrix_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            matrix = root / "latest.json"
            matrix.write_text(json.dumps({
                "status": "PARTIAL_NO_KVA_BAD_CANDIDATE",
                "run_id": "r1",
                "source_head": "a" * 40,
                "binary_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "process_launches": 9,
                "production_write_allowed": False,
                "kva_candidate_scan": [
                    {
                        "n": 5,
                        "classification": "GOOD",
                        "requests_completed": 12,
                        "counts": {"corrected_hits": 12},
                        "requested_policy_applied": True,
                        "run_dir": "/private/path/not-for-summary",
                    }
                ],
                "g5_kva_rollback": {
                    "status": "NO_BAD_CANDIDATE_WITHIN_BUDGET",
                },
                "g6_kva": {
                    "positive": {
                        "result": {
                            "status": "CANARY_PASS",
                            "decision": "CANARY_PASS_NO_AUTO_EXPANSION",
                            "rollback_required": False,
                            "auto_expand": False,
                        },
                        "reused_hardware_run": False,
                        "source_run": "kva_n7_slots4",
                    },
                },
                "g6_shared_up": {
                    "regression": {
                        "result": {
                            "status": "REGRESSION_DETECTED",
                            "decision": "ROLLBACK_REQUIRED",
                            "rollback_required": True,
                            "auto_expand": False,
                        },
                        "reused_hardware_run": False,
                        "source_run": "shared_up_n7_slots4",
                        "real_bad_candidate": True,
                    },
                },
                "batch_regression": {
                    "slots1_equals_slots2": True,
                    "slots2_equals_slots4": True,
                },
            }, sort_keys=True))
            with patch.object(f, "LATEST_RESULT", matrix):
                summary = f._latest_public_summary()
            self.assertEqual(
                summary["g5_kva_status"],
                "NO_BAD_CANDIDATE_WITHIN_BUDGET",
            )
            self.assertEqual(
                summary["g6_shared_up"]["regression"]["decision"],
                "ROLLBACK_REQUIRED",
            )
            self.assertTrue(
                summary["batch_regression"]["slots1_equals_slots2"]
            )
            self.assertEqual(len(summary["matrix_sha256"]), 64)
            self.assertNotIn(
                "run_dir",
                summary["kva_candidate_scan"][0],
            )

    def test_terminal_status_attaches_public_summary(self):
        with patch.object(
            f,
            "_load_status",
            return_value={
                "status": "COMPLETE",
                "production_write_allowed": False,
            },
        ), patch.object(
            f,
            "_latest_public_summary",
            return_value={"matrix_sha256": "d" * 64},
        ):
            got = f.status()
        self.assertEqual(
            got["coverage_summary"]["matrix_sha256"],
            "d" * 64,
        )
        self.assertFalse(got["production_write_allowed"])

    def test_progress_enforces_process_launch_budget(self):
        with tempfile.TemporaryDirectory() as td:
            p = f.Progress("test-run", Path(td))
            p.launches = f.MAX_PROCESS_LAUNCHES
            with self.assertRaises(f.CoverageError):
                p.consume_launch("one-too-many")


if __name__ == "__main__":
    unittest.main()
