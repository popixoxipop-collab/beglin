#!/usr/bin/env python3
import fcntl
import json
from pathlib import Path
import tempfile
import unittest

import gpu_shadow_pipeline as gp


def ready(candidate_id, role="shared_down_proj", layer=4, n=6):
    return {
        "candidate_id": candidate_id,
        "status": "READY",
        "role": role,
        "layer": layer,
        "n": n,
        "event": {
            "orig_token": 3268,
            "corrected_token": 1224,
            "pos": 16,
            "req": 3,
        },
        "reference": {"emitted_token": 1224},
        "provenance": {
            "manifest": "/prod/discovery/manifest.txt",
            "source_jsonl": "/prod/discovery/events.jsonl",
        },
    }


class ShadowPipelineTests(unittest.TestCase):
    def setup_fs(self, td):
        root = Path(td)
        repo = root / "repo"
        repo.mkdir()
        tools_dir = repo / "tools"
        tools_dir.mkdir()
        for rel in gp.CONTROL_IDENTITY_FILES:
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# identity fixture {rel}\n")
        mirror = root / "mirror"
        (mirror / "discovery").mkdir(parents=True)
        (mirror / "tokens").mkdir(parents=True)
        (mirror / "tokens" / "prompt.i32").write_bytes(b"\0" * (9 * 4))
        (mirror / "discovery" / "manifest.txt").write_text(
            "/prod/tokens/prompt.i32 10\n"
        )
        binary = root / "qwen_infer_gpu"
        binary.write_bytes(b"gpu")
        config = {
            "model": "deepseek-v2-lite",
            "limit": 100,
            "shadow_root": str(root / "shadow"),
            "cwd": str(repo),
            "autopilot": str(repo / "tools" / "gpu_autopilot.py"),
            "binary": str(binary),
            "checkpoint_sha256": "d" * 64,
            "moe_base": str(root / "moe"),
            "safetensors": str(root / "model.safetensors.index.json"),
            "path_maps": [f"/prod={mirror}"],
            "g6_repeats": 3,
            "forbidden_roots": [str(root / "production-control")],
        }
        return root, config

    def test_no_ready_candidate_stops_without_execution(self):
        with tempfile.TemporaryDirectory() as td:
            root, config = self.setup_fs(td)
            called = {"run": 0}
            def fail_run(*args, **kwargs):
                called["run"] += 1
                raise AssertionError("runner must not be called")
            got = gp.run_cycle(
                config,
                discover_fn=lambda model, limit: {
                    "schema": "gpu-shadow-discovery-v1",
                    "mode": "read_only",
                    "production_write_allowed": False,
                    "ready": [],
                },
                run_shadow_fn=fail_run,
            )
            self.assertEqual(got["status"], "NO_READY_CANDIDATE")
            self.assertFalse(got["production_write_allowed"])
            self.assertEqual(called["run"], 0)
            self.assertTrue((root / "shadow" / "discovery.json").exists())

    def test_one_target_is_selected_and_remaining_ready_are_deferred(self):
        with tempfile.TemporaryDirectory() as td:
            root, config = self.setup_fs(td)
            rows = [ready("c1"), ready("c2", role="shared_up_proj", layer=3)]
            captured = {}

            def fake_run(spec, **kwargs):
                captured["spec"] = spec
                captured["kwargs"] = kwargs
                return {
                    "shadow_status": "SHADOW_ADMITTED",
                    "production_write_allowed": False,
                }

            got = gp.run_cycle(
                config,
                discover_fn=lambda model, limit: {
                    "schema": "gpu-shadow-discovery-v1",
                    "mode": "read_only",
                    "production_write_allowed": False,
                    "payload_sha256": "c" * 64,
                    "ready": rows,
                },
                run_shadow_fn=fake_run,
            )
            self.assertEqual(got["status"], "SHADOW_CYCLE_COMPLETE")
            self.assertEqual(got["selected_candidate_id"], "c1")
            self.assertEqual(got["deferred_candidate_ids"], ["c2"])
            self.assertFalse(got["production_write_allowed"])
            self.assertEqual(captured["spec"]["candidate_id"], "c1")
            self.assertEqual(captured["spec"]["prompt_len"], 9)
            self.assertFalse(captured["spec"]["production_write_allowed"])
            self.assertIn("/executions", captured["kwargs"]["shadow_root"])
            saved = json.loads((root / "shadow" / "last_cycle.json").read_text())
            self.assertEqual(saved["selected_candidate_id"], "c1")


    def test_same_evidence_is_executed_once_until_candidate_or_runtime_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root, config = self.setup_fs(td)
            row = ready("c1")
            calls = {"run": 0}

            def discover(model, limit):
                return {
                    "schema": "gpu-shadow-discovery-v1",
                    "mode": "read_only",
                    "production_write_allowed": False,
                    "payload_sha256": "c" * 64,
                    "ready": [dict(row)],
                }

            def fake_run(spec, **kwargs):
                calls["run"] += 1
                return {
                    "shadow_status": "SHADOW_ADMITTED",
                    "production_write_allowed": False,
                    "result_sha256": "r" * 64,
                }

            first = gp.run_cycle(
                config,
                discover_fn=discover,
                run_shadow_fn=fake_run,
            )
            second = gp.run_cycle(
                config,
                discover_fn=discover,
                run_shadow_fn=fake_run,
            )
            self.assertEqual(first["status"], "SHADOW_CYCLE_COMPLETE")
            self.assertEqual(second["status"], "NO_NEW_READY_CANDIDATE")
            self.assertEqual(second["already_observed_candidate_ids"], ["c1"])
            self.assertEqual(calls["run"], 1)
            self.assertEqual(
                first["runtime_identity_sha256"],
                second["runtime_identity_sha256"],
            )

            # Same production evidence but a different certified runtime must
            # be revalidated instead of reusing the old shadow verdict.
            Path(config["binary"]).write_bytes(b"gpu-v2")
            third = gp.run_cycle(
                config,
                discover_fn=discover,
                run_shadow_fn=fake_run,
            )
            self.assertEqual(third["status"], "SHADOW_CYCLE_COMPLETE")
            self.assertNotEqual(
                first["runtime_identity_sha256"],
                third["runtime_identity_sha256"],
            )
            self.assertEqual(calls["run"], 2)

            fourth = gp.run_cycle(
                config,
                discover_fn=discover,
                run_shadow_fn=fake_run,
            )
            self.assertEqual(fourth["status"], "NO_NEW_READY_CANDIDATE")
            self.assertEqual(calls["run"], 2)

            row["event_count"] = 99
            fifth = gp.run_cycle(
                config,
                discover_fn=discover,
                run_shadow_fn=fake_run,
            )
            self.assertEqual(fifth["status"], "SHADOW_CYCLE_COMPLETE")
            self.assertEqual(calls["run"], 3)


    def test_control_plane_code_change_invalidates_dedupe(self):
        with tempfile.TemporaryDirectory() as td:
            root, config = self.setup_fs(td)
            row = ready("c1")
            calls = {"run": 0}

            def discover(model, limit):
                return {
                    "schema": "gpu-shadow-discovery-v1",
                    "mode": "read_only",
                    "production_write_allowed": False,
                    "payload_sha256": "c" * 64,
                    "ready": [dict(row)],
                }

            def fake_run(spec, **kwargs):
                calls["run"] += 1
                return {
                    "shadow_status": "SHADOW_ADMITTED",
                    "production_write_allowed": False,
                    "result_sha256": str(calls["run"]) * 64,
                }

            first = gp.run_cycle(
                config,
                discover_fn=discover,
                run_shadow_fn=fake_run,
            )
            second = gp.run_cycle(
                config,
                discover_fn=discover,
                run_shadow_fn=fake_run,
            )
            self.assertEqual(second["status"], "NO_NEW_READY_CANDIDATE")

            autopilot = Path(config["cwd"]) / "tools" / "gpu_autopilot.py"
            autopilot.write_text("# changed certified autopilot\n")
            third = gp.run_cycle(
                config,
                discover_fn=discover,
                run_shadow_fn=fake_run,
            )
            self.assertEqual(third["status"], "SHADOW_CYCLE_COMPLETE")
            self.assertNotEqual(
                first["runtime_identity_sha256"],
                third["runtime_identity_sha256"],
            )
            self.assertEqual(calls["run"], 2)

    def test_concurrent_cycle_lock_fails_before_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            root, config = self.setup_fs(td)
            shadow = Path(config["shadow_root"])
            shadow.mkdir(parents=True)
            lock_path = shadow / ".cycle.lock"
            handle = open(lock_path, "a+")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            called = {"discover": 0}

            def discover(model, limit):
                called["discover"] += 1
                return {}

            try:
                with self.assertRaises(gp.ShadowPipelineError):
                    gp.run_cycle(config, discover_fn=discover)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            self.assertEqual(called["discover"], 0)

    def test_corrupt_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root, config = self.setup_fs(td)
            shadow = Path(config["shadow_root"])
            shadow.mkdir(parents=True)
            (shadow / "candidate_history.json").write_text(
                json.dumps({"schema": "wrong", "candidates": {}})
            )
            with self.assertRaises(gp.ShadowPipelineError):
                gp.run_cycle(
                    config,
                    discover_fn=lambda model, limit: {
                        "schema": "gpu-shadow-discovery-v1",
                        "mode": "read_only",
                        "production_write_allowed": False,
                        "ready": [ready("c1")],
                    },
                )

    def test_shadow_root_overlap_fails_before_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            root, config = self.setup_fs(td)
            config["shadow_root"] = str(Path(config["cwd"]) / "shadow")
            called = {"discover": 0}
            def discover(*args, **kwargs):
                called["discover"] += 1
                return {}
            with self.assertRaises(Exception):
                gp.run_cycle(config, discover_fn=discover)
            self.assertEqual(called["discover"], 0)

    def test_missing_runtime_config_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            _, config = self.setup_fs(td)
            del config["binary"]
            with self.assertRaises(gp.ShadowPipelineError):
                gp.run_cycle(
                    config,
                    discover_fn=lambda model, limit: {
                        "schema": "gpu-shadow-discovery-v1",
                        "mode": "read_only",
                        "production_write_allowed": False,
                        "ready": [ready("c1")],
                    },
                )


if __name__ == "__main__":
    unittest.main()
