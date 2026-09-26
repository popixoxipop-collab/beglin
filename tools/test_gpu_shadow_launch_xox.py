#!/usr/bin/env python3
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gpu_shadow_launch_xox as gl


class FakePopen:
    pid = 4242


class ShadowXoxLaunchTests(unittest.TestCase):
    def test_whitelist_parser_reads_only_two_keys_without_shell_eval(self):
        with tempfile.TemporaryDirectory() as td:
            env = Path(td) / "env"
            env.write_text(
                "IGNORED=$(touch /tmp/should-not-run)\n"
                "export QWEN_SUPABASE_URL='https://example.supabase.co'\n"
                'QWEN_SUPABASE_KEY="secret-value"\n'
                "OTHER=ignored\n"
            )
            got = gl.load_credentials(env)
            self.assertEqual(
                got,
                {
                    "QWEN_SUPABASE_URL": "https://example.supabase.co",
                    "QWEN_SUPABASE_KEY": "secret-value",
                },
            )

    def test_missing_required_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            env = Path(td) / "env"
            env.write_text("QWEN_SUPABASE_URL=https://example.supabase.co\n")
            with self.assertRaises(gl.ShadowLaunchError):
                gl.load_credentials(env)

    def test_conflicting_duplicate_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            env = Path(td) / "env"
            env.write_text(
                "QWEN_SUPABASE_URL=https://a.supabase.co\n"
                "QWEN_SUPABASE_URL=https://b.supabase.co\n"
                "QWEN_SUPABASE_KEY=x\n"
            )
            with self.assertRaises(gl.ShadowLaunchError):
                gl.load_credentials(env)

    def test_symlink_env_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "real"
            target.write_text(
                "QWEN_SUPABASE_URL=https://example.supabase.co\n"
                "QWEN_SUPABASE_KEY=x\n"
            )
            link = root / "env"
            link.symlink_to(target)
            with self.assertRaises(gl.ShadowLaunchError):
                gl.load_credentials(link)

    def test_child_env_contains_only_whitelisted_runtime_and_credentials(self):
        with patch.dict(
            os.environ,
            {
                "PATH": "/bin",
                "HOME": "/home/x",
                "SECRET_OTHER": "must-not-pass",
            },
            clear=True,
        ):
            got = gl._safe_child_env({
                "QWEN_SUPABASE_URL": "https://example.supabase.co",
                "QWEN_SUPABASE_KEY": "secret",
            })
        self.assertNotIn("SECRET_OTHER", got)
        self.assertEqual(got["QWEN_SUPABASE_KEY"], "secret")
        self.assertEqual(got["PATH"], "/bin")


    def test_summarize_last_cycle_marks_current_previous_and_legacy(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "last_cycle.json"

            path.write_text(json.dumps({
                "status": "SHADOW_CYCLE_COMPLETE",
                "launch_id": "launch-1",
                "cycle_id": "cycle-1",
                "shadow_run_id": "run-1",
                "ready_count": 1,
            }))
            current = gl._summarize_last_cycle(path, "launch-1")
            self.assertEqual(current["relation"], "CURRENT")
            self.assertEqual(current["cycle_id"], "cycle-1")
            self.assertEqual(current["shadow_run_id"], "run-1")

            previous = gl._summarize_last_cycle(path, "launch-2")
            self.assertEqual(previous["relation"], "PREVIOUS")

            path.write_text(json.dumps({
                "status": "NO_NEW_READY_CANDIDATE",
                "ready_count": 1,
            }))
            legacy = gl._summarize_last_cycle(path, "launch-2")
            self.assertEqual(legacy["relation"], "LEGACY_UNLINKED")

    def test_summarize_last_cycle_handles_corrupt_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "last_cycle.json"
            path.write_text("{not-json")
            got = gl._summarize_last_cycle(path, "launch-1")
            self.assertEqual(got["relation"], "UNREADABLE")
            self.assertEqual(got["status"], "UNREADABLE_LAST_CYCLE")


    def test_launch_generates_id_and_passes_it_only_to_worker_env(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "launcher_status.json"
            log = root / "cycle.log"
            config = root / "config.json"
            config.write_text("{}")

            captured = {}
            class PopenFixture:
                pid = 4242
                def __init__(self, *args, **kwargs):
                    captured["args"] = args
                    captured["kwargs"] = kwargs

            with (
                patch.object(gl, "STATUS_FILE", status),
                patch.object(gl, "LOG_FILE", log),
                patch.object(gl, "DEFAULT_SHADOW_ROOT", root),
                patch.object(gl, "load_config", return_value={}),
                patch.object(
                    gl,
                    "load_credentials",
                    return_value={
                        "QWEN_SUPABASE_URL": "https://example.supabase.co",
                        "QWEN_SUPABASE_KEY": "secret",
                    },
                ),
                patch.object(gl.subprocess, "Popen", PopenFixture),
                patch.object(
                    gl.uuid,
                    "uuid4",
                    return_value=SimpleNamespace(hex="launch-fixture-1"),
                ),
            ):
                got = gl.launch(config_path=config, python_bin="python3")

            self.assertEqual(got["launch_id"], "launch-fixture-1")
            self.assertEqual(got["worker_pid"], 4242)
            self.assertEqual(
                captured["kwargs"]["env"]["GPU_SHADOW_LAUNCH_ID"],
                "launch-fixture-1",
            )
            saved = json.loads(status.read_text())
            self.assertEqual(saved["launch_id"], "launch-fixture-1")
            self.assertNotIn("secret", status.read_text())

    def test_worker_marks_mismatched_last_cycle_as_previous(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "launcher_status.json"
            log = root / "cycle.log"
            last = root / "last_cycle.json"
            last.write_text(json.dumps({
                "schema": "gpu-shadow-pipeline-v1",
                "status": "SHADOW_CYCLE_COMPLETE",
                "launch_id": "launch-old",
                "cycle_id": "cycle-old",
                "shadow_run_id": "run-old",
                "ready_count": 1,
                "selected_candidate_id": "candidate-old",
                "shadow_run": {"shadow_status": "SHADOW_ADMITTED"},
            }))

            proc = SimpleNamespace(returncode=0, stdout="", stderr="")
            with (
                patch.object(gl, "STATUS_FILE", status),
                patch.object(gl, "LOG_FILE", log),
                patch.object(gl, "DEFAULT_SHADOW_ROOT", root),
                patch.object(gl.subprocess, "run", return_value=proc),
                patch.dict(
                    os.environ,
                    {"GPU_SHADOW_LAUNCH_ID": "launch-new"},
                    clear=False,
                ),
            ):
                rc = gl.worker("/tmp/config.json", python_bin="python3")

            self.assertEqual(rc, 0)
            saved = json.loads(status.read_text())
            self.assertEqual(saved["launch_id"], "launch-new")
            self.assertIsNone(saved["cycle_id"])
            self.assertIsNone(saved["shadow_run_id"])
            self.assertEqual(saved["last_cycle"]["relation"], "PREVIOUS")

    def test_status_payload_never_contains_credential_values(self):
        with tempfile.TemporaryDirectory() as td:
            payload = {
                "schema": "gpu-shadow-xox-launch-v1",
                "status": "RUNNING",
                "credential_keys_loaded": sorted(gl.REQUIRED_KEYS),
            }
            text = json.dumps(payload)
            self.assertNotIn("secret-value", text)
            self.assertNotIn("https://example.supabase.co", text)


if __name__ == "__main__":
    unittest.main()
