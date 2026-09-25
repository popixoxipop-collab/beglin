#!/usr/bin/env python3
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import checkpoint_identity as ci
import checkpoint_identity_xox as cx


class FakePopen:
    pid = 4242


class XoxCheckpointVerifierTests(unittest.TestCase):
    def make_checkpoint(self, root: Path):
        shard = root / "model-00001-of-00001.safetensors"
        shard.write_bytes(b"checkpoint-data" * 1024)
        index = root / "model.safetensors.index.json"
        index.write_text(json.dumps({
            "weight_map": {"layer.weight": shard.name},
        }))
        return index

    def test_worker_once_runs_two_pass_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = self.make_checkpoint(root)
            output = root / "out" / "checkpoint_identity.json"
            got = cx.worker_once(
                checkpoint=index,
                approved_root=root,
                output=output,
            )
            expected = ci.verify_checkpoint_identity(
                index,
                allowed_root=root,
                passes=2,
            )
            self.assertEqual(got["checkpoint_sha256"], expected["checkpoint_sha256"])
            self.assertEqual(len(got["verification_passes"]), 2)
            self.assertEqual(
                json.loads(output.read_text())["status"],
                "VERIFIED",
            )

    def test_minimal_env_does_not_forward_unrelated_values(self):
        with patch.dict(
            os.environ,
            {
                "PATH": "/bin",
                "HOME": "/home/test",
                "SECRET_OTHER": "must-not-pass",
                "QWEN_SUPABASE_KEY": "must-not-pass",
            },
            clear=True,
        ):
            got = cx._minimal_env()
        self.assertEqual(got["PATH"], "/bin")
        self.assertNotIn("SECRET_OTHER", got)
        self.assertNotIn("QWEN_SUPABASE_KEY", got)

    def test_launch_refuses_duplicate_running_worker(self):
        with patch.object(
            cx,
            "_load_status",
            return_value={"status": "RUNNING", "pid": 123},
        ), patch.object(cx, "_pid_alive", return_value=True):
            with self.assertRaises(cx.XoxCheckpointVerifyError):
                cx.launch()

    def test_status_not_started_contains_no_production_write(self):
        with patch.object(cx, "_load_status", return_value=None):
            got = cx.status()
        self.assertEqual(got["status"], "NOT_STARTED")
        self.assertFalse(got["production_write_allowed"])

    def test_status_running_reports_liveness_without_file_contents(self):
        with patch.object(
            cx,
            "_load_status",
            return_value={
                "schema": "xox-checkpoint-verifier-v1",
                "status": "RUNNING",
                "pid": 123,
                "production_write_allowed": False,
            },
        ), patch.object(cx, "_pid_alive", return_value=True):
            got = cx.status()
        self.assertTrue(got["pid_alive"])
        self.assertNotIn("checkpoint_path", got)
        self.assertFalse(got["production_write_allowed"])


if __name__ == "__main__":
    unittest.main()
