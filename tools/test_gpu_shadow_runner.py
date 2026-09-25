#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpu_shadow_runner as gs


def spec(root):
    repo = Path(root) / "repo"
    repo.mkdir(exist_ok=True)
    tools_dir = repo / "tools"
    tools_dir.mkdir(exist_ok=True)
    (tools_dir / "gpu_autopilot.py").write_text("# certified test autopilot\n")
    return {
        "role": "shared_down_proj",
        "layer": 4,
        "n": 6,
        "event": {"orig_token": 3268, "corrected_token": 1224, "pos": 16},
        "reference": {"emitted_token": 1224},
        "prompt_len": 9,
        "g4_manifest": str(Path(root) / "g4.txt"),
        "g6_manifest": str(Path(root) / "g6.txt"),
        "cwd": str(repo),
        "binary": str(Path(root) / "qwen_infer_gpu"),
        "binary_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "moe_base": str(Path(root) / "moe"),
        "safetensors": str(Path(root) / "model.safetensors.index.json"),
    }


class FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class GpuShadowRunnerTests(unittest.TestCase):
    def test_child_env_scrubs_all_mutation_variables(self):
        inherited = {key: "/prod/live" for key in gs.MUTATION_ENV}
        inherited["KEEP_ME"] = "yes"
        got = gs.build_child_env(inherited)
        self.assertEqual(got["KEEP_ME"], "yes")
        self.assertEqual(got["QWEN_GPU_SHADOW_MODE"], "1")
        self.assertEqual(got["QWEN_GPU_PRODUCTION_WRITE"], "0")
        for key in gs.MUTATION_ENV:
            self.assertNotIn(key, got)

    def test_shadow_root_cannot_overlap_engine_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            with self.assertRaises(gs.ShadowModeError):
                gs.validate_shadow_root(
                    str(Path(s["cwd"]) / "shadow"),
                    candidate_cwd=s["cwd"],
                )

    def test_shadow_root_cannot_overlap_forbidden_control_root(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            production = Path(td) / "production-control"
            with self.assertRaises(gs.ShadowModeError):
                gs.validate_shadow_root(
                    str(production / "nested"),
                    candidate_cwd=s["cwd"],
                    forbidden_roots=[str(production)],
                )

    def test_shadow_root_parent_of_forbidden_control_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            shadow_parent = Path(td) / "all-control"
            production = shadow_parent / "production"
            with self.assertRaises(gs.ShadowModeError):
                gs.validate_shadow_root(
                    str(shadow_parent),
                    candidate_cwd=s["cwd"],
                    forbidden_roots=[str(production)],
                )

    def test_command_always_forces_supplied_scratch_control_root(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            control = Path(td) / "shadow" / "runs" / "r1" / "control"
            cmd = gs.build_autopilot_command(
                s,
                autopilot=str(Path(s["cwd"]) / "tools" / "gpu_autopilot.py"),
                control_root=control,
                python_bin="python3",
            )
            self.assertEqual(cmd[0], "python3")
            self.assertIn("--control-root", cmd)
            idx = cmd.index("--control-root")
            self.assertEqual(cmd[idx + 1], str(control))
            self.assertNotIn("/private/tmp/qng64_ctl", cmd)

    def test_admitted_child_is_shadow_admitted_not_production_approval(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            shadow = Path(td) / "shadow"
            captured = {}

            def fake_run(command, **kwargs):
                captured["command"] = command
                captured["env"] = kwargs["env"]
                return FakeProc(
                    0,
                    stdout=json.dumps({"status": "ADMITTED"}) + "\n",
                )

            with patch.object(gs.subprocess, "run", side_effect=fake_run):
                got = gs.run_shadow(
                    s,
                    shadow_root=str(shadow),
                    autopilot=str(Path(s["cwd"]) / "tools" / "gpu_autopilot.py"),
                    python_bin="python3",
                    run_id="run-1",
                    forbidden_roots=[str(Path(td) / "prod")],
                    base_env={
                        "QWEN_PRECISION_CONTROL_DIR": "/prod/live",
                        "QWEN_MOE_GPU_TXN_FILE": "/prod/txn",
                        "QWEN_SUPABASE_KEY": "secret",
                    },
                )

            self.assertEqual(got["shadow_status"], "SHADOW_ADMITTED")
            self.assertFalse(got["production_write_allowed"])
            self.assertNotIn("QWEN_PRECISION_CONTROL_DIR", captured["env"])
            self.assertNotIn("QWEN_MOE_GPU_TXN_FILE", captured["env"])
            self.assertNotIn("QWEN_SUPABASE_KEY", captured["env"])
            self.assertIn("--control-root", captured["command"])
            result_path = shadow / "runs" / "run-1" / "shadow_result.json"
            saved = json.loads(result_path.read_text())
            self.assertEqual(saved["shadow_status"], "SHADOW_ADMITTED")
            self.assertFalse(saved["production_write_allowed"])


    def test_run_shadow_rejects_noncertified_autopilot_path(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            rogue = Path(td) / "rogue.py"
            rogue.write_text("print('rogue')\n")
            with self.assertRaises(gs.ShadowModeError):
                gs.run_shadow(
                    s,
                    shadow_root=str(Path(td) / "shadow"),
                    autopilot=str(rogue),
                    run_id="rogue-run",
                )

    def test_run_shadow_rejects_symlinked_certified_autopilot(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            expected = Path(s["cwd"]) / "tools" / "gpu_autopilot.py"
            expected.unlink()
            target = Path(td) / "real-autopilot.py"
            target.write_text("# outside repo\n")
            expected.symlink_to(target)
            with self.assertRaises(gs.ShadowModeError):
                gs.run_shadow(
                    s,
                    shadow_root=str(Path(td) / "shadow"),
                    autopilot=str(expected),
                    run_id="symlink-run",
                )

    def test_pretty_printed_child_json_is_classified(self):
        payload = {"status": "ADMITTED", "nested": {"ok": True}}
        got = gs._extract_last_json("prefix log\n" + json.dumps(payload, indent=2) + "\n")
        self.assertEqual(got, payload)
        self.assertEqual(gs._shadow_status(0, got), "SHADOW_ADMITTED")

    def test_rejected_child_stays_shadow_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            with patch.object(
                gs.subprocess,
                "run",
                return_value=FakeProc(
                    0,
                    stdout=json.dumps({"status": "REJECTED_AT_G4"}) + "\n",
                ),
            ):
                got = gs.run_shadow(
                    s,
                    shadow_root=str(Path(td) / "shadow"),
                    autopilot=str(Path(s["cwd"]) / "tools" / "gpu_autopilot.py"),
                    run_id="run-2",
                )
            self.assertEqual(got["shadow_status"], "SHADOW_REJECTED")
            self.assertFalse(got["production_write_allowed"])

    def test_nonzero_child_is_shadow_error(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            with patch.object(
                gs.subprocess,
                "run",
                return_value=FakeProc(7, stderr="simulated failure"),
            ):
                got = gs.run_shadow(
                    s,
                    shadow_root=str(Path(td) / "shadow"),
                    autopilot=str(Path(s["cwd"]) / "tools" / "gpu_autopilot.py"),
                    run_id="run-3",
                )
            self.assertEqual(got["shadow_status"], "SHADOW_ERROR")
            self.assertEqual(got["child_returncode"], 7)
            self.assertFalse(got["production_write_allowed"])

    def test_invalid_hash_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            s["binary_sha256"] = "not-a-hash"
            with self.assertRaises(gs.ShadowModeError):
                gs.validate_spec(s)

    def test_missing_event_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            s = spec(td)
            del s["event"]["pos"]
            with self.assertRaises(gs.ShadowModeError):
                gs.validate_spec(s)


if __name__ == "__main__":
    unittest.main()
