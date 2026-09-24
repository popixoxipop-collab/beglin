#!/usr/bin/env python3
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import gpu_isolated_preflight as gp
import precision_context as pc


BASE = []
CAND = [{"role": "shared_down_proj", "layer": 4, "n": 6}]


class FakeProc:
    def __init__(self, rc=0, stdout="", stderr=""):
        self.returncode = rc
        self.stdout = stdout
        self.stderr = stderr


class GpuIsolatedPreflightTests(unittest.TestCase):
    def test_render_policy_is_canonical(self):
        rows = [
            {"role": "shared_up_proj", "layer": 3, "n": 5},
            {"role": "shared_down_proj", "layer": 4, "n": 6},
        ]
        self.assertEqual(
            gp.render_promotion_file(list(reversed(rows))),
            "shared_down_proj 4 6\nshared_up_proj 3 5\n",
        )

    def test_first_release_rejects_unapproved_width(self):
        with self.assertRaises(gp.GpuPreflightError):
            gp.normalize_policy([
                {"role": "shared_down_proj", "layer": 4, "n": 9}
            ])

    def test_parse_mla_and_gqa_tokens(self):
        mla = (
            "[moe gpu cb online] req 0 prompt 4 slot 0 arrive 0 "
            "admit_step 0 ttft_ms 1.0 nout 3 tokens: 10 11 12\n"
        )
        gqa = (
            "[moe gpu gqa cb online] req 0 prompt 4 slot 0 arrive 0 "
            "admit_step 0 ttft_ms 1.0 nout 2 tokens: 20 21\n"
        )
        self.assertEqual(gp.parse_emitted_tokens(mla, "mla"), [10, 11, 12])
        self.assertEqual(gp.parse_emitted_tokens(gqa, "gqa"), [20, 21])
        self.assertEqual(
            gp.emitted_token_at_position(
                mla, architecture="mla", prompt_len=4, pos=4
            ),
            11,
        )

    def test_worker_env_forces_correction_off_and_selects_one_gate(self):
        env = gp._worker_env(
            architecture="mla",
            moe_base="/m",
            manifest="/manifest",
            promotion_file="/p",
            safetensors="/st",
            ack_path="/ack",
        )
        self.assertEqual(env["QWEN_MOE_NEARTIE_CORRECT"], "0")
        self.assertEqual(env["QWEN_MOE_GPU_CBATCH_ONLINE"], "1")
        self.assertNotIn("QWEN_MOE_GPU_GQA_CBATCH_ONLINE", env)
        self.assertEqual(env["QWEN_MOE_PROMOTION_SAFETENSORS"], "/st")
        self.assertEqual(env["QWEN_MOE_GPU_VALIDATION_REPORT"], "1")

    def test_parse_validation_report_requires_matching_finite_gpu_report(self):
        got = gp.parse_validation_report(
            "GPU_VALIDATION_V1 backend=mlx_metal arch=mla correction=off "
            "finite_logits=1 logits_checked=64000 requests=1\n",
            "mla",
        )
        self.assertTrue(got["finite_logits"])
        self.assertEqual(got["logits_checked"], 64000)
        self.assertEqual(got["requests"], 1)
        with self.assertRaises(gp.GpuPreflightError):
            gp.parse_validation_report("", "mla")
        with self.assertRaises(gp.GpuPreflightError):
            gp.parse_validation_report(
                "GPU_VALIDATION_V1 backend=mlx_metal arch=gqa correction=off "
                "finite_logits=1 logits_checked=1 requests=1\n",
                "mla",
            )

    def test_parse_ack_requires_runtime_correction_mode(self):
        base = {
            "schema": gp.ACK_SCHEMA,
            "status": "STARTUP_STATE",
            "backend": "mlx_metal",
            "weight_epoch": 0,
            "changed_targets": 0,
            "snapshot_count": 0,
            "active_policy": [],
        }
        with self.assertRaises(gp.GpuPreflightError):
            gp._parse_ack_text(json.dumps(base))
        base["correction_mode"] = "off"
        got = gp._parse_ack_text(json.dumps(base))
        self.assertEqual(got["active_policy_hash"], pc.policy_hash([]))

    @patch.object(gp, "_binary_identity")
    @patch.object(gp, "_read_text")
    @patch.object(gp, "_run")
    @patch.object(gp, "_remove")
    @patch.object(gp, "_write_text")
    def test_run_worker_proves_empty_baseline(
        self, write, remove, run, read, identity
    ):
        identity.return_value = {
            "host": "XOX.local",
            "arch": "arm64",
            "binary_path": "/bin/q",
            "binary_sha256": "a" * 64,
            "binary_size": 123,
        }
        run.return_value = FakeProc(
            stdout=(
                "[moe gpu cb online] req 0 prompt 4 slot 0 arrive 0 "
                "admit_step 0 ttft_ms 1.0 nout 1 tokens: 42\n"
                "GPU_VALIDATION_V1 backend=mlx_metal arch=mla correction=off "
                "finite_logits=1 logits_checked=100 requests=1\n"
            )
        )
        read.return_value = json.dumps({
            "schema": gp.ACK_SCHEMA,
            "status": "STARTUP_STATE",
            "backend": "mlx_metal",
            "correction_mode": "off",
            "weight_epoch": 0,
            "changed_targets": 0,
            "snapshot_count": 0,
            "active_policy": [],
        })
        got = gp.run_isolated_worker(
            host="xox",
            cwd="/repo",
            binary="/bin/q",
            architecture="mla",
            moe_base="/m",
            manifest="/manifest",
            safetensors="/st",
            run_dir="/tmp/run",
            policy=[],
        )
        self.assertEqual(got["weight_epoch"], 0)
        self.assertEqual(got["applied_policy_hash"], pc.policy_hash([]))
        self.assertEqual(got["correction_mode"], "off")

    @patch.object(gp, "_binary_identity")
    @patch.object(gp, "_read_text")
    @patch.object(gp, "_run")
    @patch.object(gp, "_remove")
    @patch.object(gp, "_write_text")
    def test_run_worker_rejects_wrong_applied_policy(
        self, write, remove, run, read, identity
    ):
        identity.return_value = {
            "host": "XOX.local",
            "arch": "arm64",
            "binary_path": "/bin/q",
            "binary_sha256": "a" * 64,
            "binary_size": 123,
        }
        run.return_value = FakeProc(stdout=(
            "GPU_VALIDATION_V1 backend=mlx_metal arch=mla correction=off "
            "finite_logits=1 logits_checked=100 requests=1\n"
        ))
        read.return_value = json.dumps({
            "schema": gp.ACK_SCHEMA,
            "status": "STARTUP_STATE",
            "backend": "mlx_metal",
            "correction_mode": "off",
            "weight_epoch": 0,
            "changed_targets": 0,
            "snapshot_count": 0,
            "active_policy": [],
        })
        with self.assertRaises(gp.GpuPreflightError):
            gp.run_isolated_worker(
                host="xox",
                cwd="/repo",
                binary="/bin/q",
                architecture="mla",
                moe_base="/m",
                manifest="/manifest",
                safetensors="/st",
                run_dir="/tmp/run",
                policy=CAND,
            )

    @patch.object(gp, "_binary_identity")
    @patch.object(gp, "_read_text")
    @patch.object(gp, "_run")
    @patch.object(gp, "_remove")
    @patch.object(gp, "_write_text")
    def test_run_worker_rejects_nonfinite_validation_report(
        self, write, remove, run, read, identity
    ):
        identity.return_value = {
            "host": "XOX.local",
            "arch": "arm64",
            "binary_path": "/bin/q",
            "binary_sha256": "a" * 64,
            "binary_size": 123,
        }
        run.return_value = FakeProc(stdout=(
            "GPU_VALIDATION_V1 backend=mlx_metal arch=mla correction=off "
            "finite_logits=0 logits_checked=100 requests=1\n"
        ))
        read.return_value = json.dumps({
            "schema": gp.ACK_SCHEMA,
            "status": "STARTUP_STATE",
            "backend": "mlx_metal",
            "correction_mode": "off",
            "weight_epoch": 0,
            "changed_targets": 0,
            "snapshot_count": 0,
            "active_policy": [],
        })
        with self.assertRaises(gp.GpuPreflightError):
            gp.run_isolated_worker(
                host="xox", cwd="/repo", binary="/bin/q", architecture="mla",
                moe_base="/m", manifest="/manifest", safetensors="/st",
                run_dir="/tmp/run", policy=[],
            )

    @patch.object(gp, "run_isolated_worker")
    def test_ab_preflight_checks_baseline_and_candidate_tokens(self, worker):
        worker.side_effect = [
            {
                "output": (
                    "[moe gpu cb online] req 0 prompt 4 slot 0 arrive 0 "
                    "admit_step 0 ttft_ms 1.0 nout 2 tokens: 111 112\n"
                ),
                "worker": {"binary_sha256": "a" * 64},
                "applied_policy_hash": pc.policy_hash(BASE),
                "weight_epoch": 0,
            },
            {
                "output": (
                    "[moe gpu cb online] req 0 prompt 4 slot 0 arrive 0 "
                    "admit_step 0 ttft_ms 1.0 nout 2 tokens: 222 223\n"
                ),
                "worker": {"binary_sha256": "a" * 64},
                "applied_policy_hash": pc.policy_hash(CAND),
                "weight_epoch": 1,
            },
        ]
        got = gp.run_ab_preflight(
            host="xox",
            cwd="/repo",
            binary="/bin/q",
            architecture="mla",
            moe_base="/m",
            manifest="/manifest",
            safetensors="/st",
            root="/tmp/r",
            baseline_policy=BASE,
            candidate_policy=CAND,
            event={"orig_token": 111, "corrected_token": 222, "pos": 3},
            reference={"emitted_token": 222},
            prompt_len=4,
        )
        self.assertEqual(got["status"], "passed")
        self.assertEqual(got["baseline_emitted_token"], 111)
        self.assertEqual(got["candidate_emitted_token"], 222)

    @patch.object(gp, "run_isolated_worker")
    def test_ab_preflight_rejects_candidate_token_mismatch(self, worker):
        worker.side_effect = [
            {
                "output": (
                    "[moe gpu cb online] req 0 prompt 4 slot 0 arrive 0 "
                    "admit_step 0 ttft_ms 1.0 nout 1 tokens: 111\n"
                ),
                "worker": {"binary_sha256": "a" * 64},
                "applied_policy_hash": pc.policy_hash(BASE),
                "weight_epoch": 0,
            },
            {
                "output": (
                    "[moe gpu cb online] req 0 prompt 4 slot 0 arrive 0 "
                    "admit_step 0 ttft_ms 1.0 nout 1 tokens: 999\n"
                ),
                "worker": {"binary_sha256": "a" * 64},
                "applied_policy_hash": pc.policy_hash(CAND),
                "weight_epoch": 1,
            },
        ]
        with self.assertRaises(gp.GpuPreflightError):
            gp.run_ab_preflight(
                host="xox",
                cwd="/repo",
                binary="/bin/q",
                architecture="mla",
                moe_base="/m",
                manifest="/manifest",
                safetensors="/st",
                root="/tmp/r",
                baseline_policy=BASE,
                candidate_policy=CAND,
                event={"orig_token": 111, "corrected_token": 222, "pos": 3},
                reference={"emitted_token": 222},
                prompt_len=4,
            )

    @patch.object(gp, "run_isolated_worker")
    def test_ab_preflight_rejects_binary_identity_drift(self, worker):
        worker.side_effect = [
            {
                "output": "[moe gpu cb online] req 0 x tokens: 111\n",
                "worker": {"binary_sha256": "a" * 64},
                "applied_policy_hash": pc.policy_hash(BASE),
                "weight_epoch": 0,
            },
            {
                "output": "[moe gpu cb online] req 0 x tokens: 222\n",
                "worker": {"binary_sha256": "b" * 64},
                "applied_policy_hash": pc.policy_hash(CAND),
                "weight_epoch": 1,
            },
        ]
        with self.assertRaises(gp.GpuPreflightError):
            gp.run_ab_preflight(
                host="xox", cwd="/repo", binary="/bin/q", architecture="mla",
                moe_base="/m", manifest="/manifest", safetensors="/st",
                root="/tmp/r", baseline_policy=BASE, candidate_policy=CAND,
                event={"orig_token": 111, "corrected_token": 222, "pos": 0},
                reference={"emitted_token": 222}, prompt_len=1,
            )

    def test_to_planner_evidence_marks_restart_mode_without_live_epoch_claim(self):
        result = {
            "status": "passed",
            "backend": "mlx_metal",
            "correction_mode": "off",
            "binary_sha256": "a" * 64,
            "baseline_policy_hash": pc.policy_hash(BASE),
            "candidate_policy_hash": pc.policy_hash(CAND),
            "baseline_epoch": 0,
            "candidate_epoch": 1,
            "baseline_emitted_token": 111,
            "candidate_emitted_token": 222,
            "reference_emitted_token": 222,
        }
        got = gp.to_planner_evidence(
            result,
            context_hash="c" * 64,
            expected_epoch=7,
        )
        self.assertEqual(got["evidence_mode"], "isolated_restart")
        self.assertEqual(got["expected_epoch"], 7)
        self.assertNotIn("observed_epoch", got)
        self.assertEqual(got["isolated_candidate_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
