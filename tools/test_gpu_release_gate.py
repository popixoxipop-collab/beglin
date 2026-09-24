#!/usr/bin/env python3
import json
import pathlib
import tempfile
import unittest

import gpu_release_gate as gate
import gpu_runtime_control as grc
import precision_context as pc


CANDIDATE = [{"role": "shared_down_proj", "layer": 4, "n": 6}]


def h(ch):
    return ch * 64


def context():
    return pc.ExecutionContext(
        schema="precision-context-v3",
        model_id="deepseek-v2-lite",
        architecture="deepseek_v2_mla",
        checkpoint_sha256=h("1"),
        tokenizer_sha256=h("2"),
        base_artifact_sha256=h("3"),
        backend="mlx_metal",
        device_fingerprint="apple-m4-test",
        binary_sha256=h("4"),
        build_manifest_sha256=h("5"),
        kernel_revision="mlx-test",
        execution_mode="online",
        runtime_config_sha256=h("6"),
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        correction_mode="off",
    )


def budget():
    return gate.freeze_budget_spec({
        "scope": "xox-m4-mla-single-target",
        "workload": "paired-50-v1",
        "max_ttft_p50_ratio": 1.10,
        "max_ttft_p95_ratio": 1.15,
        "max_token_latency_p50_ratio": 1.10,
        "max_token_latency_p95_ratio": 1.15,
        "min_throughput_ratio": 0.90,
        "max_cpu_rss_peak_ratio": 1.05,
        "max_gpu_peak_memory_ratio": 1.10,
        "max_drain_ms": 250.0,
        "max_rollback_ms": 1000.0,
        "max_correction_required_rate": 0.02,
        "min_holdout_delta": -0.001,
        "max_aa_relative_spread": 0.05,
    })


def calibration(ctx, b):
    return {
        "schema": gate.CALIBRATION_SCHEMA,
        "budget_spec_sha256": b["budget_spec_sha256"],
        "context_hash": ctx.context_hash,
        "binary_sha256": ctx.binary_sha256,
        "repeat_count": 3,
        "relative_spread": {
            metric: 0.01 for metric in gate.PERF_METRICS
        },
    }


def capability(ctx):
    symbols = {
        "mlx_gpu_available": True,
        "mlx_gpu_binding_kind": True,
        "mlx_gpu_snapshot_binding": True,
        "mlx_gpu_restore_binding_snapshot": True,
        "mlx_gpu_drop_binding_snapshot": True,
        "mlx_gpu_binding_snapshot_count": True,
        "mlx_gpu_synchronize": True,
        "mlx_gpu_reset_runtime_epoch": True,
    }
    value = {
        "schema": "precision-capability-v1",
        "worker": {
            "host": "XOX.local",
            "arch": "arm64",
            "binary_path": "/bin/q",
            "binary_sha256": ctx.binary_sha256,
            "binary_size": 123456,
            "mlx_symbol_present": True,
            "mlx_control_symbols": symbols,
            "link_report": "mlx",
        },
        "backends": {
            "cpu": {"compiled": True, "status": "PRESENT"},
            "mlx_metal": {
                "compiled": True,
                "status": "IMPLEMENTED_UNVERIFIED",
                "qng64_widths": {"5": "IMPLEMENTED_UNVERIFIED", "6": "IMPLEMENTED_UNVERIFIED", "7": "IMPLEMENTED_UNVERIFIED"},
                "native_widths": [2, 3, 5, 6],
                "custom_metal_widths": [7],
                "runtime_control": {
                    "compiled": True,
                    "status": "IMPLEMENTED_UNVERIFIED",
                    "symbols": symbols,
                },
            },
        },
    }
    value["worker_identity_sha256"] = pc.sha256_json(value["worker"])
    return value


def preflight(ctx):
    return {
        "status": "passed",
        "backend": "mlx_metal",
        "architecture": "mla",
        "binary_sha256": ctx.binary_sha256,
        "baseline_policy_hash": pc.policy_hash([]),
        "candidate_policy_hash": pc.policy_hash(CANDIDATE),
        "baseline_epoch": 0,
        "candidate_epoch": 1,
        "baseline_emitted_token": 111,
        "candidate_emitted_token": 222,
        "reference_emitted_token": 222,
        "correction_mode": "off",
        "evidence_bundle": {
            "capability_path": "/run/capability.json",
            "capability_sha256": h("7"),
            "inputs_path": "/run/inputs.json",
            "inputs_sha256": h("8"),
            "verdict_path": "/run/verdict.json",
            "verdict_payload_sha256": h("9"),
        },
    }


def canary_plan(ctx):
    return {
        "schema": "gpu-restart-canary-plan-v1",
        "backend": "mlx_metal",
        "context_hash": ctx.context_hash,
        "binary_sha256": ctx.binary_sha256,
        "baseline_policy": [],
        "baseline_policy_hash": pc.policy_hash([]),
        "candidate_policy": CANDIDATE,
        "candidate_policy_hash": pc.policy_hash(CANDIDATE),
        "limits": {
            "min_requests": 50,
            "max_requests": 50,
            "auto_expand": False,
        },
    }


def canary_result():
    return {
        "status": "CANARY_PASS",
        "decision": "CANARY_PASS_NO_AUTO_EXPANSION",
        "rollback_required": False,
        "auto_expand": False,
        "requests_completed": 50,
    }


def runtime_ack():
    return grc.normalize_runtime_ack({
        "schema": grc.ACK_SCHEMA,
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
        "active_policy": CANDIDATE,
    })


def measurements():
    return {
        "baseline": {
            "ttft_p50_ms": 100.0,
            "ttft_p95_ms": 150.0,
            "token_latency_p50_ms": 10.0,
            "token_latency_p95_ms": 15.0,
            "throughput_tok_s": 100.0,
            "cpu_rss_peak_bytes": 1_000_000,
            "gpu_peak_memory_bytes": 2_000_000,
        },
        "candidate": {
            "ttft_p50_ms": 103.0,
            "ttft_p95_ms": 155.0,
            "token_latency_p50_ms": 10.2,
            "token_latency_p95_ms": 15.5,
            "throughput_tok_s": 98.0,
            "cpu_rss_peak_bytes": 1_010_000,
            "gpu_peak_memory_bytes": 2_050_000,
        },
        "quality": {
            "finite_logits": True,
            "target_replay_pass": True,
            "regression_panel_pass": True,
            "holdout_delta": 0.0,
            "correction_required_rate": 0.0,
        },
        "drain_ms": 100.0,
    }


def rollback_drill():
    return {
        "status": "ROLLBACK_APPLIED",
        "rollback_complete": True,
        "rollback_latency_ms": 500.0,
        "reconcile": {"status": "IN_SYNC"},
        "ack_sha256": h("a"),
    }


def cpu_identity():
    return {
        "host": "BOB.local",
        "binary_sha256": h("b"),
        "promotion_policy_sha256": h("c"),
        "service_config_sha256": h("d"),
    }


class GpuReleaseGateTests(unittest.TestCase):
    def qualify(self, **overrides):
        ctx = overrides.pop("context", context())
        b = overrides.pop("budget_spec", budget())
        args = {
            "context": ctx,
            "budget_spec": b,
            "calibration": calibration(ctx, b),
            "capability": capability(ctx),
            "preflight": preflight(ctx),
            "canary_plan": canary_plan(ctx),
            "canary_result": canary_result(),
            "runtime_ack": runtime_ack(),
            "candidate_policy": CANDIDATE,
            "measurements": measurements(),
            "rollback_drill": rollback_drill(),
            "cpu_identity_before": cpu_identity(),
            "cpu_identity_after": cpu_identity(),
        }
        args.update(overrides)
        return gate.evaluate_release(**args)

    def test_frozen_budget_detects_tampering(self):
        b = budget()
        b["max_ttft_p50_ratio"] = 9.0
        with self.assertRaises(gate.GpuReleaseGateError):
            gate.require_frozen_budget(b)

    def test_release_qualifies_scope_but_never_enables_auto_promotion(self):
        got = self.qualify()
        self.assertEqual(got["status"], "QUALIFIED_FOR_SCOPE")
        self.assertFalse(got["gpu_auto_promotion_enabled"])
        self.assertTrue(got["requires_explicit_opt_in_for_auto_promotion"])
        self.assertEqual(got["candidate_policy_hash"], pc.policy_hash(CANDIDATE))
        self.assertEqual(len(got["release_manifest_sha256"]), 64)

    def test_budget_failure_is_explicit_rejection(self):
        m = measurements()
        m["candidate"]["ttft_p95_ms"] = 999.0
        got = self.qualify(measurements=m)
        self.assertEqual(got["status"], "REJECTED_BUDGET")
        self.assertFalse(got["budget"]["checks"]["ttft_p95_ms_ratio"]["pass"])

    def test_aa_calibration_must_match_frozen_budget_and_be_repeatable(self):
        ctx = context()
        b = budget()
        c = calibration(ctx, b)
        c["relative_spread"]["ttft_p95_ms"] = 0.5
        with self.assertRaises(gate.GpuReleaseGateError):
            self.qualify(context=ctx, budget_spec=b, calibration=c)

    def test_wrong_applied_policy_rejects_release(self):
        bad_ack = dict(runtime_ack())
        bad_ack["active_policy"] = []
        # Force re-normalization from the altered raw payload.
        bad_ack.pop("active_policy_hash", None)
        bad_ack.pop("ack_sha256", None)
        with self.assertRaises(gate.GpuReleaseGateError):
            self.qualify(runtime_ack=bad_ack)

    def test_failed_or_expanding_canary_rejects_release(self):
        result = canary_result()
        result["auto_expand"] = True
        with self.assertRaises(gate.GpuReleaseGateError):
            self.qualify(canary_result=result)

    def test_cpu_identity_drift_rejects_release(self):
        after = cpu_identity()
        after["promotion_policy_sha256"] = h("e")
        with self.assertRaises(gate.GpuReleaseGateError):
            self.qualify(cpu_identity_after=after)

    def test_rollback_drill_must_be_in_sync_and_within_budget(self):
        drill = rollback_drill()
        drill["reconcile"] = {"status": "DRIFT"}
        with self.assertRaises(gate.GpuReleaseGateError):
            self.qualify(rollback_drill=drill)
        drill = rollback_drill()
        drill["rollback_latency_ms"] = 9999.0
        with self.assertRaises(gate.GpuReleaseGateError):
            self.qualify(rollback_drill=drill)

    def test_no_candidate_is_valid_fail_closed_outcome(self):
        got = gate.no_approved_candidate(
            scope="xox-m4-mla-single-target",
            workload="paired-50-v1",
            reason="no GPU candidate passed G4/G5 evidence gates",
        )
        self.assertEqual(got["status"], "NO_APPROVED_CANDIDATE")
        self.assertFalse(got["gpu_auto_promotion_enabled"])

    def test_release_bundle_writes_hash_manifest(self):
        got = self.qualify()
        with tempfile.TemporaryDirectory() as td:
            bundle = gate.write_release_bundle(
                td,
                release_manifest=got,
                budget_spec=budget(),
                measurements=measurements(),
                rollback_drill=rollback_drill(),
                cpu_identity_before=cpu_identity(),
                cpu_identity_after=cpu_identity(),
            )
            self.assertTrue((pathlib.Path(td) / "release_manifest.json").exists())
            sums = (pathlib.Path(td) / "SHA256SUMS").read_text()
            self.assertIn("release_manifest.json", sums)
            self.assertEqual(len(bundle["sha256sums_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
