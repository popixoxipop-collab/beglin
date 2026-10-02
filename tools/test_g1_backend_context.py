#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = str(ROOT / "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import precision_context as pctx
import backend_adapters
import autopilot_full
import autopilot_live_preflight as live_preflight


def make_context(backend="mlx_metal", device="apple-m1-max", binary="bin-a"):
    return pctx.build_context(
        model_id="deepseek-v2-lite",
        architecture="deepseek-v2",
        checkpoint_sha256="checkpoint-a",
        tokenizer_sha256="tokenizer-a",
        base_artifact_sha256="base-a",
        backend=backend,
        device_fingerprint=device,
        binary_sha256=binary,
        build_manifest_sha256="manifest-a",
        kernel_version="kernel-a",
        precision_mode="mixed",
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        encoder_version="qng64-ef-v1",
        execution_mode="serving",
        runtime_config={"batch": 1},
    )


class TestContextIdentity(unittest.TestCase):
    def test_backend_changes_context_hash(self):
        cpu = make_context("cpu")
        gpu = make_context("mlx_metal")
        self.assertNotEqual(cpu["context_id"], gpu["context_id"])

    def test_binary_and_device_change_context_hash(self):
        base = make_context()
        other_binary = make_context(binary="bin-b")
        other_device = make_context(device="apple-m2-ultra")
        self.assertNotEqual(base["context_id"], other_binary["context_id"])
        self.assertNotEqual(base["context_id"], other_device["context_id"])

    def test_legacy_cpu_or_other_context_cannot_authorize_gpu(self):
        gpu = make_context()
        with self.assertRaises(pctx.ContextError):
            pctx.require_evidence_match(
                {"status": "passed", "pass": True},
                backend="mlx_metal",
                context_hash=gpu["context_id"],
            )
        with self.assertRaises(pctx.ContextError):
            pctx.require_evidence_match(
                {
                    "backend": "cpu",
                    "context_hash": make_context("cpu")["context_id"],
                    "status": "passed",
                    "pass": True,
                },
                backend="mlx_metal",
                context_hash=gpu["context_id"],
            )
        with self.assertRaises(pctx.ContextError):
            pctx.require_evidence_match(
                {
                    "backend": "mlx_metal",
                    "context_hash": make_context(binary="bin-b")["context_id"],
                    "status": "passed",
                    "pass": True,
                },
                backend="mlx_metal",
                context_hash=gpu["context_id"],
            )

    def test_backend_state_paths_are_disjoint(self):
        cpu = backend_adapters.backend_scoped_paths("/ctl", "rev-a", "cpu")
        gpu = backend_adapters.backend_scoped_paths("/ctl", "rev-a", "mlx_metal")
        self.assertNotEqual(cpu["root"], gpu["root"])
        self.assertIn("/cpu", cpu["root"])
        self.assertIn("/mlx_metal", gpu["root"])


class TestPlannerFailClosed(unittest.TestCase):
    def test_gpu_planner_requires_context(self):
        env = {
            "QWEN_AUTOPILOT_BACKEND": "mlx_metal",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires QWEN_PRECISION_CONTEXT_JSON"):
                autopilot_full._planner_backend_context()

    def test_gpu_planner_accepts_matching_context(self):
        context = make_context()
        env = {
            "QWEN_AUTOPILOT_BACKEND": "mlx_metal",
            "QWEN_PRECISION_CONTEXT_JSON": json.dumps(context),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            backend, got, context_hash = autopilot_full._planner_backend_context()
        self.assertEqual(backend, "mlx_metal")
        self.assertEqual(context_hash, context["context_id"])
        self.assertEqual(got["context_id"], context["context_id"])

    def test_gpu_gate_rejects_cpu_pass(self):
        gpu = make_context()
        cpu_evidence = {
            "backend": "cpu",
            "context_id": make_context("cpu")["context_id"],
            "status": "passed",
            "pass": True,
        }
        with mock.patch.object(
            live_preflight, "fetch_latest_evidence_v3", return_value=cpu_evidence
        ):
            action, detail = autopilot_full._live_evidence_gate(
                "m", "q_proj", 0, 5, "preimage",
                backend="mlx_metal", context=gpu,
            )
        self.assertEqual(action, "EVIDENCE_CONTEXT_MISMATCH")
        self.assertIn("backend mismatch", detail["reason"])

    def test_gpu_gate_db_unavailable_blocks_admission(self):
        gpu = make_context()
        with mock.patch.object(
            live_preflight,
            "fetch_latest_evidence_v3",
            side_effect=live_preflight.EvidenceStoreUnavailable("db down"),
        ):
            action, detail = autopilot_full._live_evidence_gate(
                "m", "q_proj", 0, 5, "preimage",
                backend="mlx_metal", context=gpu,
            )
        self.assertEqual(action, "EVIDENCE_STORE_UNAVAILABLE")
        self.assertIn("db down", detail["reason"])


class TestStaticG1Contracts(unittest.TestCase):
    def test_v3_migration_contains_backend_scoped_tables(self):
        sql = (ROOT / "supabase_migration_precision_context_v3.sql").read_text()
        for table in (
            "moe_execution_contexts_v3",
            "moe_validation_runs_v3",
            "moe_attribution_provenance_v3",
            "moe_live_preflight_results_v3",
            "moe_precision_transactions_v3",
        ):
            self.assertIn(table, sql)
        self.assertIn("context_id", sql)
        self.assertIn("backend", sql)
        self.assertIn("binary_sha256", sql)
        self.assertIn("build_manifest_sha256", sql)

    def test_real_sweep_blocks_v3_fallthrough_to_legacy_write(self):
        source = (ROOT / "tools" / "autopilot_real_sweep.py").read_text()
        blocker = source.index(
            "v3-scoped sweep completed but legacy qng64_real DB write is blocked"
        )
        legacy_write = source.index("qsn.push_sweep_results_atomic")
        self.assertLess(blocker, legacy_write)
        self.assertIn("_fetch_scoped_provenance", source)
        self.assertIn('row.get("context_id")', source)


if __name__ == "__main__":
    unittest.main()
