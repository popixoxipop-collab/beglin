#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = str(ROOT / "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import backend_adapters as adapters
import precision_context as pc


def make_context(**overrides):
    args = {
        "model_id": "deepseek-v2-lite",
        "architecture": "deepseek_v2",
        "checkpoint_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "base_artifact_sha256": "c" * 64,
        "backend": "cpu",
        "device_fingerprint": "apple-m4",
        "binary_sha256": "d" * 64,
        "build_manifest_sha256": "e" * 64,
        "kernel_version": "cpu-v1",
        "precision_mode": "role_layer",
        "quant_format": "qng64_g64_ef_v1",
        "group_size": 64,
        "encoder_version": "qng64-ef-v1",
        "execution_mode": "online_b1",
        "runtime_config": {"threads": 8, "batch": 1},
    }
    args.update(overrides)
    return pc.build_context(**args)


def make_evidence(context, preimage=None, candidate=None):
    preimage = preimage or {('shared_down_proj', 4): 5}
    candidate = candidate or {('shared_down_proj', 4): 6}
    return pc.evidence_scope(
        context=context,
        run_id="run-1",
        role="shared_down_proj",
        layer=4,
        n=6,
        preimage_policy=preimage,
        candidate_policy=candidate,
    )


class TestPrecisionContext(unittest.TestCase):
    def test_runtime_dict_order_does_not_change_context_id(self):
        a = make_context(runtime_config={"threads": 8, "batch": 1})
        b = make_context(runtime_config={"batch": 1, "threads": 8})
        self.assertEqual(a["context_id"], b["context_id"])

    def test_cpu_evidence_cannot_be_reused_for_mlx(self):
        cpu = make_context()
        gpu = make_context(
            backend="mlx_metal",
            device_fingerprint="apple-m1-max",
            kernel_version="mlx-custom-metal-v1",
        )
        evidence = make_evidence(cpu)
        with self.assertRaises(pc.ContextMismatch):
            pc.assert_evidence_compatible(evidence, gpu)

    def test_different_binary_cannot_reuse_evidence(self):
        old = make_context(binary_sha256="1" * 64)
        new = make_context(binary_sha256="2" * 64)
        evidence = make_evidence(old)
        with self.assertRaises(pc.ContextMismatch):
            pc.assert_evidence_compatible(evidence, new)

    def test_different_preimage_policy_cannot_reuse_evidence(self):
        ctx = make_context()
        evidence = make_evidence(ctx)
        with self.assertRaises(pc.ContextMismatch):
            pc.assert_evidence_compatible(
                evidence, ctx,
                preimage_policy={('shared_down_proj', 4): 7},
            )

    def test_legacy_unscoped_row_is_rejected(self):
        ctx = make_context()
        legacy = {"backend": "cpu", "context_id": None}
        with self.assertRaises(pc.ContextMismatch):
            pc.assert_evidence_compatible(legacy, ctx)

    def test_same_context_and_preimage_is_reusable(self):
        ctx = make_context()
        preimage = {('shared_down_proj', 4): 5}
        evidence = make_evidence(ctx, preimage=preimage)
        self.assertTrue(
            pc.assert_evidence_compatible(evidence, ctx, preimage_policy=preimage)
        )


class TestBackendAdapters(unittest.TestCase):
    def test_mlx_qng64_n7_uses_custom_metal(self):
        self.assertEqual(
            adapters.quant_path("mlx_metal", "qng64_g64_ef_v1", 7),
            "custom_metal_bitplane",
        )

    def test_mlx_qng64_n8_is_rejected(self):
        with self.assertRaises(adapters.BackendCapabilityError):
            adapters.quant_path("mlx_metal", "qng64_g64_ef_v1", 8)

    def test_cpu_qng64_n8_is_distinctly_supported(self):
        self.assertEqual(
            adapters.quant_path("cpu", "qng64_g64_ef_v1", 8),
            "native",
        )

    def test_mlx_q8g64_n8_is_a_different_format(self):
        self.assertEqual(
            adapters.quant_path("mlx_metal", "q8g64", 8),
            "mlx_native_repack",
        )

    def test_initial_gpu_auto_ladder_is_5_6_7(self):
        self.assertEqual(adapters.initial_auto_ladder("mlx_metal"), [5, 6, 7])


if __name__ == "__main__":
    unittest.main()
