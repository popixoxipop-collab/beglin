#!/usr/bin/env python3
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = str(ROOT / "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import backend_adapters as ba
import precision_context as pc


def make_context(backend="cpu", binary="bin-a", device="dev-a"):
    return pc.build_context(
        model_id="deepseek-v2-lite",
        architecture="deepseek_v2_mla",
        checkpoint_sha256="checkpoint",
        tokenizer_sha256="tokenizer",
        base_artifact_sha256="base",
        backend=backend,
        device_fingerprint=device,
        binary_sha256=binary,
        build_manifest_sha256="manifest",
        kernel_version="kernel-v1",
        precision_mode="qng64",
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        encoder_version="encoder-v1",
        execution_mode="online-b1",
        runtime_config={"threads": 8, "correction": False},
    )


class TestPrecisionContext(unittest.TestCase):
    def test_context_hash_is_stable(self):
        a = make_context()
        b = make_context()
        self.assertEqual(a["context_id"], b["context_id"])
        self.assertEqual(pc.context_id(a), a["context_id"])

    def test_backend_changes_context(self):
        cpu = make_context("cpu")
        gpu = make_context("mlx_metal")
        self.assertNotEqual(cpu["context_id"], gpu["context_id"])

    def test_binary_changes_context(self):
        a = make_context(binary="bin-a")
        b = make_context(binary="bin-b")
        self.assertNotEqual(a["context_id"], b["context_id"])

    def test_policy_hash_is_order_independent(self):
        a = {("shared_down_proj", 4): 6, ("shared_up_proj", 3): 5}
        b = [
            {"role": "shared_up_proj", "layer": 3, "n": 5},
            {"role": "shared_down_proj", "layer": 4, "n": 6},
        ]
        self.assertEqual(pc.policy_hash(a), pc.policy_hash(b))

    def test_cpu_pass_is_rejected_for_gpu(self):
        cpu = make_context("cpu")
        gpu = make_context("mlx_metal")
        evidence = {
            "context_id": cpu["context_id"],
            "backend": "cpu",
            "status": "passed",
            "pass": True,
        }
        self.assertFalse(ba.admission_evidence_ok(evidence, gpu))

    def test_same_backend_different_binary_is_rejected(self):
        old = make_context("mlx_metal", binary="old")
        new = make_context("mlx_metal", binary="new")
        evidence = {
            "context_id": old["context_id"],
            "backend": "mlx_metal",
            "status": "passed",
            "pass": True,
        }
        self.assertFalse(ba.admission_evidence_ok(evidence, new))

    def test_preimage_mismatch_is_rejected(self):
        ctx = make_context("mlx_metal")
        before = {("shared_down_proj", 4): 6}
        other = {("shared_down_proj", 4): 7}
        evidence = {
            "context_id": ctx["context_id"],
            "backend": "mlx_metal",
            "preimage_policy_sha256": pc.policy_hash(other),
            "status": "passed",
            "pass": True,
        }
        self.assertFalse(
            ba.admission_evidence_ok(evidence, ctx, preimage_policy=before)
        )

    def test_backend_state_directories_do_not_collide(self):
        cpu = ba.backend_scoped_paths("/ctl", "modelrev", "cpu")
        gpu = ba.backend_scoped_paths("/ctl", "modelrev", "mlx_metal")
        self.assertNotEqual(cpu["root"], gpu["root"])
        self.assertTrue(cpu["root"].endswith("/modelrev/cpu"))
        self.assertTrue(gpu["root"].endswith("/modelrev/mlx_metal"))


if __name__ == "__main__":
    unittest.main()
