#!/usr/bin/env python3
"""JE01 regression for actual M1 MLX 384 fused K/V allocation evidence.

Receipt was obtained from real macOS ARM64 GitHub Actions run 44017766019.
The test audits it; it does not substitute for a fresh model forward.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import je01_isolated_context384_stage as context


class NativeGpuReceiptTests(unittest.TestCase):
    def test_actual_fused_metal_memory_equals_240_mib(self):
        r = json.loads((ROOT / "docs" / "research" /
                        "JE01_ACTUAL_METAL384_DEVICE_KV_20261117.json").read_text())
        d = r["native_observation"]
        self.assertEqual(r["ci_run_id"], 44017766019)
        self.assertEqual(r["ci_checked_commit"], "cdc72d723716d3162b96c4b8ec2d1fea2308e575")
        self.assertEqual(r["isolated_binary_sha256"],
                         "bd7d79661d25370d50529b92cddaa3871cd60be6211263dbc29f35aeacd35b6f")
        self.assertEqual(d["status"], "PASS_NO_MODEL")
        self.assertTrue(d["metal_full_fused_kv_materialized"])
        self.assertTrue(d["mlx_device_available"])
        self.assertTrue(d["single_slot_accepted"])
        self.assertTrue(d["two_or_more_slots_rejected"])
        self.assertTrue(d["mla_host_configured"])
        self.assertEqual(d["measured_mlx_gpu_active_before_bytes"], 0)
        self.assertEqual(d["measured_mlx_gpu_active_delta_bytes"], context.kv_bytes())
        self.assertEqual(d["measured_mlx_gpu_active_after_bytes"], 251_658_240)
        self.assertEqual(d["measured_mlx_gpu_peak_after_bytes"], 251_658_496)
        self.assertEqual(d["theoretical_kv_bytes_per_single_slot"], context.kv_bytes())
        self.assertNotEqual(d["measured_process_peak_rss_delta_bytes"], 0)
        self.assertFalse(d["model_weights_loaded"])
        self.assertEqual(d["pretrained_model_forward"], "NOT_RUN")
        self.assertFalse(d["production_modified"])
        self.assertFalse(r["limits"]["production_promotion_allowed"])
        self.assertEqual(r["limits"]["short_prefix_logit_parity"], "NOT_RUN")
        self.assertEqual(r["limits"]["routed_expert_nll"], "NOT_MEASURED")
        self.assertEqual(r["frozen_input_manifest_sha256"],
                         context.EXPECTED_DATASET_MANIFEST_SHA)
        self.assertEqual(r["original_bf16_sha256"],
                         context.EXPECTED_MODEL_SHA)

    def test_hook_only_inserted_in_isolated_copy(self):
        q = (ROOT / "qwen_infer.c").read_text()
        mlx = (ROOT / "mlx_moe.cpp").read_text()
        cmake = (ROOT / "CMakeLists.txt").read_text()
        original_token = "mlx_gpu_je01_materialize_fused_mla_kv"
        self.assertNotIn(original_token, mlx)
        altered = context.patch_sources(q, mlx, cmake)
        self.assertIn(original_token, altered["mlx_moe.cpp"])
        self.assertEqual(altered["mlx_moe.cpp"].count(
            'extern "C" int mlx_gpu_je01_materialize_fused_mla_kv'), 1)
        self.assertNotIn("install(", altered["CMakeLists.txt"])
        self.assertNotIn(original_token, (ROOT / "mlx_moe.cpp").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
