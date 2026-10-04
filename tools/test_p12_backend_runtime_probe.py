#!/usr/bin/env python3
import copy
import unittest

import backend_adapter_v2 as bav2
import backend_runtime_probe_p12 as brp


class BinaryProbeTests(unittest.TestCase):
    def legacy(self):
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
        return {
            "schema": "precision-capability-v1",
            "collected_at": "2026-10-04T00:00:00Z",
            "worker": {
                "host": "xox",
                "arch": "arm64",
                "binary_path": "/a/qwen_infer_gpu",
                "binary_sha256": "a" * 64,
                "binary_size": 123,
                "mlx_symbol_present": True,
                "mlx_control_symbols": symbols,
                "link_report": "volatile text",
            },
            "backends": {
                "cpu": {"compiled": True, "status": "PRESENT"},
                "mlx_metal": {
                    "compiled": True,
                    "status": "IMPLEMENTED_UNVERIFIED",
                    "qng64_widths": {
                        "2": "IMPLEMENTED_UNVERIFIED",
                        "3": "IMPLEMENTED_UNVERIFIED",
                        "5": "IMPLEMENTED_UNVERIFIED",
                        "6": "IMPLEMENTED_UNVERIFIED",
                        "7": "IMPLEMENTED_UNVERIFIED",
                        "9": "IMPLEMENTED_UNVERIFIED",
                    },
                    "runtime_control": {
                        "compiled": True,
                        "status": "IMPLEMENTED_UNVERIFIED",
                        "symbols": symbols,
                    },
                },
            },
        }

    def test_probe_identity_ignores_observation_location_metadata(self):
        a = self.legacy()
        b = copy.deepcopy(a)
        b["collected_at"] = "later"
        b["worker"]["host"] = "another-host"
        b["worker"]["binary_path"] = "/different/path"
        b["worker"]["link_report"] = "different"
        pa = brp.normalize_binary_capability(a)
        pb = brp.normalize_binary_capability(b)
        self.assertEqual(pa["probe_sha256"], pb["probe_sha256"])

    def test_mlx_widths_and_control_are_preserved(self):
        probe = brp.normalize_binary_capability(self.legacy())
        mlx = probe["backends"]["mlx_metal"]
        self.assertEqual(mlx["supported_n"], [2, 3, 5, 6, 7, 9])
        self.assertTrue(mlx["runtime_control_compiled"])
        self.assertIn("HOT_REBIND_SINGLE", mlx["mutation_modes"])


class SymmetricSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.policy = [
            {"role": "shared_down_proj", "layer": 26, "n": 5},
            {"role": "shared_up_proj", "layer": 3, "n": 6},
        ]
        self.target = [
            {"role": "shared_down_proj", "layer": 26, "n": 5},
            {"role": "shared_up_proj", "layer": 3, "n": 5},
        ]
        self.ack = {
            "weight_epoch": 11,
            "active_policy": self.policy,
            "active_policy_hash": bav2.policy_hash(self.policy),
            "model_id": "m",
        }
        legacy = BinaryProbeTests().legacy()
        self.probe = brp.normalize_binary_capability(legacy)

    def test_cpu_and_mlx_query_same_state_shape_and_plan_schema(self):
        cpu = brp.ReadOnlyBackendSurfaceV2(
            backend="cpu",
            binary_probe=self.probe,
            state_provider=lambda: self.ack,
        )
        mlx = brp.ReadOnlyBackendSurfaceV2(
            backend="mlx_metal",
            binary_probe=self.probe,
            state_provider=lambda: self.ack,
            verified_hot_targets={"m/L3/shared_up_proj"},
        )
        cs = cpu.query_applied_state()
        ms = mlx.query_applied_state()
        self.assertEqual(cs.epoch, ms.epoch)
        self.assertEqual(cs.policy_hash, ms.policy_hash)
        cp = cpu.plan_transition(self.target)
        mp = mlx.plan_transition(self.target)
        self.assertEqual(cp["schema"], mp["schema"])
        self.assertEqual(cp["target_policy_hash"], mp["target_policy_hash"])
        self.assertEqual(cp["action"], "RESTART_REQUIRED")
        self.assertEqual(mp["action"], "HOT_REBIND_SINGLE")

    def test_runtime_evidence_is_deterministic(self):
        surface = brp.ReadOnlyBackendSurfaceV2(
            backend="mlx_metal",
            binary_probe=self.probe,
            state_provider=lambda: self.ack,
            verified_hot_targets={"m/L3/shared_up_proj"},
        )
        a = surface.collect_runtime_evidence()
        b = surface.collect_runtime_evidence()
        self.assertEqual(a["evidence_sha256"], b["evidence_sha256"])

    def test_mutation_is_explicitly_disabled(self):
        surface = brp.ReadOnlyBackendSurfaceV2(
            backend="cpu",
            binary_probe=self.probe,
            state_provider=lambda: self.ack,
        )
        with self.assertRaisesRegex(
            bav2.BackendPlanError, "does not enable runtime mutation"
        ):
            surface.apply_transition(self.target)


if __name__ == "__main__":
    unittest.main()
