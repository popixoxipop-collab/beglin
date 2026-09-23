#!/usr/bin/env python3
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CPP = (ROOT / "mlx_moe.cpp").read_text()
HDR = (ROOT / "mlx_moe.h").read_text()


class BindingSourceGuardTests(unittest.TestCase):
    def test_dense_rebind_clears_custom_qng64(self):
        block = CPP.split("if (bits == 16 || bits == 32)", 1)[1]
        block = block.split("// D-metal-4:", 1)[0]
        self.assertIn("g_tensors.erase", block)
        self.assertIn("g_qng64_tensors.erase", block)

    def test_custom_qng64_rebind_clears_other_registries(self):
        block = CPP.split("if (bits == 7 || (bits >= 9 && bits <= 15))", 1)[1]
        block = block.split("if (ng <= 0 || (in % 8) != 0)", 1)[0]
        self.assertIn("g_tensors.erase", block)
        self.assertIn("g_dtensors.erase", block)

    def test_native_quant_rebind_clears_dense_and_custom(self):
        tail = CPP.split("// insert_or_assign, not operator[]=", 1)[1]
        block = tail.split("g_bound_count++;", 1)[0]
        self.assertIn("g_dtensors.erase", block)
        self.assertIn("g_qng64_tensors.erase", block)

    def test_binding_kind_probe_is_public(self):
        self.assertIn("int mlx_gpu_binding_kind(", CPP)
        self.assertIn("int mlx_gpu_binding_kind(", HDR)

    def test_snapshot_restore_api_is_public(self):
        for symbol in (
            "mlx_gpu_snapshot_binding",
            "mlx_gpu_restore_binding_snapshot",
            "mlx_gpu_drop_binding_snapshot",
            "mlx_gpu_binding_snapshot_count",
        ):
            self.assertIn(symbol, CPP)
            self.assertIn(symbol, HDR)

    def test_restore_clears_all_representation_maps_before_insert(self):
        block = CPP.split("int mlx_gpu_restore_binding_snapshot", 1)[1]
        block = block.split("int mlx_gpu_drop_binding_snapshot", 1)[0]
        self.assertIn("g_tensors.erase(key)", block)
        self.assertIn("g_dtensors.erase(key)", block)
        self.assertIn("g_qng64_tensors.erase(key)", block)

    def test_restore_requires_quiescence_by_contract_comment(self):
        block = CPP.split("int mlx_gpu_restore_binding_snapshot", 1)[0]
        self.assertIn("pause admission", block)
        self.assertIn("drain existing requests", block)
        self.assertIn("synchronize pending MLX/Metal work", block)


if __name__ == "__main__":
    unittest.main()
