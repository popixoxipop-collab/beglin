#!/usr/bin/env python3
"""Synthetic-only test of source-locked JE01 B/C native rolewise MLX loader."""
import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).resolve().parents[1]/"tools/je01_eoe_postnll_bc_native_overlay_patch.py"
s=importlib.util.spec_from_file_location("je01_bc_patch",P)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class TestJE01PostNLLBCNativePatch(unittest.TestCase):
    def test_BC_pre_registered_six_role_q4_masks(self):
        d=m.arm_roles()
        self.assertEqual(set(d),{"B_PROTECT_EARLY_GATE","C_PROTECT_EARLY_UP"})
        self.assertEqual([len(d[k]) for k in d],[6,6])
        b=d["B_PROTECT_EARLY_GATE"]
        c=d["C_PROTECT_EARLY_UP"]
        self.assertIn(("12","13","up_proj"),b)
        self.assertNotIn(("12","13","gate_proj"),b)
        self.assertIn(("12","13","gate_proj"),c)
        self.assertNotIn(("12","13","up_proj"),c)

    def test_synthetic_patch_contains_exact_B_C_branch_and_fails_unknown(self):
        cpp="\n".join([
            "static std::vector<uint8_t> g_je01_real_native_bundle;",
            "if (bias_off==-1)",
            "mlx_gpu_je01_overlay_register",
            "(t.bits!=4 && t.bits!=5)",
            "routed_sum=je01_scratch_adjust_routed_sum(",
            "int mlx_gpu_je01_overlay_load_bundle(){return 0;}",
            "int mlx_gpu_layer_step_lazy(){return 0;}"
        ])
        c="mlx_gpu_je01_overlay_count()!=4\nJE01_REAL_AF4_KEEP4_Q4Q5_MIXED4_REGISTERED\n"
        out,mark=m.patch(cpp,c)
        self.assertIn('std::getenv("JE01_CAL240_POSTNLL_ARM")',out)
        self.assertIn('arm!="B_PROTECT_EARLY_GATE" && arm!="C_PROTECT_EARLY_UP"',out)
        self.assertIn('arm=="B_PROTECT_EARLY_GATE"',out)
        self.assertIn('arm=="C_PROTECT_EARLY_UP"',out)
        self.assertEqual(out.count("const bool q4="),1)
        self.assertEqual(out.count("altered_experts==7"),1)
        self.assertEqual(mark.count(m.NEW_MARK),1)
        self.assertEqual(mark.count("mlx_gpu_je01_overlay_count()!=7"),1)
        self.assertNotIn("mlx_gpu_je01_overlay_count()!=4",mark)

    def test_original_frozen_A_code_or_marker_not_accepted(self):
        with self.assertRaisesRegex(ValueError,"ANCHOR"):
            m.patch("int mlx_gpu_je01_overlay_load_bundle(){}","bad")


if __name__=="__main__":
    unittest.main()
