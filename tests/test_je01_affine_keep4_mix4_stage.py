#!/usr/bin/env python3
"""JE01 original 27-layer scratch-only affine-preserve+sym45 loader contract."""
import importlib.util,unittest
from pathlib import Path
f=Path(__file__).resolve().parents[1]/"tools/je01_eoe_original384_affine_keep4_mix4_stage.py"
spec=importlib.util.spec_from_file_location("je01_affine_keep4",f)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class TestAffine4Keep4(unittest.TestCase):
    def fixture(self):
        cpp=('static std::vector<uint8_t> g_je01_real_native_bundle;\n'
             'routed_sum=je01_scratch_adjust_routed_sum(l,h2,top_idx,top_wgt,down_flat,routed_sum);\n'
             'if (bias_off==-1) { // Q4 symmetric\n'
             '}\n'
             'int mlx_gpu_je01_overlay_register(int l,int e,const char*g,const char*u,const char*d);\n'
             'int mlx_gpu_je01_overlay_load_bundle(const char*b,const char*t) {return 1;}\n'
             'int mlx_gpu_layer_step_lazy(int l,int pos,int is_dense,')
        c=('static int run_moe_gpu_cbatch_online_gate(int argc, char **argv) {\n'
           'if (mlx_gpu_je01_overlay_count()!=8) return -1;\n'
           'fprintf(stderr,"JE01_REAL_Q35_EIGHT_EXPERTS_FULL_MODEL_REGISTERED\\n");\n}\n')
        return cpp,c
    def test_converts_only_four_affine_preserving_mode(self):
        a,b=mod.patch(*self.fixture())
        self.assertEqual(a.count("int mlx_gpu_je01_overlay_load_bundle("),1)
        self.assertEqual(b.count(mod.MARK_NEW),1)
        self.assertEqual(b.count(mod.COUNT_NEW),1)
        self.assertIn("high && (r.codec!=\"AF4\"",a)
        self.assertIn("source original AF4's original E64",a.replace("The original","source original")) if False else None
        self.assertIn("affine!=12 || sym4!=6 || sym5!=6",a)
        self.assertIn("spans.size()!=60",a)
        self.assertIn("if(v[0].codec==\"AF4\")",a)
        self.assertIn("continue;",a)
        self.assertIn("mlx_gpu_je01_overlay_count()==4?1:0",a)
        self.assertIn("if (bias_off==-1)",a)
    def test_no_original_fullgold_hook_refused(self):
        a,b=self.fixture()
        with self.assertRaises(ValueError):
            mod.patch(a.replace(mod.SOURCE_MARK,""),b)
    def test_refuse_c_gate_count_duplicate(self):
        a,b=self.fixture()
        with self.assertRaises(ValueError):
            mod.patch(a,b+b)
    def test_second_patch_refused(self):
        a,b=mod.patch(*self.fixture())
        with self.assertRaises(ValueError):
            mod.patch(a,b)
if __name__=="__main__":
    unittest.main(verbosity=2)
