#!/usr/bin/env python3
import importlib.util,unittest
from pathlib import Path
f=Path(__file__).resolve().parents[1]/"tools/je01_scratch_cb384_overlay_stage.py"
s=importlib.util.spec_from_file_location("je01_scratch_overlay",f)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class TestJE01Overlay(unittest.TestCase):
    def setUp(self):
        self.c='#include <algorithm>\n'+m.CXX_ANCHOR+'\n'+m.CB_ANCHOR+'\n'
        self.h=m.HEADER_ANCHOR+'\n'
    def test_patch_unique(self):
        a,b=m.patch(self.c,self.h)
        self.assertEqual(a.count("routed_sum=je01_scratch_adjust_routed_sum("),1)
        self.assertIn("JE01_SCRATCH_OVERLAY_B1_SHAPE_FENCE",a)
        self.assertIn("mlx_gpu_je01_overlay_register",b)
        self.assertEqual(len(m.PAIRS),8)
    def test_cb_missing(self):
        with self.assertRaisesRegex(ValueError,"ONLY_ORIGINAL_ONLINE_CBATCH_HOOK"):
            m.patch(self.c.replace(m.CB_ANCHOR,""),self.h)
    def test_cb_duplicate(self):
        with self.assertRaisesRegex(ValueError,"ONLY_ORIGINAL_ONLINE_CBATCH_HOOK"):
            m.patch(self.c+m.CB_ANCHOR,self.h)
    def test_cpp_changed(self):
        with self.assertRaisesRegex(ValueError,"ONLY_CPP_LAZY_ENTRY"):
            m.patch(self.c.replace(m.CXX_ANCHOR,""),self.h)
    def test_header_changed(self):
        with self.assertRaisesRegex(ValueError,"C_HEADER_API"):
            m.patch(self.c,"missing")
if __name__=="__main__":unittest.main(verbosity=2)
