#!/usr/bin/env python3
"""Synthetic JE01 original27 CAL240 mixed-role overlay guards; no Metal run."""
from __future__ import annotations
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "tools" / "je01_eoe_original384_cal240_sse_role_specific_overlay_stage.py"
spec = importlib.util.spec_from_file_location("je01_rolewise_overlay", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def source_fixture():
    cpp = """
#include <array>
static std::vector<uint8_t> g_je01_real_native_bundle;
if (bias_off==-1) {}
int mlx_gpu_je01_overlay_register(int l,int e,const char *g,const char *u,const char *d);
if ((t.bits!=4 && t.bits!=5)) {}
routed_sum=je01_scratch_adjust_routed_sum(
int mlx_gpu_je01_overlay_load_bundle(const char *p,const char *t) {return 0;}
int mlx_gpu_layer_step_lazy(int layer) {return 1;}
"""
    c = """
if (mlx_gpu_je01_overlay_count()!=4) { exit(1); }
fprintf(stderr,"JE01_REAL_AF4_KEEP4_Q4Q5_MIXED4_REGISTERED");
"""
    return cpp, c


class TestJE01RoleSpecificScratchOverlay(unittest.TestCase):
    def test_exact_scratch_only_source_patch_changes_seven_experts(self):
        cpp, c = mod.patch(*source_fixture())
        self.assertEqual(c.count("mlx_gpu_je01_overlay_count()!=7"), 1)
        self.assertEqual(c.count(mod.NEW_MARK), 1)
        self.assertNotIn(mod.OLD_MARK, c)
        self.assertEqual(cpp.count("altered_experts==7"), 1)
        self.assertIn("mlx_gpu_je01_overlay_register(", cpp)
        self.assertIn("if(group.first!=std::make_pair(14,63))return 0;", cpp)

    def test_binding_handles_mixed_af4_bias_and_q4_sentinel(self):
        cpp, _ = mod.patch(*source_fixture())
        self.assertIn('long bias=(r.codec=="AF4") ? (long)r.bb : -1L;', cpp)
        self.assertIn("r.name.c_str(),1,r.out,r.in,r.in/64", cpp)
        self.assertIn("r.bb>=r.be", cpp)
        self.assertIn("r.bb!=0 || r.be!=0", cpp)
        self.assertIn('r.codec!="Q4"', cpp)
        self.assertIn('r.codec!="Q5"', cpp)
        self.assertIn('r.codec!="AF4"', cpp)

    def test_role_specific_frozen_tiers_and_physical_span_checks(self):
        cpp, _ = mod.patch(*source_fixture())
        self.assertIn("(r.layer==12 && r.expert==13)", cpp)
        self.assertIn("(r.layer==13 && r.expert==22)", cpp)
        self.assertIn("(r.layer==13 && r.expert==28)", cpp)
        self.assertIn("(r.layer==24 && r.expert==26 && idx>=0)", cpp)
        self.assertIn("nAF4!=12 || nQ4!=6 || nQ5!=6", cpp)
        self.assertIn("spans.size()!=60", cpp)
        self.assertIn("fs::file_size(bin)!=43253760", cpp)
        self.assertIn("je01_slot_frozen(r.layer,r.expert)", cpp)

    def test_missing_original_bias_sentinel_fails_closed(self):
        cpp, c = source_fixture()
        with self.assertRaisesRegex(ValueError, "NOT_READY"):
            mod.patch(cpp.replace("if (bias_off==-1)", "if (missing)"), c)

    def test_missing_unique_cbatch_marker_fails_closed(self):
        cpp, c = source_fixture()
        with self.assertRaisesRegex(ValueError, "SCRATCH_ANCHOR"):
            mod.patch(cpp, c.replace(mod.OLD_COUNT, "changed_count"))

    def test_no_nll_or_production_promotion_claim(self):
        data = PATH.read_text()
        for marker in (
            '"native_binder_compilation": "NOT_RUN"',
            '"original_E1_AF4_vs_E64_exact_gold_parity": "NOT_RUN"',
            '"candidate_full_27layer_NLL": "NOT_RUN"',
            '"production_modified": False',
            "resolved.is_relative_to(WORK.resolve())"
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, data)


if __name__ == "__main__":
    unittest.main()
