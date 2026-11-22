#!/usr/bin/env python3
"""Fail-closed contract tests for JE01 post-NLL B/C byte-frozen arms."""
import importlib.util
from pathlib import Path
import unittest

SRC=Path(__file__).resolve().parents[1]/"tools/je01_postnll_bc_exactbyte_native_bundle.py"
spec=importlib.util.spec_from_file_location("je01_postnll_bc_bundle",SRC)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class TestJE01BCPhysicalPrereg(unittest.TestCase):
    def test_B_C_exact_source_and_presealed_set(self):
        p,b,B=mod.frozen_policy("B")
        _,c,C=mod.frozen_policy("C")
        self.assertEqual(len(B),len(C))
        self.assertEqual(len(B),24)
        for assignment in (B,C):
            counts={k:list(assignment.values()).count(k) for k in ("AF4","Q4","Q5")}
            self.assertEqual(counts,{"AF4":12,"Q4":6,"Q5":6})
            self.assertEqual(sum(mod_path for mod_path in [
                1802240*counts["AF4"],1622016*counts["Q4"],1982464*counts["Q5"]]),43253760)
        self.assertEqual(b["id"],"B_PROTECT_EARLY_GATE")
        self.assertEqual(c["id"],"C_PROTECT_EARLY_UP")
        self.assertEqual({(l,e,r) for (l,e,r),t in B.items() if t=="Q4" and l in (12,13)},
                         {(12,13,"up_proj"),(13,22,"up_proj"),(13,28,"up_proj")})
        self.assertEqual({(l,e,r) for (l,e,r),t in C.items() if t=="Q4" and l in (12,13)},
                         {(12,13,"gate_proj"),(13,22,"gate_proj"),(13,28,"gate_proj")})
        self.assertEqual(sum(v=="Q5" for v in B.values()),6)
        self.assertEqual({r for r,t in B.items() if t=="Q5"},
                         {r for r,t in C.items() if t=="Q5"})
        self.assertEqual(B[(14,63,"gate_proj")],"AF4")
        self.assertEqual(B[(14,63,"up_proj")],"AF4")
        self.assertEqual(B[(14,63,"down_proj")],"AF4")

    def test_ablation_not_bayesian_gradient_claim(self):
        s=SRC.read_text()
        self.assertIn('"actual_fisher_or_parameter_gradients":"NOT_MEASURED"',s)
        self.assertIn('"native_original27_true_gold_NLL":"NOT_RUN"',s)
        self.assertIn('"heldout_validation_OOD_used":False',s)
        self.assertIn('original.load_ledger()',s)
        self.assertIn('original.build(dest)',s)

    def test_unpreregistered_arm_rejected(self):
        with self.assertRaisesRegex(ValueError,"ONLY_PREREG_B_C"):
            mod.frozen_policy("D")


if __name__=="__main__":
    unittest.main()
