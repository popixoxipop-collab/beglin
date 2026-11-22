#!/usr/bin/env python3
"""Independent reproducibility test of real EOE DeepSeek-V2-Lite 27-layer NLL.

Uses immutable native OFF/ON+original signed source receipts, not synthetic
substitutes, and reports in-sample TRAIN16 pilot data only. Does NOT establish
population quality, a Fisher/Hessian estimate, or validation/OOD improvement.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"docs"/"research"
AUDITOR=ROOT/"tools"/"je01_cal240_rolewise_fullgold16_independent_audit.py"
SOURCE={
    "baseline":("JE01_REAL27_ORIGINAL_AF4_656GOLD_BASELINE_20261122.json",
                "1289088dd3cff691b9a2a8120442e221346b3fd281656fe129ea30443ae9d09e"),
    "off":("JE01_REAL27_CAL240_ROLEWISE_OFF_656GOLD_NLL_20261122.json",
           "22f5ce69b63208e3bd0a6a6bb29e119183110b90517c4e2afb3f7d57095fe661"),
    "on":("JE01_REAL27_CAL240_ROLEWISE_ON_656GOLD_NLL_20261122.json",
          "54afc4f7e7faa86b6c4a9c21286a95cae02856de2d6e3c8dfa309bb1011c0510"),
    "audit":("JE01_REAL27_CAL240_ROLEWISE_656GOLD_INDEPENDENT_NLL_AUDIT_20261122.json",
             "93b256b8470ce2a988f68d1c71e6a6634171544a72ba071a53ab76989585a369"),
}
spec=importlib.util.spec_from_file_location("je01_cal240_real_native_656gold_audit",AUDITOR)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestJE01Real27NativeOffOnNLL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths={}
        for key,(name,expected_sha) in SOURCE.items():
            path=DATA/name
            assert path.is_file() and not path.is_symlink()
            raw=path.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==expected_sha,(key,"UNEXPECTED_REAL_SOURCE_SHA")
            cls.paths[key]=path

    def test_all_original_receipt_signatures_and_output_sha(self):
        original=mod.signed_json(self.paths["baseline"],mod.BASELINE_RECEIPT)
        off=mod.signed_json(self.paths["off"])
        on=mod.signed_json(self.paths["on"])
        self.assertEqual(original["native_output_sha256"],mod.ORIGINAL_OUTPUT_SHA)
        self.assertEqual(off["real_native_output_sha256"],mod.ORIGINAL_OUTPUT_SHA)
        self.assertEqual(off["receipt_sha256"],
                         "e8c9e76d4bbf5e0b0846a4c48c03afca1042bff04869cfd46ec8bcce649baec6")
        self.assertEqual(on["receipt_sha256"],
                         "a73442c5cc2fd4fe8fdacee96fb8b4e5b74939412fd2ef6de09f9b1001b8d845")
        self.assertIs(on["production_checkpoint_modified"],False)
        self.assertIs(on["true_source_heldout_ood_labels_accessed"],False)

    def test_real_native_656gold_paired_nll_recomputed(self):
        rec=mod.audit(self.paths["baseline"],self.paths["off"],self.paths["on"])
        self.assertEqual(rec["status"],"JE01_REAL_CAL240_ROLEWISE_TRAIN16_EXPLORATORY_NLL_AUDIT")
        self.assertEqual(rec["gold_tokens"],656)
        self.assertEqual(rec["cases"],16)
        self.assertEqual((rec["worsened_cases"],rec["improved_cases"],rec["unchanged_cases"]),(10,6,0))
        self.assertAlmostEqual(rec["baseline_NLL"],2.291799702354307,places=14)
        self.assertAlmostEqual(rec["candidate_NLL"],2.294761262072885,places=14)
        self.assertAlmostEqual(rec["paired_delta_candidate_minus_baseline"],
                               0.002961559718578117,places=13)
        self.assertAlmostEqual(rec["relative_NLL_change_pct"],
                               0.12922419509592323,places=11)
        self.assertLess(rec["paired_mean_per_token_delta_cluster_bootstrap_percentile95"][0],0)
        self.assertGreater(rec["paired_mean_per_token_delta_cluster_bootstrap_percentile95"][1],0)
        self.assertAlmostEqual(rec["bootstrap_fraction_lower_NLL"],0.07904,places=5)
        self.assertTrue(rec["no_independent_holdout"])
        self.assertEqual(rec["full_240_prompt_independent_NLL"],"NOT_RUN")

    def test_independent_eoe_signed_audit_matches_archived_recompute(self):
        saved=json.loads(self.paths["audit"].read_bytes())
        actual=mod.audit(self.paths["baseline"],self.paths["off"],self.paths["on"])
        # Equal exact deterministic evidence, including seeded 50k bootstrap.
        self.assertEqual(saved,actual)


if __name__=="__main__":
    unittest.main()
