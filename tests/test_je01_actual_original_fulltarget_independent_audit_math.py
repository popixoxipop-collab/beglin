#!/usr/bin/env python3
"""JE01 native full-target raw NLL independent audit statistical safety tests."""
import math,pathlib,sys,unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import je01_original256_fulltarget_independent_audit as audit

class NativeRealAFCalGoldBaselineIndependentContracts(unittest.TestCase):
    def test_case_bootstrap_baseline_no_false_paired_bayesian_CI(self):
        rows=[{"nll_sum":1.5*k,"tokens":k} for k in ([1,3,5,9]*64)]
        r=audit.resample_ci(rows,200,12345)
        self.assertEqual(r["n_case_resamples"],200)
        self.assertTrue(r["not_paired_precision_allocation_improvement"])
        self.assertEqual(r["real_source_train_mean_gold_token_NLL_CI_95"],[1.5,1.5])
        with self.assertRaisesRegex(ValueError,"NOT_256"):
            audit.resample_ci(rows[:255],200,1)

    def test_case_level_confidence_percentiles_deterministic(self):
        a=list(range(1,257))
        self.assertEqual(audit.percentile(a,.025),6)
        self.assertEqual(audit.percentile(a,.975),249)
        self.assertEqual(audit.percentile(a,.50),128)

    def test_native_real_pinned_training_source_and_scope(self):
        src=(ROOT/"tools"/"je01_original256_fulltarget_independent_audit.py").read_text()
        self.assertIn('"actual_full_target_per_prompt_NLL_p95"',src)
        self.assertIn('"all_11024_real_gold_tokens_independently_checked"',src)
        self.assertIn('"native_Bayesian_expert_bit_quality_alternatives":"NOT_MEASURED"',src)
        self.assertIn('"baseline_CI_is_not_a_precision_improvement_CI":True',src)
        self.assertIn('JE01_ORIGINAL_AF_FULL_TARGET_GOLD_NLL_ALL256.json',src)
        self.assertIn('JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1',src)

if __name__=="__main__":
    unittest.main(verbosity=2)
