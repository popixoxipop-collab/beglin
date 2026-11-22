#!/usr/bin/env python3
"""Real sealed JE01 full27 NLL secant source and Bayesian bootstrap tests."""
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/"tools/je01_real27_nll_paired_secant_bayesian_bootstrap.py"
spec=importlib.util.spec_from_file_location("je01_secant",PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestJE01Real27SecantBayesianEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=m.audit()

    def test_exact_provenance_and_no_holdout(self):
        d=self.d
        self.assertEqual(d["status"],"JE01_ACTUAL_REAL27_LOSS_SECANT_NOT_EMPIRICAL_FISHER")
        self.assertEqual(d["gold_tokens"],656)
        self.assertEqual(d["sentence_clusters"],16)
        self.assertEqual(d["off_original_AF_output_sha256"],m.mod.ORIGINAL_OUTPUT_SHA)
        self.assertEqual(d["fisher_diagonal_24_roles"],"NOT_MEASURED")
        self.assertEqual(d["independent_validation128"],"NOT_ACCESSED")
        self.assertEqual(d["independent_OOD128"],"NOT_ACCESSED")
        self.assertIs(d["production_modified"],False)

    def test_real_nll_and_sense_of_loss_direction(self):
        d=self.d
        self.assertAlmostEqual(d["baseline_NLL"],2.291799702354307,places=13)
        self.assertAlmostEqual(d["mixed_NLL"],2.294761262072885,places=13)
        self.assertAlmostEqual(d["mean_delta_NLL"],0.002961559718578117,places=13)
        self.assertGreater(d["secant_score_squared_mean"],0)
        self.assertGreater(d["secant_score_variance_about_mean"],0)
        self.assertEqual(d["receipt_sha256"],m.canon({k:v for k,v in d.items() if k!="receipt_sha256"}))
        self.assertEqual(len(d["per_sentence"]),16)
        self.assertEqual(sum(a["gold_tokens"] for a in d["per_sentence"]),656)
        self.assertAlmostEqual(sum(a["delta_NLL_sum"] for a in d["per_sentence"]),
                               656*d["mean_delta_NLL"],places=10)

    def test_seeded_bayesian_bootstrap_deterministic(self):
        data=[-1.0]*8+[1.0]*8
        n=[41]*16
        a=m.bayesian_bootstrap(data,n,draws=150,seed=20261123)
        b=m.bayesian_bootstrap(data,n,draws=150,seed=20261123)
        self.assertEqual(a,b)
        self.assertLess(a["conditional_95pct_interval_delta_NLL"][0],0)
        self.assertGreater(a["conditional_95pct_interval_delta_NLL"][1],0)
        self.assertLess(abs(a["posterior_fraction_candidate_mean_NLL_lower"]-.5),.2)

    def test_pinned_eoe_files_reject_tamper(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,"JE01_REAL_SOURCE_SHA_DRIFT"):
                m.pinned_sources(Path(folder))


if __name__=="__main__":
    unittest.main()
