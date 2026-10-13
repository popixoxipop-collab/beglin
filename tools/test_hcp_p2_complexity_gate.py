#!/usr/bin/env python3
"""CI regression: real P2 result does not justify added hierarchy or native switch."""
import copy,json,pathlib,unittest
from hcp_p2_complexity_gate import decide,P2GateError

RESULT=pathlib.Path(__file__).resolve().parents[1]/"experiments/hcp_p2/observed_120prompt_proxy_summary_20261014.json"

def real():
    return json.loads(RESULT.read_text())

class P2DepthSafetyTests(unittest.TestCase):
    def test_real_120_prompt_data_rejects_twolevel(self):
        x=decide(real())
        self.assertEqual(x["status"],"REJECT_COMPLEXITY_KEEP_FLAT_REFERENCE")
        self.assertEqual(x["selected_proxy_prediction_baseline"],"flat")
        self.assertFalse(x["native_policy_change_allowed"])
        self.assertEqual(x["source_heldout_prompt_count"],24)
        self.assertEqual(x["source_heldout_sample_count"],278)
        self.assertEqual(len(x["alternatives"]),2)
        self.assertTrue(all(not r["predictive_depth_gate_pass"] for r in x["alternatives"]))
    def test_optimistic_point_with_ci_crossing_zero_does_not_pass(self):
        x=real()
        x["paired_improvement_vs_flat"]["separate_twolevel"]["improvement_vs_flat_prompt_rmse"]=.03
        x["final_heldout"]["separate_twolevel"]["prompt_macro_rmse"]=.019
        y=decide(x)
        self.assertEqual(y["selected_proxy_prediction_baseline"],"flat")
    def test_even_significant_proxy_is_not_native_authority(self):
        x=real()
        p=x["paired_improvement_vs_flat"]["separate_twolevel"]
        p["improvement_vs_flat_prompt_rmse"]=.15
        p["paired_prompt_bootstrap_95ci"]=[.08,.20]
        x["final_heldout"]["separate_twolevel"]["prompt_macro_rmse"]=.017
        y=decide(x)
        self.assertEqual(y["status"],"PROXY_DEPTH_CANDIDATE_ONLY")
        self.assertEqual(y["selected_proxy_prediction_baseline"],"separate_twolevel")
        self.assertFalse(y["native_policy_change_allowed"])
        self.assertFalse(y["automatic_live_promotion"])
    def test_missing_authority_or_holdout_rejected(self):
        x=real();x["native_policy_rebinding_allowed"]=True
        with self.assertRaisesRegex(P2GateError,"read-only"):
            decide(x)
        x=real();x["final_heldout"]["flat"]["prompt_count"]=23
        with self.assertRaisesRegex(P2GateError,"coverage"):
            decide(x)
    def test_invalid_paired_credibility_rejected(self):
        x=real()
        x["paired_improvement_vs_flat"]["shared_twolevel"]["paired_prompt_bootstrap_95ci"]=[float("nan"),.2]
        with self.assertRaisesRegex(P2GateError,"invalid paired"):
            decide(x)

if __name__=="__main__":unittest.main()
