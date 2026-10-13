"""Regression for high-retention true channel-count and strict research gates."""
import copy
import unittest
from high_retention_gate import keep_count,inspect_result,FRACTIONS,WIDTHS

def fixture():
    payload={"schema":"beglin-prompt-holdout-micro-router-v1",
       "status":"EVALUATED_NOT_PROMOTABLE",
       "prompt_count":120,
       "prompt_split_counts":{"train":72,"validation":24,"holdout":24},
       "raw_hidden_uploaded":False,"production_touched":False,
       "native_qng64_parity_tested":False,"intermediate":1408,
       "dataset_manifest_sha256":"a"*64,"checkpoint_index_sha256":"b"*64,
       "results":{}}
    for width in WIDTHS:
        ng=1408//int(width)
        rows={}
        for f in FRACTIONS:
            k=keep_count(1408,int(width),float(f))
            err=0.0 if k==ng else 0.025
            methods={m:{"relative_l2":err} for m in
                ("static","dynamic","dynamic_ridge_residual","dynamic_rank8_residual")}
            rows[f]={"tupper_roundtrip":True,
                "idealized_channel_compute_fraction_excluding_gate_and_corrector":k/ng,
                "method_selected_by_validation":"dynamic_ridge_residual",
                "locked_holdout":{"relative_l2":err,"prompt_bootstrap_95ci":[err,err+.005],"prompt_count":24},
                "holdout_methods":methods}
        payload["results"][width]=rows
    return payload

class HighRetentionTests(unittest.TestCase):
    def test_true_discrete_retention_geometry(self):
        self.assertEqual([keep_count(1408,16,f) for f in (.9,.95,.99)],[80,84,88])
        self.assertEqual([keep_count(1408,64,f) for f in (.9,.95,.99)],[20,21,22])
    def test_noop_99_percent_is_not_a_speed_candidate(self):
        row=inspect_result(fixture())
        z=[r for r in row["results"] if r["requested_retention"]==.99]
        self.assertEqual(len(z),2)
        self.assertTrue(all(r["no_op_full_execution"] for r in z))
        self.assertTrue(all(not r["local_quality_robust_gate"] for r in z))
        self.assertEqual(len(row["eligible_for_further_shadow_validation"]),6)
        self.assertTrue(row["all_native_model_quality_gates_blocked"])
    def test_reject_wrong_runtime_fraction(self):
        payload=fixture()
        payload["results"]["16"]["0.95"]["idealized_channel_compute_fraction_excluding_gate_and_corrector"]=.95
        with self.assertRaisesRegex(ValueError,"fraction/cost mismatch"):
            inspect_result(payload)
    def test_reject_bad_or_missing_evidence(self):
        payload=fixture()
        payload["raw_hidden_uploaded"]=True
        with self.assertRaisesRegex(ValueError,"privacy"):
            inspect_result(payload)
        payload=fixture()
        payload["results"]["64"]["0.9"]["tupper_roundtrip"]=False
        with self.assertRaisesRegex(ValueError,"Tupper"):
            inspect_result(payload)
    def test_robust_quality_gate_uses_holdout_ci(self):
        payload=fixture()
        payload["results"]["16"]["0.9"]["locked_holdout"]["prompt_bootstrap_95ci"]=[.024,.06]
        result=inspect_result(payload)
        k=next(r for r in result["results"] if r["width"]==16 and r["requested_retention"]==.9)
        self.assertTrue(k["local_quality_point_gate"])
        self.assertFalse(k["local_quality_robust_gate"])
        self.assertFalse(k["auto_promotion_allowed"])

if __name__=="__main__":unittest.main()
