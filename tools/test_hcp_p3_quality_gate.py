#!/usr/bin/env python3
"""A statistically positive true-boundary forecast still cannot deploy itself."""
import copy,unittest
from hcp_p3_quality_gate import decide,QualityGateError,ARMS

def fixture():
    r={"schema":"beglin-hcp-p3-real-topk-boundary-forecast-v1",
       "status":"TRUE_ROUTER_RISK_FORECAST_EVALUATED_RESEARCH_ONLY",
       "true_K_vs_Kplus1_measured":True,
       "full_64_expert_probabilities_before_renormalization":True,
       "native_policy_rebind_allowed":False,"production_touched":False,
       "auto_live_promotion":False,"certified_precision_trigger_training_labels":False,
       "train_prompts":72,"validation_prompts":24,"holdout_prompts":24,
       "supervised_token_positions":{"train":873,"validation":284,"holdout":278},
       "per_arm_validation_choice":{x:{"lam":1,"slow_alpha":0.5} for x in ARMS},
       "heldout":{x:{"prompt_macro_rmse":0.1,"token_rmse":0.1,"mae":0.1,
                       "prompt_count":24,"token_position_count":278,
                       "near_tie_auroc":.60,"near_tie_average_precision":.3} for x in ARMS},
       "paired_vs_flat":{
         "shared_twolevel":{"bootstrap_draws":500,"improvement_vs_flat_prompt_rmse":-.02,
                            "paired_prompt_bootstrap_95ci":[-.08,.06]},
         "separate_twolevel":{"bootstrap_draws":500,"improvement_vs_flat_prompt_rmse":.005,
                              "paired_prompt_bootstrap_95ci":[-.02,.03]}}}
    return r

class TrueBoundaryGateTests(unittest.TestCase):
    def test_nonsignificant_depth_keeps_simple_model(self):
        p=decide(fixture())
        self.assertEqual(p["status"],"NO_VALIDATED_HIERARCHY_GAIN")
        self.assertFalse(p["supports_live_promotion"])
        self.assertFalse(p["supports_real_qng64_policy_switch"])
        self.assertEqual(p["qualified_deeper_arms"],[])
    def test_significant_surrogate_only_research(self):
        o=fixture()
        o["paired_vs_flat"]["separate_twolevel"]={
           "bootstrap_draws":500,"improvement_vs_flat_prompt_rmse":.08,
           "paired_prompt_bootstrap_95ci":[.03,.12]}
        o["heldout"]["separate_twolevel"]["prompt_macro_rmse"]=.092
        o["heldout"]["separate_twolevel"]["near_tie_auroc"]=.63
        d=decide(o)
        self.assertEqual(d["status"],"PROXY_HIERARCHY_RESEARCH_CANDIDATE_ONLY")
        self.assertEqual(d["qualified_deeper_arms"],["separate_twolevel"])
        self.assertFalse(d["supports_live_promotion"])
    def test_auc_regression_blocks_despite_rmse_win(self):
        o=fixture()
        o["paired_vs_flat"]["separate_twolevel"]={
           "bootstrap_draws":500,"improvement_vs_flat_prompt_rmse":.08,
           "paired_prompt_bootstrap_95ci":[.03,.12]}
        o["heldout"]["separate_twolevel"]["prompt_macro_rmse"]=.092
        o["heldout"]["separate_twolevel"]["near_tie_auroc"]=.53
        self.assertEqual(decide(o)["qualified_deeper_arms"],[])
    def test_false_precision_authority_or_selected_only_scores_blocks(self):
        o=fixture()
        o["native_policy_rebind_allowed"]=True
        with self.assertRaisesRegex(QualityGateError,"policy authority"):
            decide(o)
        o=fixture()
        o["true_K_vs_Kplus1_measured"]=False
        with self.assertRaisesRegex(QualityGateError,"not true"):
            decide(o)
    def test_nonfinite_and_short_holdout_rejected(self):
        o=fixture()
        o["heldout"]["flat"]["token_rmse"]=float("nan")
        with self.assertRaisesRegex(QualityGateError,"nonfinite"):
            decide(o)
        o=fixture()
        o["heldout"]["flat"]["prompt_count"]=23
        with self.assertRaisesRegex(QualityGateError,"identities"):
            decide(o)

if __name__=="__main__":unittest.main()
