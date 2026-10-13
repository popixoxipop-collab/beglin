#!/usr/bin/env python3
"""Synthetic-shape P4 full experimental evidence and strict no-promotion gates."""
from __future__ import annotations
import copy,math,unittest
from hcp_p4_evidence_gate import audit,A,B

def cases():
    groups=("astro","physics","chem","biology","math","computing",
            "ml","robotics","energy","economics","history","language")
    risk={"schema":"beglin-hcp-p4-long-actual-af-risk-dataset-v1",
          "status":"ACTUAL_DEEPSEEK_LONGISH_CONTEXT_RISK_TRACES_ONLY",
          "record_count":36,"token_position_count":104*36,
          "train_val_holdout":{"train":12,"validation":12,"holdout":12},
          "production_touched":False,"raw_prompt_token_ids_exported":False,
          "full_logits_exported":False,"native_precision_changed":False,
          "runs":[],"sequences":[]}
    ab={"schema":"beglin-hcp-p4-matched-native-qng64-ab-v1",
        "status":"ACTUAL_MATCHED_NATIVE_POLICY_A_B_AGREEMENT_COST_ONLY",
        "sample_pairs":36,"observed_policy_A":A,"observed_policy_B":B,
        "native_accuracy_vs_human_reference_measured":False,
        "production_touched":False,"active_route_sha_before_after_equal":True,
        "a_b_generated_sequence_exact_agreement_rate":.75,
        "per_policy":{"A":{},"B":{}},"records":[]}
    for i in range(36):
        pid=f"p{i:03d}";domain=groups[i//3]
        split=("train","validation","holdout")[i%3]
        row={"prompt_id":pid,"split":split,"category":domain,
             "prompt_length":104,"rank7_observation_count":104*26,
             "complete_future_32_labels":72,
             "real_min_rank6_rank7_gap_min":.001}
        seq=[]
        for t in range(103):
            seq.append({"position":t,"fast":[.1]*8,"slow4":[.2]*8,"slow16":[.3]*8,
                        "true_future_gap_min":{h:(.001 if t+int(h)<104 else None) for h in ("4","16","32")}})
        risk["runs"].append(row)
        risk["sequences"].append({"prompt_id":pid,"split":split,"domain":domain,"rows":seq})
        cost=lambda d:{"engine_wall_ms":150+d,"roundtrip_ms":180+d,
                       "transition_wall_ms":5,"cache_bytes_added":0,"rss_bytes":20000000}
        ab["records"].append({"prompt_id":pid,"split":split,"domain":domain,"prompt_length":104,
             "matched_policy_order":["A","B"] if i%2==0 else ["B","A"],
             "a_af_reference_next_token_agreement":i%3!=2,
             "b_af_reference_next_token_agreement":i%4!=0,
             "a_b_next_token_agreement":i%4!=0,
             "a_b_generated_sequence_exact_match":i%4!=0,
             "policy_A":cost(0),"policy_B":cost(5)})
    return risk,ab

class EvidenceAuditTest(unittest.TestCase):
    def test_complete_shape_is_study_only(self):
        a,b=cases();out=audit(a,b,bootstrap=40)
        self.assertEqual(out["prompts"],36)
        self.assertEqual(out["complete_input_positions"],3744)
        self.assertEqual(out["future_target_count"]["32"],36*72)
        self.assertEqual(out["paired_policyB_minus_A_mean_ms"]["policyB_minus_A_engine_ms"],5.)
        self.assertFalse(out["native_policy_change_allowed"])
        self.assertFalse(out["automatic_live_promotion"])
    def test_missing_future_observation_blocks(self):
        a,b=cases()
        a["sequences"][0]["rows"][76]["true_future_gap_min"]["32"]=.005
        with self.assertRaisesRegex(ValueError,"incomplete horizon"):
            audit(a,b,bootstrap=30)
    def test_mismatched_policy_or_live_mutation_blocks(self):
        a,b=cases()
        b["active_route_sha_before_after_equal"]=False
        with self.assertRaisesRegex(ValueError,"live serving"):
            audit(a,b,bootstrap=30)
        a,b=cases()
        b["observed_policy_B"]=A
        with self.assertRaisesRegex(ValueError,"policies"):
            audit(a,b,bootstrap=30)
    def test_raw_hidden_not_allowed(self):
        a,b=cases()
        a["raw_prompt_token_ids_exported"]=True
        with self.assertRaisesRegex(ValueError,"raw export"):
            audit(a,b,bootstrap=30)
    def test_missing_a_b_pair_rejected(self):
        a,b=cases();b["records"].pop()
        with self.assertRaisesRegex(ValueError,"missing full real native"):
            audit(a,b,bootstrap=30)

if __name__=="__main__":unittest.main()
