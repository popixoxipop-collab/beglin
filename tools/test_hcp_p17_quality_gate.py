#!/usr/bin/env python3
"""P17 real-source research quality gate adversarial and paired bootstrap tests."""
from __future__ import annotations
import copy,json,pathlib,sys,unittest
from hcp_p17_quality_gate import gate,one_sided_bounds,longest_common_prefix_score,FROZEN_SHA

REPO=pathlib.Path(__file__).resolve().parents[1]
FROZEN=REPO/"experiments/hcp_p17/P17_FROZEN_TOKENS.json"

class P17GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank=json.loads(FROZEN.read_text())
        assert cls.bank["question_count"]==12
    def example(self):
        records=[]
        for item in self.bank["items"]:
            gold=item["gold_answer_ids"]
            n=len(gold)
            cell={"gold_token_nll_nats":[1.0]*n,
                  "native_greedy_token_ids":(gold+[0]*8)[:8],
                  "real_source_logits_manifest_sha256":"a"*64}
            records.append({"item_id":item["id"],"domain":item["domain"],
                "gold_answer_sha256":item["gold_answer_sha256"],"gold_token_count":n,
                "original_BF16_teacher_gold_nll_nats":[.8]*n,
                "A":copy.deepcopy(cell),"B":copy.deepcopy(cell)})
        return {"schema":"beglin-hcp-p17-independent-real-teacher-native-question-scores-v1",
            "frozen_token_sha256":FROZEN_SHA,
            "verified_actual_BF16_gold_logit_positions":31,
            "verified_actual_native_gold_logit_positions":62,
            "verified_native_real_greedy_generations":24,
            "independent_actual_logits_source_audit_passed":True,
            "production_touched":False,"records":records}
    def check(self,d):return gate(FROZEN,d)
    def rejects(self,d):
        with self.assertRaises((ValueError,RuntimeError,KeyError,TypeError)):
            self.check(d)
    def test_01_paired_zero_delta_pass_internal_only(self):
        v=self.check(self.example());self.assertTrue(v["research_internal_gate_pass"])
        self.assertFalse(v["automatic_live_promotion"])
        self.assertFalse(v["actual_end_to_end_task_accuracy_certified"])
    def test_02_bad_loss_margin_blocked(self):
        d=self.example()
        for rec in d["records"]:rec["B"]["gold_token_nll_nats"]=[1.4]*rec["gold_token_count"]
        v=self.check(d);self.assertFalse(v["research_internal_gate_pass"])
        self.assertGreater(v["B_minus_A_nll_question_bootstrap"]["upper_one_sided_95"],.05)
    def test_03_bad_greedy_prefix_accuracy_blocked(self):
        d=self.example()
        for rec in d["records"]:rec["B"]["native_greedy_token_ids"]=[101999]*8
        v=self.check(d);self.assertFalse(v["research_internal_gate_pass"])
        self.assertLess(v["B_minus_A_prefix_accuracy_question_bootstrap"]["lower_one_sided_95"],-.05)
    def test_04_missing_question_refused(self):
        d=self.example();d["records"].pop();self.rejects(d)
    def test_05_duplicate_item_refused(self):
        d=self.example();d["records"][1]["item_id"]=d["records"][0]["item_id"];self.rejects(d)
    def test_06_gold_answer_sha_mismatch_refused(self):
        d=self.example();d["records"][1]["gold_answer_sha256"]="0"*64;self.rejects(d)
    def test_07_wrong_gold_length_refused(self):
        d=self.example();d["records"][0]["B"]["gold_token_nll_nats"]=[1.];self.rejects(d)
    def test_08_zero_nan_gold_loss_refused(self):
        d=self.example();d["records"][0]["B"]["gold_token_nll_nats"][0]=float('nan');self.rejects(d)
    def test_09_incorrect_p15_reference_count_refused(self):
        d=self.example();d["verified_actual_BF16_gold_logit_positions"]=30;self.rejects(d)
    def test_10_missing_native_prefixes_refused(self):
        d=self.example();d["verified_actual_native_gold_logit_positions"]=61;self.rejects(d)
    def test_11_missing_native_greedy_refused(self):
        d=self.example();d["verified_native_real_greedy_generations"]=23;self.rejects(d)
    def test_12_silent_source_not_audited_refused(self):
        d=self.example();d["independent_actual_logits_source_audit_passed"]=False;self.rejects(d)
    def test_13_invalid_raw_prefix_ids_refused(self):
        d=self.example();d["records"][0]["B"]["native_greedy_token_ids"]=[102400];self.rejects(d)
    def test_14_early_EOS_single_output_valid(self):
        d=self.example();d["records"][0]["B"]["native_greedy_token_ids"]=[d["records"][0]["A"]["native_greedy_token_ids"][0]]
        v=self.check(d);self.assertEqual(v["question_units"],12)
    def test_15_longer_than_max_generation_refused(self):
        d=self.example();d["records"][0]["B"]["native_greedy_token_ids"]=[7]*9;self.rejects(d)
    def test_16_untrusted_orig_checkpoint_claim_refused(self):
        d=self.example();d["frozen_token_sha256"]="f"*64;self.rejects(d)
    def test_17_wrong_orig_domain_refused(self):
        d=self.example();d["records"][0]["domain"]="untrusted";self.rejects(d)
    def test_18_clustering_by_question_not_gold_token_count(self):
        d=self.example()
        # All 12 question differences identical regardless of 1..4 target tokens.
        for x in d["records"]:x["B"]["gold_token_nll_nats"]=[1.02]*x["gold_token_count"]
        v=self.check(d)
        self.assertEqual(v["B_minus_A_nll_question_bootstrap"]["cluster_unit"],"question_not_token")
        self.assertAlmostEqual(v["B_minus_A_nll_question_bootstrap"]["mean"],.02)
    def test_19_exact_margins_and_bootstrap_seed(self):
        v=self.check(self.example())
        self.assertEqual(v["noninferiority_nll_margin_nats_per_token"],.05)
        self.assertEqual(v["B_minus_A_prefix_accuracy_question_bootstrap"]["seed"],20261015)
        self.assertEqual(v["B_minus_A_prefix_accuracy_question_bootstrap"]["replicates"],10000)
    def test_20_reject_token_level_pseudo_replication(self):
        with self.assertRaises(ValueError):one_sided_bounds([0.]*31)
    def test_21_reject_unsupported_bootstrap_seed(self):
        with self.assertRaises(ValueError):one_sided_bounds([0.]*12,seed=123)
    def test_22_reject_unsupported_bootstrap_resample_count(self):
        with self.assertRaises(ValueError):one_sided_bounds([0.]*12,reps=100)
    def test_23_longest_correct_prefix(self):
        self.assertAlmostEqual(longest_common_prefix_score([1,2,9,4],[1,2,3,4]),.5)
    def test_24_native_source_provenance_missing_refused(self):
        d=self.example();d["records"][0]["A"]["real_source_logits_manifest_sha256"]="";self.rejects(d)

if __name__=="__main__":unittest.main(verbosity=2)
