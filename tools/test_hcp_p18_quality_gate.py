#!/usr/bin/env python3
"""P18 pure synthetic tests: no model, dataset, network, or production writes."""
import copy,hashlib,json,pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import hcp_p18_quality_gate as g

def sha_obj(obj):
    return hashlib.sha256((json.dumps(obj,sort_keys=True)+"\n").encode()).hexdigest()

class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=pathlib.Path(self.temp.name)
        self.protocol={
          "schema":g.PROTOCOL_SCHEMA,"status":"FROZEN_BEFORE_EVALUATION",
          "incumbent":"A","challenger":"B","automatic_live_promotion":False,
          "statistics":{"seed":g.SEED,"bootstrap_replicates":10000,
            "familywise_alpha":0.05,"bonferroni_endpoints":3,
            "nll_margin_nats_per_gold_token":0.05,
            "accuracy_margin_B_minus_A":-0.05,
            "latency_geomean_ratio_ceiling":1.10,
            "p95_latency_ratio_ceiling":1.15},
          "datasets":{s:{"revision":"a"*64,"raw_dataset_sha256":"b"*64,
                         "license":"TEST_ONLY","split":"test","n":160}
                      for s in g.SOURCES}}
        qs=[];scores=[]
        for s in g.SOURCES:
            for i in range(160):
                ident=f"{s}-{i:03d}"
                cap=(8,32,128)[i%3];shots=(0,3,5)[i%3]
                q={"item_id":ident,"dataset":s,"dataset_revision":"a"*64,"split":"test",
                   "record_sha256":"c"*64,"prompt_token_sha256":"d"*64,
                   "gold_token_sha256":"e"*64,"gold_token_count":2,
                   "prompt_token_count":100,"max_new_tokens":cap,
                   "decode_steps":cap,"shot_count":shots}
                qs.append(q)
                trial={"prefill_ms":25.0,"decode_ms":75.0,"total_ms":100.0,
                       "decode_steps":cap,"precision_switch_excluded":True,"cache_mode":"warm"}
                tcorrect=(i < 64 if s!="gsm8k" else i<32)
                a={"gold_token_nll_nats":[2.0,2.0],
                   "independently_verified_task_correct":i<80,
                   "latency_trials":[copy.deepcopy(trial) for _ in range(3)],
                   "raw_evidence_sha256":"f"*64}
                b=copy.deepcopy(a); b["gold_token_nll_nats"]=[2.01,2.01]
                scores.append({"item_id":ident,"dataset":s,"record_sha256":q["record_sha256"],
                  "prompt_token_sha256":q["prompt_token_sha256"],"gold_token_sha256":q["gold_token_sha256"],
                  "BF16":{"gold_token_nll_nats":[1.0,1.0],
                    "independently_verified_task_correct":tcorrect,
                    "raw_evidence_sha256":"1"*64},"A":a,"B":b})
        self.cohort={"schema":g.COHORT_SCHEMA,"items":qs}
        self.scores={"schema":g.SCHEMA,"protocol_sha256":"","cohort_sha256":"",
          "independent_raw_full_logits_audit_passed":True,
          "independent_task_output_audit_passed":True,
          "production_touched":False,"original_serving_unchanged":True,
          "policy_A_B_identity_verified":True,"original_BF16_identity_verified":True,
          "independent_audit_manifest_sha256":"1"*64,
          "teacher_source_manifest_sha256":"2"*64,
          "native_source_manifest_sha256":"3"*64,
          "records":scores}
    def write(self,name,obj):
        path=self.root/name
        path.write_text(json.dumps(obj,sort_keys=True)+"\n",encoding="utf-8")
        return path
    def check(self):
        protocol=self.write("protocol.json",self.protocol)
        cohort=self.write("cohort.json",self.cohort)
        self.scores["protocol_sha256"]=g.sha_file(protocol)
        self.scores["cohort_sha256"]=g.sha_file(cohort)
        scores=self.write("scores.json",self.scores)
        return g.gate(protocol,cohort,scores,self.scores["protocol_sha256"],self.scores["cohort_sha256"])
    def rejects(self):
        with self.assertRaises((ValueError,TypeError,KeyError,ZeroDivisionError)):
            self.check()
    def test_01_synthetic_zero_margin_pass_never_promotes(self):
        result=self.check()
        self.assertTrue(result["research_gate_pass"])
        self.assertFalse(result["automatic_live_promotion"])
        self.assertFalse(result["raw_logits_reverification_performed_here"])
        self.assertEqual(result["question_cluster_units"],480)
    def test_02_nll_noninferiority_exceeded(self):
        for r in self.scores["records"]:
            r["B"]["gold_token_nll_nats"]=[2.25,2.25]
        self.assertFalse(self.check()["research_gate_pass"])
    def test_03_task_accuracy_drop(self):
        for r in self.scores["records"]:
            r["B"]["independently_verified_task_correct"]=False
        self.assertFalse(self.check()["research_gate_pass"])
    def test_04_latency_regression(self):
        for r in self.scores["records"]:
            for t in r["B"]["latency_trials"]:
                t["decode_ms"]=125.0;t["total_ms"]=150.0
        self.assertFalse(self.check()["research_gate_pass"])
    def test_05_positive_control_failure(self):
        for r in self.scores["records"]:
            r["BF16"]["independently_verified_task_correct"]=False
        self.assertFalse(self.check()["research_gate_pass"])
    def test_06_unfrozen_protocol_fails(self):
        self.protocol["status"]="DRAFT_SOURCE_IDENTITY_PENDING"
        self.rejects()
    def test_07_changed_margin_fails(self):
        self.protocol["statistics"]["nll_margin_nats_per_gold_token"]=0.2
        self.rejects()
    def test_08_wrong_dataset_revision_fails(self):
        self.cohort["items"][0]["dataset_revision"]="9"*64
        self.rejects()
    def test_09_duplicate_question_fails(self):
        self.cohort["items"][0]["item_id"]=self.cohort["items"][1]["item_id"]
        self.rejects()
    def test_10_missing_question_fails(self):
        self.cohort["items"].pop()
        self.rejects()
    def test_11_wrong_split_fails(self):
        self.protocol["datasets"]["mmlu"]["split"]="train"
        self.rejects()
    def test_12_no_raw_audit_fails(self):
        self.scores["independent_raw_full_logits_audit_passed"]=False
        self.rejects()
    def test_13_wrong_independent_source_identity_fails(self):
        self.scores["records"][0]["record_sha256"]="5"*64
        self.rejects()
    def test_14_missing_teacher_evidence_fails(self):
        self.scores["records"][0]["BF16"]["raw_evidence_sha256"]=None
        self.rejects()
    def test_15_nonfinite_logits_rejected(self):
        self.scores["records"][0]["A"]["gold_token_nll_nats"]=[2.0,float("nan")]
        self.rejects()
    def test_16_wrong_gold_token_count_fails(self):
        self.scores["records"][0]["A"]["gold_token_nll_nats"]=[2.0]
        self.rejects()
    def test_17_inconsistent_time_breakdown_fails(self):
        self.scores["records"][0]["A"]["latency_trials"][0]["total_ms"]=50.0
        self.rejects()
    def test_18_latency_precision_transition_included_fails(self):
        self.scores["records"][0]["A"]["latency_trials"][0]["precision_switch_excluded"]=False
        self.rejects()
    def test_19_latency_unfixed_decode_steps_fails(self):
        self.scores["records"][0]["A"]["latency_trials"][0]["decode_steps"]=2
        self.rejects()
    def test_20_missed_output_audit_fails(self):
        self.scores["independent_task_output_audit_passed"]=False
        self.rejects()
    def test_21_not_untouched_production_fails(self):
        self.scores["production_touched"]=True
        self.rejects()
    def test_22_missing_24_tokens_long_cap_stratum_fails(self):
        for q in self.cohort["items"]:
            if q["dataset"]=="mmlu" and q["max_new_tokens"]==128:
                q["max_new_tokens"]=32
        self.rejects()
    def test_23_substitute_tokens_for_questions_fails(self):
        self.cohort["items"]=self.cohort["items"][:31]
        self.rejects()
    def test_24_cannot_make_unpinned_protocol_pass(self):
        self.protocol["datasets"]["mmlu"]["raw_dataset_sha256"]=None
        self.rejects()
    def test_25_wrong_source_evidence_manifest_fails(self):
        self.scores["native_source_manifest_sha256"]="not-a-hash"
        self.rejects()

if __name__=="__main__":unittest.main(verbosity=2)
