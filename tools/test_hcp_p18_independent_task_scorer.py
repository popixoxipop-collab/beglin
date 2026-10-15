#!/usr/bin/env python3
"""Negative/source-tamper tests for P18 independent task scoring.
Synthetic predictions only; no BF16 or A/B inference is claimed by this QA.
"""
import copy,hashlib,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import hcp_p18_independent_task_scorer as a

class FakeRefs:
 def __init__(self):
  self.tok,self.byid=a.pin_authority()
  self.mmlu=self.byid["p18-mmlu-000"]
  self.arc=self.byid["p18-arc_challenge-000"]
  self.gsm=self.byid["p18-gsm8k-000"]
 def full(self,rr,ids=None,finish_reason="eos"):
  ref,f=rr
  return a.score_complete(self.tok,ref,f,
        list(f["gold_ids"] if ids is None else ids),finish_reason)
 def encode(self,text):
  return self.tok.encode(text,add_special_tokens=False).ids
 def signed(self,rr,policy="A",answer=None):
  ref,f=rr
  tokens=list(f["gold_ids"] if answer is None else answer)+[a.EOS_TOKEN_ID]
  return {"item_id":f["item_id"],"dataset":f["dataset"],
          "record_sha256":f["record_sha256"],
          "prompt_token_sha256":f["prompt_token_sha256"],
          "policy":policy,"generated_token_ids":tokens,
          "generated_token_sha256":a.shaids(tokens),
          "finish_reason":"eos"}

class TestP18TaskRules(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.refs=FakeRefs()
 def test_01_all480_pin_and_P17_id_replay(self):
  self.assertEqual(len(self.refs.byid),480)
 def test_02_all_3_datasets_complete_gold_with_EOS(self):
  for rr in (self.refs.mmlu,self.refs.arc,self.refs.gsm):
   expected=rr[1]["gold_ids"]+[a.EOS_TOKEN_ID]
   with self.subTest(rr[1]["item_id"]):
    self.assertTrue(self.refs.full(rr,expected)["correct"])
 def test_03_mmlu_wrong_single_letter_is_incorrect(self):
  ref,f=self.refs.mmlu;wrong=next(i for i in "ABCD" if i!=ref["reference_answer"])
  self.assertFalse(self.refs.full(self.refs.mmlu,self.refs.encode(wrong)+[a.EOS_TOKEN_ID])["correct"])
 def test_04_arc_wrong_single_letter_is_incorrect(self):
  ref,f=self.refs.arc;wrong=next(i for i in "ABCDE" if i!=ref["reference_answer"])
  self.assertFalse(self.refs.full(self.refs.arc,self.refs.encode(wrong)+[a.EOS_TOKEN_ID])["correct"])
 def test_05_mmlu_extra_explanation_is_wrong_not_correct(self):
  text=self.refs.mmlu[0]["reference_answer"]+" because"
  self.assertFalse(self.refs.full(self.refs.mmlu,self.refs.encode(text)+[a.EOS_TOKEN_ID])["correct"])
 def test_06_mmlu_question_prefix_is_wrong(self):
  text="Answer: "+self.refs.mmlu[0]["reference_answer"]
  self.assertFalse(self.refs.full(self.refs.mmlu,self.refs.encode(text)+[a.EOS_TOKEN_ID])["correct"])
 def test_07_gsm_exacts_decimal_numeric_equivalence(self):
  text=self.refs.gsm[0]["reference_answer"]
  self.assertTrue(self.refs.full(self.refs.gsm,self.refs.encode("0"+text if not text.startswith("-") else text)+[a.EOS_TOKEN_ID])["correct"])
 def test_08_gsm_decimal_with_units_is_incorrect(self):
  txt=self.refs.gsm[0]["reference_answer"]+" dollars"
  self.assertFalse(self.refs.full(self.refs.gsm,self.refs.encode(txt)+[a.EOS_TOKEN_ID])["correct"])
 def test_09_gsm_chain_of_thought_is_incorrect(self):
  txt="#### "+self.refs.gsm[0]["reference_answer"]
  self.assertFalse(self.refs.full(self.refs.gsm,self.refs.encode(txt)+[a.EOS_TOKEN_ID])["correct"])
 def test_10_numeric_exponent_refused(self):
  self.assertIsNone(a.numeric_value("1e3"))
 def test_11_numeric_comma_grouping_valid(self):
  self.assertEqual(a.numeric_value("1,000.0"),a.numeric_value("1000"))
 def test_12_nonfinite_ignored(self):
  self.assertIsNone(a.numeric_value("NaN"))
  self.assertIsNone(a.numeric_value("Infinity"))
 def test_13_incomplete_one_token_in_8_cap_rejected(self):
  with self.assertRaises(a.SourceOrOutputError):
   self.refs.full(self.refs.mmlu,list(self.refs.mmlu[1]["gold_ids"]),"max_new_tokens")
 def test_14_no_finish_reason_rejected(self):
  with self.assertRaises(a.SourceOrOutputError):
   self.refs.full(self.refs.mmlu,self.refs.mmlu[1]["gold_ids"],"unknown")
 def test_15_claim_eos_without_eos_rejected(self):
  with self.assertRaises(a.SourceOrOutputError):
   self.refs.full(self.refs.mmlu,self.refs.mmlu[1]["gold_ids"],"eos")
 def test_16_bos_generated_rejected(self):
  with self.assertRaises(a.SourceOrOutputError):
   self.refs.full(self.refs.mmlu,[a.BOS_TOKEN_ID]+self.refs.mmlu[1]["gold_ids"]+[a.EOS_TOKEN_ID])
 def test_17_interior_eos_rejected(self):
  with self.assertRaises(a.SourceOrOutputError):
   self.refs.full(self.refs.mmlu,[a.EOS_TOKEN_ID]+self.refs.mmlu[1]["gold_ids"]+[a.EOS_TOKEN_ID])
 def test_18_out_of_vocab_rejected(self):
  with self.assertRaises(a.SourceOrOutputError):
   self.refs.full(self.refs.mmlu,[a.MAX_VOCAB,a.EOS_TOKEN_ID])
 def test_19_bool_token_rejected(self):
  with self.assertRaises(a.SourceOrOutputError):
   self.refs.full(self.refs.mmlu,[True,a.EOS_TOKEN_ID])
 def test_20_over_cap_rejected(self):
  with self.assertRaises(a.SourceOrOutputError):
   self.refs.full(self.refs.mmlu,list(self.refs.mmlu[1]["gold_ids"])+[3]*9+[a.EOS_TOKEN_ID])
 def test_21_correct_source_and_raw_generation_hash_only_not_attested(self):
  row=self.refs.signed(self.refs.arc,policy="B")
  result=a.audit_raw_generation_record(self.refs.tok,self.refs.byid,row)
  self.assertTrue(result["correct"])
  self.assertFalse(result["authoritative_task_accuracy_allowed"])
 def test_22_bad_generation_sha_rejected(self):
  row=self.refs.signed(self.refs.arc);row["generated_token_sha256"]="0"*64
  with self.assertRaises(a.SourceOrOutputError):
   a.audit_raw_generation_record(self.refs.tok,self.refs.byid,row)
 def test_23_bad_prompt_sha_rejected(self):
  row=self.refs.signed(self.refs.arc);row["prompt_token_sha256"]="0"*64
  with self.assertRaises(a.SourceOrOutputError):
   a.audit_raw_generation_record(self.refs.tok,self.refs.byid,row)
 def test_24_unknown_question_rejected(self):
  row=self.refs.signed(self.refs.arc);row["item_id"]="unknown"
  with self.assertRaises(a.SourceOrOutputError):
   a.audit_raw_generation_record(self.refs.tok,self.refs.byid,row)
 def test_25_wrong_policy_rejected(self):
  row=self.refs.signed(self.refs.arc,policy="unknown")
  with self.assertRaises(a.SourceOrOutputError):
   a.audit_raw_generation_record(self.refs.tok,self.refs.byid,row)
 def test_26_manually_specified_text_not_accepted_as_decoded_evidence(self):
  row=self.refs.signed(self.refs.arc);row["generated_token_ids"]=[]
  with self.assertRaises(a.SourceOrOutputError):
   a.audit_raw_generation_record(self.refs.tok,self.refs.byid,row)
 def test_27_wrong_benchmark_type_rejected(self):
  ref,f=copy.deepcopy(self.refs.gsm)
  f["task_scoring"]="unknown"
  with self.assertRaises(a.SourceOrOutputError):
   a.score_complete(self.refs.tok,ref,f,f["gold_ids"]+[a.EOS_TOKEN_ID],"eos")
 def test_28_corrupt_reference_does_not_score_correct(self):
  ref,f=copy.deepcopy(self.refs.arc);ref["reference_answer"]="UNKNOWN"
  out=a.score_complete(self.refs.tok,ref,f,f["gold_ids"]+[a.EOS_TOKEN_ID],"eos")
  self.assertFalse(out["correct"])
 def test_29_maxnew_correctly_requires_exact_cap(self):
  ref,f=copy.deepcopy(self.refs.mmlu)
  f["max_new_tokens"]=len(f["gold_ids"])
  self.assertTrue(a.score_complete(self.refs.tok,ref,f,f["gold_ids"],"max_new_tokens")["correct"])
 def test_30_numeric_ambiguity_is_false(self):
  gold=self.refs.gsm[0]["reference_answer"]
  self.assertFalse(self.refs.full(self.refs.gsm,self.refs.encode(gold+" "+gold)+[a.EOS_TOKEN_ID])["correct"])

if __name__=="__main__":unittest.main(verbosity=2)
