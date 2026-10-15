#!/usr/bin/env python3
"""P18 raw evidence integrity regression tests (read-only origin; synthetic tampering).
Run with exactly the same source and tokenizer SHA as the frozen P18 contract.
"""
from __future__ import annotations
import copy,json,pathlib,sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import hcp_p18_real_raw_evidence_coverage as p
import hcp_p18_independent_task_scorer as task

class SourceTamperAuditTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tok,cls.refs=task.pin_authority()
  cls.raw_arc=json.loads((p.M/"native_ARC_GSM_8AB_RECEIPT.json").read_text())
  cls.raw_single=json.loads((p.E/"P18_ONE_REAL_AB_RECEIPT.json").read_text())
 def test_01_real_BF16_recorded_teacher_four(self):
  self.assertEqual(len(p.TEACHER),4)
 def test_02_real_native_BF16_teacher_original_exact_position_sum(self):
  self.assertEqual(sum(self.refs[k][1]["gold_token_count"] for k in p.TEACHER),6)
 def test_03_all_real_full_native_gold_receipts_10(self):
  self.assertEqual(len(self.raw_arc["rows"])+len(self.raw_single["native_rows"]),10)
 def test_04_real_capture_log_one_parse_two(self):
  self.assertEqual(len(p.parse_actual_native_log(p.E/"p18_AB_capture_worker.log",p.GPU1_LOG_SHA)),2)
 def test_05_real_capture_log_eight_parse(self):
  self.assertEqual(len(p.parse_actual_native_log(p.M/"native_ARC_GSM_8AB_worker.log",p.GPU_MULTI_LOG_SHA)),8)
 def test_06_log_tampering_fails_sha_before_parse(self):
  with tempfile.TemporaryDirectory() as t:
   path=pathlib.Path(t)/"unsafe.log";path.write_text("synthetic changed source\n")
   with self.assertRaises(ValueError):p.parse_actual_native_log(path,p.GPU1_LOG_SHA)
 def test_07_original_serving_runtime_consistent(self):
  p.check_before_after(self.raw_arc)
  p.check_before_after(self.raw_single)
 def test_08_original_stage_digest_mutation_rejected(self):
  q=copy.deepcopy(self.raw_arc);q["original_serving_after"]["stage_sha256"]="0"*64
  with self.assertRaises(ValueError):p.check_before_after(q)
 def test_09_original_stage_PID_mutation_rejected(self):
  q=copy.deepcopy(self.raw_arc);q["original_serving_before"]["stage_pids"]=[35481]
  with self.assertRaises(ValueError):p.check_before_after(q)
 def test_10_scratch_PID_leak_mutation_rejected(self):
  q=copy.deepcopy(self.raw_arc)
  q["original_serving_before"]["scratch_pids"]=[12345]
  q["original_serving_after"]["scratch_pids"]=[12345]
  with self.assertRaises(ValueError):p.check_before_after(q)
 def test_11_incomplete_stage_snapshot_rejected(self):
  with self.assertRaises(ValueError):p.check_before_after({})
 def test_12_logprob_requires_102400_logits(self):
  with self.assertRaises(ValueError):p.logprobs(np.zeros(102399,dtype=np.float32))
 def test_13_logprob_rejects_nan(self):
  x=np.zeros(102400,dtype=np.float32);x[0]=np.nan
  with self.assertRaises(ValueError):p.logprobs(x)
 def test_14_logprob_rejects_infinity(self):
  x=np.zeros(102400,dtype=np.float32);x[0]=np.inf
  with self.assertRaises(ValueError):p.logprobs(x)
 def test_15_uniform_distribution_gives_log_vocab(self):
  x=np.zeros(102400,dtype=np.float32)
  self.assertAlmostEqual(float(-p.logprobs(x)[0]),float(np.log(102400)),places=10)
 def test_16_missing_raw_file_fails(self):
  with self.assertRaises(ValueError):p.checksha(pathlib.Path("/nonexistent/P18_raw.f32"),"0"*64)
 def test_17_tampered_raw_bytes_fail(self):
  with tempfile.TemporaryDirectory() as t:
   f=pathlib.Path(t)/"P18_wrong.f32"
   f.write_bytes(b"\x00\x00\x00\x00"*102400)
   with self.assertRaises(ValueError):p.checksha(f,p.GPU1_RECEIPT)
 def test_18_symlink_source_fails(self):
  with tempfile.TemporaryDirectory() as t:
   root=pathlib.Path(t);a=root/"a";a.write_bytes(b"a");b=root/"b";b.symlink_to(a)
   with self.assertRaises(task.SourceOrOutputError):task.digest(b)
 def test_19_one_token_under_cap_eight_not_task_complete(self):
  r,f=self.refs["p18-mmlu-000"]
  self.assertGreater(f["max_new_tokens"],1)
  with self.assertRaises(task.SourceOrOutputError):
   task.score_complete(self.tok,r,f,list(f["gold_ids"]),"max_new_tokens")
 def test_20_source_original_tensor_hashes_same(self):
  values={p.scorer.digest(entry[4]) for entry in p.TEACHER.values()}
  self.assertEqual(values,{p.TENSOR_SHA})
 def test_21_evidence_partial_truthfulness(self):
  j=json.loads((p.ROOT/"P18_REAL_RAW_LOGIT_EVIDENCE_COVERAGE.json").read_text())
  self.assertEqual(j["original_BF16_question_windows_verified"],4)
  self.assertEqual(j["native_paired_A_B_question_windows_verified"],3)
  self.assertEqual(j["native_full_vocab_float32_A_B_vectors_verified"],10)
  self.assertFalse(j["statistical_480_question_noninferiority_certified"])
  self.assertEqual(j["complete_greedy_generations_with_native_finish_reason_verified"],0)
 def test_22_no_raw_evidence_from_gold_source_fabrication(self):
  z=self.refs["p18-gsm8k-000"][1]
  with self.assertRaises(task.SourceOrOutputError):
   task.score_complete(self.tok,*self.refs["p18-gsm8k-000"],list(z["gold_ids"]),"not_observed")
 def test_23_source_teacher_no_bf16_generated_task_claim(self):
  j=json.loads((p.ROOT/"P18_REAL_RAW_LOGIT_EVIDENCE_COVERAGE.json").read_text())
  self.assertEqual(j["BF16_original_full_generations_with_native_finish_reason_verified"],0)
  self.assertIs(j["independent_full_480_task_accuracy_verified"],False)
 def test_24_full_480_gold_positions_still_incomplete(self):
  j=json.loads((p.ROOT/"P18_REAL_RAW_LOGIT_EVIDENCE_COVERAGE.json").read_text())
  self.assertLess(j["native_paired_gold_positions_verified"],j["native_paired_gold_positions_required"])
  self.assertLess(j["original_BF16_gold_positions_verified"],j["original_BF16_gold_positions_required"])
 def test_25_unmodified_original_header_sha(self):
  self.assertEqual(p.scorer.digest(p.ROOT/"tokenizer_exact_XOX/tokenizer.json"),
                   task.SOURCES["tokenizer_exact_XOX/tokenizer.json"])

if __name__=="__main__":unittest.main(verbosity=2)
