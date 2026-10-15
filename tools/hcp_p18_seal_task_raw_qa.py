#!/usr/bin/env python3
"""Seal P18 independent task-output and REAL native raw source audit regression.

P18 raw benchmark text, source token arrays and raw logits remain private D50.
Only source SHA, summary coverage, test counts and no-production status may
be committed to the private research repository. No inference launched.
"""
from __future__ import annotations
import hashlib,json,pathlib,subprocess,sys
R=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
SOURCES={
 "hcp_p18_independent_task_scorer.py":"588a7b03f1f931e6a79a0358dc2bc85ef8cc6818e3235ef53a8113264bc6e28f",
 "test_hcp_p18_independent_task_scorer.py":"fe62818b2486713818276034f235818ef420a109d9c82f44979e52b5825a94c0",
 "hcp_p18_real_raw_evidence_coverage.py":"239e6b476c9a163c68063283280a7e3b719bf55563c905eb7f44531f76f59c55",
 "test_hcp_p18_real_raw_evidence_coverage.py":"cffb080dc8be5e8974a037322d3866ce2c2e643f3b4a79c0a4f7fc28afaa4e88",
 "P18_REAL_RAW_LOGIT_EVIDENCE_COVERAGE.json":"02d440e8a73603b230a7c39f7d2ea65ae938414f616d25dbbd1c0f8034e47d10",
 "hcp_p18_quality_gate.py":"237d6ce68c714f8e5098bf212796bde965abc669f3fc03147eb50ffd9bba18de",
 "p18_protocol_FROZEN.json":"77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c",
 "p18_gate_cohort_PRIVATE.json":"19e7080b7208b8c7dd6fd06a656bb400dd2a9f72ca5f5674a80f5cc9a654cf65"}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 if not R.is_dir() or R.is_symlink():raise RuntimeError("P18 D50 source root missing")
 for fn,expected in SOURCES.items():
  p=R/fn
  if p.is_symlink() or not p.is_file() or sha(p)!=expected:
   raise RuntimeError("P18 evaluation source SHA changed: "+fn)
 receipt=R/"P18_TASK_AND_REAL_RAW_QA_RECEIPT.json";log=R/"P18_TASK_AND_REAL_RAW_55_TESTS.log"
 if receipt.exists() or log.exists():raise FileExistsError("Refuse replacement of sealed QA evidence")
 results=[];full=[]
 for fn,count in (("test_hcp_p18_independent_task_scorer.py",30),
                  ("test_hcp_p18_real_raw_evidence_coverage.py",25)):
  p=subprocess.run([sys.executable,str(R/fn)],capture_output=True,text=True,timeout=45)
  output=p.stdout+p.stderr
  if p.returncode!=0 or f"Ran {count} tests" not in output or "\nOK\n" not in output:
   raise RuntimeError("P18 actual test suite unexpectedly failed: "+fn)
  results.append({"script":fn,"expected_tests":count,"exit_code":p.returncode})
  full.append("$ python3 "+fn+"\n"+output)
 # Verify quality gate cannot certify fake or absent 480 scored generation.
 sys.path.insert(0,str(R))
 import hcp_p18_quality_gate as gate
 protocol=gate.frozen_protocol(json.loads((R/"p18_protocol_FROZEN.json").read_text()))
 cohort=gate.frozen_cohort(json.loads((R/"p18_gate_cohort_PRIVATE.json").read_text()),protocol)
 if len(cohort)!=480:raise RuntimeError("Scoring gate no longer ties to 480 questions")
 incomplete={"schema":gate.SCHEMA,"protocol_sha256":SOURCES["p18_protocol_FROZEN.json"],
  "cohort_sha256":SOURCES["p18_gate_cohort_PRIVATE.json"],
  "independent_raw_full_logits_audit_passed":False,
  "independent_task_output_audit_passed":False,"records":[]}
 try:
  gate.validate_rows(protocol,cohort,incomplete,SOURCES["p18_protocol_FROZEN.json"],SOURCES["p18_gate_cohort_PRIVATE.json"])
 except gate.GateInputError as exc:
  gate_refusal=str(exc)
 else:raise RuntimeError("Incomplete original BF16/native A/B raw evidence was incorrectly admitted to quality gate")
 full.append("P18 real 480-question gate incomplete-source fail-closed: "+gate_refusal+"\n")
 with log.open("x") as f:f.write("\n".join(full))
 raw=json.loads((R/"P18_REAL_RAW_LOGIT_EVIDENCE_COVERAGE.json").read_text())
 if(raw.get("original_BF16_question_windows_verified")!=4
   or raw.get("native_paired_A_B_question_windows_verified")!=3
   or raw.get("native_full_vocab_float32_A_B_vectors_verified")!=10
   or raw.get("complete_greedy_generations_with_native_finish_reason_verified")!=0
   or raw.get("statistical_480_question_noninferiority_certified") is not False):
  raise RuntimeError("P18 source coverage receipt fails closure policy")
 out={"schema":"beglin-hcp-p18-task-and-real-raw-evidence-QA-v1",
  "status":"P18_INDEPENDENT_TASK_AND_RAW_EVIDENCE_QA_PASS_NOT_480_QUALITY",
  "question_authority":480,"source_task_rules":["letter_exact","numeric_exact"],
  "QA_cases":55,"QA_exit_code":0,"test_groups":results,
  "gate_rejection_of_incomplete_480_raw_evidence":True,
  "raw_generation_native_witness_score_not_attested_by_task_grammar_alone":True,
  "actual_original_BF16_question_windows":4,
  "actual_native_A_B_question_windows":3,
  "original_full_vocab_A_B_vectors":10,
  "real_task_accuracy_certified":False,"statistical_noninferiority_certified":False,
  "source_files_sha256":SOURCES,
  "test_log_sha256":sha(log),
  "production_touched":False,"automatic_live_promotion":False}
 with receipt.open("x") as f:json.dump(out,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":out["status"],"tests":out["QA_cases"],"real_BF16_questions":4,
   "real_native_A_B_questions":3,"real_full_vocab_vectors":10,
   "gate_refusal":gate_refusal,"QA_receipt_sha256":sha(receipt),
   "QA_log_sha256":sha(log)},sort_keys=True))
if __name__=="__main__":main()
