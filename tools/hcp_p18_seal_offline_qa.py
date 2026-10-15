#!/usr/bin/env python3
"""Seal actual offline P18 synthetic-gate QA and public selection audit receipts."""
import hashlib,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 tests=ROOT/"test_hcp_p18_quality_gate.py"
 p=subprocess.run([sys.executable,str(tests)],cwd=str(ROOT),capture_output=True,text=True,timeout=55)
 log=p.stdout+p.stderr
 if p.returncode or "Ran 25 tests" not in log or "\nOK\n" not in log:
  raise RuntimeError("P18 synthetic QA did not pass; refusing receipt "+log[-1500:])
 selection=json.loads((ROOT/"candidate_independent_audit.json").read_text())
 if selection["status"]!="P18_SOURCE_SELECTION_INDEPENDENT_AUDIT_PASS" or selection["audited"]!=480:
  raise RuntimeError("independent public selection incomplete")
 logfile=ROOT/"P18_SYNTHETIC_GATE_25_TESTS.log"
 with logfile.open("x") as f:f.write(log)
 manifest={"schema":"beglin-hcp-p18-offline-preparation-qa-receipt-v1",
   "state":"P18_OFFLINE_PREP_QA_PASS_NOT_NATIVE",
   "synthetic_unit_tests_passed":25,"unit_test_exit_code":p.returncode,
   "unit_test_log_sha256":sha(logfile),
   "source_independent_audit_n":480,"source_independent_audit_sha256":sha(ROOT/"candidate_independent_audit.json"),
   "source_manifest_sha256":sha(ROOT/"source_manifest.json"),
   "candidate_bank_sha256":sha(ROOT/"candidate_bank_PRIVATE.json"),
   "selection_receipt_sha256":sha(ROOT/"candidate_selection_receipt.json"),
   "gate_code_sha256":sha(ROOT/"hcp_p18_quality_gate.py"),
   "test_code_sha256":sha(tests),
   "sampler_code_sha256":sha(ROOT/"hcp_p18_sample_public.py"),
   "download_code_sha256":sha(ROOT/"hcp_p18_download_sources.py"),
   "source_auditor_code_sha256":sha(ROOT/"hcp_p18_independent_source_audit.py"),
   "draft_protocol_sha256":sha(ROOT/"p18_protocol_DRAFT.json"),
   "external_inference_done":False,"BF16_P18_done":False,
   "native_A_B_P18_done":False,"production_touched":False,"automatic_live_promotion":False,
   "completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
 receipt=ROOT/"P18_OFFLINE_PREP_QA_RECEIPT.json"
 with receipt.open("x") as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":manifest["state"],"tests":25,"selection_audit":480,
  "QA_SHA256":sha(receipt),"log_sha256":sha(logfile)},sort_keys=True))
if __name__=="__main__":main()
