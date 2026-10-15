#!/usr/bin/env python3
"""Seal corrected P18 Git-SHA-1/Parquet-SHA-256 real-source gate regression."""
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
GATE_SHA="237d6ce68c714f8e5098bf212796bde965abc669f3fc03147eb50ffd9bba18de"
TEST_SHA="61d40e514b9ab007bcc4ea74d1ad74185407a107e71904df8495831c25b848b9"
PROTO_SHA="77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c"
COHORT_SHA="19e7080b7208b8c7dd6fd06a656bb400dd2a9f72ca5f5674a80f5cc9a654cf65"
CONTRACT_AUDIT_SHA="80d39b127ef6b9606ea6066cca5d4af5fcf52e6ed60bfe3dfb9c0d5c273090c4"
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def main():
 pairs=[("hcp_p18_quality_gate.py",GATE_SHA),("test_hcp_p18_quality_gate.py",TEST_SHA),
        ("p18_protocol_FROZEN.json",PROTO_SHA),("p18_gate_cohort_PRIVATE.json",COHORT_SHA),
        ("p18_frozen_protocol_contract_receipt.json",CONTRACT_AUDIT_SHA)]
 if any(sha(ROOT/name)!=h for name,h in pairs):raise RuntimeError("corrected gate/protocol source SHA drift")
 proc=subprocess.run([sys.executable,str(ROOT/"test_hcp_p18_quality_gate.py")],capture_output=True,text=True,timeout=45)
 log=proc.stdout+proc.stderr
 if proc.returncode!=0 or "Ran 25 tests" not in log or "\nOK\n" not in log:
  raise RuntimeError("corrected P18 synthetic QA failed")
 logpath=ROOT/"P18_CORRECTED_GATE_25_TESTS.log";outpath=ROOT/"P18_CORRECTED_GATE_QA_RECEIPT.json"
 if logpath.exists() or outpath.exists():raise FileExistsError("no mutable QA log replacement")
 with logpath.open("x") as f:f.write(log)
 payload={"schema":"beglin-hcp-p18-git-revision-fix-regression-v1",
  "status":"P18_CORRECTED_GATE_AND_FROZEN_COHORT_QA_PASS_NOT_NATIVE",
  "source_git_revision_format":"40 lowercase hex",
  "parquet_sha256_format":"64 lowercase hex",
  "new_gate_sha256":GATE_SHA,"new_test_sha256":TEST_SHA,
  "frozen_protocol_sha256":PROTO_SHA,"frozen_cohort_sha256":COHORT_SHA,
  "frozen_contract_audit_sha256":CONTRACT_AUDIT_SHA,
  "qa_cases":25,"qa_exit":proc.returncode,"qa_log_sha256":sha(logpath),
  "real_native_BF16_forward_count":0,"real_native_AB_forward_count":0,
  "production_touched":False,"automatic_live_promotion":False}
 with outpath.open("x") as f:json.dump(payload,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":payload["status"],"qa_pass":25,
   "testlog_sha256":sha(logpath),"receipt_sha256":sha(outpath)},sort_keys=True))
if __name__=="__main__":main()
