#!/usr/bin/env python3
"""Qv13 durable, nonadaptive native C 256-item fixed-mask research sweep.
Batch order/policies are fixed and immutable; no result-dependent reranking.
"""
import argparse, hashlib, json, os, pathlib, subprocess, sys, time
from typing import Dict
ROOT=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023")
NATIVE=ROOT/"tools/qv13_native_256.py"
RECHECK=ROOT/"tools/qv13_native_log_crosscheck.py"
PRE=ROOT/"configs/research/BEGLIN_QV13_ACTUAL_NATIVE_EVAL_PRELOCK_20261023.json"
FREEZE=ROOT/"docs/research/BEGLIN_QV13_ORIGINAL_NATIVE_256_PREEXEC_LOCK_PUBLIC_20261023.json"
LOCK=LAB/"qv13_lock_private.json"
PROGRESS=LAB/"qv13_full256_sequential_progress.jsonl"
EXPECTED={
"native":"e6a48a2b8a0da8300cd3caa7690ec748122adf610ecdb6d542247cdf85fa552c",
"recheck":"f7391223e791325ca9b203479776954b240a7aa7a78d8f3da58d40337a992f36",
"pre":"5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a",
"freeze":"9d045e1056e5ec8bc3a093a8a3d198277cbbc4083fb054a929b566cb2bb03c8c",
"lock":"79826eb944ac2a4f87b637a430471f069a3789da37cdb12a9b7d1060d49db9b8"}
FIXED_BATCHES=("arc128","ob128")
FIXED_ARMS=("q4","source_qv8","random73","random137","random211","random389","random577","original_v2")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1<<18),b""):h.update(chunk)
 return h.hexdigest()
def public_checks():
 for title,file in [("native",NATIVE),("recheck",RECHECK),("pre",PRE),("freeze",FREEZE),("lock",LOCK)]:
  if not file.is_file() or sha(file)!=EXPECTED[title]:
   raise RuntimeError("Qv13 fixed sweep prohibited, authority drift: "+title)
 lock=json.loads(LOCK.read_text())
 if lock["status"]!="FROZEN_BEFORE_QV13_NATIVE_C_FORWARD":
  raise RuntimeError("preexecution lock not frozen")
 if tuple(lock["dataset_groups"])!=FIXED_BATCHES or len(lock["masks"])!=7:
  raise RuntimeError("frozen data/policy cardinality changed")
 for batch,n in (("pilot16",16),("arc128",128),("ob128",128)):
  manifest=LAB/"native_runs"/batch/"input_manifest.json"
  if sha(manifest)!=lock["batches"][batch]["manifest_sha256"] or lock["batches"][batch]["count"]!=n:
   raise RuntimeError("frozen prompt cohort changed "+batch)
 return lock
def audit(batch,require_complete):
 run=subprocess.run([sys.executable,str(RECHECK),"--batch",batch],capture_output=True,text=True,timeout=60)
 if run.returncode:
  raise RuntimeError("independent raw C audit error "+run.stderr[-750:])
 report=json.loads(run.stdout)
 if require_complete and (report["status"]!="ALL_EIGHT_NATIVE_C_LOG_REPARSE_PASS" or len(report["tested_arms"])!=8):
  raise RuntimeError("cannot treat incomplete native logs as terminal")
 return report
def append_event(item):
 PROGRESS.parent.mkdir(parents=True,exist_ok=True)
 with PROGRESS.open("ab") as f:
  f.write((json.dumps(item,sort_keys=True,ensure_ascii=False)+"\n").encode())
  f.flush();os.fsync(f.fileno())
 PROGRESS.chmod(0o600)
def already(batch,arm,lock):
 receipt=LAB/"native_runs"/batch/arm/"native_qv13_original_C_receipt.json"
 logfile=LAB/"native_runs"/batch/arm/"native_qv13_original_C_run.log"
 if not receipt.exists():
  if logfile.exists():raise RuntimeError("incomplete log without receipt, STOP, no overwrite "+batch+":"+arm)
  return False
 data=json.loads(receipt.read_text())
 if not logfile.exists() or sha(logfile)!=data["native_log_sha256"] or data["status"]!="ORIGINAL_NATIVE_28L_REAL_Q4_OR_28N8_SCORING_PASS" or data["preexecution_lock_sha256"]!=EXPECTED["lock"] or data["input_manifest_sha256"]!=lock["batches"][batch]["manifest_sha256"] or data["arm"]!=arm:
  raise RuntimeError("preexisting receipt inconsistent; STOP "+batch+":"+arm)
 return True
def progress():
 lock=public_checks()
 masks={b:{a:already(b,a,lock) for a in FIXED_ARMS} for b in FIXED_BATCHES}
 return {"status":"SOURCE_FROZEN_RESEARCH_NO_LIVE_PROMOTION","completed":masks,"total_passed":sum(v for a in masks.values() for v in a.values()),"registered_runs":16,"progress_path":str(PROGRESS)}
def run():
 lock=public_checks()
 audit("pilot16",True)
 print("Qv13_FIXED_FULL_256_BEGIN",json.dumps(progress(),sort_keys=True),flush=True)
 try:
  os.nice(10)  # best effort lower CPU priority; original native math/kernel unchanged
 except OSError:pass
 for batch in FIXED_BATCHES:
  for arm in FIXED_ARMS:
   public_checks()
   if already(batch,arm,lock):
    print("QV13_FULL256_SKIP_VERIFIED",batch,arm,flush=True)
    continue
   append_event({"step":"START","batch":batch,"arm":arm,"epoch_s":int(time.time())})
   t=time.monotonic()
   result=subprocess.run([sys.executable,str(NATIVE),"eval","--batch",batch,"--arm",arm],
              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=3600)
   if result.returncode:
    append_event({"step":"FAIL","batch":batch,"arm":arm,"epoch_s":int(time.time()),
       "exit_code":result.returncode,"stderr_tail":result.stderr[-300:]})
    raise RuntimeError("one frozen original C job failed; stopped without retry: "+batch+" "+arm)
   if not already(batch,arm,lock):
    raise RuntimeError("native returned SUCCESS but receipt missing: "+batch+" "+arm)
   rec=LAB/"native_runs"/batch/arm/"native_qv13_original_C_receipt.json"
   msg={"step":"PASS","batch":batch,"arm":arm,"wall_s":round(time.monotonic()-t,2),
        "receipt_sha256":sha(rec),"epoch_s":int(time.time())}
   append_event(msg)
   print("QV13_FULL256_PASS",json.dumps(msg,sort_keys=True),flush=True)
  summary=audit(batch,True)
  append_event({"step":"INDEPENDENT_BATCH_PASS","batch":batch,"n_native_rows":summary["original_C_rows_reparsed"],"epoch_s":int(time.time())})
  print("QV13_FULL256_BATCH_INDEPENDENT_PASS",batch,summary["original_C_rows_reparsed"],flush=True)
 print("QV13_FULL256_ALL_NATIVE_16_ARMS_PASS",json.dumps(progress(),sort_keys=True),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser()
 parser.add_argument("action",choices=["check","run"])
 args=parser.parse_args()
 if args.action=="check":
  print(json.dumps(progress(),sort_keys=True))
 else:run()
