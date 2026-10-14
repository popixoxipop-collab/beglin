#!/usr/bin/env python3
"""P12 exact P9 cached native test: terminal ACK archived under exclusive lease.

One real DeepSeek MLX worker at a time, same run-owned work root reused safely,
two genuine terminal B transitions archived + three new B/A/B P6-exact
outputs after restart. No live binary/route/worker or release changes.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,shutil,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import gpu_runtime_control as grc
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,sha
from hcp_p12_prestart_journal import PrestartLease,verify_archive_folder,BINARY_SHA,SOURCE_SHA,MODEL_SHA

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p11-inflight-p12-startup-20261014")
P6=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104")
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
P6_QUALITY_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
MAX_RSS=14*1024**3

def shaseq(seq):return hashlib.sha256(",".join(map(str,seq)).encode()).hexdigest()
def liveproof():
 txt=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True)
 return {"stage_sha":sha(STAGE),"route_sha":sha(ROUTE),
 "PIDs":sorted(int(line.split()[0]) for line in txt.splitlines() if str(STAGE) in line)}
def init():
 if not ROOT.is_dir() or ROOT.is_symlink() or ROOT.parent.name!=".tailnet-tmp":
  raise ValueError("P12 research root invalid")
 if sha(ROOT/"build/qwen_infer_gpu")!=BINARY_SHA:
  raise RuntimeError("P12 source-specific native binary differs from P9 certified SHA")
 if sha(ROOT/"src/mlx_moe.cpp")!=SOURCE_SHA or sha(MODEL/"model.safetensors.index.json")!=MODEL_SHA:
  raise RuntimeError("P12 original MLX model or source identity mismatch")
 if sha(P6/"repository/experiments/hcp_p6/evidence/quality.json")!=P6_QUALITY_SHA:
  raise RuntimeError("P12 original native P6 sequences changed")
 if shutil.disk_usage(ROOT).free<20_300_000_000:
  raise RuntimeError("20GB XOX reserve + 300MB margin not available")
 before=liveproof()
 if before!={"stage_sha":STAGE_SHA,"route_sha":ROUTE_SHA,"PIDs":[35481,35493]}:
  raise RuntimeError("P12 original deployment identity mismatch")
 ref=prepare(P6/"inputs/bank",MODEL)["prompts"][0]
 golden={x["prompt_id"]:x for x in
   json.loads((P6/"repository/experiments/hcp_p6/evidence/quality.json").read_text())["records"]}
 expected={p:golden[ref["id"]][p]["sequence_sha256"] for p in ("A","B")}
 return before,ref,expected

def run():
 before,ref,expected=init()
 workerroot=ROOT/"workers"/"native-qualification"
 if workerroot.exists():raise RuntimeError("P12 test never reuses foreign worker")
 workerroot.mkdir()
 old=(ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO)
 ps.PERSISTENT_BINARY=ROOT/"build/qwen_infer_gpu"
 ps.EXPECTED_PERSISTENT_BINARY_SHA=BINARY_SHA
 ps.REQUEST_TIMEOUT_SECONDS=180
 base.REPO=ROOT/"src"
 records=[]
 def newworker():
  w=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=workerroot)
  oldenv=w._env
  def env():
   t=oldenv()
   t["QWEN_MOE_CB_SLOTS"]="1";t["QWEN_MOE_CB_REQS"]="1"
   t["OMP_NUM_THREADS"]="2";t["OPENBLAS_NUM_THREADS"]="2"
   t["BEGLIN_HCP_P9_PREPARED_CACHE"]="1"
   return t
  w._env=env
  return w
 def inference(w,p,number):
  ret=w.submit_with_precision_policy([(ref["token_ids"],4)],
            target_policy={"A":A,"B":B}[p],
            admission_id="p12-native-terminal-"+str(number)+"-"+p)
  rows=ret.get("responses")
  if not isinstance(rows,list) or len(rows)!=1 or not rows[0]:
   raise RuntimeError("P12 missing native inference output")
  seq=shaseq(rows[0])
  if seq!=expected[p]:raise RuntimeError("P12 generated token SHA differs from P6")
  ack=grc.read_runtime_ack(w.ack_path)
  z=w.health()
  ep=ret.get("precision_epoch") or {}
  if (ack["status"]!="REBIND_SET_APPLIED" or
      ack["active_policy_hash"]!=pc.policy_hash({"A":A,"B":B}[p]) or
      ack["txn_id"]!=ep.get("txn_id") or
      len(ep.get("changed_targets",[]))!=2 or
      not z["alive"] or z["rss_bytes"]>=MAX_RSS):
   raise RuntimeError("P12 real native terminal policy/epoch/ACK mismatch")
  return {"policy":p,"seq_sha256":seq,"PID":z["pid"],
          "native_ACK_status":ack["status"],"epoch":ack["weight_epoch"],
          "txn_id_sha256":hashlib.sha256(ack["txn_id"].encode()).hexdigest(),
          "transitioned":True}
 try:
  with PrestartLease(ROOT,workerroot) as lease:
   d=lease.inspect_for_start()
   if d["state"]!="CLEAN_FRESH_START":raise RuntimeError("P12 first startup not clean")
   w=newworker()
   try:
    first=w.start()
    if first["runtime_policy_hash"]!=pc.policy_hash(A):raise RuntimeError("P12 startup policy not A")
    initial_pid=first["pid"]
    records.append(inference(w,"B",0))
   finally:w.stop(force=False)
  with PrestartLease(ROOT,workerroot) as lease:
   d=lease.inspect_for_start()
   if (d["state"]!="KNOWN_TERMINAL_JOURNALED" or
       d["terminal"]!="REBIND_SET_APPLIED"):
    raise RuntimeError("P12 completed B terminal not journaled before restart")
   archived=sorted((workerroot/"p12_journal").iterdir())
   if len(archived)!=1:raise RuntimeError("P12 archived command count mismatch")
   first_receipt=verify_archive_folder(archived[0])
   if first_receipt["decision"]!="TERMINAL_COMMITTED":raise RuntimeError("P12 receipt wrong")
   # Critical: no old txn file survives after durable archival.
   if (workerroot/"txn.cmd").exists():raise RuntimeError("P12 old txn would be replayed")
   w=newworker()
   try:
    restarted=w.start()
    if restarted["pid"]==initial_pid or restarted["runtime_policy_hash"]!=pc.policy_hash(A):
     raise RuntimeError("P12 worker reused wrong PID/policy instead of new isolated startup")
    ack=grc.read_runtime_ack(w.ack_path)
    if ack["status"]!="PROMOTION_APPLIED" or ack.get("txn_id"):
     raise RuntimeError("P12 previous command unexpectedly applied during restart")
    second_pid=restarted["pid"]
    for i,p in enumerate(("B","A","B"),start=1):records.append(inference(w,p,i))
   finally:w.stop(force=False)
  with PrestartLease(ROOT,workerroot) as lease:
   d=lease.inspect_for_start()
   if d["state"]!="KNOWN_TERMINAL_JOURNALED":raise RuntimeError("P12 second terminal not archived")
   folders=sorted((workerroot/"p12_journal").iterdir())
   if len(folders)!=2:raise RuntimeError("P12 expected distinct two real terminal transactions")
   receipt_sha=[sha(f/"VERIFIED.json") for f in folders]
   escrow_hashes=[sha(f/"original.cmd") for f in folders]
   if len(set(escrow_hashes))!=2:raise RuntimeError("P12 reused old transaction instead of fresh")
   # Can safely re-enter prestart gate repeatedly after journal is fully sealed.
   d_again=lease.inspect_for_start()
   if not d_again["ready_for_fresh_start"]:raise RuntimeError("P12 idempotent prestart failed")
  after=liveproof()
  if before!=after:raise RuntimeError("P12 real prestart modified production")
  doc={"schema":"beglin-hcp-p12-real-native-acknowledged-prestart-journal-v1",
    "status":"XOX_SAME_PINNED_GPU_BINARY_TERMINAL_ARCHIVED_AND_FRESH_RESTART_PASS",
    "real_native_p9_binary_sha256":BINARY_SHA,
    "model_index_sha256":MODEL_SHA,
    "p9_mlx_source_sha256":SOURCE_SHA,
    "real_native_first_start_policy":"A",
    "real_native_restart_policy":"A",
    "native_previous_terminal_B_journaled_before_start":True,
    "old_terminal_transactions_escrowed":2,
    "distinct_escrow_command_sha256":escrow_hashes,
    "verified_receipt_sha256":receipt_sha,
    "restart_new_pid":int(second_pid)!=int(initial_pid),
    "archived_old_transaction_was_never_replayed":True,
    "new_P6_matched_generations":len(records),
    "responses":records,
    "real_ack_verified":True,
    "prior_crash_ambiguous_transactions_not_auto_retried":True,
    "P12_gate_never_installed_in_production":True,
    "no_full_BF16_reference":True,
    "live_before":before,"live_after":after,
    "production_touched":False,"automatic_promotion":False}
  out=ROOT/"evidence/P12_native_prestart.json"
  with out.open("x") as f:json.dump(doc,f,sort_keys=True,indent=2);f.write("\n")
  print("P12_REAL_NATIVE_PRESTART_GATE_PASS",json.dumps({
    "committed_txns_journaled":2,"fresh_new_generations":len(records)-1,
    "output_matches_P6":True,"changed_PID":True,
    "receipt_sha256":sha(out),"no_live_modification":True},sort_keys=True),flush=True)
 finally:
  ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO=old

if __name__=="__main__":run()
