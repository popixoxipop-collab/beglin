#!/usr/bin/env python3
"""P18 one-item native A/B actual 102400-logit gold NLL probe (not a 480 gate).
Uses SHA-pinned 1024-position scratch binary and the precommitted P18 item
p18-mmlu-000 (107 prompt tokens, 1 gold token). No production promotion.
"""
from __future__ import annotations
import hashlib,json,pathlib,re,sys,time,subprocess
import numpy as np
OLD_ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-native1024-20261015")
sys.path.insert(0,str(OLD_ROOT))
import pilot_driver as pd
ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015")
P18_NATIVE_C_SHA="82475145a4351b934cfccda6763112cdf25751e2ca470ac8417a16e6dc8fdb1c"
P18_NATIVE_BIN_SHA="ed2af527c1291567b0e9e67a9940cb5f13b298d4a645d11742ff1cc34248bc6e"
from hcp_p4_native_policies_ab import B

PILOT_SHA="89f3ac38e032b5b0761e7da788d85c63f0f165009342eb6c75fb4147b6b111e1"
PROTO_SHA="77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c"
PAT=re.compile(r"^P18_NATIVE_GOLD_PREFIX_LOGITS rid=([a-f0-9]{32}) inputsha=([a-f0-9]{64}) plen=(\d+) pos=(\d+) next=(\d+) vocab=(102400) bytes=(409600) floatsha=([a-f0-9]{64})$")
TASK="p18-mmlu-000"

def native_nll(a,target):
 z=a.astype(np.float64)
 m=float(z.max())
 out=m+float(np.log(np.exp(z-m).sum()))-float(z[target])
 if not 0<=out<150:raise RuntimeError("Not valid genuine full-logit NLL")
 return out

def p18_scratch_pids():
 t=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=8)
 name=str(ROOT/"build/qwen_infer_gpu")
 return sorted(int(line.split()[0]) for line in t.splitlines() if name in line)

def main():
 bank,before,resources=pd.preflight()
 if (pd.sha(ROOT/"source/qwen_infer.c")!=P18_NATIVE_C_SHA or
     pd.sha(ROOT/"source/mlx_moe.cpp")!="a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b" or
     pd.sha(ROOT/"build/qwen_infer_gpu")!=P18_NATIVE_BIN_SHA):
  raise RuntimeError("P18 new research observer C/MLX/binary SHA drift")
 if pd.sha(ROOT/"input/pilot.json")!=PILOT_SHA or pd.sha(ROOT/"input/frozen_protocol.json")!=PROTO_SHA:
  raise RuntimeError("Pilot or frozen P18 scoring authority SHA drift")
 item=next((x for x in bank["items"] if x["item_id"]==TASK),None)
 if (not item or len(item["prompt_ids"])!=107 or len(item["gold_ids"])!=1
     or pd.tid(item["prompt_ids"])!=item["prompt_token_sha256"]
     or pd.tid(item["gold_ids"])!=item["gold_token_sha256"]):
  raise RuntimeError("P18 item token SHA or test pilot scope invalid")
 ids=item["prompt_ids"];target=item["gold_ids"][0]
 if p18_scratch_pids():raise RuntimeError("P18 isolated capture process already active")
 work=ROOT/"p18_one_gold_logits_worker";capture=ROOT/"capture"
 if work.exists() or capture.exists():raise RuntimeError("One-shot actual logits already exist; refuse retry")
 capture.mkdir(mode=0o700)
 old=(pd.ps.PERSISTENT_BINARY,pd.ps.EXPECTED_PERSISTENT_BINARY_SHA,pd.ps.WORKER_READY_TIMEOUT_SECONDS,
      pd.ps.REQUEST_TIMEOUT_SECONDS,pd.base.REPO)
 pd.ps.PERSISTENT_BINARY=ROOT/"build/qwen_infer_gpu"
 pd.ps.EXPECTED_PERSISTENT_BINARY_SHA=P18_NATIVE_BIN_SHA
 pd.ps.WORKER_READY_TIMEOUT_SECONDS=180
 pd.ps.REQUEST_TIMEOUT_SECONDS=180
 pd.base.REPO=ROOT/"source"
 w=pd.ps.AdaptivePersistentRouteWorker(route=pd.base.candidate_route(),root=work)
 oldenv=w._env
 def new_env():
  env=oldenv()
  env.update({"BEGLIN_P18_SCRATCH_ONLY":"1","QWEN_MOE_CB_SLOTS":"1",
      "QWEN_MOE_CB_REQS":"1","BEGLIN_HCP_P9_PREPARED_CACHE":"1",
      "BEGLIN_P18_LOGITS_DIR":str(capture),
      "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
      "TOKENIZERS_PARALLELISM":"false"})
  return env
 w._env=new_env
 rows=[]
 try:
  initial=w.start()
  ack=pd.grc.read_runtime_ack(w.ack_path)
  if not initial.get("alive") or ack.get("active_policy_hash")!=pd.pc.policy_hash(pd.A):
   raise RuntimeError("Initial A policy ACK invalid")
  pid=int(initial["pid"])
  prior_epoch=int(ack["weight_epoch"])
  for policy,expected in (("A",pd.A),("B",B)):
   existing={p.name for p in capture.iterdir()}
   before_log_len=len(w.log_path.read_text(errors="replace"))
   start=time.monotonic()
   result=w.submit_with_precision_policy([(ids,1)],target_policy=expected,admission_id="p18-one-gold-real-"+policy)
   trace=[line.strip() for line in w.log_path.read_text(errors="replace")[before_log_len:].splitlines() if "P18_NATIVE_GOLD_PREFIX_LOGITS" in line]
   if len(trace)!=1:raise RuntimeError("Original native C full-vocab capture missing")
   match=PAT.fullmatch(trace[0])
   if match is None:raise RuntimeError("Original native C capture line schema drift")
   rid,token_hash,plen,pos,pred,_,_,expected_sha=match.groups()
   path=capture/(rid+".f32")
   extra={p.name for p in capture.iterdir()}-existing
   if (extra!={path.name} or not path.is_file() or path.is_symlink()
       or path.stat().st_size!=409600 or pd.sha(path)!=expected_sha
       or token_hash!=item["prompt_token_sha256"]
       or int(plen)!=len(ids) or int(pos)!=len(ids)-1):
    raise RuntimeError("Real full logit byte/token identity mismatch")
   z=np.fromfile(path,dtype="<f4")
   health=w.health()
   ack=pd.grc.read_runtime_ack(w.ack_path)
   response=result.get("responses")
   if (z.shape!=(102400,) or not np.isfinite(z).all() or not isinstance(response,list)
       or len(response)!=1 or len(response[0])!=1
       or int(np.argmax(z))!=int(pred) or response[0][0]!=int(pred)
       or ack["active_policy_hash"]!=pd.pc.policy_hash(expected)
       or result.get("finite_logits") is not True or not health.get("alive")
       or int(health["pid"])!=pid or health["rss_bytes"]>pd.MAX_RSS):
    raise RuntimeError("Native actual 102400-logit/ACK/PID/RSS cross-check failed")
   is_switch=(policy=="B")
   epoch=result.get("precision_epoch") or {}
   if (bool(epoch.get("transitioned"))!=is_switch
       or int(ack["weight_epoch"])!=prior_epoch+int(is_switch)):
    raise RuntimeError("Policy/weight epoch transition mismatch")
   if is_switch and (ack.get("status")!="REBIND_SET_APPLIED" or len(epoch.get("changed_targets") or [])!=2
       or epoch.get("txn_id")!=ack.get("txn_id")):
    raise RuntimeError("B terminal ACK not exact")
   prior_epoch=int(ack["weight_epoch"])
   rows.append({"item_id":TASK,"policy":policy,"token_sha256":token_hash,
     "gold_next_token_id":target,"gold_nll_nats":native_nll(z,target),
     "generated_next_token_id":int(pred),"generated_first_matches_gold":int(pred)==target,
     "raw_original_full_logits":path.name,"raw_full_logits_sha256":expected_sha,
     "vocab":102400,"float32_bytes":409600,
     "worker_pid":pid,"ack_sha256":ack["ack_sha256"],"precision_epoch":prior_epoch,
     "B_transition_A_to_B":is_switch,"resident_bytes":int(health["rss_bytes"]),
     "observed_wall_ms_not_latency_certified":round((time.monotonic()-start)*1000,2)})
 finally:
  try:w.stop(force=True)
  finally:
   (pd.ps.PERSISTENT_BINARY,pd.ps.EXPECTED_PERSISTENT_BINARY_SHA,
    pd.ps.WORKER_READY_TIMEOUT_SECONDS,pd.ps.REQUEST_TIMEOUT_SECONDS,pd.base.REPO)=old
 after=pd.live()
 if after!=before or p18_scratch_pids() or len(rows)!=2:raise RuntimeError("Original stage/worker process drift or incomplete A/B capture")
 out={"schema":"beglin-hcp-p18-1-item-673-allowed-native-ab-gold-logits-v1",
  "status":"P18_ONE_ITEM_REAL_NATIVE_A_B_FULL_LOGITS_CAPTURE_PASS_NOT_480_GATE",
  "item_id":TASK,"source_480_token_bank_sha256":bank["source_full_480_token_bank_sha256"],
  "frozen_protocol_sha256":PROTO_SHA,"pilot_source_sha256":PILOT_SHA,
  "binary_sha256":P18_NATIVE_BIN_SHA,"policy_A_B_source":"P9 native precision policy",
  "native_rows":rows,"native_calls":2,"original_live_before":before,"original_live_after":after,
  "actual_BF16_teacher_forwards_in_this_run":0,
  "P18_full_480_NLL_noninferiority_certified":False,
  "production_touched":False,"automatic_live_promotion":False,
  "resource_preflight":resources}
 result=ROOT/"p18-one-gold-native-AB-logits-receipt.json"
 with result.open("x") as f:json.dump(out,f,indent=2,sort_keys=True);f.write("\n")
 print("P18_ONE_ITEM_REAL_AB_FULL_LOGITS_PASS",json.dumps({
   "item_id":TASK,"A_NLL":rows[0]["gold_nll_nats"],"B_NLL":rows[1]["gold_nll_nats"],
   "B_minus_A":rows[1]["gold_nll_nats"]-rows[0]["gold_nll_nats"],
   "raw_logits_SHA256":[x["raw_full_logits_sha256"] for x in rows],
   "receipt_SHA256":pd.sha(result),"production_unchanged":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
