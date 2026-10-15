#!/usr/bin/env python3
"""P18 one frozen 480-cohort question: original P9 A/B GPU full-vocab F32 logits.

One *real* 27-layer native source, 673 SHA-pinned gold prefixes only, safe
P18 1024-position observer. Original production stage/route/PIDs unchanged.
The resulting forced-prefix argmax is NOT a complete task generation.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,pathlib,re,subprocess,sys,time
import numpy as np,psutil

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015")
BATCH=ROOT/"batch480-work"
OLDER=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-native1024-20261015")
sys.path.insert(0,str(OLDER))
import pilot_driver as pd
from hcp_p4_native_policies_ab import B
FROZEN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
C_SOURCE="fa8a328bedc1e56e79938460e1d4a171f0fd363535fc610183b4022e41ce0bfa"
MLX_SOURCE="a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b"
C_BIN="bf8a70ec5f0e449c3e2c35aeeccc3932dbafe0a376322570d0ebd0a0cae8f81d"
PAT=re.compile(r"^P18_NATIVE_GOLD_PREFIX_LOGITS rid=([a-f0-9]{32}) inputsha=([a-f0-9]{64}) plen=(\d+) pos=(\d+) next=(\d+) vocab=(102400) bytes=(409600) floatsha=([a-f0-9]{64})$")
BYTES=409600
RESERVE=20*1024**3+85*1024**2

def sha(p):
 p=pathlib.Path(p)
 if p.is_symlink() or not p.is_file():raise RuntimeError("refuse missing or symlink native source "+str(p))
 h=hashlib.sha256()
 with p.open("rb") as f:
  for part in iter(lambda:f.read(1024*1024),b""):h.update(part)
 return h.hexdigest()
def tok(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def nll(v,g):
 a=np.asarray(v,dtype=np.float64)
 if a.shape!=(102400,) or not np.isfinite(a).all():raise ValueError("not real full finite float32 102400 original vocab")
 m=float(np.max(a));x=m+math.log(float(np.exp(a-m).sum()))-float(a[g])
 if not 0<=x<150:raise ValueError("native full logits gold loss not finite")
 return x
def live_check(expected):
 now=pd.live()
 if now!=expected:raise RuntimeError("original production stage or PID changed")
def native_actors():
 out=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=8)
 target=str(ROOT/"build_long/qwen_infer_gpu")
 return [int(line.split()[0]) for line in out.splitlines() if target in line]
def pre(index):
 if BATCH.is_symlink() or not BATCH.is_dir() or ROOT.is_symlink() or ROOT.parent.name!=".tailnet-tmp":
  raise RuntimeError("unregistered P18 research source root")
 if sha(BATCH/"p18_multishot_frozen_PRIVATE.json")!=FROZEN:raise RuntimeError("480 original source SHA drift")
 for path,digest in ((ROOT/"source_long/qwen_infer.c",C_SOURCE),
                     (ROOT/"source_long/mlx_moe.cpp",MLX_SOURCE),
                     (ROOT/"build_long/qwen_infer_gpu",C_BIN)):
  if sha(path)!=digest:raise RuntimeError("real native source or binary identity drift")
 b=json.loads((BATCH/"p18_multishot_frozen_PRIVATE.json").read_text())
 if len(b["items"])!=480 or not 0<=index<480:raise ValueError("frozen test count/index invalid")
 q=b["items"][index];ids=q["prompt_ids"];gold=q["gold_ids"]
 if(not 1<=len(gold)<=8 or len(ids)+len(gold)>1024 or
    not 2<=len(ids) or q["prompt_token_sha256"]!=tok(ids) or
    q["gold_token_sha256"]!=tok(gold) or
    q["gold_token_count"]!=len(gold) or q["prompt_token_count"]!=len(ids)):
  raise RuntimeError("invalid gold source token preimage")
 if native_actors():raise RuntimeError("other P18 source-bound native worker still running")
 if psutil.disk_usage(str(ROOT)).free<RESERVE or psutil.virtual_memory().available<17*1024**3:
  raise RuntimeError("XOX real resource admission blocked")
 before=pd.live()
 if before!={"stage_sha256":pd.STAGE_SHA,"route_sha256":pd.ROUTE_SHA,
             "stage_pids":[35481,35493],"scratch_pids":[]}:
  raise RuntimeError("P18 normal stage or live serving different from pinned original")
 capdir=ROOT/"capture"
 if not capdir.is_dir() or capdir.is_symlink():raise RuntimeError("native raw source-bound 673 prefix capture dir missing")
 return q,before
def run(index):
 q,before=pre(index)
 ids=q["prompt_ids"];gold=q["gold_ids"]
 outdir=BATCH/("unit_%03d"%index)/"native"
 if outdir.exists():raise FileExistsError("P18 native one-shot unit already exists")
 if not outdir.parent.exists():outdir.parent.mkdir(mode=0o700)
 outdir.mkdir(mode=0o700)
 old=(pd.ps.PERSISTENT_BINARY,pd.ps.EXPECTED_PERSISTENT_BINARY_SHA,
      pd.ps.WORKER_READY_TIMEOUT_SECONDS,pd.ps.REQUEST_TIMEOUT_SECONDS,pd.base.REPO)
 pd.ps.PERSISTENT_BINARY=ROOT/"build_long/qwen_infer_gpu"
 pd.ps.EXPECTED_PERSISTENT_BINARY_SHA=C_BIN
 pd.ps.WORKER_READY_TIMEOUT_SECONDS=180
 pd.ps.REQUEST_TIMEOUT_SECONDS=240
 pd.base.REPO=ROOT/"source_long"
 worker=pd.ps.AdaptivePersistentRouteWorker(route=pd.base.candidate_route(),root=outdir/"worker")
 prior=worker._env
 def env():
  vars=prior()
  vars.update({"BEGLIN_P18_SCRATCH_ONLY":"1","QWEN_MOE_CB_SLOTS":"1",
   "QWEN_MOE_CB_REQS":"1","BEGLIN_HCP_P9_PREPARED_CACHE":"1",
   "BEGLIN_P18_LOGITS_DIR":str(ROOT/"capture"),
   "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
   "TOKENIZERS_PARALLELISM":"false"})
  vars.pop("BEGLIN_P17_LOGITS_DIR",None)
  return vars
 worker._env=env
 rows=[];started=time.monotonic();last_policy="A"
 try:
  start=worker.start();ack=pd.grc.read_runtime_ack(worker.ack_path)
  if(not start.get("alive") or start["rss_bytes"]>pd.MAX_RSS or
     ack.get("active_policy_hash")!=pd.pc.policy_hash(pd.A)):
   raise RuntimeError("original P9 GPU policy A startup not attested")
  pid=int(start["pid"]);last_epoch=int(ack["weight_epoch"])
  for step,g in enumerate(gold):
   prefix=ids+gold[:step]
   choices=(("A",pd.A),("B",B)) if index%2==0 else (("B",B),("A",pd.A))
   for name,precision in choices:
    old_files={p.name for p in (ROOT/"capture").iterdir()}
    log_size=len(worker.log_path.read_text(errors="replace"))
    request_start=time.monotonic()
    real=worker.submit_with_precision_policy([(prefix,1)],target_policy=precision,
       admission_id="p18-480-source-%03d-g%d-%s"%(index,step,name))
    terminal=pd.grc.read_runtime_ack(worker.ack_path);health=worker.health()
    trace=[line.strip() for line in worker.log_path.read_text(errors="replace")[log_size:].splitlines()
          if "P18_NATIVE_GOLD_PREFIX_LOGITS" in line]
    if len(trace)!=1:raise RuntimeError("exact one original native C GPU logit capture absent")
    m=PAT.fullmatch(trace[0])
    if m is None:raise RuntimeError("actual original C source capture signature invalid")
    rid,seen_sha,plen,pos,pr,_,_,expected=m.groups()
    f=(ROOT/"capture")/(rid+".f32")
    if(seen_sha!=tok(prefix) or int(plen)!=len(prefix) or int(pos)!=len(prefix)-1 or
       {p.name for p in (ROOT/"capture").iterdir()}-old_files!={f.name} or
       sha(f)!=expected or f.stat().st_size!=BYTES or f.is_symlink()):
     raise RuntimeError("real native XOX F32 provenance and SHA mismatch")
    z=np.fromfile(f,dtype="<f4")
    if(z.shape!=(102400,) or not np.isfinite(z).all() or
       int(np.argmax(z))!=int(pr) or real.get("responses")!=[[int(pr)]] or
       real.get("finite_logits") is not True or not health.get("alive") or
       health["rss_bytes"]>pd.MAX_RSS or health["pid"]!=pid or
       terminal["active_policy_hash"]!=pd.pc.policy_hash(precision)):
      raise RuntimeError("original GPU logits / ACK / PID / RSS mismatch")
    changed=(last_policy!=name)
    epoch=real.get("precision_epoch") or {}
    if bool(epoch.get("transitioned"))!=changed or int(terminal["weight_epoch"])!=last_epoch+int(changed):
     raise RuntimeError("real native P9 A/B weight epoch mismatch")
    if changed and (terminal.get("status")!="REBIND_SET_APPLIED" or
       terminal.get("txn_id")!=epoch.get("txn_id") or len(epoch.get("changed_targets",[]))!=2):
     raise RuntimeError("real native 2-target policy epoch ACK missing")
    last_epoch=int(terminal["weight_epoch"]);last_policy=name
    rows.append({"item_id":q["item_id"],"index":index,"gold_step":step,
      "policy":name,"record_sha256":q["record_sha256"],
      "gold_token_id":g,"prefix_token_sha256":seen_sha,"prefix_len":len(prefix),
      "raw_C_capture_filename":f.name,"raw_C_float32_sha256":expected,"raw_file_bytes":BYTES,
      "actual_logits_vocab_count":102400,"real_gold_NLL_nats":nll(z,g),
      "generated_argmax_id":int(pr),"precision_epoch":last_epoch,
      "actual_A_B_policy_transition":changed,"terminal_ACK_sha256":terminal["ack_sha256"],
      "native_worker_pid":pid,"native_RSS_bytes":int(health["rss_bytes"]),
      "wall_request_ms_includes_rebind":round((time.monotonic()-request_start)*1000,2)})
   print("P18_NATIVE_GPU_REAL_FORCED_GOLD_PAIR",json.dumps({"index":index,"gold_step":step,
     "total_vector_count":len(rows)},sort_keys=True),flush=True)
 finally:
  try:worker.stop(force=True)
  finally:
   pd.ps.PERSISTENT_BINARY,pd.ps.EXPECTED_PERSISTENT_BINARY_SHA,\
   pd.ps.WORKER_READY_TIMEOUT_SECONDS,pd.ps.REQUEST_TIMEOUT_SECONDS,pd.base.REPO=old
 after=pd.live()
 if after!=before or native_actors() or len(rows)!=2*len(gold):
  raise RuntimeError("P18 native not complete, process leaked or original serving identity changed")
 out={"schema":"beglin-hcp-p18-frozen-480-real-native-full-gold-prefix-paired-v1",
  "status":"REAL_P18_NATIVE_A_B_FULL_F32_GOLD_PREFIX_PAIRS_ITEM_COMPLETE",
  "index":index,"item_id":q["item_id"],"dataset":q["dataset"],
  "record_sha256":q["record_sha256"],
  "source_frozen_480_sha256":FROZEN,"prompt_token_sha256":q["prompt_token_sha256"],
  "gold_token_sha256":q["gold_token_sha256"],"native_gpu_binary_sha256":C_BIN,
  "source_native_C_sha256":C_SOURCE,"source_MLX_sha256":MLX_SOURCE,
  "actual_gold_token_count":len(gold),"native_real_GPU_logits_102400_F32_captures":len(rows),
  "rows":rows,"original_serving_before":before,"original_serving_after":after,
  "elapsed_seconds":time.monotonic()-started,
  "native_BF16_teacher_scored_here":False,
  "full_generation_task_accuracy_scored":False,"warm_three_latency_trials_scored":False,
  "quality_noninferiority_certified":False,"production_touched":False,
  "automatic_live_promotion":False}
 receipt=outdir/"native_receipt.json"
 with receipt.open("x") as o:json.dump(out,o,sort_keys=True,indent=2);o.write("\n")
 print("P18_NATIVE_ONE_PAIRED_ITEM_PASS",json.dumps({"index":index,"item_id":q["item_id"],
  "rows":len(rows),"receipt_SHA256":sha(receipt)},sort_keys=True),flush=True)
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--index",type=int,required=True)
 a=parser.parse_args()
 lock=(BATCH/".native_lock").open("a+")
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit("P18_NATIVE_480_PROCESS_ALREADY_RUNNING")
 run(a.index)
if __name__=="__main__":main()
