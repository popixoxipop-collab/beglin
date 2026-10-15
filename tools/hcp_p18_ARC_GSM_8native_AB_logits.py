#!/usr/bin/env python3
"""8 real native P18 original-full-vocab (102400 F32) captures:
ARC 1 gold token + GSM8K 3 gold tokens; 4 gold positions × policy A/B.
Frozen before scores: two public benchmark items, all prefix token SHA
are in the P18 673-allowed native source, no P17 source or serving mutation.
This is a 2/480 capability pilot, NOT a statistically certified benchmark.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,subprocess,sys,time
import numpy as np

LEGACY=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-native1024-20261015")
ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015")
sys.path.insert(0,str(LEGACY))
import pilot_driver as pd
from hcp_p4_native_policies_ab import B

BANK_SHA="7ffc9bd15eba15ff853c0d6dfd4e16ef204e6992b340a18a7e15ea5b0eb12979"
P18_FROZEN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
PROTOCOL_SHA="77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c"
C_SHA="82475145a4351b934cfccda6763112cdf25751e2ca470ac8417a16e6dc8fdb1c"
MLX_SHA="a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b"
BIN_SHA="ed2af527c1291567b0e9e67a9940cb5f13b298d4a645d11742ff1cc34248bc6e"
LOG_RE=re.compile(r"^P18_NATIVE_GOLD_PREFIX_LOGITS rid=([a-f0-9]{32}) inputsha=([a-f0-9]{64}) plen=(\d+) pos=(\d+) next=(\d+) vocab=(102400) bytes=(409600) floatsha=([a-f0-9]{64})$")

def sseq(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def require_file(path,expected):
 p=pathlib.Path(path)
 if p.is_symlink() or not p.is_file() or pd.sha(p)!=expected:raise RuntimeError("Frozen source file SHA/identity mismatch: "+str(path))
def native_logits_nll(z,gold):
 x=np.asarray(z,dtype=np.float64); peak=float(x.max())
 out=peak+math.log(float(np.exp(x-peak).sum()))-float(x[gold])
 if not 0<=out<150:raise RuntimeError("Native F32 full-vocab actual loss invalid")
 return out
def capture_pid_list():
 output=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=8)
 which=str(ROOT/"build/qwen_infer_gpu")
 return [int(line.split()[0]) for line in output.splitlines() if which in line]
def main():
 original,live_before,resources=pd.preflight()
 if (ROOT.is_symlink() or ROOT.parent.name!=".tailnet-tmp" or
     not ROOT.is_dir() or capture_pid_list()):
  raise RuntimeError("P18 exact isolated research capture path or existing worker invalid")
 for p,h in ((ROOT/"source/qwen_infer.c",C_SHA),
             (ROOT/"source/mlx_moe.cpp",MLX_SHA),
             (ROOT/"build/qwen_infer_gpu",BIN_SHA),
             (ROOT/"input/ARC_GSM_two_pilot.json",BANK_SHA),
             (ROOT/"input/frozen_protocol.json",PROTOCOL_SHA)):
  require_file(p,h)
 bank=json.loads((ROOT/"input/ARC_GSM_two_pilot.json").read_text())
 if (bank.get("original_full_480_sha256")!=P18_FROZEN or
     bank.get("schema")!="beglin-hcp-p18-ARC-GSM-two-original-gold-multistep-pilot-v1" or
     len(bank.get("items",[]))!=2):
  raise RuntimeError("P18 two-source original gold corpus drift")
 expected={"p18-arc_challenge-000":(57,1),"p18-gsm8k-000":(59,3)}
 tokens=[]
 for item in bank["items"]:
  ident=item["item_id"]
  if ident not in expected:raise RuntimeError("wrong preregistered P18 source id")
  ids,gold=item["prompt_ids"],item["gold_ids"]
  if (tuple([len(ids),len(gold)])!=expected[ident] or
      item.get("shot_count")!=0 or
      item["prompt_token_sha256"]!=sseq(ids) or
      item["gold_token_sha256"]!=sseq(gold) or
      not all(0<=t<102400 for t in ids+gold) or len(ids)+len(gold)>112):
   raise RuntimeError("Unfrozen or incorrect P18 source gold token sequence")
  for step,t in enumerate(gold):
   prefix=ids+gold[:step]
   tokens.append((ident,step,prefix,t,sseq(prefix)))
 if len(tokens)!=4 or len({s for _,_,_,_,s in tokens})!=4:raise RuntimeError("Not four source gold prefixes")
 capturedir=ROOT/"capture"
 if capturedir.is_symlink() or not capturedir.is_dir():raise RuntimeError("Original dedicated P18 capture dir absent")
 originals={p.name for p in capturedir.iterdir()}
 if len(originals)!=2 or any(not p.endswith(".f32") for p in originals):
  raise RuntimeError("Already-captured one-question source changed; refuse unexpected artefacts")
 workerroot=ROOT/"p18_ARC_GSM_multistep_worker"
 receipt=ROOT/"p18_ARC_GSM_native_AB_raw_full_vocab_receipt.json"
 if workerroot.exists() or receipt.exists():raise RuntimeError("This P18 8-request pilot already ran")
 old=(pd.ps.PERSISTENT_BINARY,pd.ps.EXPECTED_PERSISTENT_BINARY_SHA,
      pd.ps.WORKER_READY_TIMEOUT_SECONDS,pd.ps.REQUEST_TIMEOUT_SECONDS,pd.base.REPO)
 pd.ps.PERSISTENT_BINARY=ROOT/"build/qwen_infer_gpu"
 pd.ps.EXPECTED_PERSISTENT_BINARY_SHA=BIN_SHA
 pd.ps.WORKER_READY_TIMEOUT_SECONDS=180
 pd.ps.REQUEST_TIMEOUT_SECONDS=180
 pd.base.REPO=ROOT/"source"
 w=pd.ps.AdaptivePersistentRouteWorker(route=pd.base.candidate_route(),root=workerroot)
 prev=w._env
 def env():
  x=prev()
  x.update({"BEGLIN_P18_SCRATCH_ONLY":"1","QWEN_MOE_CB_SLOTS":"1",
      "QWEN_MOE_CB_REQS":"1","BEGLIN_HCP_P9_PREPARED_CACHE":"1",
      "BEGLIN_P18_LOGITS_DIR":str(capturedir),
      "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false"})
  x.pop("BEGLIN_P17_LOGITS_DIR",None)
  return x
 w._env=env
 rows=[];last="A";started=time.monotonic();pid=None
 try:
  h=w.start();ack=pd.grc.read_runtime_ack(w.ack_path)
  if (not h.get("alive") or h["rss_bytes"]>=pd.MAX_RSS or
      ack.get("active_policy_hash")!=pd.pc.policy_hash(pd.A)):
   raise RuntimeError("Native research worker start integrity invalid")
  pid=int(h["pid"]);old_epoch=int(ack["weight_epoch"])
  for ident,step,prefix,target,hash_input in tokens:
   for kind,pol in (("A",pd.A),("B",B)):
    before_files={p.name for p in capturedir.iterdir()}
    before_log=len(w.log_path.read_text(errors="replace"))
    call_started=time.monotonic()
    result=w.submit_with_precision_policy([(prefix,1)],target_policy=pol,
       admission_id=f"p18-arc-gsm-gold-{ident}-{step}-{kind}")
    ack=pd.grc.read_runtime_ack(w.ack_path)
    health=w.health()
    req=result.get("responses")
    new_lines=[q.strip() for q in w.log_path.read_text(errors="replace")[before_log:].splitlines()
               if "P18_NATIVE_GOLD_PREFIX_LOGITS" in q]
    if len(new_lines)!=1:raise RuntimeError("Real P18 native C capture witness missing")
    m=LOG_RE.fullmatch(new_lines[0])
    if m is None:raise RuntimeError("Real P18 native C capture schema invalid")
    rid,seen_hash,plen,pos,nexttok,_,_,dig=m.groups()
    path=capturedir/(rid+".f32")
    if (seen_hash!=hash_input or int(plen)!=len(prefix) or int(pos)!=len(prefix)-1 or
        {p.name for p in capturedir.iterdir()}-before_files!={path.name} or
        path.is_symlink() or not path.is_file() or path.stat().st_size!=409600 or pd.sha(path)!=dig):
     raise RuntimeError("Raw F32 capture/owner/input byte SHA contract invalid")
    data=np.fromfile(path,dtype="<f4")
    if (data.shape!=(102400,) or not np.isfinite(data).all() or
        not req or len(req)!=1 or len(req[0])!=1 or
        int(np.argmax(data))!=int(nexttok) or req[0][0]!=int(nexttok) or
        not result.get("finite_logits") or not health.get("alive") or
        int(health["pid"])!=pid or int(health["rss_bytes"])>=pd.MAX_RSS or
        ack["active_policy_hash"]!=pd.pc.policy_hash(pol)):
     raise RuntimeError("Native full-vocab/ACK/model runtime integrity invalid")
    switched=last!=kind;epoch=result.get("precision_epoch") or {}
    if bool(epoch.get("transitioned"))!=switched or int(ack["weight_epoch"])!=old_epoch+int(switched):
     raise RuntimeError("Actual A/B policy weight epoch inconsistent")
    if switched and (ack.get("status")!="REBIND_SET_APPLIED" or
                     ack.get("txn_id")!=epoch.get("txn_id") or
                     len(epoch.get("changed_targets") or [])!=2):
     raise RuntimeError("Actual A/B two-target atomic rebinding missing")
    last=kind;old_epoch=int(ack["weight_epoch"])
    rows.append({"item_id":ident,"step":step,"policy":kind,
      "prefix_token_sha256":hash_input,"prefix_len":len(prefix),"gold_token_id":target,
      "gold_nll_nats":native_logits_nll(data,target),"generated_token":int(nexttok),
      "actual_409600_F32_logits":path.name,"raw_float32_logits_sha256":dig,
      "vocab_size":102400,"terminal_ack_sha256":ack["ack_sha256"],
      "weight_epoch":old_epoch,"changed_policy":switched,
      "worker_pid":pid,"observed_RSS_bytes":int(health["rss_bytes"]),
      "observed_ms_not_warm_latency_certified":round((time.monotonic()-call_started)*1000,2)})
   print("P18_NATIVE_MULTISOURCE_GOLD_POSITION_CAPTURED",
      json.dumps({"item_id":ident,"step":step,"captured_total":len(rows)},sort_keys=True),flush=True)
 finally:
  try:w.stop(force=True)
  finally:
   (pd.ps.PERSISTENT_BINARY,pd.ps.EXPECTED_PERSISTENT_BINARY_SHA,
    pd.ps.WORKER_READY_TIMEOUT_SECONDS,pd.ps.REQUEST_TIMEOUT_SECONDS,pd.base.REPO)=old
 after=pd.live()
 if after!=live_before or capture_pid_list() or len(rows)!=8:
  raise RuntimeError("P18 run incomplete or live service changed")
 result={"schema":"beglin-hcp-p18-ARC-GSM-paired-original-gold-logits-v1",
  "status":"P18_TWO_MORE_REAL_ORIGINAL_A_B_FULL_LOGIT_MULTISTEP_PILOTS_PASS_NOT_480_GATE",
  "P18_frozen_480_sha256":P18_FROZEN,"input_pilot_sha256":BANK_SHA,
  "input_frozen_protocol_sha256":PROTOCOL_SHA,"original_native_binary_sha256":BIN_SHA,
  "source_C_sha256":C_SHA,"source_MLX_sha256":MLX_SHA,
  "full_vocab_logits_raw_dtype":"float32","native_source_items":2,
  "native_gold_prefixes":4,"native_policy_calls":len(rows),"rows":rows,
  "runtime_seconds":round(time.monotonic()-started,2),
  "original_serving_before":live_before,"original_serving_after":after,
  "original_27_layer_BF16_forward_count_in_this_run":0,
  "statistical_noninferiority_480_certified":False,
  "automatic_live_promotion":False,"production_touched":False}
 with receipt.open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
 print("P18_ARC_GSM_EIGHT_REAL_A_B_F32_LOGIT_CAPTURES_COMPLETE",
       json.dumps({"actual_native_calls":8,"receipt_sha256":pd.sha(receipt),
        "production_unchanged":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
