#!/usr/bin/env python3
"""P18 EOE standalone MLX Metal A/B DEV ONLY parity against original XOX.

One 203-token frozen development probe, not a P18 test item; BF16 teacher
and test-bank 480 never accessed by this smoke. Requires fully SHA-verified
EOE D50 original model and native base receipts. Exact original XOX dev
hardware receipts are independently SHA verified before checking 1 token.
No full-logit capture allowed for out-of-673 dev input. No live promotion.
"""
from __future__ import annotations
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
sys.path.insert(0,str(ROOT))
import hcp_p18_eoe_d50_native_AB_unit as native
pd=native.pd
B=native.B
A=pd.A
INPUT_SHA="b628ec3247cfc7a94785a93e539562a7a2bc7726f133c21162479749b8f0ac2c"
ORIGINAL={
 "A":("p18_devonly203_native_gpu_receipt.json","997ac9934555107872600c19d7b8bcf0f1e564bf83d6299292344d0448cd2ae8"),
 "B":("p18_devonly203_B_native_gpu_receipt.json","09bd83bbb68ac1f090c0c1713a770fb0a266110c66c487fd016860c08feb75f9")}
SUCCESS="P18_EOE_D50_NATIVE_DEV_ONLY_203_GPU_PARITY_WITH_XOX_A_B_PASS_NOT_TEST_QUALITY"
def sha(path):
 p=pathlib.Path(path)
 if not p.is_file() or p.is_symlink():raise ValueError("source missing/not regular: "+str(p))
 return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def main():
 src=ROOT/"p18_devonly_203_hardware_input_PRIVATE.json"
 if sha(src)!=INPUT_SHA:raise RuntimeError("Not the original frozen P18 development-only 203 test source")
 bank=json.loads(src.read_text())
 ids=bank["prompt_ids"]
 if (bank.get("source_role")!="development_only_not_480_test" or
     bank.get("prompt_token_count")!=203 or len(ids)!=203 or
     bank.get("prompt_token_sha256")!=thash(ids) or
     bank.get("gold_answer_not_scored") is not True):
  raise ValueError("Not the original P18 dev-only unscored 203 tokens")
 expected={}
 for name,(fn,h) in ORIGINAL.items():
  path=ROOT/"native1024_real_evidence"/fn
  if sha(path)!=h:raise RuntimeError("Original XOX dev source GPU reference changed")
  r=json.loads(path.read_text())
  if (r.get("candidate_binary_sha256")!="12a50d4282596ebcf4297f28694698e97b6f4fac11ebe2117bf1701f94dd4591"
      or r.get("input_dev_SHA256")!=INPUT_SHA or r.get("observation",{}).get("input_tokens")!=203 or
      r.get("production_touched") is not False):
   raise ValueError("XOX original A/B dev source not genuine")
  expected[name]=r["observation"]["generated_token_sha256"]
 # Independently enforce both model-copy receipts and active production.
 for filename,status in (("EOE_P18_BF16_MODEL_COPY_RECEIPT.json","ORIGINAL_BF16_FOUR_SHARDS_COPY_SHA_VERIFIED"),
                         ("EOE_P18_MOE_BASE_COPY_RECEIPT.json","ORIGINAL_MOE_NATIVE_BASE_COPY_SHA_VERIFIED")):
  p=ROOT/filename
  if not p.is_file() or json.loads(p.read_text()).get("status")!=status:
   raise RuntimeError("Model source copy not yet SHA sealed: "+filename)
 before=pd.live()
 if before!={"stage_sha256":pd.STAGE_SHA,"route_sha256":pd.ROUTE_SHA,
             "stage_pids":[35481,35493],"scratch_pids":[]}:
  raise RuntimeError("Original XOX production route/PIDs changed")
 for path,h in ((ROOT/"eoe_p18_native_capture_build/qwen_infer_gpu",native.C_BIN),
                (ROOT/"eoe_p18_native_capture_buildroot/qwen_infer.c",native.C_SOURCE)):
  if sha(path)!=h:raise RuntimeError("EOE dev smoke native original SHA mismatch")
 work=ROOT/"eoe_p18_devonly203_one_shot_worker"
 if work.exists():raise FileExistsError("Do not repeat already-run development-only EOE GPU smoke")
 ps=pd.ps;prev=(ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,
                  ps.WORKER_READY_TIMEOUT_SECONDS,ps.REQUEST_TIMEOUT_SECONDS,pd.base.REPO)
 ps.PERSISTENT_BINARY=ROOT/"eoe_p18_native_capture_build/qwen_infer_gpu"
 ps.EXPECTED_PERSISTENT_BINARY_SHA=native.C_BIN
 ps.WORKER_READY_TIMEOUT_SECONDS=180
 ps.REQUEST_TIMEOUT_SECONDS=240
 pd.base.REPO=ROOT/"eoe_p18_native_capture_buildroot"
 w=ps.AdaptivePersistentRouteWorker(route=pd.base.candidate_route(),root=work)
 oldenv=w._env
 def isolated():
  env=oldenv()
  env.update({"BEGLIN_P18_SCRATCH_ONLY":"1","QWEN_MOE_CB_SLOTS":"1",
   "QWEN_MOE_CB_REQS":"1","BEGLIN_HCP_P9_PREPARED_CACHE":"1",
   "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
   "TOKENIZERS_PARALLELISM":"false"})
  env.pop("BEGLIN_P18_LOGITS_DIR",None) # 673 pinned TEST-token observer is NOT valid for DEV input
  env.pop("BEGLIN_P17_LOGITS_DIR",None)
  return env
 w._env=isolated
 observations=[]
 try:
  st=w.start();initial=pd.grc.read_runtime_ack(w.ack_path)
  if not st.get("alive") or initial["active_policy_hash"]!=pd.pc.policy_hash(A):
   raise RuntimeError("EOE native A policy start failed")
  pid=int(st["pid"]);last_epoch=int(initial["weight_epoch"])
  for name,policy in (("A",A),("B",B)):
   start=time.monotonic()
   result=w.submit_with_precision_policy([(ids,1)],target_policy=policy,
    admission_id="p18-eoe-d50-203-devonly-"+name)
   ack=pd.grc.read_runtime_ack(w.ack_path);health=w.health()
   responses=result.get("responses")
   if (result.get("finite_logits") is not True or not isinstance(responses,list)
       or len(responses)!=1 or len(responses[0])!=1 or
       not health.get("alive") or int(health["pid"])!=pid or
       health["rss_bytes"]>pd.MAX_RSS or
       ack["active_policy_hash"]!=pd.pc.policy_hash(policy)):
    raise RuntimeError("Real EOE Metal GPU dev source output and ACK invalid")
   token=responses[0][0];tokhash=thash([token])
   if tokhash!=expected[name]:
    raise RuntimeError("EOE dev-only GPU token SHA differs from pinned XOX reference")
   switched=name=="B";epoch=result.get("precision_epoch") or {}
   if bool(epoch.get("transitioned"))!=switched or int(ack["weight_epoch"])!=last_epoch+int(switched):
    raise RuntimeError("Original P9 A/B precision epoch mismatch")
   if switched and (ack.get("status")!="REBIND_SET_APPLIED"
        or len(epoch.get("changed_targets") or [])!=2
        or ack.get("txn_id")!=epoch.get("txn_id")):
    raise RuntimeError("Real EOE A→B set rebind not verified")
   last_epoch=int(ack["weight_epoch"])
   observations.append({"policy":name,"input_tokens":203,"generated_tokens":1,
       "generated_token_sha256":tokhash,"original_XOX_token_SHA_match":True,
       "native_runtime_ACK_SHA256":ack["ack_sha256"],
       "native_GPU_RSS_bytes":int(health["rss_bytes"]),
       "worker_PID":pid,"wall_seconds_not_speed_qualified":time.monotonic()-start})
 finally:
  try:w.stop(force=True)
  finally:
   (ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,
    ps.WORKER_READY_TIMEOUT_SECONDS,ps.REQUEST_TIMEOUT_SECONDS,pd.base.REPO)=prev
 after=pd.live()
 if before!=after or len(observations)!=2:
  raise RuntimeError("EOE research worker leaked or XOX live stage changed")
 receipt=ROOT/"P18_EOE_D50_DEV203_A_B_METAL_PARITY_RECEIPT.json"
 out={"schema":"beglin-hcp-p18-d50-EOE-203-devonly-real-Metal-parity-v1",
   "status":SUCCESS,"source_dev_input_sha256":INPUT_SHA,
   "native_EOE_C_SHA256":native.C_SOURCE,
   "native_EOE_MLX_binary_SHA256":native.C_BIN,
   "original_XOX_dev_test_only_SHA256":{k:v[1] for k,v in ORIGINAL.items()},
   "real_GPU_results":observations,
   "source_external_test_questions_used":0,
   "full_480_gold_NLL_evaluated":False,
   "full_480_task_accuracy_certified":False,
   "original_serving_before":before,"original_serving_after":after,
   "automatic_live_promotion":False,"production_touched":False}
 with receipt.open("x") as f:json.dump(out,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":SUCCESS,"receipt_sha256":sha(receipt),
  "policy_results":len(observations),"original_XOX_invariant":True},sort_keys=True))
if __name__=="__main__":main()
