#!/usr/bin/env python3
"""P18 one-question original BF16 teacher pilot, 27-layer/5291 original tensors.
Source: immutable P17 GoldWindowTeacher/TensorStore. No P17 output modification,
no A/B quantized-model route mutation, no production release operations.
"""
from __future__ import annotations
import gc,hashlib,json,os,pathlib,resource,subprocess,sys,threading,time
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np,torch,psutil

XOX_ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-native1024-20261015")
P17_ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015")
sys.path.insert(0,str(P17_ROOT/"src/tools"))
import hcp_p17_teacher_gold as original
from hcp_p15_streamed_teacher import TensorStore,original_classes,sha,tensor_sha
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
TEACHER_SHA="40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3"
STREAMER_SHA="46510526dbaf21884bca30a85ad3d1131099ccf8a1d41063064c1b1fd6683f56"
PILOT_SHA="89f3ac38e032b5b0761e7da788d85c63f0f165009342eb6c75fb4147b6b111e1"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
TARGET=XOX_ROOT/"teacher_p18_pilot_one"
MIN_FREE=20*1024**3+100*1024**2
MAX_RSS=6*1024**3

def thash(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def live():
 out=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=6)
 return {"stage_sha256":sha(STAGE),"route_sha256":sha(ROUTE),
  "stage_pids":sorted(int(line.split()[0]) for line in out.splitlines() if str(STAGE) in line)}
class Watch:
 def __init__(self):
  self.stop=threading.Event()
  self.peak=0;self.baseline_swap=psutil.swap_memory().used;self.delta=0
 def check(self):
  p=psutil.Process()
  rss=p.memory_info().rss
  self.peak=max(self.peak,rss)
  self.delta=max(self.delta,psutil.swap_memory().used-self.baseline_swap)
  if rss>MAX_RSS:return "P18_BF16_PILOT_RSS_ABOVE_6GIB"
  if psutil.virtual_memory().available<8*1024**3:return "P18_HOST_AVAILABLE_RAM_BELOW_8GIB"
  if self.delta>1024**3:return "P18_NEW_SWAP_OVER_1GIB"
  if psutil.disk_usage(str(XOX_ROOT)).free<20*1024**3:return "P18_XOX_DISK_BELOW_20GIB"
  return None
 def _run(self):
  while not self.stop.wait(.25):
   err=self.check()
   if err:
    print("P18_BF16_ORIGINAL_TEACHER_RESOURCE_ABORT",err,flush=True)
    os._exit(92)
 def __enter__(self):
  err=self.check()
  if err:raise RuntimeError("Preflight "+err)
  self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
  return self
 def __exit__(self,*args):
  self.stop.set();self.thread.join(timeout=3)
  if args[0] is None and self.check():raise RuntimeError("Terminal resource fail: "+str(self.check()))

def main():
 if XOX_ROOT.is_symlink() or XOX_ROOT.parent.name!=".tailnet-tmp" or not XOX_ROOT.is_dir() or TARGET.exists():
  raise RuntimeError("One-shot research scratch path invalid/existing")
 for path,expected in [
  (P17_ROOT/"src/tools/hcp_p17_teacher_gold.py",TEACHER_SHA),
  (P17_ROOT/"src/tools/hcp_p15_streamed_teacher.py",STREAMER_SHA),
  (MODEL/"model.safetensors.index.json",MODEL_SHA),
  (XOX_ROOT/"input/pilot.json",PILOT_SHA)]:
  if sha(path)!=expected:raise RuntimeError("Original source or pilot SHA drift: "+path.name)
 before=live()
 if before!={"stage_sha256":STAGE_SHA,"route_sha256":ROUTE_SHA,"stage_pids":[35481,35493]}:
  raise RuntimeError("Active original model/pids changed: refuse teacher")
 b=json.loads((XOX_ROOT/"input/pilot.json").read_text())
 item=next((r for r in b["items"] if r["item_id"]=="p18-mmlu-000"),None)
 if not item or len(item["prompt_ids"])!=107 or len(item["gold_ids"])!=1:
  raise RuntimeError("P18 frozen one-item BF16 pilot invalid")
 ids=item["prompt_ids"];gold=item["gold_ids"];completed=ids+gold
 if item["prompt_token_sha256"]!=thash(ids) or item["gold_token_sha256"]!=thash(gold) or len(completed)>112:
  raise RuntimeError("P18 source original P17-supported BF16 context validation failed")
 if psutil.disk_usage(str(XOX_ROOT)).free<MIN_FREE or psutil.virtual_memory().available<16*1024**3:
  raise RuntimeError("Insufficient bounded BF16 pilot preflight disk/RAM")
 TARGET.mkdir()
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 try:os.nice(10)
 except PermissionError:pass
 start=time.monotonic()
 with Watch() as guard:
  classes,original_sources=original.original_classes(MODEL/"modeling_deepseek.py")
  store=TensorStore(MODEL)
  teacher=original.GoldWindowTeacher(store,classes,dtype=torch.bfloat16,
   progress=lambda row:print("P18_ORIGINAL_BF16_LAYER",
                             json.dumps({"layer":row["layer"],"total":27,"secs":round(row["seconds"],2)}),flush=True))
  tensor=torch.tensor([completed],dtype=torch.int64)
  logits=teacher.forward_gold(tensor,len(ids),len(gold))
  raw=logits.numpy()
  if raw.dtype!=np.float32 or raw.shape!=(1,102400) or not np.isfinite(raw).all():
   raise RuntimeError("P18 full-vocab BF16 original logit absent")
  target=TARGET/"p18-mmlu-000.teacher_full_logits.npy"
  with target.open("xb") as f:np.save(f,raw,allow_pickle=False)
  m=raw[0].astype(np.float64);lse=float(m.max()+np.log(np.exp(m-m.max()).sum()))
  nll=lse-float(m[gold[0]])
  if not 0<=nll<150:raise RuntimeError("Original teacher invalid token gold loss")
  provenance=TARGET/"p18-mmlu-000.original_5291_tensor_hashes.json"
  with provenance.open("x") as f:json.dump(store.value_hashes,f,sort_keys=True,indent=2);f.write("\n")
  if len(teacher.layer_records)!=27 or len(store.value_hashes)!=5291:
   raise RuntimeError("Original BF16 full-model provenance invalid")
  after=live()
  if before!=after:raise RuntimeError("Original stage drift during BF16 forward")
  record={"schema":"beglin-hcp-p18-original-BF16-27L-5291T-pilot-v1",
    "status":"P18_REAL_ONE_FULL_BF16_GOLD_LOGIT_PASS_NOT_FULL_480",
    "source_480_token_bank_sha256":b["source_full_480_token_bank_sha256"],
    "item_id":item["item_id"],"pilot_source_sha256":PILOT_SHA,
    "original_checkpoint_index_sha256":MODEL_SHA,"original_P17_teacher_source_sha256":TEACHER_SHA,
    "original_streamer_source_sha256":STREAMER_SHA,
    "input_token_sha256":item["prompt_token_sha256"],"gold_token_sha256":item["gold_token_sha256"],
    "input_count":len(ids),"gold_count":len(gold),"gold_NLL_nats":[nll],
    "teacher_argmax_matches_gold":int(m.argmax())==gold[0],
    "original_model_full_vocab":102400,"original_decoder_layers":len(teacher.layer_records),
    "original_model_source_tensors_exactly_once":len(store.value_hashes),
    "raw_full_logits_npy_sha256":sha(target),"raw_source_tensor_hashes_sha256":sha(provenance),
    "source_original_torch_functions_sha256":original_sources,
    "per_layer":[{"layer":row["layer"],"source_rss_peak_bytes":row["source_rss_peak_bytes"],
                  "hidden_sha256":row["hidden_sha256"]} for row in teacher.layer_records],
    "elapsed_seconds":time.monotonic()-start,
    "process_peak_RSS_bytes":guard.peak,
    "additional_swap_bytes":guard.delta,
    "original_serving_before":before,"original_serving_after":after,
    "end_to_end_task_accuracy_certified":False,
    "full_480_noninferiority_certified":False,"production_touched":False,
    "automatic_live_promotion":False}
  dest=TARGET/"p18-mmlu-000.teacher_receipt.json"
  with dest.open("x") as f:json.dump(record,f,sort_keys=True,indent=2);f.write("\n")
  print("P18_ORIGINAL_BF16_ONE_REAL_FORWARD_COMPLETE",json.dumps({
   "item_id":record["item_id"],"gold_nll":round(nll,6),
   "teacher_argmax_matches_gold":record["teacher_argmax_matches_gold"],
   "time_sec":round(record["elapsed_seconds"],2),
   "peak_RSS_GiB":round(record["process_peak_RSS_bytes"]/1024**3,2),
   "teacher_logits_sha256":record["raw_full_logits_npy_sha256"],
   "receipt_sha256":sha(dest),"production_unchanged":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
