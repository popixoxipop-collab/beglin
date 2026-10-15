#!/usr/bin/env python3
"""First complete 27-layer original BF16 teacher P18 context beyond original112.
Exactly source-pinned item p18-mmlu-018 (203 prompt +1 gold).
Derived original GoldWindowTeacher has only 112→1024 admission bound change.
No P17 sealed output or operator service modifications.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,sys,time
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np,torch,psutil

P17=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015")
P18=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015")
sys.path.insert(0,str(P17/"src/tools"))
sys.path.insert(0,str(P18))
import hcp_p17_teacher_gold as og
import hcp_p18_extended_original_bf16_teacher_core as extended
from hcp_p15_streamed_teacher import TensorStore,original_classes
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
INPUT_SHA="374c74797344d68164ee84d3be8932ec86f7b2ffcea51b7cdbe4d6c4a72cde71"
CORE_SHA="c2a631ee7d82feef83efce59087b0a144f7561b9da26db920e1201b45b689820"
MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
FROZEN_SHA="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
PRESET_ID="p18-mmlu-018"
TARGET=P18/"teacher_extended_long_p18_mmlu_018"
MIN_DISK=20*1024**3 + 40*1024**2
def sha(p):
 p=pathlib.Path(p)
 if p.is_symlink() or not p.is_file():raise ValueError("Untrusted source file")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def tokhash(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def main():
 if(P18.is_symlink() or P18.parent.name!=".tailnet-tmp" or not P18.is_dir() or TARGET.exists()):
  raise RuntimeError("Wrong or reused one-shot P18 teacher output")
 if sha(P18/"input/p18_long203_teacher_pilot.json")!=INPUT_SHA or sha(P18/"hcp_p18_extended_original_bf16_teacher_core.py")!=CORE_SHA:
  raise RuntimeError("P18 extended BF16 source/input SHA differs")
 if sha(MODEL/"model.safetensors.index.json")!=MODEL_SHA:
  raise RuntimeError("Original DeepSeek checkpoint index changed")
 d=json.loads((P18/"input/p18_long203_teacher_pilot.json").read_text())
 q=d.get("item")
 if(d.get("schema")!="beglin-hcp-p18-original-extended-long-teacher-pilot-v1"
    or d.get("source_480_sha256")!=FROZEN_SHA or q.get("item_id")!=PRESET_ID
    or q.get("shot_count")!=0 or q.get("prompt_token_count")!=203
    or q.get("gold_token_count")!=1):
  raise RuntimeError("P18 frozen long-context pilot authority mismatch")
 ids=q["prompt_ids"];gold=q["gold_ids"]
 if len(ids)!=203 or len(gold)!=1 or tokhash(ids)!=q["prompt_token_sha256"] or tokhash(gold)!=q["gold_token_sha256"]:
  raise RuntimeError("P18 frozen exact token preimage failed")
 if psutil.disk_usage(str(P18)).free<MIN_DISK or psutil.virtual_memory().available<16*1024**3:
  raise RuntimeError("XOX protected SSD or RAM insufficient")
 before=og.live()
 if(before!={"stage_sha256":og.LIVE_SHA,"route_sha256":og.ROUTE_SHA,"pids":list(og.PIDS)}):
  raise RuntimeError("Original production stage or PIDs drift")
 TARGET.mkdir(mode=0o700)
 torch.set_num_threads(2)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 try:os.nice(10)
 except PermissionError:pass
 start=time.monotonic()
 with og.Watch(PRESET_ID) as watch:
  cls,fn_hash=original_classes(MODEL/"modeling_deepseek.py")
  store=TensorStore(MODEL)
  teacher=extended.P18LongGoldWindowTeacher(store,cls,dtype=torch.bfloat16,
    progress=lambda r:print("P18_BF16_203_LONG_LAYER",json.dumps({"layer":r["layer"],"total":27,
      "secs":round(r["seconds"],2)},sort_keys=True),flush=True))
  t=torch.tensor([ids+gold],dtype=torch.int64)
  result=teacher.forward_gold(t,len(ids),len(gold))
  raw=result.numpy()
  if raw.dtype!=np.float32 or raw.shape!=(1,102400) or not np.isfinite(raw).all():
   raise RuntimeError("Original 27 layer BF16 complete source full logits absent")
  vals=raw[0].astype(np.float64)
  m=float(vals.max());loss=m+float(np.log(np.exp(vals-m).sum()))-float(vals[gold[0]])
  if not 0<=loss<150 or len(store.value_hashes)!=5291 or len(teacher.layer_records)!=27:
   raise RuntimeError("P18 original long-context gold NLL/27L/5291T coverage invalid")
  after=og.live()
  if after!=before:raise RuntimeError("Original operator model changed during long-context teacher")
  npy=TARGET/"original_p18-mmlu-018_203BF16_logits.npy"
  with npy.open("xb") as f:np.save(f,raw,allow_pickle=False)
  src=TARGET/"original_p18-mmlu-018_5291_tensors.json"
  with src.open("x") as f:json.dump(store.value_hashes,f,sort_keys=True,indent=2);f.write("\n")
  receipt={
   "schema":"beglin-hcp-p18-extended-original-BF16-204position-v1",
   "status":"P18_FULL_ORIGINAL_BF16_203_PROMPT_PLUS_ONE_GOLD_PASS_NOT_480_QUALITY",
   "model_source":"original DeepSeek-V2-Lite BF16 weights, unchanged 27 layers",
   "original_model_index_sha256":MODEL_SHA,"original_P17_torch_source_sha256":og.sha(P17/"src/tools/hcp_p17_teacher_gold.py"),
   "extended_teacher_exact_source_sha256":CORE_SHA,
   "item_id":PRESET_ID,"source_frozen_480_sha256":FROZEN_SHA,"source_pilot_sha256":INPUT_SHA,
   "prompt_tokens":203,"gold_tokens":1,"full_original_position_length":204,
   "prompt_token_sha256":q["prompt_token_sha256"],"gold_token_sha256":q["gold_token_sha256"],
   "gold_NLL_nats":loss,"gold_argmax_exact":int(np.argmax(raw[0]))==gold[0],
   "source_real_decoder_layers":len(teacher.layer_records),
   "source_real_tensors_exactly_once":len(store.value_hashes),
   "source_tensor_hash_sha256":sha(src),"real_BF16_logits_npy_sha256":sha(npy),
   "per_layer":[{"layer":r["layer"],"hidden_sha256":r["hidden_sha256"]} for r in teacher.layer_records],
   "original_torch_classes_sha256":fn_hash,"observed_process_peak_RSS_bytes":watch.peak,
   "observed_additional_swap_bytes":watch.max_swap_delta,
   "observed_elapsed_seconds":time.monotonic()-start,
   "original_serving_before":before,"original_serving_after":after,
   "P18_480_quality_certified":False,"production_touched":False,"automatic_live_promotion":False
  }
  dest=TARGET/"original_p18-mmlu-018_BF16_teacher_receipt.json"
  with dest.open("x") as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write("\n")
  print("P18_FIRST_FULL_ORIGINAL_BF16_OVER_112_SOURCE_PASS",json.dumps({"item_id":PRESET_ID,
   "full_original_positions":204,"NLL":round(loss,7),"rss_GiB":round(watch.peak/2**30,3),
   "source_tensors":len(store.value_hashes),"receipt_sha256":sha(dest),
   "raw_logits_sha256":sha(npy),"production_unchanged":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
