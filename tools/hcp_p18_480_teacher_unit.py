#!/usr/bin/env python3
"""BEGLIN P18 frozen-480 single original BF16 full-model unit.
Exactly original 27 layers and 5291 BF16 source tensors. One immutable
P18 test item, run-owned temporary XOX output; EOE owns post-copy cleanup.
No training, no synthetic logits, no P17 source/serving writes.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,psutil,subprocess,sys,time,fcntl
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ["TRANSFORMERS_OFFLINE"]="1";os.environ["HF_HUB_OFFLINE"]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np,torch
ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015")
BATCH=ROOT/"batch480-work"
P17=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015")
sys.path.insert(0,str(P17/"src/tools"))
sys.path.insert(0,str(ROOT))
import hcp_p17_teacher_gold as og
import hcp_p18_extended_original_bf16_teacher_core as ext
from hcp_p15_streamed_teacher import TensorStore,original_classes
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
FROZEN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
CORE="c2a631ee7d82feef83efce59087b0a144f7561b9da26db920e1201b45b689820"
CKPT="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
PREIMAGE_5291="88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda"
LIVE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
EXISTING_PIDS=[35481,35493]
DRESERVE=20*1024**3+90*1024**2
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def tok(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def pre(index):
 if ROOT.is_symlink() or BATCH.is_symlink() or not BATCH.is_dir():raise ValueError("wrong P18 one-run-owned scratch path")
 if sha(BATCH/"p18_multishot_frozen_PRIVATE.json")!=FROZEN:raise ValueError("real original frozen corpus SHA drift")
 if sha(ROOT/"hcp_p18_extended_original_bf16_teacher_core.py")!=CORE or sha(MODEL/"model.safetensors.index.json")!=CKPT:
  raise ValueError("BF16 torch or original DeepSeek model drift")
 d=json.loads((BATCH/"p18_multishot_frozen_PRIVATE.json").read_text())
 if len(d["items"])!=480 or sum(x["gold_token_count"] for x in d["items"])!=673:raise ValueError("not 480 frozen original questions")
 if not 0<=index<480:raise ValueError("item index outside frozen corpus")
 q=d["items"][index];ids=q["prompt_ids"];gold=q["gold_ids"]
 if(not 1<=len(gold)<=8 or not 2<=len(ids) or len(ids)+len(gold)>1024 or
    q["prompt_token_count"]!=len(ids) or q["gold_token_count"]!=len(gold) or
    q["prompt_token_sha256"]!=tok(ids) or q["gold_token_sha256"]!=tok(gold) or
    q.get("split")!="test"):raise ValueError("frozen token original source shape/hash mismatch")
 before=og.live()
 if before!={"stage_sha256":LIVE,"route_sha256":ROUTE,"pids":EXISTING_PIDS}:raise RuntimeError("production stage/PID changed")
 if psutil.disk_usage(str(ROOT)).free<DRESERVE or psutil.virtual_memory().available<15*1024**3:
  raise RuntimeError("XOX physical reserve or RAM failed")
 return q,before
def run(index):
 q,original=pre(index)
 resultdir=BATCH/("unit_%03d"%index)/"teacher"
 if resultdir.parent.exists():raise FileExistsError("Immutable question unit already started")
 resultdir.mkdir(parents=True,mode=0o700)
 torch.set_num_threads(2)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 try:os.nice(10)
 except PermissionError:pass
 started=time.monotonic()
 with og.Watch(index) as watch:
  classes,sources=original_classes(MODEL/"modeling_deepseek.py")
  store=TensorStore(MODEL)
  teacher=ext.P18LongGoldWindowTeacher(store,classes,dtype=torch.bfloat16,
       progress=lambda x:print("P18_STREAM_REAL_BF16_LAYER",json.dumps({"index":index,"layer":x["layer"]}),flush=True))
  vals=teacher.forward_gold(torch.tensor([q["prompt_ids"]+q["gold_ids"]],dtype=torch.int64),len(q["prompt_ids"]),len(q["gold_ids"]))
  z=vals.numpy()
  if z.dtype!=np.float32 or z.shape!=(len(q["gold_ids"]),102400) or not np.isfinite(z).all():raise RuntimeError("full original 27L BF16 logits absent")
  if len(teacher.layer_records)!=27 or len(store.value_hashes)!=5291:raise RuntimeError("original 5291 source tensors incomplete")
  sourcejson=json.dumps(store.value_hashes,sort_keys=True,indent=2)+"\n"
  src_digest=hashlib.sha256(sourcejson.encode()).hexdigest()
  if src_digest!=PREIMAGE_5291:raise RuntimeError("original P17 tensor value SHA mismatch")
  tmp=z.astype(np.float64);m=tmp.max(axis=1)
  losses=m+np.log(np.exp(tmp-m[:,None]).sum(axis=1))-tmp[np.arange(len(q["gold_ids"])),q["gold_ids"]]
  if not np.isfinite(losses).all() or not (losses>=0).all() or not (losses<150).all():raise RuntimeError("teacher source NLL not finite")
  after=og.live()
  if after!=original:raise RuntimeError("original production route/PIDs drifted")
  f=resultdir/"teacher_logits.npy"
  with f.open("xb") as out:np.save(out,z,allow_pickle=False)
  out={"schema":"beglin-hcp-p18-480-teacher-item-v1",
   "status":"REAL_27L_5291T_BF16_ORIGINAL_FULL_GOLD_LOGITS",
   "item_id":q["item_id"],"dataset":q["dataset"],"index":index,
   "frozen_480_source_sha256":FROZEN,"prompt_sha256":q["prompt_token_sha256"],
   "gold_sha256":q["gold_token_sha256"],"record_sha256":q["record_sha256"],
   "original_checkpoint_index_sha256":CKPT,"original_bf16_full_decoder_layers":27,
   "source_tensors_loaded_once":5291,"original_tensor_values_manifest_sha256":src_digest,
   "prompt_tokens":len(q["prompt_ids"]),"gold_tokens":len(q["gold_ids"]),
   "native_F32_teacher_full_vocab_logits_shape":list(z.shape),"native_F32_teacher_logit_file":"teacher_logits.npy",
   "original_logits_sha256":sha(f),"gold_NLL_nats":[float(v) for v in losses],
   "wall_seconds":time.monotonic()-started,"observed_process_RSS_peak_bytes":watch.peak,
   "observed_swap_delta_bytes":watch.max_swap_delta,
   "original_live_before":original,"original_live_after":after,
   "source_bf16_full_model_real":True,"A_B_measured":False,
   "task_accuracy_measured":False,"latency_3trials_measured":False,
   "quality_noninferiority_certified":False,"automatic_live_promotion":False,
   "production_touched":False}
  dest=resultdir/"teacher_receipt.json"
  with dest.open("x") as w:json.dump(out,w,sort_keys=True,indent=2);w.write("\n")
  print("P18_STREAM_BF16_ITEM_DONE",json.dumps({"index":index,"id":q["item_id"],
     "gold":len(q["gold_ids"]),"NLL_mean":float(np.mean(losses)),
     "receipt_sha":sha(dest),"raw_sha":sha(f)},sort_keys=True),flush=True)
def main():
 a=argparse.ArgumentParser()
 a.add_argument("--index",type=int,required=True)
 opt=a.parse_args()
 BATCH.mkdir(exist_ok=True)
 lock=(BATCH/".bf16_lock").open("a+")
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  raise SystemExit("P18_BF16_RESEARCH_LOCK_OCCUPIED")
 run(opt.index)
if __name__=="__main__":main()
