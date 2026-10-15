#!/usr/bin/env python3
"""P18 ARC/GSM8K first-source per-dataset full original DeepSeek BF16 teacher.
27 original layers, 5291 original BF16 tensors; P17 teacher inherited unchanged.
Exactly one SHA-pinned frozen source per invocation. No operating route changes.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,sys,time
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np,torch,psutil

P17=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015")
P18=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015")
sys.path.insert(0,str(P17/"src/tools"))
import hcp_p17_teacher_gold as og
from hcp_p15_streamed_teacher import TensorStore,original_classes
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
SOURCE_SHA="7ffc9bd15eba15ff853c0d6dfd4e16ef204e6992b340a18a7e15ea5b0eb12979"
INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
P17_TEACHER_SHA="40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3"
P17_STREAMER_SHA="46510526dbaf21884bca30a85ad3d1131099ccf8a1d41063064c1b1fd6683f56"
FROZEN_SHA="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
PROTO_SHA="77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c"
FIXED=("p18-arc_challenge-000","p18-gsm8k-000")
REQUIRED_BYTES=20*1024**3 + 50*1024**2

def shafile(path):
 p=pathlib.Path(path)
 if p.is_symlink() or not p.is_file():raise ValueError("original file path not regular")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def tid(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def main(item_id):
 if item_id not in FIXED:raise ValueError("Only two frozen question identifiers permitted")
 if P18.is_symlink() or P18.parent.name!=".tailnet-tmp" or not P18.is_dir():
  raise RuntimeError("P18 research workspace identity incorrect")
 checks=((P17/"src/tools/hcp_p17_teacher_gold.py",P17_TEACHER_SHA),
         (P17/"src/tools/hcp_p15_streamed_teacher.py",P17_STREAMER_SHA),
         (MODEL/"model.safetensors.index.json",INDEX_SHA),
         (P18/"input/ARC_GSM_two_pilot.json",SOURCE_SHA),
         (P18/"input/frozen_protocol.json",PROTO_SHA))
 for p,h in checks:
  if shafile(p)!=h:raise RuntimeError("P18 original BF16 teacher source identity drift: "+p.name)
 items=json.loads((P18/"input/ARC_GSM_two_pilot.json").read_text())
 if (items.get("schema")!="beglin-hcp-p18-ARC-GSM-two-original-gold-multistep-pilot-v1"
    or items.get("original_full_480_sha256")!=FROZEN_SHA
    or len(items.get("items",[]))!=2):
  raise RuntimeError("P18 frozen corpus source changed")
 rows=[z for z in items["items"] if z["item_id"]==item_id]
 if len(rows)!=1:raise RuntimeError("P18 source item repeated or absent")
 row=rows[0];ids=row["prompt_ids"];gold=row["gold_ids"]
 if (row["shot_count"]!=0 or row["prompt_token_count"]!=len(ids)
    or row["gold_token_count"]!=len(gold)
    or not 1<=len(gold)<=8 or not 1<=len(ids)+len(gold)<=112
    or tid(ids)!=row["prompt_token_sha256"] or tid(gold)!=row["gold_token_sha256"]):
  raise ValueError("Selected P18 source gold token preimage invalid")
 before=og.live()
 if before!={"stage_sha256":og.LIVE_SHA,"route_sha256":og.ROUTE_SHA,"pids":list(og.PIDS)}:
  raise RuntimeError("original P17 live serving changed")
 free=psutil.disk_usage(str(P18)).free
 if free<REQUIRED_BYTES or psutil.virtual_memory().available<16*1024**3:
  raise RuntimeError("P18 protected source disk/memory insufficient")
 target=P18/("original_bf16_teacher_"+item_id)
 if target.exists():raise FileExistsError("P18 one-shot original BF16 output already occupied")
 target.mkdir(mode=0o700)
 torch.set_num_threads(2)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 try:os.nice(10)
 except PermissionError:pass
 start=time.monotonic()
 with og.Watch(item_id) as watch:
  classes,functions=original_classes(MODEL/"modeling_deepseek.py")
  store=TensorStore(MODEL)
  teacher=og.GoldWindowTeacher(store,classes,dtype=torch.bfloat16,
   progress=lambda x:print("P18_ARC_GSM_ORIGINAL_BF16_LAYER",json.dumps(
    {"id":item_id,"layer":x["layer"],"n":27,"s":round(x["seconds"],2)},sort_keys=True),flush=True))
  both=torch.tensor([ids+gold],dtype=torch.int64)
  logits=teacher.forward_gold(both,len(ids),len(gold))
  raw=logits.numpy()
  if raw.dtype!=np.float32 or raw.shape!=(len(gold),102400) or not np.isfinite(raw).all():
   raise ValueError("Original complete 27 layer 102400 BF16 gold logit span absent")
  expanded=raw.astype(np.float64)
  peak=expanded.max(axis=1)
  loss=peak+np.log(np.exp(expanded-peak[:,None]).sum(axis=1))-expanded[np.arange(len(gold)),gold]
  if not np.isfinite(loss).all() or not (loss>=0).all() or not (loss<150).all():
   raise ValueError("Gold NLL values not valid")
  if len(store.value_hashes)!=5291 or len(teacher.layer_records)!=27:
   raise RuntimeError("Original tensor and layer source coverage incomplete")
  after=og.live()
  if after!=before:raise RuntimeError("original serving changed during P18 BF16")
  npy=target/("original_"+item_id+"_teacher_logits.npy")
  with npy.open("xb") as f:np.save(f,raw,allow_pickle=False)
  source=target/("original_"+item_id+"_tensor_hashes.json")
  with source.open("x") as f:json.dump(store.value_hashes,f,sort_keys=True,indent=2);f.write("\n")
  report={
    "schema":"beglin-hcp-p18-multisource-full-original-BF16-teacher-v1",
    "status":"P18_ONE_REAL_FULL_BF16_TEACHER_MULTISOURCE_PILOT_NOT_480_GATE",
    "item_id":item_id,"dataset":row["dataset"],"source_token_bank_sha256":FROZEN_SHA,
    "source_shortlist_sha256":SOURCE_SHA,
    "prompt_token_sha256":row["prompt_token_sha256"],"gold_token_sha256":row["gold_token_sha256"],
    "prompt_tokens":len(ids),"gold_tokens":len(gold),"gold_ids":gold,
    "gold_NLL_nats":[float(x) for x in loss],"gold_mean_NLL_nats":float(np.mean(loss)),
    "source_original_checkpoint_index_sha256":INDEX_SHA,
    "original_teacher_source_sha256":P17_TEACHER_SHA,
    "original_decoder_layers":len(teacher.layer_records),
    "original_tensors_exactly_once":len(store.value_hashes),
    "raw_logits_shape":list(raw.shape),"raw_logits_f32_sha256":shafile(npy),
    "original_tensor_values_sha256":shafile(source),
    "original_torch_class_source_sha256":functions,
    "per_layer":[{"layer":int(x["layer"]),"hidden_sha256":x["hidden_sha256"]} for x in teacher.layer_records],
    "peak_source_RSS_bytes":max(watch.peak,0),
    "additional_swap_bytes":watch.max_swap_delta,
    "elapsed_seconds":time.monotonic()-start,
    "original_serving_before":before,"original_serving_after":after,
    "P18_480_quality_certified":False,
    "automatic_live_promotion":False,"production_touched":False
  }
  dest=target/("original_"+item_id+"_teacher_receipt.json")
  with dest.open("x") as f:json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
  print("P18_ARC_GSM_REAL_BF16_ONE_SOURCE_COMPLETED",json.dumps({
   "item_id":item_id,"gold_tokens":len(gold),"mean_NLL":round(float(np.mean(loss)),7),
   "receipt_sha256":shafile(dest),"raw_logits_sha256":shafile(npy),
   "peak_RSS_GiB":round(watch.peak/(1024**3),2),"production_unchanged":True},sort_keys=True),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--item",choices=FIXED,required=True)
 main(p.parse_args().item)
