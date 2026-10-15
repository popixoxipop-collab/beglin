#!/usr/bin/env python3
"""P18 EOE/D50 original DeepSeek-V2-Lite 27L/5291T BF16 reference, 480 frozen items.

No XOX writes. SHA-fixed P17 torch calculation, extended solely to 1024
positions, real original BF16 checkpoint copied read-only to D50. Outputs
full 102400-vocab float32 logits for the original teacher-forced gold positions.

Each item runs once; positive evidence is not task-generation accuracy.
No release/promotion or policy A/B changes. Fail closed on capacity/source drift.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,pathlib,psutil,resource,sys,threading,time
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np
import torch

ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
MODEL=ROOT/"teacher_checkpoint_source_copy"
OUTROOT=ROOT/"eoe_p18_bf16_480_runs"
P17=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/hcp-p17-unseen-multitoken-20261015/repository/tools")
sys.path.insert(0,str(P17))
sys.path.insert(0,str(ROOT))
import hcp_p17_teacher_gold as og
import hcp_p18_eoe_d50_original_bf16_core as ext
from hcp_p15_streamed_teacher import TensorStore,original_classes,tensor_sha
FROZEN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
CORE="295ce59380f83162d20f619bf9d0cccc961f0c1484d16f5031fdc0040bb6bb78"
P17_TEACHER="40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3"
P17_STREAM="46510526dbaf21884bca30a85ad3d1131099ccf8a1d41063064c1b1fd6683f56"
INDEX="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
CONFIG="f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"
MODEL_IMPL="7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c"
ALL_SOURCE_VALUES="88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda"
MAX_RSS=6*1024**3
MIN_HOST_FREE=11*1024**3
MIN_D50_FREE=100*1024**3
MAX_SWAP_GROWTH=1024**3

def sha(path):
 p=pathlib.Path(path)
 if p.is_symlink() or not p.is_file():raise ValueError("not a regular source file")
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def seq_sha(ids):return hashlib.sha256(",".join(map(str,ids)).encode("ascii")).hexdigest()
def inputs(index):
 if(ROOT.is_symlink() or not ROOT.is_dir() or index<0 or index>=480
     or sha(ROOT/"p18_multishot_frozen_PRIVATE.json")!=FROZEN):
  raise RuntimeError("P18 original frozen cohort missing or index invalid")
 bank=json.loads((ROOT/"p18_multishot_frozen_PRIVATE.json").read_text())
 if bank.get("question_count")!=480 or len(bank.get("items",[]))!=480:
  raise RuntimeError("not original frozen 480 questions")
 item=bank["items"][index]
 ids=item["prompt_ids"]; gold=item["gold_ids"]
 if (len(gold)<1 or len(gold)>8 or len(ids)<2 or len(ids)+len(gold)>1024 or
     item["prompt_token_count"]!=len(ids) or item["gold_token_count"]!=len(gold) or
     seq_sha(ids)!=item["prompt_token_sha256"] or seq_sha(gold)!=item["gold_token_sha256"] or
     any(type(v)!=int or v<0 or v>=102400 for v in ids+gold)):
  raise RuntimeError("invalid original frozen item token evidence")
 return item
def checkpoint():
 if not MODEL.is_dir() or MODEL.is_symlink():
  raise RuntimeError("D50 original BF16 model not transferred")
 for name,expected in (("model.safetensors.index.json",INDEX),("config.json",CONFIG),
                       ("modeling_deepseek.py",MODEL_IMPL)):
  if sha(MODEL/name)!=expected:
   raise RuntimeError("original P17 model index/config/original modeling SHA mismatch: "+name)
 if sha(P17/"hcp_p17_teacher_gold.py")!=P17_TEACHER or sha(P17/"hcp_p15_streamed_teacher.py")!=P17_STREAM or sha(ROOT/"hcp_p18_eoe_d50_original_bf16_core.py")!=CORE:
  raise RuntimeError("original P17 torch calculation or P18 input-only bound modified")
 idx=json.loads((MODEL/"model.safetensors.index.json").read_text())
 shards=sorted(set(idx["weight_map"].values()))
 if len(shards)!=4 or any(not (MODEL/file).is_file() or (MODEL/file).is_symlink() for file in shards):
  raise RuntimeError("all four original BF16 safetensor shards required")
 ts=TensorStore(MODEL)
 if len(ts.meta)!=5291 or len(ts.mapping)!=5291:
  raise RuntimeError("original BF16 tensor metadata incomplete")
 return ts
class Guard:
 def __init__(self):
  self.stop=threading.Event()
  self.start_swap=psutil.swap_memory().used
  self.peak_rss=0;self.swap_delta=0
 def check(self):
  ram=psutil.virtual_memory()
  rss=psutil.Process().memory_info().rss
  self.peak_rss=max(self.peak_rss,rss)
  self.swap_delta=max(self.swap_delta,max(0,psutil.swap_memory().used-self.start_swap))
  if rss>MAX_RSS:return "teacher RSS >6GiB"
  if ram.available<MIN_HOST_FREE:return "EOE available RAM <11GiB"
  if self.swap_delta>MAX_SWAP_GROWTH:return "new swap growth >1GiB"
  if psutil.disk_usage(str(ROOT)).free<MIN_D50_FREE:return "D50 free <100GiB"
  return None
 def _run(self):
  while not self.stop.wait(.5):
   err=self.check()
   if err:
    print("P18_EOE_BF16_WATCH_FAIL_CLOSED",err,flush=True)
    os._exit(94)
 def __enter__(self):
  err=self.check()
  if err:raise RuntimeError(err)
  self.thread=threading.Thread(target=self._run,daemon=True)
  self.thread.start();return self
 def __exit__(self,*exc):
  self.stop.set();self.thread.join(timeout=2)
  if exc[0] is None:
   err=self.check()
   if err:raise RuntimeError("P18 EOE BF16 guard after forward "+err)
def run(index,preflight_only=False):
 q=inputs(index)
 if psutil.virtual_memory().available<17*1024**3:
  raise RuntimeError("EOE original teacher preflight RAM <17GiB")
 store=checkpoint()
 if preflight_only:
  print(json.dumps({"status":"P18_EOE_480_ORIGINAL_27L_BF16_TEACHER_PREFLIGHT_PASS",
    "index":index,"item_id":q["item_id"],"gold_count":len(q["gold_ids"]),
    "tensor_metadata":len(store.meta),"checkpoint_index_sha256":INDEX,
    "production_touched":False},sort_keys=True),flush=True)
  return
 out=OUTROOT/("unit_%03d"%index)
 if out.exists() or out.is_symlink():
  raise FileExistsError("source teacher one-shot item already present")
 out.mkdir(parents=True,mode=0o700)
 torch.set_num_threads(2)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 try:os.nice(10)
 except PermissionError:pass
 started=time.monotonic()
 with Guard() as guard:
  original,pinned_source=original_classes(MODEL/"modeling_deepseek.py")
  teacher=ext.P18LongGoldWindowTeacher(store,original,dtype=torch.bfloat16,
    progress=lambda p:print("P18_EOE_ORIGINAL_BF16_LAYER",json.dumps(
      {"index":index,"layer":p["layer"],"total":27},sort_keys=True),flush=True))
  gold=q["gold_ids"];ids=q["prompt_ids"]
  raw=teacher.forward_gold(torch.tensor([ids+gold],dtype=torch.int64),len(ids),len(gold)).numpy()
  if(raw.dtype!=np.float32 or raw.shape!=(len(gold),102400) or
     not np.isfinite(raw).all() or len(teacher.layer_records)!=27 or len(store.value_hashes)!=5291):
   raise RuntimeError("not the complete original DeepSeek 27L teacher")
  valuesjson=json.dumps(store.value_hashes,sort_keys=True,indent=2)+"\n"
  values_digest=hashlib.sha256(valuesjson.encode()).hexdigest()
  if values_digest!=ALL_SOURCE_VALUES:
   raise RuntimeError("P17 original 5291 values differ from XOX")
  z=raw.astype(np.float64)
  top=z.max(axis=1)
  nll=top+np.log(np.exp(z-top[:,None]).sum(axis=1))-z[np.arange(len(gold)),gold]
  if not np.isfinite(nll).all() or not (nll>=0).all() or not (nll<150).all():
   raise RuntimeError("full original gold loss nonfinite")
  target=out/"teacher_logits.npy"
  with target.open("xb") as f:np.save(f,raw,allow_pickle=False)
  report={"schema":"beglin-hcp-p18-EOE-original-BF16-teacher-real-v1",
   "status":"P18_EOE_REAL_ORIGINAL_27L_5291T_BF16_FULL_GOLD_LOGITS_PASS",
   "item_id":q["item_id"],"index":index,"dataset":q["dataset"],
   "frozen_480_sha256":FROZEN,
   "prompt_token_sha256":q["prompt_token_sha256"],
   "gold_token_sha256":q["gold_token_sha256"],
   "original_record_sha256":q["record_sha256"],
   "source_original_checkpoint_index_sha256":INDEX,
   "source_original_model_functions_sha256":pinned_source,
   "original_full_decoder_layers_forwarded":27,
   "original_source_BF16_tensors_read_exactly_once":5291,
   "source_original_tensor_values_manifest_sha256":values_digest,
   "input_tokens":len(ids),"gold_tokens":len(gold),
   "gold_NLL_nats":[float(x) for x in nll],
   "full_float32_logit_shape":list(raw.shape),
   "real_BF16_full_vocab_logits_file":"teacher_logits.npy",
   "real_BF16_full_vocab_logits_sha256":sha(target),
   "observed_elapsed_seconds":time.monotonic()-started,
   "observed_process_RSS_peak_bytes":guard.peak_rss,
   "observed_swap_growth_bytes":guard.swap_delta,
   "real_original_XOX_operator_runtime_mutated":False,
   "P18_full_480_complete":False,"full_generation_task_accuracy_certified":False,
   "quality_noninferiority_certified":False,
   "production_touched":False,"automatic_live_promotion":False}
  rec=out/"teacher_receipt.json"
  with rec.open("x") as f:json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
  print("P18_EOE_ORIGINAL_BF16_ONE_ITEM_REAL_PASS",json.dumps({
   "index":index,"item_id":q["item_id"],"gold_mean_NLL":round(float(np.mean(nll)),8),
   "source_5291_hash":values_digest,"F32_SHA256":sha(target),
   "receipt_SHA256":sha(rec),"rss_GiB":round(guard.peak_rss/2**30,3)},sort_keys=True),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--index",type=int,required=True)
 p.add_argument("--preflight-only",action="store_true")
 a=p.parse_args()
 run(a.index,a.preflight_only)
