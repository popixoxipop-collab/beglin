#!/usr/bin/env python3
"""P18 original 27L BF16 DEVELOPMENT-ONLY staged parity evaluator (resumable).

Uses the exact SHA-pinned, previously distributed v2/qualifier code.
Stages keep each real forward immutable so an interruption does not repeat
the 5291-tensor source stream. No frozen test row or production API touched.

Usage: --stage standalone_short | standalone_long | shared_short_long |
       shared_long_short | audit
Every model stage REQUIRES a resource_plan admission before invocation.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,pathlib,sys,time
import numpy as np
import torch
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
sys.path.insert(0,str(ROOT))
import hcp_p18_devonly_shared_bf16_gold_v2_qualify as q

TARGET=ROOT/"P18_DEVONLY_BF16_STAGED_PARITY_20261016"
SCHEMA="beglin-hcp-p18-27layer-5291-devonly-staged-parity-v1"
WORKERS=("standalone_short","standalone_long","shared_short_long","shared_long_short")
GOLDS=(3,5)
POSITIONS=(101,198)

def source():
 if (not ROOT.is_dir() or ROOT.is_symlink() or
     q.sha(ROOT/"hcp_p18_shared_bf16_gold_window_v2.py")!=q.PIN["v2_core"] or
     q.sha(ROOT/"hcp_p18_eoe_d50_original_bf16_core.py")!=q.PIN["P18_core"]):
  raise RuntimeError("Original/research BF16 implementation/source identity drift")
 for p,h in ((q.DEV,q.PIN["dev"]),
             (q.MODEL/"model.safetensors.index.json",q.PIN["index"]),
             (q.MODEL/"config.json",q.PIN["config"]),
             (q.MODEL/"modeling_deepseek.py",q.PIN["implementation"]),
             (q.MODEL/"tokenizer.json",q.PIN["tokenizer"]),
             (q.P17/"hcp_p17_teacher_gold.py",q.PIN["teacher"]),
             (q.P17/"hcp_p15_streamed_teacher.py",q.PIN["streamer"])):
  if q.sha(p)!=h:raise RuntimeError("P18 mandatory original pinned file drift: "+p.name)
 d=json.loads(q.DEV.read_text())
 tokens=d["prompt_ids"]
 if (d.get("source_role")!="development_only_not_480_test" or
     d.get("gold_answer_not_scored") is not True or
     len(tokens)!=203 or d.get("prompt_token_count")!=203 or
     q.tokhash(tokens)!=d["prompt_token_sha256"] or
     d.get("tokenizer_sha256")!=q.PIN["tokenizer"]):
  raise RuntimeError("P18 source not immutable development-only 203 prompt")
 short=tokens[:104];long=tokens[:203]
 if len(short)!=104 or len(long)!=203 or [len(short)-101,len(long)-198]!=list(GOLDS):
  raise RuntimeError("Fixed development pseudo-gold windows drift")
 return (short,long)

def target(stage):
 t=TARGET/stage
 if t.is_symlink():raise RuntimeError("Symlinked model stage cannot be trusted")
 return t

def audited(stage):
 dest=target(stage)
 p=dest/"stage_receipt.json"
 if not p.is_file() or p.is_symlink():raise RuntimeError("Missing immutable stage receipt: "+stage)
 j=json.loads(p.read_text())
 expected_indices={
  "standalone_short":[0], "standalone_long":[1],
  "shared_short_long":[0,1], "shared_long_short":[1,0],
 }[stage]
 if(j.get("schema")!=SCHEMA or j.get("status")!="ONE_REAL_ORIGINAL_BF16_DEVONLY_STAGE_PASS"
    or j.get("stage")!=stage or j.get("source_role")!="development_only_not_480_test"
    or j.get("source_original_dev203_sha256")!=q.PIN["dev"]
    or j.get("P17_original_source_sha256")!=q.PIN["teacher"]
    or j.get("v2_source_sha256")!=q.PIN["v2_core"]
    or j.get("original_27layer_source_SHA256")!=q.PIN["implementation"]
    or j.get("original_source_tensor_values_sha256")!=q.PIN["original_tensor_values"]
    or j.get("original_tensor_count")!=5291 or j.get("original_decoder_layers")!=27
    or j.get("gold_positions_for_unique_development_inputs")!=list(GOLDS)
    or [v.get("original_index") for v in j.get("raw_files",[])]!=expected_indices
    or j.get("production_touched") is not False or j.get("automatic_live_promotion") is not False):
  raise RuntimeError("Unverified stage source/model/receipt: "+stage)
 logits=[]
 for k,f in enumerate(j["raw_files"]):
  p=dest/f["file"]
  if p.is_symlink() or not p.is_file() or q.sha(p)!=f["sha256"]:
   raise RuntimeError("Actual F32 raw output SHA mismatch "+stage)
  a=np.load(p,allow_pickle=False)
  if a.dtype!=np.float32 or a.shape!=(GOLDS[f["original_index"]],102400) or not np.isfinite(a).all():
   raise RuntimeError("Not full original BF16 gold-position logits")
  logits.append(a)
 return j,logits

def prepare(stage):
 if stage not in WORKERS:raise ValueError("unsupported source stage")
 samples=source()
 if not TARGET.is_dir():TARGET.mkdir(mode=0o700)
 dest=target(stage)
 if dest.exists():raise FileExistsError("Existing stage is immutable, use read-only audit")
 if stage.startswith("shared_"):
  # Both controls MUST finish before allowing any batch optimization.
  audited("standalone_short")
  audited("standalone_long")
 return samples

def run(stage):
 inputs=prepare(stage)
 if q.psutil.virtual_memory().available<17*1024**3:raise RuntimeError("Insufficient safe host RAM")
 torch.set_num_threads(2)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 try:os.nice(10)
 except PermissionError:pass
 originals,_=q.original_classes(q.MODEL/"modeling_deepseek.py")
 begun=time.monotonic()
 with q.Guard() as guard:
  if stage.startswith("standalone_"):
   index=0 if stage=="standalone_short" else 1
   raw,secs=q.original_once(inputs[index],POSITIONS[index],originals)
   arrays=[raw] if index==0 else [raw]
   count=5291
   rows=[index]
  else:
   reverse=stage.endswith("long_short")
   order=[1,0] if reverse else [0,1]
   t=q.P18LongGoldWindowTeacher(q.TensorStore(q.MODEL),originals,dtype=torch.bfloat16)
   data=q.forward_two_original_gold_windows(t,[inputs[i] for i in order],
            [POSITIONS[i] for i in order],max_length=1024)
   sec=time.monotonic()-begun
   hashes=hashlib.sha256((json.dumps(t.store.value_hashes,sort_keys=True,indent=2)+"\n").encode()).hexdigest()
   if(hashes!=q.PIN["original_tensor_values"] or
      len(t.store.loaded_names)!=5291 or len(set(t.store.loaded_names))!=5291):
    raise RuntimeError("Shared original 5291 weight-value digest/once-only invariant failed")
   arrays=[v.detach().cpu().numpy() for v in data]
   rows=order
   count=len(t.store.loaded_names)
  if not isinstance(secs if stage.startswith("standalone_") else sec,(int,float)):
   raise RuntimeError("No measured original forward duration")
  dest=target(stage)
  dest.mkdir(mode=0o700)
  files=[]
  for original_index,a in zip(rows,arrays):
   if a.dtype!=np.float32 or a.shape!=(GOLDS[original_index],102400):
    raise RuntimeError("Original real float32 gold window shape differs")
   fn="original_devonly_%d_f32.npy"%original_index
   path=dest/fn
   with path.open("xb") as f:np.save(f,a,allow_pickle=False)
   files.append({"original_index":original_index,"file":fn,"sha256":q.sha(path)})
  elapsed=(secs if stage.startswith("standalone_") else sec)
  proof={"schema":SCHEMA,"status":"ONE_REAL_ORIGINAL_BF16_DEVONLY_STAGE_PASS",
    "source_role":"development_only_not_480_test","stage":stage,
    "source_original_dev203_sha256":q.PIN["dev"],
    "P17_original_source_sha256":q.PIN["teacher"],
    "v2_source_sha256":q.PIN["v2_core"],
    "original_27layer_source_SHA256":q.PIN["implementation"],
    "original_source_tensor_values_sha256":q.PIN["original_tensor_values"],
    "original_tensor_count":count,"original_decoder_layers":27,
    "gold_positions_for_unique_development_inputs":list(GOLDS),
    "raw_files":files,"actual_original_full_model_wall_seconds":elapsed,
    "peak_RSS_bytes":guard.peak_rss,"new_swap_bytes":guard.max_new_swap,
    "training_or_holdout_tuning":False,"task_accuracy_certified":False,
    "frozen_480_quality_certified":False,
    "production_touched":False,"automatic_live_promotion":False}
  with (dest/"stage_receipt.json").open("x") as f:
   json.dump(proof,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
  print("P18_STAGED_ORIGINAL_DEVONLY_FORWARD_PASS",json.dumps({
    "stage":stage,"actual_full_vocabulary_gold_windows":len(files),
    "wall_seconds":round(elapsed,3),"receipt_SHA256":q.sha(dest/"stage_receipt.json"),
    "max_RSS_GiB":round(guard.peak_rss/1024**3,3)},sort_keys=True),flush=True)

def audit():
 source()
 if (not TARGET.is_dir() or TARGET.is_symlink() or
     (TARGET/"P18_DEVONLY_STAGED_PARITY_FINAL_RECEIPT.json").exists()):
  raise RuntimeError("No readable stage data or already finalised")
 stage={name:audited(name) for name in WORKERS}
 originals={0:stage["standalone_short"][1][0],1:stage["standalone_long"][1][0]}
 results=[]
 for key in ("shared_short_long","shared_long_short"):
  meta,arrays=stage[key]
  for block,data in zip(meta["raw_files"],arrays):
   idx=block["original_index"]
   baseline=originals[idx]
   diff=float(np.max(np.abs(data.astype(np.float64)-baseline.astype(np.float64))))
   passed=bool(np.array_equal(data,baseline))
   results.append({"stage":key,"dev_sequence_index":idx,"gold_count":GOLDS[idx],
     "logits_number":int(data.size),"bitwise_equal":passed,
     "maximum_absolute_F32_difference":diff})
 complete=all(z["bitwise_equal"] for z in results)
 baseline_sec=sum(stage[name][0]["actual_original_full_model_wall_seconds"] for name in ("standalone_short","standalone_long"))
 shared_sec=stage["shared_short_long"][0]["actual_original_full_model_wall_seconds"]
 rev_sec=stage["shared_long_short"][0]["actual_original_full_model_wall_seconds"]
 result={"schema":"beglin-hcp-p18-staged-original-27L-BF16-real-parity-summary-v1",
    "status":"P18_REAL_DEVONLY_ORIGINAL_SHARED_BF16_ALL_GOLD_BITEXACT_PASS" if complete else "P18_REAL_DEVONLY_ORIGINAL_SHARED_BF16_PARITY_FAIL_DO_NOT_ADOPT",
    "only_development_203_token_source_sha256":q.PIN["dev"],
    "v2_source_sha256":q.PIN["v2_core"],"original_P17_source_SHA256":q.PIN["teacher"],
    "validated_F32_elements_each_pass":int((GOLDS[0]+GOLDS[1])*102400),
    "two_development_gold_windows":[3,5],"all_logit_comparisons":results,
    "standalone_two_original_weight_streams_seconds":baseline_sec,
    "shared_one_original_weight_stream_seconds":shared_sec,
    "reverse_shared_one_original_weight_stream_seconds":rev_sec,
    "measured_speedup_ratio":baseline_sec/shared_sec if shared_sec>0 else None,
    "frozen_480_quality_certified":False,"production_touched":False,
    "automatic_live_promotion":False}
 path=TARGET/"P18_DEVONLY_STAGED_PARITY_FINAL_RECEIPT.json"
 with path.open("x") as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
 print("P18_STAGED_DEVONLY_SHARED_BF16_FINAL",json.dumps({"status":result["status"],
   "speedup_observed":result["measured_speedup_ratio"],
   "all_gold_logit_f32_bitwise_equal":complete,
   "receipt_SHA256":q.sha(path)},sort_keys=True),flush=True)
 if not complete:raise RuntimeError("Shared candidate FAIL: production and frozen cohort remain blocked")

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--stage",choices=(*WORKERS,"audit"),required=True)
 args=ap.parse_args()
 if args.stage=="audit":
  audit()
  return
 # File lock prevents overlapping partial stage owners or different frozen dev refs.
 TARGET.mkdir(mode=0o700,exist_ok=True)
 with (TARGET/".worker.lock").open("a+") as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run(args.stage)

if __name__=="__main__":main()
