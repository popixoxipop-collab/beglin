#!/usr/bin/env python3
"""Independent read-only four-stage full BF16 source/research parity audit.

Only 203-token DEVELOPMENT-ONLY source; each stage is an original 27-layer,
5291-tensor full float32 102400-vocab logits source. Requires all 4 unique
full sequence traces and 3+5 gold token spans. Original BF16 reference and
shared short->long / long->short must be *byte exact*. No 480 quality claims,
and no inference is launched by this verifier.
"""
from __future__ import annotations
import hashlib,json,pathlib,math
import numpy as np
R=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
P=R/"P18_DEVONLY_BF16_STAGED_PARITY_20261016"
EXPECTED={
 "source_dev":"b628ec3247cfc7a94785a93e539562a7a2bc7726f133c21162479749b8f0ac2c",
 "shared_core":"c160e79ecb527ccfcc7c908e7fc64f67abf8938a4186c2f7033eac753c7cde2f",
 "qualifier":"5b50cd59f7b056177a5ef696e8603299c722081ae00f6d21ba634d15af2620a1",
 "staged":"cf5b20128bfa93d151d24eb1969397fa9e1def35025c13f295b392a953f6f7ff",
 "original_teacher":"40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3",
 "original_tensor_value_sha256":"88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda",
}
STAGES={
 "standalone_short":[0],"standalone_long":[1],
 "shared_short_long":[0,1],"shared_long_short":[1,0]
}
COUNTS=[3,5]
def sha(path):
 p=pathlib.Path(path)
 if not p.is_file() or p.is_symlink():raise RuntimeError("Missing/symlink source: "+str(p))
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
 return h.hexdigest()
def require(p,dig):
 if sha(p)!=dig:raise RuntimeError("Exact original source SHA drift: "+str(p))
def main():
 for key,file in [("source_dev","p18_devonly_203_hardware_input_PRIVATE.json"),
                  ("shared_core","hcp_p18_shared_bf16_gold_window_v2.py"),
                  ("qualifier","hcp_p18_devonly_shared_bf16_gold_v2_qualify.py"),
                  ("staged","hcp_p18_devonly_staged_shared_bf16_parity.py")]:
  require(R/file,EXPECTED[key])
 if P.is_symlink() or not P.is_dir():raise RuntimeError("No canonical staged research scope")
 bank=json.loads((R/"p18_devonly_203_hardware_input_PRIVATE.json").read_text())
 ids=bank["prompt_ids"]
 seq_sha=hashlib.sha256(",".join(map(str,ids)).encode("ascii")).hexdigest()
 if (bank.get("source_role")!="development_only_not_480_test" or
     bank.get("gold_answer_not_scored") is not True or len(ids)!=203 or
     seq_sha!=bank["prompt_token_sha256"]):
  raise RuntimeError("No pre-existing development-only original source")
 rows={};meta={}
 for stage,expected_order in STAGES.items():
  folder=P/stage
  manifest=folder/"stage_receipt.json"
  j=json.loads(manifest.read_text())
  if (j.get("schema")!="beglin-hcp-p18-27layer-5291-devonly-staged-parity-v1"
      or j.get("status")!="ONE_REAL_ORIGINAL_BF16_DEVONLY_STAGE_PASS"
      or j.get("stage")!=stage
      or j.get("source_role")!="development_only_not_480_test"
      or j.get("source_original_dev203_sha256")!=EXPECTED["source_dev"]
      or j.get("P17_original_source_sha256")!=EXPECTED["original_teacher"]
      or j.get("v2_source_sha256")!=EXPECTED["shared_core"]
      or j.get("original_source_tensor_values_sha256")!=EXPECTED["original_tensor_value_sha256"]
      or j.get("original_tensor_count")!=5291
      or j.get("original_decoder_layers")!=27
      or j.get("gold_positions_for_unique_development_inputs")!=COUNTS
      or j.get("production_touched") is not False
      or j.get("automatic_live_promotion") is not False):
   raise RuntimeError("Invalid source receipt/model/provenance: "+stage)
  f=j.get("raw_files",[])
  if [x.get("original_index") for x in f]!=expected_order:
   raise RuntimeError("Incorrect original source control order for "+stage)
  if not isinstance(j.get("actual_original_full_model_wall_seconds"),(int,float)) or not (0<j["actual_original_full_model_wall_seconds"]<3600):
   raise RuntimeError("Stage time is not a real bounded forward measurement")
  meta[stage]={"receipt_sha256":sha(manifest),
               "measured_original_forward_seconds":j["actual_original_full_model_wall_seconds"],
               "original_tensors_once":5291}
  for entry in f:
   n=entry["original_index"]
   source=folder/entry["file"]
   require(source,entry["sha256"])
   raw=np.load(source,allow_pickle=False)
   if (raw.shape!=(COUNTS[n],102400) or raw.dtype!=np.float32 or
       not np.isfinite(raw).all()):
    raise RuntimeError("Real original 102400 floats not valid in "+stage)
   rows[(stage,n)]=raw
 comparisons=[]
 for stage in ("shared_short_long","shared_long_short"):
  for n in (0,1):
   actual=rows[(stage,n)]
   ctrl=rows[("standalone_short" if n==0 else "standalone_long",n)]
   bit_exact=actual.tobytes()==ctrl.tobytes()
   max_err=float(np.max(np.abs(actual.astype(np.float64)-ctrl.astype(np.float64))))
   comparisons.append({"shared_stage":stage,"dev_source_index":n,
       "gold_positions":COUNTS[n],"full_vocab_float32_values_compared":int(actual.size),
       "byte_exact":bit_exact,"maximum_abs_float32_logit_diff":max_err})
 all_pass=all(x["byte_exact"] for x in comparisons)
 alone=meta["standalone_short"]["measured_original_forward_seconds"]+meta["standalone_long"]["measured_original_forward_seconds"]
 shared=meta["shared_short_long"]["measured_original_forward_seconds"]
 rev=meta["shared_long_short"]["measured_original_forward_seconds"]
 result={"schema":"beglin-hcp-p18-development-only-real-original-bf16-independent-four-stage-audit-v1",
    "status":"P18_SHARED_27L_BF16_REAL_FULL_GOLD_VOCAB_BOTH_ORDERS_BYTE_EXACT_DEVONLY" if all_pass else "P18_SHARED_27L_BF16_REAL_FULL_GOLD_VOCAB_PARITY_FAIL",
    "source_role":"development_only_not_480_test","input_original_tokens":203,
    "original_tensor_value_reference_SHA256":EXPECTED["original_tensor_value_sha256"],
    "stage_receipts":meta,"comparisons":comparisons,
    "comparison_count":len(comparisons),
    "comparison_float32_logits_total":sum(x["full_vocab_float32_values_compared"] for x in comparisons),
    "baseline_2_separate_original_BF16_forward_seconds":alone,
    "shared_short_long_original_BF16_forward_seconds":shared,
    "shared_long_short_original_BF16_forward_seconds":rev,
    "original_forward_speedup_one_run_ratio":alone/shared,
    "reverse_order_ratio":alone/rev,
    "frozen_480_research_quality_certified":False,
    "official_original_EOE_A_B_native_forward_unrun":True,
    "full_generation_task_accuracy_certified":False,
    "production_touched":False,"automatic_live_promotion":False}
 outfile=P/"P18_INDEPENDENT_STAGED_REAL_DEVONLY_BF16_AUDIT.json"
 if outfile.exists():raise FileExistsError("Independent 4stage audit already exists")
 with outfile.open("x") as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
 print(json.dumps({"status":result["status"],"bit_exact":all_pass,
 "full_vocab_values_compared":result["comparison_float32_logits_total"],
 "speedup":round(alone/shared,4),"reverse_speedup":round(alone/rev,4),
 "receipt_SHA256":sha(outfile)},sort_keys=True),flush=True)
 if not all_pass:raise RuntimeError("Real shared dev BF16 failed: do not promote to frozen eval")
if __name__=="__main__":main()
