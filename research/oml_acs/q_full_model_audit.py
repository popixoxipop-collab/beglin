#!/usr/bin/env python3
"""Audit isolated native BEGLIN full Qwen model precision modes.
The RSS, wall times and perplexities are manually transcribed FROM observed
live /usr/bin/time -l process outputs (process IDs attached). Logits,
compressed storage and source hashes are independently re-read from files.
Small synthetic prompts only; final Q hypothesis not tested.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,struct
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_full_model_20261021_v2"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
VOCAB=151936
OUT=LAB/"full_model_Q_aggregate_research_receipt.json"
PPL={
 "f32":dict(process_id="proc_477e9d9eba963b248a52309e452e7ab7",perplexity=1988.5681,token_count=21,wall_s=5.70,peak_RSS_bytes=9273802752,exit_code=0),
 "n8":dict(process_id="proc_863c70a3877229f2dcc839029171de98",perplexity=2148.1101,token_count=21,wall_s=167.91,peak_RSS_bytes=5490196480,exit_code=0),
 "q4":dict(process_id="proc_9cbaac402f4d3464b5b601e9af1a1aec",perplexity=4649.3209,token_count=21,wall_s=13.79,peak_RSS_bytes=4828217344,exit_code=0)
}
DUMP={
 "f32":"proc_b680921b8c5e77b57d04a4ad50b7c78b",
 "n8":"proc_9991e43bf1c73bb59650c15b399cb975",
 "q4":"proc_d0b8f835cd57f0918b4b6c4281a22528"
}
def sha(data):return hashlib.sha256(data).hexdigest()
def private(f):
 if not f.is_file() or f.is_symlink() or (f.stat().st_mode&0o777)!=0o600:
  raise RuntimeError("unsecured/missing input: "+str(f))
def compare_logits():
 vals={};checks={};tokens=[]
 for name in ("f32","n8","q4"):
  f=LAB/name/"results/c_logits_last.f32";private(f)
  raw=f.read_bytes()
  if len(raw)!=VOCAB*4:raise ValueError("unexpected full-vocab shape")
  vec=np.frombuffer(raw,dtype="<f4").astype(np.float64)
  if not np.isfinite(vec).all():raise ValueError("nonfinite logits")
  vals[name]=vec
  checks[name]=dict(logits_sha256=sha(raw),top1=int(np.argmax(vec)))
  ids=LAB/name/"prompt_ids.i32";private(ids);tokens.append(ids.read_bytes())
 if tokens[0]!=tokens[1] or tokens[0]!=tokens[2]:
  raise RuntimeError("token input identity mismatch across modes")
 orig=vals["f32"]
 p=np.exp(orig-orig.max());p/=p.sum()
 result={}
 for name in ("n8","q4"):
  z=vals[name];q=np.exp(z-z.max());q/=q.sum()
  diff=z-orig
  result[name]=dict(
   reference_top1=int(np.argmax(orig)),top1_agreement=bool(np.argmax(z)==np.argmax(orig)),
   top10_overlap=len(set(np.argsort(z)[-10:].tolist())&set(np.argsort(orig)[-10:].tolist())),
   relative_logit_l2=float(np.linalg.norm(diff)/np.linalg.norm(orig)),
   logit_rmse=float(np.sqrt(np.mean(diff**2))),
   logit_max_abs=float(np.max(np.abs(diff))),
   KL_reference_to_quantized_nats=float(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))))),
   probability_reference_top1_under_quantized=float(q[np.argmax(orig)]))
 return dict(original_top1=int(np.argmax(orig)),baseline_top1_probability=float(p.max()),
  prompt_token_ids_sha256=sha(tokens[0]),prompt_token_length=len(tokens[0])//4,
  output_checks=checks,comparison=result)

def memory_accounting():
 with (SNAP/"model.safetensors").open("rb") as fp:
  meta=json.loads(fp.read(struct.unpack("<Q",fp.read(8))[0]))
 roles={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
 count=0;n_values=0
 for name,m in meta.items():
  if not name.startswith("model.layers.") or not name.endswith(".weight") or name.split(".")[-2] not in roles:continue
  if m["dtype"]!="BF16" or len(m["shape"])!=2 or m["shape"][1]%64!=0:raise ValueError("unsupported source")
  count+=1
  n_values+=int(m["shape"][0])*int(m["shape"][1])
 if count!=196 or "lm_head.weight" in meta:raise RuntimeError("model role or tied-head drift")
 return dict(projection_tensors=count,projection_elements=n_values,
  source_BF16_projection_bytes=2*n_values,
  widened_F32_projection_bytes=4*n_values,
  qNg64_8_projection_bytes_including_f32_scales=n_values*17//16,
  Q4G64_projection_bytes_including_f32_scales=n_values*9//16,
  common_nonprojection_weights_not_counted=True,tied_output_head_is_FP32_all_modes=True)
def main():
 os.umask(0o077)
 if OUT.exists():raise FileExistsError("do not overwrite prior experiment")
 if not LAB.is_dir() or LAB.stat().st_mode&0o777!=0o700:raise RuntimeError("not private")
 metadata=LAB/"lab_metadata.json";source=LAB/"qwen_infer_q_lab.c";binary=LAB/"beglin_q_lab"
 private(metadata);private(source)
 info=json.loads(metadata.read_text())
 if sha(source.read_bytes())!=info["lab_modified_source_sha256"]:raise RuntimeError("patched C source differs")
 if sha(binary.read_bytes())!=info["binary_sha256"]:raise RuntimeError("native binary differs")
 if sha((ROOT/"qwen_infer.c").read_bytes())!=info["original_BEGLIN_source_sha256"]:raise RuntimeError("tracked source differs")
 val=compare_logits()
 baseline_nll=math.log(PPL["f32"]["perplexity"])
 data={name:dict(r) for name,r in PPL.items()}
 for mode,d in data.items():
  d["NLL_nats_per_token"]=math.log(d["perplexity"])
  if mode!="f32":
   d["relative_NLL_increase_vs_F32"]=(d["NLL_nats_per_token"]-baseline_nll)/baseline_nll
   d["perplexity_ratio_vs_F32"]=d["perplexity"]/PPL["f32"]["perplexity"]
   d["wall_ratio_vs_F32"]=d["wall_s"]/PPL["f32"]["wall_s"]
   d["RSS_saving_vs_F32"]=1-d["peak_RSS_bytes"]/PPL["f32"]["peak_RSS_bytes"]
 r=dict(schema="oml-acs-q-beglin-native-fullmodel-triple/1",
  status="PASS_NATIVE_ALL_THREE_TIERS_SMALL_SYNTHETIC_TEST",
  model="Qwen/Qwen2.5-1.5B-Instruct",
  official_HF_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  local_BF16_snapshot_revision=info["local_BF16_revision"],
  selected_qproj_original_BF16_sha256=info["source_q_proj_bf16_sha256"],
  original_BEGILN_C_sha256=info["original_BEGLIN_source_sha256"],
  lab_C_sha256=info["lab_modified_source_sha256"],lab_binary_sha256=info["binary_sha256"],
  baseline_definition="BF16-source F32 projection arithmetic, NOT BF16 runtime math",
  mode_tensors={"f32":{"F32":196,"n8":0,"q4":0},"n8":{"F32":0,"n8":196,"q4":0},"q4":{"F32":0,"n8":0,"q4":196}},
  tied_FP32_output_head_unchanged=True,
  projection_storage_memory=memory_accounting(),
  native_last_token_logits=val,
  small_teacher_forced_PPL_and_process_observations=data,
  dump_process_ids=DUMP,
  experiment_uses_only_two_public_synthetic_chat_windows=True,
  source_checkpoint_not_modified=True,main_and_production_untouched=True,
  V6_24_consumed_document_ids_still_unknown=True,final_Q_hypothesis_verdict="NO_VERDICT",
  limitations=["PPL 21 tokens and one final logit input cannot establish general task accuracy",
   "Baseline FP32 arithmetic from BF16-source weights; not true BF16 inference runtime",
   "Wall timing and RSS include source loading, weight transcode and paging; not decode tokens/second",
   "Only one run per PPL condition; no timing CI or cross-chip evaluation",
   "Source quantizer behavior tested n8 CPU scalar and Q4 NEON, not BEGLIN Metal GPU kernels",
   "No actual BEGLIN adaptive per-layer bits, C1-D behavioral AUROC under quant, or final IID/OOD evaluation",
   "Old V6 used 24 document IDs and third-party dataset licensing still require independent audit"])
 with OUT.open("x",encoding="utf-8") as f:
  json.dump(r,f,indent=2,ensure_ascii=True,sort_keys=True)
  f.write("\n");f.flush();os.fsync(f.fileno())
 OUT.chmod(0o600)
 print("Q_FULLMODEL_AUDIT_PASS",json.dumps(r,sort_keys=True),flush=True)
if __name__=="__main__":main()
