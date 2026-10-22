#!/usr/bin/env python3
"""Qv9 post-freeze original BF16 choice-output trust-region diagnostic.

GGN = 0.5 Var_p(J delta) is second-order conditional choice-space KL proxy.
Exact KL(p || softmax(log p + Jdelta)) for the *linearized logits* diagnoses
when the quadratic Taylor truncation is no longer trustworthy. This is NOT
the native quantized model's actual KL, and diagnostic data must not tune mask.
"""
import hashlib,json,math,os,pathlib
import numpy as np
from scipy.special import logsumexp
R=pathlib.Path(__file__).resolve().parents[2];L=R/"lab_local/qv9_ggn_20261023";D=R/"docs/research";C=R/"configs/research"
OUT=D/"BEGLIN_QV9_GGN_QUADRATIC_TRUST_REGION_DIAGNOSTIC_20261023.json"
P=D/"BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
MODEL={
"allQ4":None,
"Fisher64":C/"BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",
"Fisher128":C/"BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt",
"Qv8signed96":C/"BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt",
"Qv9_GGN24":C/"BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt"}
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(b):return hashlib.sha256(b).hexdigest()
def private(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:raise RuntimeError("original source GGN not sealed")
 return p.read_bytes()
def choice_delta(obj,mask):
 q=np.asarray(obj["q4_signed_relative_logit_delta"],dtype=float)
 n=np.asarray(obj["n8_signed_relative_logit_delta"],dtype=float)
 p=np.asarray(obj["original_conditional_four_letter_probabilities"],dtype=float)
 bit=np.zeros(196)
 if mask:
  for r,layer in [line.split() for line in mask.read_text().splitlines()]:
   bit[ROLES.index(r)*28+int(layer)]=1
 d=np.sum(q+bit[None,:,None]*(n-q),axis=1)
 z=np.column_stack([d,np.zeros(24)])
 prob=np.maximum(p,1e-300);prob/=prob.sum(axis=1,keepdims=True)
 m=(z*prob).sum(axis=1)
 ggn=np.maximum(0,.5*((z*z*prob).sum(axis=1)-m*m))
 log_kl=logsumexp(np.log(prob)+z,axis=1)-m
 if np.min(log_kl)<-1e-8:raise RuntimeError("negative exact KL from linearized choice-output")
 f=np.abs(z).max(axis=1)
 return {"mean_conditional_GGN":float(ggn.mean()),
   "mean_linearized_4choice_exact_KL":float(np.maximum(log_kl,0).mean()),
   "fraction_abs_4choice_relative_logit_shift_gt_1":float(np.mean(f>1)),
   "fraction_abs_4choice_relative_logit_shift_gt_4":float(np.mean(f>4)),
   "median_max_abs_relative_logit_shift":float(np.median(f)),
   "KL_divided_by_second_order_GGN_mean_ratio":float(np.mean(log_kl)/np.mean(ggn)) if ggn.mean()>1e-12 else None}
def run():
 if OUT.exists():raise FileExistsError("no overwrite Qv9 trust region scientific caveat")
 p=json.loads(P.read_text())
 if p["status"]!="GGN_SOURCE_ONLY_FROZEN_BEFORE_DIAG_OR_MMLU" or sha(MODEL["Qv9_GGN24"].read_bytes())!=p["mask_sha256"]:
  raise RuntimeError("source mask was not original source-only frozen")
 cases={}
 for cohort,file in (("train","original_BF16_choice_GGN_train24_private.json"),("diagnostic","original_BF16_choice_GGN_diagnostic24_private.json")):
  raw=private(L/file);r=json.loads(raw)
  if r["source_group"]!=cohort or r["n_source_examples"]!=24:raise RuntimeError("original BF16 conditional GGN sample set wrong")
  cases[cohort]={k:choice_delta(r,v) for k,v in MODEL.items()}
 report={"schema":"BEGLIN-Qv9-original-choice-GGN-quadratic-trust-regime-analysis/1",
   "status":"POST_MASK_FREEZE_OUTPUT_CHOICE_LINEARIZED_KL_CROSSCHECK",
   "source_policy_sha256":sha(P.read_bytes()),"original_choice_model_rev":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
   "source_and_diagnostic":cases,
   "exact_choice_KL_is_only_on_original_first_order_Jdelta_logits_not_fullmodel_native_logits":True,
   "original_model_output_deviation_not_directly_measured":True,
   "no_held_MMLU_logit_used":True,"not_used_for_mask_reoptimization":True,"no_production_promotion":True}
 b=(json.dumps(report,indent=2,sort_keys=True)+"\n").encode()
 with OUT.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 print("QV9_CHOICENLL_TRUST_REGION_POSTFREEZE_DIAGNOSTIC",json.dumps({
    "sha":sha(b),
    "train_GGN":{k:round(v["mean_conditional_GGN"],4) for k,v in cases["train"].items()},
    "diag_GGN":{k:round(v["mean_conditional_GGN"],4) for k,v in cases["diagnostic"].items()},
    "train_logitshift_gt4":{k:round(v["fraction_abs_4choice_relative_logit_shift_gt_4"],3) for k,v in cases["train"].items()}
 },sort_keys=True),flush=True)
if __name__=="__main__":run()
