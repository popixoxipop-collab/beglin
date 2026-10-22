#!/usr/bin/env python3
"""Independent exact grouped-output-KL audit, source 24 and separate diagnostic 24."""
import hashlib,json,os,pathlib
import numpy as np
from scipy.special import logsumexp
ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv9-ggn-20261023")
LAB=ROOT/"lab_local/qv10_topk_trust_20261023"
DOC=ROOT/"docs/research"
POL=DOC/"BEGLIN_QV10_TOPK_OTHER_TRUST_SOURCE_POLICY_20261023.json"
MASK=ROOT/"configs/research/BEGLIN_QV10_TOP2OTHER_TRUST_28N8_SOURCE_POLICY_20261023.txt"
MASK64=ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt"
OUT=LAB/"qv10_48originalBF16_7group_independent_audit.json"
PUB=DOC/"BEGLIN_QV10_48SOURCE_DIAGNOSTIC_INDEPENDENT_PUBLIC_20261023.json"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:raise RuntimeError("original private source not 0600 "+str(p))
 return p.read_bytes()
def load(stage):
 path=LAB/f"original_BF16_top2other_{stage}24_private.json"
 obj=json.loads(sealed(path))
 prior=json.loads(sealed(OLD/f"lab_local/qv9_ggn_20261023/original_BF16_choice_GGN_{stage}24_private.json"))
 q4=np.asarray(obj["signed_Q4_relative_logit_groups"],dtype=float)
 n8=np.asarray(obj["signed_n8_relative_logit_groups"],dtype=float)
 p=np.asarray(obj["source_original_BF16_output_group_probabilities"],dtype=float)
 if q4.shape!=(24,196,6) or n8.shape!=(24,196,6) or p.shape!=(24,7):raise RuntimeError("wrong 7-group BF16 signed source dimensions")
 if not np.array_equal(q4[:,:,:3],prior["q4_signed_relative_logit_delta"]) or not np.array_equal(n8[:,:,:3],prior["n8_signed_relative_logit_delta"]):raise RuntimeError("old Qv9 original A/B/C-D columns changed")
 merged=p[:,[0,1,2,6]]/p[:,[0,1,2,6]].sum(axis=1,keepdims=True)
 if np.max(abs(merged-np.asarray(prior["original_conditional_four_letter_probabilities"])))>5e-4:raise RuntimeError("old BF16 four-answer probabilities different")
 proofs=obj["original_case_proof_SHA256"];folder=LAB/f"{stage}24_top2other_native_signed"
 for i in range(24):
  f=sealed(folder/f"group_{i:02d}.json");case=json.loads(f)
  if sha(f)!=proofs[i] or case["index"]!=i or case["new_BF16_backward_calls"]!=3 or not np.array_equal(q4[i,:,3:],case["Q4_signed_top2other_196x3"]) or not np.array_equal(n8[i,:,3:],case["n8_signed_top2other_196x3"]):raise RuntimeError("per-question source BF16 new gradient SHA differs")
 if not np.isfinite(q4).all() or not np.isfinite(n8).all() or not np.isfinite(p).all() or np.max(abs(p.sum(axis=1)-1))>5e-5:raise RuntimeError("nonfinite grouped masses")
 return q4,n8,p,sha(sealed(path))
def select(path):
 bits=np.zeros(196)
 if path:
  pairs=[tuple(x.split()) for x in path.read_text().splitlines()]
  if len(pairs)!=28 or len(set(pairs))!=28:raise RuntimeError("28 native n8 projections required")
  for role,layer in pairs:
   if role not in ROLES or not 0<=int(layer)<28:raise RuntimeError("unknown tensor")
   bits[ROLES.index(role)*28+int(layer)]=1
 return bits
def risk(a,b,p,bits):
 z=a+bits[None,:,None]*(b-a)
 v=z.sum(axis=1)
 v7=np.concatenate((v,np.zeros((24,1))),axis=1)
 kl=logsumexp(np.log(np.clip(p,1e-300,1))+v7,axis=1)-(p*v7).sum(axis=1)
 if np.min(kl)<-1e-7:raise RuntimeError("grouped KL negative")
 kl=np.maximum(kl,0)
 trust=.05*(p*np.maximum(0,abs(v7)-2)**2).sum(axis=1)
 expanded=np.concatenate((z,np.zeros((24,196,1))),axis=-1)
 m=np.einsum("sg,sig->si",p,expanded)
 diag=.5*(np.einsum("sg,sig->si",p,expanded**2)-m*m)
 if np.min(diag)<-1e-7:raise RuntimeError("PSD grouped output curvature violated")
 ridge=np.maximum(0,diag).mean(axis=0).sum()/196
 return {"objective":float(max(np.mean((kl+trust)[::2]),np.mean((kl+trust)[1::2]))+.1*ridge),
  "mean_KL":float(kl.mean()),"mean_trust":float(trust.mean()),
  "fraction_large_shift":float(np.mean(np.max(abs(v),axis=1)>2)),
  "original_fullvocab_answer_mass":float(np.mean(p[:,[0,1,2,6]].sum(axis=1)))}
def main():
 if OUT.exists() or PUB.exists():raise FileExistsError("independent audit already finalized")
 pol=json.loads(POL.read_text())
 if pol["status"]!="SOURCE_ONLY_REAL_TOPK_OTHER_GROUPED_KL_28N8_MASK_FROZEN_BEFORE_DIAG_OR_RACE" or pol["mask_sha256"]!=sha(MASK.read_bytes()):raise RuntimeError("source-only policy authority invalid")
 d={s:load(s) for s in ("train","diagnostic")}
 stats={name:{s:risk(*row[:3],select(mask)) for s,row in d.items()}
   for name,mask in {"q4":None,"fisher64":MASK64,"qv10":MASK}.items()}
 if abs(stats["qv10"]["train"]["objective"]-pol["selected_source_train_objective"]["objective"])>1e-8 or abs(stats["q4"]["train"]["objective"]-pol["prior_mask_source_only_objectives"]["all_Q4"]["objective"])>1e-8:raise RuntimeError("independent optimizer objective disagrees")
 better=stats["qv10"]["diagnostic"]["objective"]<min(stats["q4"]["diagnostic"]["objective"],stats["fisher64"]["diagnostic"]["objective"])
 doc={"schema":"BEGLIN-Qv10-48-original-BF16-source7group-independent-audit/1",
  "status":"PASS_SOURCE_24_AND_DIAG_24_BF16_7GROUP_SIGNED_NATIVE_RESIDUAL_RECALC",
  "mask_sha256":pol["mask_sha256"],"source_policy_sha256":sha(POL.read_bytes()),
  "source24_original_private_SHA256":d["train"][3],
  "diagnostic24_original_private_SHA256":d["diagnostic"][3],
  "source_train_diagnostic_previous_Qv9_used_before":True,
  "grouped_7way_probability_masses_original_fullvocab":True,
  "within_OTHER_curvature_missing":True,
  "diagnostic_robustly_beats_both_q4_and_fisher64":better,
  "source_to_diagnostic":stats,
  "not_true_quantized_fullmodel_KL":True,"no_RACE_eval_outputs_used":True,
  "no_live_promotion_allowed":True}
 os.umask(0o077)
 raw=(json.dumps(doc,sort_keys=True,indent=2)+"\n").encode()
 with OUT.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 OUT.chmod(0o600)
 public=dict(doc);public["independent_private_receipt_sha256"]=sha(raw)
 result=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
 with PUB.open("xb") as f:f.write(result);f.flush();os.fsync(f.fileno())
 print("QV10_INDEPENDENT_48_ORIGINAL_BF16_GROUPED_KL_AUDIT_PASS",
  json.dumps({"train":{x:round(v["train"]["objective"],4) for x,v in stats.items()},
   "diag":{x:round(v["diagnostic"]["objective"],4) for x,v in stats.items()},
   "diagnostic_beats_both":better,"sha":sha(result)},sort_keys=True))
if __name__=="__main__":main()
