"""Independent replay of 24+24 original BF16 choice-GGN Jacobians."""
import hashlib,json,math,os,pathlib
import numpy as np
R=pathlib.Path(__file__).resolve().parents[2]
L=R/"lab_local/qv9_ggn_20261023"
P=R/"docs/research"
C=R/"configs/research"
POL=P/"BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
OUT=P/"BEGLIN_QV9_INDEPENDENT_OUTPUT_GGN_TRAIN_DIAGNOSTIC_20261023.json"
TARGETS={"q4":None,"fisher64":C/"BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",
"fisher128":C/"BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt",
"signed96":C/"BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt",
"ggn24":C/"BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt"}
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:raise RuntimeError("Qv9 private source protection failed")
 return p.read_bytes()
def load(tag):
 file=L/f"original_BF16_choice_GGN_{tag}24_private.json"
 frozen=L/("ARC_train24_GGN_source_frozen.json" if tag=="train" else "ARC_diag24_independent_frozen.json")
 pub=P/f"BEGLIN_QV9_{'TRAIN' if tag=='train' else 'DIAGNOSTIC'}24_CONDITIONAL_GGN_PUBLIC_20261023.json"
 raw=sealed(file);obj=json.loads(raw);info=json.loads(pub.read_text())
 if info["full_private_receipt_sha256"]!=sha(raw) or obj["source_group"]!=tag:raise RuntimeError("source/diag official SHA mismatch")
 names=json.loads(sealed(frozen))["questions"]
 q=np.array(obj["q4_signed_relative_logit_delta"],dtype=np.float64)
 n=np.array(obj["n8_signed_relative_logit_delta"],dtype=np.float64)
 p=np.array(obj["original_conditional_four_letter_probabilities"],dtype=np.float64)
 if q.shape!=(24,196,3) or n.shape!=(24,196,3) or p.shape!=(24,4) or not np.isfinite(q).all() or not np.isfinite(n).all() or not np.isfinite(p).all():raise RuntimeError("source GGN shape/nan")
 if np.max(abs(p.sum(axis=1)-1))>1e-5:raise RuntimeError("4-option BF16 source probabilities drift")
 for i,row in enumerate(names):
  f=L/f"{'train' if tag=='train' else 'diag'}24_choice_Jacobians"/f"choice_jacobian_{i:02d}.json"
  blob=sealed(f);x=json.loads(blob)
  if sha(blob)!=obj["per_case_original_BF16_forward_and_backward_receipts_sha256"][i] or x["source_question_id_SHA256"]!=sha(row["id"].encode()):raise RuntimeError("per-item original BF16 Jacobian SHA drift")
  if not np.array_equal(q[i],np.array(x["signed_Q4_relative_jacobian_196x3"])) or not np.array_equal(n[i],np.array(x["signed_n8_relative_jacobian_196x3"])):raise RuntimeError("per-item contract inconsistency")
 return q,n,p,sha(raw)
def ggn(a,p):
 z=np.concatenate((a,np.zeros((len(a),1))),axis=1)
 m=np.sum(p*z,axis=1)
 values=.5*(np.sum(p*z*z,axis=1)-m*m)
 if not np.isfinite(values).all() or np.min(values)<-1e-8:raise RuntimeError("PSD Gauss Newton violation")
 return np.maximum(values,0)
def risk(q,n,p,mask):
 idx={r:i for i,r in enumerate(ROLES)};bits=np.zeros(196)
 if mask:
  rows=[tuple(line.split()) for line in mask.read_text().splitlines()]
  if len(rows)!=28 or len(set(rows))!=28:raise RuntimeError("28 projections required")
  for role,layer in rows:bits[idx[role]*28+int(layer)]=1
 selected=q+(n-q)*bits[None,:,None]
 together=ggn(selected.sum(axis=1),p)
 isolated=np.sum([ggn(selected[:,i,:],p) for i in range(196)],axis=0)
 return {"joint_GGN_mean":float(together.mean()),
   "ridge_GGN_mean":float(isolated.mean()),
   "robust_GGN":float(.65*max(together[::2].mean(),together[1::2].mean())+.35*isolated.mean())}
def main():
 if OUT.exists():raise FileExistsError("independent source GGN audit already exists")
 policy=json.loads(POL.read_text())
 if policy["status"]!="GGN_SOURCE_ONLY_FROZEN_BEFORE_DIAG_OR_MMLU" or sha(TARGETS["ggn24"].read_bytes())!=policy["mask_sha256"]:raise RuntimeError("new GGN policy source not frozen")
 train=load("train");diag=load("diagnostic")
 scores={k:{"train":risk(*train[:3],v),"diagnostic":risk(*diag[:3],v)} for k,v in TARGETS.items()}
 if abs(scores["ggn24"]["train"]["robust_GGN"]-policy["selected_regularized_GGN_risk"])>1e-9:raise RuntimeError("new optimizer source objective arithmetic not independent")
 if abs(scores["q4"]["train"]["robust_GGN"]-policy["prior_mask_source_objectives"]["allQ4"]["objective"])>1e-9:raise RuntimeError("full Q4 risk discrepancy")
 result={"schema":"BEGLIN-Qv9-independent-24plus24-conditional-GGN-196-projection-source-audit/1",
 "status":"PASS_48_ACTUAL_BF16_SOURCE_CASES_AND_144_JACOBIAN_BACKWARDS_REPLAYED",
 "train_private_receipt_SHA256":train[3],"diagnostic_private_receipt_SHA256":diag[3],
 "policy_SHA256":sha(POL.read_bytes()),"mask_SHA256":policy["mask_sha256"],
 "only_source24_train_selects_mask":True,"diagnostic24_never_used_to_fit":True,
 "source_proxy_vs_diagnostic_metrics":scores,"full_vocabulary_Hessian_NOT_computed":True,
 "quality_superiority_unproven":True,"auto_promotion":False}
 raw=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode()
 with OUT.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 print("QV9_INDEPENDENT_CONDITIONAL_GGN_SOURCE_DIAG_PASS",json.dumps({"sha":sha(raw),
 "train_risk":{k:round(v["train"]["robust_GGN"],5) for k,v in scores.items()},
 "diagnostic_risk":{k:round(v["diagnostic"]["robust_GGN"],5) for k,v in scores.items()}},sort_keys=True),flush=True)
if __name__=="__main__":main()
