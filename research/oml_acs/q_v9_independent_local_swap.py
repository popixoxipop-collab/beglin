#!/usr/bin/env python3
"""Independent exhaustive 28×168 Qv9 source GGN 1-swap local certificate.

Separate arithmetic and loop from Qv9 mask generator. Evaluates all possible
Q4↔n8 swaps of frozen mask under original measured Apple kernel constraints.
Only local optimum, NEVER a globally optimal nonconvex mixed-precision proof.
No MMLU held data or model logits accessed.
"""
import hashlib,json,math,os,pathlib,sys
import numpy as np
R=pathlib.Path(__file__).resolve().parents[2];L=R/"lab_local/qv9_ggn_20261023";D=R/"docs/research";C=R/"configs/research"
sys.path.insert(0,str(R/"research/oml_acs"))
import q_v8_signed_joint_optimizer as q8
POL=D/"BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
TRAIN=L/"original_BF16_choice_GGN_train24_private.json"
MASK=C/"BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt"
OUT=D/"BEGLIN_QV9_GGN_EXHAUSTIVE_LOCAL_SWAP_CERT_20261023.json"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
IDS=[(r,l) for r in ROLES for l in range(28)]
INDEX={k:i for i,k in enumerate(IDS)}
def sha(b):return hashlib.sha256(b).hexdigest()
def frozen(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:raise RuntimeError("unsealed original BF16 GGN source")
 return p.read_bytes()
def independent_objective(q,n,p,mask):
 bits=np.zeros(196,dtype=np.float64)
 for i in mask:bits[INDEX[i]]=1
 z3=(q+(n-q)*bits[None,:,None]).sum(axis=1)
 zz=np.concatenate([z3,np.zeros((24,1))],axis=1)
 m=(p*zz).sum(axis=1)
 g=.5*((p*zz*zz).sum(axis=1)-m*m)
 if np.min(g)<-1e-8:raise RuntimeError("GGN variance below PSD bound")
 diag=0.
 for i in range(196):
  x=q[:,i,:]+bits[i]*(n[:,i,:]-q[:,i,:])
  x=np.concatenate((x,np.zeros((24,1))),axis=1)
  mu=(p*x).sum(axis=1)
  diag+=.5*float(((p*x*x).sum(axis=1)-mu*mu).mean())
 return .65*float(max(g[::2].mean(),g[1::2].mean()))+.35*diag
def main():
 if OUT.exists():raise FileExistsError("no overwrite local search certificate")
 data=json.loads(frozen(TRAIN));pol=json.loads(POL.read_text())
 q=np.asarray(data["q4_signed_relative_logit_delta"],dtype=float)
 n=np.asarray(data["n8_signed_relative_logit_delta"],dtype=float)
 p=np.asarray(data["original_conditional_four_letter_probabilities"],dtype=float)
 if q.shape!=(24,196,3) or p.shape!=(24,4) or pol["status"]!="GGN_SOURCE_ONLY_FROZEN_BEFORE_DIAG_OR_MMLU":
  raise RuntimeError("source-only frozen GGN source invalid")
 text=MASK.read_bytes()
 if sha(text)!=pol["mask_sha256"]:raise RuntimeError("mask source drift")
 selected={(x[0],int(x[1])) for x in (row.split() for row in text.decode().splitlines())}
 if len(selected)!=28:raise RuntimeError("source mask not 28 unique")
 physical,exchange,feasible=q8.make_engine()
 current=physical(selected)
 if not feasible(current):raise RuntimeError("frozen policy not original kernel feasible")
 old=independent_objective(q,n,p,selected)
 if abs(old-pol["selected_regularized_GGN_risk"])>1e-9:raise RuntimeError("original source risk not independently recomputed")
 allcount=0;valid=0;best=(float("inf"),None)
 for removed in sorted(selected,key=lambda x:INDEX[x]):
  for new in IDS:
   if new in selected:continue
   allcount+=1
   if exchange(selected,current,removed,new) is None:continue
   valid+=1
   value=independent_objective(q,n,p,(selected-{removed})|{new})
   improve=value-old
   if improve<best[0]:
    best=(float(improve),(removed,new))
 if allcount!=4704 or valid<1 or best[0]<-1e-8:
  raise RuntimeError("Qv9 source-only 28n8 constrained one-swap optimizer not locally optimal")
 o={"schema":"BEGLIN-Qv9-independent-source-GGN-28N8-full-1-swap-certificate/1",
  "status":"PASS_EXHAUSTIVE_ALL_4704_SOURCE_1SWAP_CANDIDATES_NO_IMPROVEMENT",
  "policy_original_SHA256":sha(POL.read_bytes()),
  "actual_native_Q4N8_mask_SHA256":sha(text),
  "original_BF16_24x196x3_signed_choice_jacobian_SHA256":sha(frozen(TRAIN)),
  "source_train24_GGN_objective":old,
  "source_one_swap_total_eligible_pairs":allcount,
  "source_feasible_Apple_cpu_kernel_pairs":valid,
  "minimum_feasible_next_1swap_GGN_change":best[0],
  "best_feasible_swap":[list(v) for v in best[1]],
  "GGN_4choice_not_exact_fullvocab_Hessian":True,
  "source_local_not_global_binary_optimum":True,
  "source_held_MMLU_labels_never_accessed":True,
  "production_precision_promotion_forbidden":True}
 raw=(json.dumps(o,sort_keys=True,indent=2)+"\n").encode()
 with OUT.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 print("QV9_GGN_EXHAUSTIVE_INDEPENDENT_SOURCE_1SWAP_CERT_PASS",
   json.dumps({"total":allcount,"feasible":valid,"min_next_delta":best[0],
               "risk":old,"sha":sha(raw)},sort_keys=True),flush=True)
if __name__=="__main__":main()
