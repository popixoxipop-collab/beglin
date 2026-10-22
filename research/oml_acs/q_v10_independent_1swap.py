#!/usr/bin/env python3
"""Qv10 independent all 28x168 feasible 1-swap contract, source-only.

Strict 28 n8, original CPU NEON memory/serial/stage budgets. Independently
evaluate KL/absolute-relative-trust/individual-GGN ridge from raw original
source, scanning every 4704 source-only exchange without held RACE scores.
No global QUBO/MILP optimality assertion.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,sys
import numpy as np
from scipy.special import logsumexp
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v8_signed_joint_optimizer as q8

LAB=ROOT/"lab_local/qv10_topk_trust_20261023"
RAW=LAB/"original_BF16_top2other_train24_private.json"
POL=ROOT/"docs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_SOURCE_POLICY_20261023.json"
MASK=ROOT/"configs/research/BEGLIN_QV10_TOP2OTHER_TRUST_28N8_SOURCE_POLICY_20261023.txt"
OUT=ROOT/"docs/research/BEGLIN_QV10_SOURCE_LOCAL_SWAP_INDEPENDENT_20261023.json"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
IDS=[(r,i) for r in ROLES for i in range(28)]
def sha(b):return hashlib.sha256(b).hexdigest()
def source():
 if OUT.exists():raise FileExistsError("1-swap result already sealed")
 obj=json.loads(RAW.read_text());pub=json.loads(POL.read_text())
 if pub["mask_sha256"]!=sha(MASK.read_bytes()) or pub["status"]!="SOURCE_ONLY_REAL_TOPK_OTHER_GROUPED_KL_28N8_MASK_FROZEN_BEFORE_DIAG_OR_RACE":
  raise RuntimeError("original source mask SHA unbound")
 a=np.asarray(obj["signed_Q4_relative_logit_groups"],dtype=np.float64)
 b=np.asarray(obj["signed_n8_relative_logit_groups"],dtype=np.float64)
 p=np.asarray(obj["source_original_BF16_output_group_probabilities"],dtype=np.float64)
 if a.shape!=(24,196,6) or b.shape!=(24,196,6) or p.shape!=(24,7):raise RuntimeError("not authentic source matrix")
 return a,b,p,pub
def risk(u,bits,p,r4,r8):
 z=np.concatenate((u,np.zeros((24,1))),axis=1)
 kl=np.maximum(0,logsumexp(np.log(np.clip(p,1e-300,1))+z,axis=1)-np.sum(p*z,axis=1))
 trust=.05*np.sum(p*np.maximum(0,abs(z)-2)**2,axis=1)
 diag=float((r4+(r8-r4)*bits).sum()/196)
 return float(max((kl+trust)[::2].mean(),(kl+trust)[1::2].mean())+.1*diag)
def run():
 a,b,p,document=source()
 costs,step,feasible=q8.make_engine()
 inverse={key:i for i,key in enumerate(IDS)}
 chosen=set()
 for row in MASK.read_text().splitlines():
  role,layer=row.split();chosen.add((role,int(layer)))
 if len(chosen)!=28 or not feasible(costs(chosen)):raise RuntimeError("original cost-constrained 28 selected invalid")
 z=a+(b-a)*np.asarray([int(key in chosen) for key in IDS])[None,:,None]
 u=z.sum(axis=1)
 def diag_var(t):
  s=np.concatenate((t,np.zeros((24,196,1))),axis=2)
  m=np.einsum("sg,sig->si",p,s)
  return .5*(np.einsum("sg,sig->si",p,s*s)-m*m)
 r4=np.maximum(0,diag_var(a)).mean(axis=0)
 r8=np.maximum(0,diag_var(b)).mean(axis=0)
 bits=np.asarray([int(key in chosen) for key in IDS],dtype=np.float64)
 initial=risk(u,bits,p,r4,r8)
 if abs(initial-document["selected_source_train_objective"]["objective"])>1e-8:
  raise RuntimeError("independent source KL trust/ridge arithmetic mismatch")
 cur=costs(chosen)
 feasibles=0;min_diff=math.inf;improve=0
 for drop in sorted(chosen,key=lambda r:inverse[r]):
  ix=inverse[drop]
  for insert in IDS:
   if insert in chosen:continue
   jx=inverse[insert]
   c=step(chosen,cur,drop,insert)
   if c is None:continue
   feasibles+=1
   trialbits=bits.copy();trialbits[ix]=0;trialbits[jx]=1
   trial=u-(b[:,ix,:]-a[:,ix,:])+(b[:,jx,:]-a[:,jx,:])
   v=risk(trial,trialbits,p,r4,r8)
   delta=v-initial
   if delta<-1e-8:improve+=1
   min_diff=min(min_diff,delta)
 if feasibles<1 or not math.isfinite(min_diff):
  raise RuntimeError("original CPU kernel cost constraints had no feasible 1-swap")
 if improve>0:raise RuntimeError("source-only selected policy NOT local-optimal; do not assert local certificate")
 out={"schema":"BEGLIN-Qv10-independent-all4704-tensor-mask-1swap-source-certificate/1",
  "status":"PASS_4704_ALL_POSSIBLE_SOURCE_1SWAPS_NO_FEASIBLE_IMPROVEMENT",
  "source_train24_private_sha256":sha(RAW.read_bytes()),
  "source_mask_sha256":document["mask_sha256"],
  "source_only_optimizer_policy_sha256":sha(POL.read_bytes()),
  "all_possible_swaps":28*168,
  "feasible_swaps_under_native_CPU_budget":feasibles,
  "source_objective_Qv10_reproduced":initial,
  "best_feasible_swap_objective_delta":min_diff,
  "number_improving_feasible_swaps":improve,
  "not_global_QUBO_or_full_model_loss_optimality_proof":True,
  "native_held_RACE48_not_read":True,
  "no_live_auto_promotion":True}
 b=(json.dumps(out,sort_keys=True,indent=2)+"\n").encode()
 with OUT.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 print("QV10_INDEPENDENT_TOPK_OTHER_SOURCE_4704_SWAPS_PASS",
     json.dumps({"candidate_swaps":4704,"feasible":feasibles,
       "minimum_feasible_delta":min_diff,"public_SHA":sha(b)},sort_keys=True),flush=True)
if __name__=="__main__":run()
