#!/usr/bin/env python3
"""Independent exhaustive 1-for-1 exchange certificate for Qv8 source QP.

Enumerates every 28 selected x 168 unselected Qwen projection pair. Checks
original measured packed bytes, calibrated serial CPU kernel proxy and
isolated async native stage DAG before evaluating the exact ORIGINAL BF16
signed first-order loss-change objective. No import from heuristic optimizer,
no held HellaSwag test responses, and no global-optimality claim.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_async_activation_milp as origin
LAB=ROOT/"lab_local/qv8_joint_signed_20261022"
RAW=LAB/"original_BF16_signed_196x96_source_private.json"
POL=ROOT/"docs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_OPTIMIZER_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
PUB=ROOT/"docs/research/BEGLIN_QV8_SOURCE_LOCAL_OPTIMALITY_INDEPENDENT_20261022.json"
ORIGINAL_SIGNED_SHA="e693fdf6798690c86ad2b53bb4a8e90cc208994956944f519841968ce680ed83"
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
    if PUB.exists():raise FileExistsError("never overwrite independent source swap certificate")
    if sha(RAW.read_bytes())!=ORIGINAL_SIGNED_SHA:raise RuntimeError("signed native BF16 source altered")
    source=json.loads(RAW.read_text());pol=json.loads(POL.read_text())
    roles=origin.ROLES
    if pol["status"]!="SOURCE_ONLY_28N8_STRICT_FEASIBLE_LOCAL_SWAP_DONE_NO_EVAL_OPENED":
        raise RuntimeError("source-only original bit mask not locked")
    cells=[(r,l) for r in roles for l in range(28)]
    selected={tuple(row.split()) for row in MASK.read_text().splitlines()}
    chosen={(r,int(l)) for r,l in selected}
    if len(chosen)!=28 or sha(MASK.read_bytes())!=pol["mask_sha256"]:
        raise RuntimeError("original exact 28 unique n8 mask absent")
    unselected=[k for k in cells if k not in chosen]
    old=[k for k in cells if k in chosen]
    A=np.asarray(source["original_tensor_grad_dot_q4"],dtype=np.float64)
    B=np.asarray(source["original_tensor_grad_dot_n8"],dtype=np.float64)
    D=B-A
    mapping={c:i for i,c in enumerate(cells)}
    q4=A.sum(axis=1)
    x=np.zeros(196,dtype=np.float64);x[[mapping[z] for z in chosen]]=1.
    r=q4+D@x;value=float(np.mean(r*r))
    if abs(value-pol["selected_source_signed_loss_squared"])>1e-6:
        raise RuntimeError("first-order source 96 signed objective differs")
    _,_,stages,serial,mem=origin.verified_sources()
    mb={key:int(mem[key[0]]["n8"]-mem[key[0]]["q4"]) for key in cells}
    tc={key:float(serial[key[0]]["delta"]) for key in cells}
    def single(stage,layer,rr,active):
        label="".join("8" if (role,layer) in active else "4" for role in rr)
        return (float(stages["data"][stage][label]["best_ms"])
            if len(rr)>1 else float(serial[rr[0]]["n8" if label=="8" else "q4"]))
    def allstage(active):
        return sum(single(st,layer,rr,active) for layer in range(28) for st,rr in origin.GROUPS)
    original_mem=736985088+sum(mb[z] for z in chosen)
    original_serial=origin.FULL_Q4_MS+origin.GAMMA_SERIAL*sum(tc[z] for z in chosen)
    original_stage=allstage(chosen)
    if original_mem>830570496 or original_serial>100 or original_stage>50.00001:
        raise RuntimeError("claimed first-order source policy violates original native costs")
    nfeasible=0;nblocked_mem=0;nblocked_serial=0;nblocked_stage=0
    best_delta=float("inf");best_pair=None
    for removed in old:
        for added in unselected:
            new_mem=original_mem-mb[removed]+mb[added]
            if new_mem>830570496:
                nblocked_mem+=1;continue
            new_serial=original_serial+origin.GAMMA_SERIAL*(-tc[removed]+tc[added])
            if new_serial>100.00001:
                nblocked_serial+=1;continue
            alt=(chosen-{removed})|{added}
            total=allstage(alt)
            if total>50.00001:
                nblocked_stage+=1;continue
            nfeasible+=1
            delta=D[:,mapping[added]]-D[:,mapping[removed]]
            proposed=float(np.mean((r+delta)**2))
            diff=proposed-value
            if diff<best_delta:
                best_delta=diff;best_pair=(removed,added)
    if nfeasible<1:
        raise RuntimeError("no feasible improvement candidate to independently test")
    if best_delta< -1e-7:
        raise RuntimeError(f"Qv8 claimed 1-swap local source optimum is false: delta {best_delta} {best_pair}")
    result={
      "schema":"BEGLIN-Qv8-source-only-exhaustive-feasible-pairwise-local-certificate/1",
      "status":"PASS_4704_POSSIBLE_ONE_SWAP_PAIRS_NO_FEASIBLE_IMPROVEMENT",
      "source_96_signed_private_SHA256":ORIGINAL_SIGNED_SHA,
      "frozen_selected_mask_SHA256":sha(MASK.read_bytes()),
      "frozen_source_optimizer_receipt_SHA256":sha(POL.read_bytes()),
      "possible_selected_to_unselected_swaps":len(old)*len(unselected),
      "independently_measured_feasible_swaps":nfeasible,
      "blocked_memory":nblocked_mem,"blocked_serial_cost":nblocked_serial,
      "blocked_async_stage_DAG":nblocked_stage,
      "independent_source_J":value,
      "best_feasible_exchange_J_change":best_delta,
      "best_feasible_exchange_roles":best_pair,
      "no_global_optimality_claim":True,
      "no_test_responses_or_new_evaluation_loaded":True,
      "no_immutable_release_modification":True}
    b=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode()
    with PUB.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    print("QV8_INDEPENDENT_EXHAUSTIVE_FEASIBLE_ONE_SWAP_LOCAL_PASS",json.dumps({
      "feasible_pairs":nfeasible,"best_source_J_delta":best_delta,"public_SHA":sha(b)},sort_keys=True),flush=True)
if __name__=="__main__":run()
