#!/usr/bin/env python3
"""BEGLIN qNg64 n7 immutable-source host packed memo working-set admission.

Research-only offline scalar calculator: uses an already-approved original
quantization policy manifest and observed *aggregate* source-role count.
No tensor contents, expert IDs, source prompts, secrets, model checkpoint
reads, GPU actions or deployment. Emits a candidate host cache budget, NOT
an automatically authorized serving setting or inference speed claim.

DeepSeek-V2-Lite frozen research contract: 27 layers; first dense FFN;
26 MoE layers x 64 experts x 3 FFN projections, plus 7 fixed roles/layer.
A native cache stores per-tensor qNg64 group64/n7 bitplanes plus F32 scales.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

GIB=1024**3
ATT=("self_attn.q_proj","self_attn.kv_a_proj_with_mqa",
     "self_attn.kv_b_proj","self_attn.o_proj")
FFN=("gate_proj","up_proj","down_proj")
LAYERS=27
ROUTED_LAYERS=26
ROUTED_EXPERTS=64
EFFECTIVE_NUM_ROWS=LAYERS*7+ROUTED_LAYERS*ROUTED_EXPERTS*3
NONKVB_FIXED=LAYERS*6

def fail(desc):
    raise ValueError("BEGLIN_N7_SOURCE_BUDGET_FAIL_CLOSED:"+desc)

def estimate(policy,actual_nonk_vb_source_role_count,available_RAM_bytes,
             headroom_percent=20,max_cache_GiB=8):
    if type(actual_nonk_vb_source_role_count) is not int or \
       actual_nonk_vb_source_role_count<NONKVB_FIXED or \
       actual_nonk_vb_source_role_count>NONKVB_FIXED+ROUTED_LAYERS*ROUTED_EXPERTS*3:
        fail("actual source role count cannot match 27L source MoE")
    if (actual_nonk_vb_source_role_count-NONKVB_FIXED)%3:
        fail("real original source routed-expert roles are not complete gate/up/down trios")
    if type(available_RAM_bytes) is not int or available_RAM_bytes<0:
        fail("available RAM must be nonnegative scalar from trusted caller")
    if type(headroom_percent) is not int or not 0<=headroom_percent<=50:
        fail("headroom must be conservative integer percent 0-50")
    if type(max_cache_GiB) is not int or not 1<=max_cache_GiB<=8:
        fail("source n7 in-RAM compact memo supports at most 8 GiB")
    if not isinstance(policy,dict) or \
       policy.get("qng64_matrix_count")!=EFFECTIVE_NUM_ROWS or \
       not isinstance(policy.get("rows"),list) or len(policy["rows"])!=EFFECTIVE_NUM_ROWS:
        fail("original source policy matrix-count mismatch")
    rows={}
    for r in policy["rows"]:
        if not isinstance(r,dict) or not isinstance(r.get("name"),str) or \
           r["name"] in rows:
            fail("missing/duplicate original quantization role")
        bits=r.get("bits")
        packed=r.get("packed_bytes")
        if not isinstance(bits,dict) or bits.get("uniform7")!=7 or \
           not isinstance(packed,dict) or \
           type(packed.get("uniform7")) is not int or packed["uniform7"]<=0:
            fail("invalid original 7-bit frozen packed-source authority")
        rows[r["name"]]=packed["uniform7"]
    def b(name):
        if name not in rows:fail("expected frozen source tensor role missing")
        return rows[name]

    fixed=0
    for lid in range(LAYERS):
        keys=[f"model.layers.{lid}.{role}.weight" for role in ATT]
        keys.extend([f"model.layers.0.mlp.{role}.weight" for role in FFN] if lid==0
                    else [f"model.layers.{lid}.mlp.shared_experts.{role}.weight"
                          for role in FFN])
        fixed+=sum(b(key) for key in keys)
    unit=tuple(b(f"model.layers.1.mlp.experts.0.{role}.weight") for role in FFN)
    # Verify the original 26x64 all-expert roles are packed-homogeneous:
    # Otherwise inferring unique bytes from only aggregate role count is wrong.
    checked=0
    for lid in range(1,LAYERS):
        for expert in range(ROUTED_EXPERTS):
            vals=tuple(b(f"model.layers.{lid}.mlp.experts.{expert}.{role}.weight")
                      for role in FFN)
            if vals!=unit:fail("nonhomogeneous expert compressed footprint; aggregate inference invalid")
            checked+=1
    unique=(actual_nonk_vb_source_role_count-NONKVB_FIXED)//3
    actual_distinct_B=fixed+unique*sum(unit)
    required=(actual_distinct_B*(100+headroom_percent)+99)//100
    cap_required_GiB=(required+GIB-1)//GIB
    available_reserve=2*GIB
    available_target_B=cap_required_GiB*GIB
    if cap_required_GiB>max_cache_GiB:
        status="NOT_READY_EXCEEDS_SOURCE_MEMO_MAXIMUM"
    elif available_RAM_bytes<available_target_B+available_reserve:
        status="NOT_READY_AVAILABLE_RAM_INSUFFICIENT_WITH_2GIB_RESERVE"
    else:
        status="ADMISSIBLE_RESEARCH_HOST_COMPACT_CAP_CANDIDATE"
    return{
      "schema":"beglin.source-n7-group64-memory-workingset-budget/1",
      "status":status,
      "model_source_layers":LAYERS,
      "source_expert_policies_verified":checked,
      "all_source_quantized_matrix_policies_verified":len(rows),
      "inferred_unique_source_experts":unique,
      "observed_aggregate_nonkv_b_qNg64_roles":actual_nonk_vb_source_role_count,
      "source_fixed_27layer_7roles_packed_bytes":fixed,
      "source_one_expert_three_n7roles_packed_bytes":sum(unit),
      "inferred_original_model_live_compact_n7_bytes":actual_distinct_B,
      "host_compact_headroom_percent":headroom_percent,
      "candidate_host_cache_GiB":cap_required_GiB,
      "candidate_host_cache_capacity_bytes":available_target_B,
      "caller_reported_available_RAM_B":available_RAM_bytes,
      "minimum_remaining_RAM_reserve_B":available_reserve,
      "source_sha_and_model_weights_verified_by_this_calculator":False,
      "original_moE_ids_or_weight_arrays_examined":False,
      "prospective_new_prompt_source_role_count_unknown":True,
      "model_forward_or_live_TPOT_certified":False,
      "automatic_GPU_promotion_allowed":False
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--policy-file",required=True)
    parser.add_argument("--policy-sha256",required=True)
    parser.add_argument("--observed-nonkvb-roles",type=int,required=True)
    parser.add_argument("--available-ram-bytes",type=int,required=True)
    parser.add_argument("--headroom-percent",type=int,default=20)
    args=parser.parse_args()
    path=Path(args.policy_file)
    if len(args.policy_sha256)!=64 or any(x not in "0123456789abcdef"
             for x in args.policy_sha256.lower()) or \
        sha256(path.read_bytes()).hexdigest()!=args.policy_sha256.lower():
        fail("input immutable source quant policy SHA256 not matched")
    policy=json.loads(path.read_text())
    result=estimate(policy,args.observed_nonkvb_roles,args.available_ram_bytes,
                    args.headroom_percent)
    result["source_policy_manifest_SHA256"]=args.policy_sha256.lower()
    print(json.dumps(result,sort_keys=True))
    if result["status"].startswith("NOT_READY"):return 2
    return 0
if __name__=="__main__":
    raise SystemExit(main())
