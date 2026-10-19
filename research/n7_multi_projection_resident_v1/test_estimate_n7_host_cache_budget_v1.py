#!/usr/bin/env python3
"""No-model BEGLIN n7 27L MoE source-host-cache admission planner tests.

Synthetic policy records only: 27x7 fixed / 26x64x3 routed roles. Forces
source role count, shape-homogeneity, admission RAM headroom, 8-GiB hard
cap and immutability/duplicate/malformed policy checks to fail closed.
"""
from __future__ import annotations

from copy import deepcopy
import json

from estimate_n7_host_cache_budget_v1 import estimate, GIB, ATT, FFN

def make():
    rows=[]
    for layer in range(27):
        roles=[f"model.layers.{layer}.{role}.weight" for role in ATT]
        roles.extend([f"model.layers.0.mlp.{role}.weight" for role in FFN]
                     if layer==0 else
                     [f"model.layers.{layer}.mlp.shared_experts.{role}.weight" for role in FFN])
        for role in roles:
            rows.append({"name":role,"bits":{"uniform7":7},
                         "packed_bytes":{"uniform7":4000000}})
    # Source-shaped aggregate fixed packed bytes matches frozen scalar sum
    # without including any original weights, tensor values or model data.
    rows[188]["packed_bytes"]["uniform7"]+=833126400-189*4000000
    for layer in range(1,27):
        for expert in range(64):
            for role in FFN:
                rows.append({"name":f"model.layers.{layer}.mlp.experts.{expert}.{role}.weight",
                             "bits":{"uniform7":7},
                             "packed_bytes":{"uniform7":2703360}})
    assert len(rows)==5181
    return{"qng64_matrix_count":5181,"rows":rows}

def rejected(callback,title):
    try:callback()
    except ValueError as e:
        assert "BEGLIN_N7_SOURCE_BUDGET_FAIL_CLOSED" in str(e),(title,str(e))
        return title
    raise RuntimeError("BEGLIN_BUDGET_SYNTHETIC_UNREJECTED:"+title)

def main():
    policy=make()
    cases=[]
    result=estimate(policy,2367,64*GIB)
    assert result["status"]=="ADMISSIBLE_RESEARCH_HOST_COMPACT_CAP_CANDIDATE"
    assert result["source_expert_policies_verified"]==1664
    assert result["inferred_unique_source_experts"]==735
    assert result["inferred_original_model_live_compact_n7_bytes"]==6794035200
    assert result["candidate_host_cache_GiB"]==8
    cases.append("ADMIT_8GIB_SOURCE_QUANT_POLICY_735_EXPERTS")
    over=estimate(policy,2367,1*GIB)
    assert over["status"]=="NOT_READY_AVAILABLE_RAM_INSUFFICIENT_WITH_2GIB_RESERVE"
    cases.append("BLOCK_WHEN_AVAILABLE_RAM_BELOW_ADMISSION_AND_2GIB_RESERVE")
    cap=estimate(policy,2367,64*GIB,max_cache_GiB=6)
    assert cap["status"]=="NOT_READY_EXCEEDS_SOURCE_MEMO_MAXIMUM"
    cases.append("BLOCK_6GIB_CAP_352MB_BELOW_REAL_WORKING_SET")
    reduced=estimate(policy,2367,64*GIB,headroom_percent=0)
    assert reduced["candidate_host_cache_GiB"]==7
    cases.append("SEVEN_GIB_MATH_ONLY_NOT_REAL_SOURCE_CERTIFICATION")
    cases.append(rejected(lambda:estimate(policy,2368,64*GIB),
                         "REJECT_INCOMPLETE_THREE_ROLE_EXPERT_UNION"))
    cases.append(rejected(lambda:estimate(policy,5182,64*GIB),
                         "REJECT_TOO_MANY_SOURCE_GPU_ROLE_ROWS"))
    broken=deepcopy(policy)
    broken["rows"][-1]["packed_bytes"]["uniform7"]+=1
    cases.append(rejected(lambda:estimate(broken,2367,64*GIB),
                         "REJECT_NONHOMOGENEOUS_EXPERT_PACKED_SIZE"))
    broken2=deepcopy(policy)
    broken2["rows"][-1]["name"]=broken2["rows"][-2]["name"]
    cases.append(rejected(lambda:estimate(broken2,2367,64*GIB),
                         "REJECT_DUPLICATE_OR_MISSING_EXPERT_ROLE"))
    broken3=deepcopy(policy)
    broken3["rows"][0]["bits"]["uniform7"]=8
    cases.append(rejected(lambda:estimate(broken3,2367,64*GIB),
                         "REJECT_UNAUTHORIZED_PRECISION_CHANGE"))
    cases.append(rejected(lambda:estimate(policy,2367,64*GIB,max_cache_GiB=9),
                         "REJECT_UNBOUNDED_HOST_MEMO"))
    assert len(cases)==10
    print(json.dumps({"status":"PASS_SYNTHETIC_N7_ORIGINAL_SOURCE_MEMO_ADMISSION_POLICY",
      "cases":len(cases),"checks":cases,
      "verified_aggregate_working_set_B":result["inferred_original_model_live_compact_n7_bytes"],
      "candidate_budget_GiB":result["candidate_host_cache_GiB"],
      "original_checkpoint_loaded":False,"real_production_memory_available_verified":False,
      "actual_serving_promotion_allowed":False},sort_keys=True))
if __name__=="__main__":main()
