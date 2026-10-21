#!/usr/bin/env python3
"""BEGLIN source-only Apple CPU kernel-aware tensor precision planning CLI.

Only plans a 28/196 tensor q4/n8 split for user-supplied memory and predicted
latency constraints. NEVER opens model outputs, labels, holdouts or changes
running model weights. No runtime promotion; output is an offline research plan.
"""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import q_kernel_aware_allocator as a

def plan(memory_limit:int,latency_limit:float)->dict:
    if not math.isfinite(latency_limit) or not 75.26<latency_limit<=200.:
        raise ValueError("latency ceiling must be > known native Q4 reference and <=200ms")
    if not 736985088<=memory_limit<=2620391424:
        raise ValueError("memory ceiling outside source BF16 projection weight bounds")
    scan=a.read_secure(a.QSCAN,a.EXPECTED_SCAN)
    profile=a.read_secure(a.PROFILE,a.EXPECTED_PROFILE)
    if profile["nroles"]!=7:raise ValueError("unsupported hardware/kernel profile")
    role=a.mk_roles(scan,profile)
    q4bytes,gamma,_=a.build_optimizer(role)
    before=(a.MEMORY_MAX,a.KERNEL_AWARE_MAX_MS)
    try:
        a.MEMORY_MAX=memory_limit
        a.KERNEL_AWARE_MAX_MS=latency_limit
        policies,stats,_=a.exhaustive_groups(role,q4bytes,gamma)
    finally:
        a.MEMORY_MAX,a.KERNEL_AWARE_MAX_MS=before
    r=policies["kernel_aware"]
    selected=a.policy(role,r)
    kernels={name:dict(bits4=role[name]["Q4_kernel"],bits8=role[name]["N8_kernel"]) for name in a.ROLES}
    return dict(schema="BEGLIN-kernel-aware-offline-planner/1",
      status="SOURCE_ONLY_PLAN_NOT_EVALUATED_OR_DEPLOYED",
      hardware="Apple Silicon Mac13,1 original 7 role BEGLIN native CPU profile",
      source_weight_sha256=a.EXPECTED_SCAN,
      source_kernel_cost_sha256=a.EXPECTED_PROFILE,
      source_label_data_used=False,
      n8_upgrades=len(selected),
      n4_tensors=196-len(selected),
      actual_bytes_constraint=memory_limit,
      predicted_ms_constraint=latency_limit,
      selected_role_layer_counts=dict(zip(a.ROLES,r["counts"])),
      selected_role_layer_pairs=selected,
      selected_total_projection_bytes=r["packed_projection_memory_bytes"],
      estimated_single_decode_ms=r["native_approx_total_decode_ms"],
      source_only_weight_distortion_gain=r["source_proxy_gain"],
      chosen_kernel_by_role=kernels,
      source_only_memory_feasible_choices=stats["memory_feasible"],
      kernel_feasible_choices=stats["kernel_time_feasible"],
      source_only_no_actual_quality_claim=True,
      production_or_main_modified=False)

def main():
    p=argparse.ArgumentParser(description="Source-only Qwen 196 tensor kernel-aware q4/n8 allocation")
    p.add_argument("--memory-limit-bytes",type=int,default=830570496)
    p.add_argument("--predicted-latency-ceiling-ms",type=float,default=100.0)
    args=p.parse_args()
    result=plan(args.memory_limit_bytes,args.predicted_latency_ceiling_ms)
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
