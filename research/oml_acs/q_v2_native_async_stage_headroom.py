#!/usr/bin/env python3
"""BEGLIN Qv2 native pthread async micro-stage Amdahl headroom gate.

Computes optimistic (NOT mathematically guaranteed) stage-level savings
using previously source-frozen bitmask and measured Apple native QKV/GU
serial and fastest isolated parallel stage costs. Prior native CPU serial
reference 93.62ms comes from a separate SHA-attested ABBA experiment.

Output is a source-microbenchmark engineering upper PROJECTION only,
not an observed whole-model speedup nor safety permission to deploy.
"""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
ASYNC=ROOT/"lab_local/q_v2_async_stage_20261022/async_sibling_stage_costs_private.json"
SCHEDULE=ROOT/"configs/research/BEGLIN_Q_V2_FROZEN_PARALLEL_STAGE_SCHEDULE_20261022.json"
OLD=ROOT/"lab_local/q_v2_actual_native_async_retry1_20261022/actual_async_guard_receipt.json"
OUT=ROOT/"lab_local/q_v2_selective_native_async_20261022/optimistic_async_stage_headroom_receipt.json"
SHA={
 "micro":"380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808",
 "schedule":"c36e76aea0b051e85bf94f95fa6973ad8918342213067c45e805674bdc47bfc5",
 "native":"6e9ed2794964aa4c3efba307744db61388730ada2aa18c81f4f65838c5da5341"
}
REQUIRED_GAIN=1.05
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def read(name,p):
    raw=p.read_bytes()
    if sha(raw)!=SHA[name]:raise RuntimeError("pinned QKV/GU native stage evidence changed "+name)
    return json.loads(raw)
def compute():
    prof=read("micro",ASYNC)
    plan=read("schedule",SCHEDULE)
    actual=read("native",OLD)
    bits=plan["bitmask_by_layer"]
    if len(bits)!=28 or len(prof["stages"]["qkv"])!=8 or len(prof["stages"]["ffn_gu"])!=4:
        raise RuntimeError("incomplete q4/n8 stage experiment")
    results={}
    total_saving=0.
    for stage,indices,table in (("qkv",(0,1,2),prof["stages"]["qkv"]),
                                ("gate_up",(4,5),prof["stages"]["ffn_gu"])):
        samples=[]
        for layer,bitset in enumerate(bits):
            key="".join("8" if bitset>>i&1 else "4" for i in indices)
            row=table[key]
            serial=row["serial_q4_6t"]["median_stage_ms"]
            choice=min(("parallel_q4_1t","parallel_q4_6t"),
                       key=lambda n:row[n]["median_stage_ms"])
            par=row[choice]["median_stage_ms"]
            samples.append(dict(layer=layer,bits=key,serial_stage_ms=serial,
                   best_isolated_parallel_ms=par,
                   best_parallel_q4_thread_mode=choice,
                   optimistic_stage_saving_ms=max(0.,serial-par)))
        saving=sum(v["optimistic_stage_saving_ms"] for v in samples)
        results[stage]=dict(source_only_role_stage_pattern_counts=dict(Counter(v["bits"] for v in samples)),
             original_isolated_stage_cost_ms=sum(v["serial_stage_ms"] for v in samples),
             optimistic_stage_level_saving_ms=saving,
             per_layer=samples)
        total_saving+=saving
    vals=actual["means_ms"]
    serial=vals["off"]
    if not math.isclose(serial,93.62,abs_tol=.01):raise RuntimeError("reference native serial mean drift")
    required=serial*(1-1/REQUIRED_GAIN)
    optimistic_speedup=serial/(serial-total_saving)
    return dict(schema="BEGLIN-Qv2-native-QKV-GU-stage-headroom-projection/1",
       status="PASS_OPTIMISTIC_STAGE_HEURISTIC_NOT_NATIVE_FULL_SPEED",
       original_sources_sha256=SHA,required_engineering_speed_ratio=REQUIRED_GAIN,
       original_full_model_native_serial_mean_ms=serial,
       required_total_saving_for_5pct_ms=required,
       total_optimistic_independent_stage_saving_ms=total_saving,
       remaining_scheduling_join_overhead_allowance_ms_at_5pct=total_saving-required,
       idealized_full_model_speed_ratio_with_zero_async_overhead=optimistic_speedup,
       actual_prior_native_onoff_ratio=actual["measured_overall_ratio_OFF_div_ON"],
       stages=results,
       not_a_hard_speed_upper_bound=True,
       no_production_async_auto_admission=True,
       final_model_quality_verdict="NO_VERDICT",
       immutable_runtime_unchanged=True)
def main():
    if OUT.exists():raise FileExistsError("not overwrite research QKV/GU source projection")
    result=compute()
    os.umask(0o077)
    with OUT.open("xb") as f:
        f.write((json.dumps(result,indent=2,sort_keys=True)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    print("QV2_ASYNC_HEADROOM_SOURCE_MODEL_PASS",json.dumps({
       "optimistic_saving_ms":result["total_optimistic_independent_stage_saving_ms"],
       "required_5pct_ms":result["required_total_saving_for_5pct_ms"],
       "headroom_for_contention_ms":result["remaining_scheduling_join_overhead_allowance_ms_at_5pct"],
       "ideal_speed_ratio":result["idealized_full_model_speed_ratio_with_zero_async_overhead"],
       "private_receipt_sha256":sha(OUT.read_bytes())
    },sort_keys=True),flush=True)
if __name__=="__main__":main()
