#!/usr/bin/env python3
"""Fail-closed high-retention research gate. Never authorizes live model changes.

A 99% requested fraction is a full execution at 88 or 22 channels. Explicitly
reject it as a speed candidate, even if the output error is numerically zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "beglin-high-retention-fail-closed-v1"
EXPECTED = {"train":72,"validation":24,"holdout":24}
FRACTIONS = ("0.5","0.9","0.95","0.99")
WIDTHS = ("16","64")
MAX_ACCEPTABLE_LOCAL_L2 = 0.05

def keep_count(intermediate:int, width:int, retention:float)->int:
    if intermediate <= 0 or width <= 0 or intermediate % width or not 0 < retention <= 1:
        raise ValueError("invalid shape or retention")
    return max(1,math.ceil((intermediate // width)*retention))

def inspect_result(payload:dict)->dict:
    if payload.get("schema") != "beglin-prompt-holdout-micro-router-v1":
        raise ValueError("unexpected experiment schema")
    if payload.get("status") != "EVALUATED_NOT_PROMOTABLE":
        raise ValueError("input is not research-only")
    if payload.get("prompt_count") != 120 or payload.get("prompt_split_counts") != EXPECTED:
        raise ValueError("wrong or missing independent prompt splits")
    if payload.get("raw_hidden_uploaded") is not False or payload.get("production_touched") is not False:
        raise ValueError("source privacy or production contract invalid")
    if payload.get("native_qng64_parity_tested") is not False:
        raise ValueError("native parity must have a separate exact-source approval gate")
    intermediate = int(payload.get("intermediate",-1))
    if intermediate != 1408:
        raise ValueError("unexpected DeepSeek expert intermediate dimension")
    configs=[]
    for width in WIDTHS:
        ng=intermediate//int(width)
        by_ret=payload.get("results",{}).get(width)
        if not isinstance(by_ret,dict):
            raise ValueError(f"missing width {width}")
        for fraction in FRACTIONS:
            row=by_ret.get(fraction)
            if not isinstance(row,dict) or row.get("tupper_roundtrip") is not True:
                raise ValueError("missing Tupper grid roundtrip")
            kept=keep_count(intermediate,int(width),float(fraction))
            observed=float(row["idealized_channel_compute_fraction_excluding_gate_and_corrector"])
            theoretical=kept/ng
            if abs(observed-theoretical)>1e-10:
                raise ValueError(f"fraction/cost mismatch width={width} keep={fraction}")
            method=row.get("method_selected_by_validation")
            if method not in ("static","static_mean_residual","dynamic","dynamic_mean_residual","dynamic_ridge_residual","dynamic_rank8_residual"):
                raise ValueError("unknown validation-selected method")
            z=row["locked_holdout"]
            metric=float(z["relative_l2"])
            ci=z.get("prompt_bootstrap_95ci")
            if not (math.isfinite(metric) and metric>=0 and isinstance(ci,list) and len(ci)==2 and
                    all(math.isfinite(float(x)) for x in ci) and 0 <=float(ci[0])<=float(ci[1])):
                raise ValueError("invalid heldout output uncertainty")
            # Dataset quality first; token/logit/e2e checks are separate and absent.
            nontrivial=kept<ng
            point_pass=nontrivial and metric<=MAX_ACCEPTABLE_LOCAL_L2
            robust_pass=point_pass and float(ci[1])<=MAX_ACCEPTABLE_LOCAL_L2
            methods=row["holdout_methods"]
            if not all(k in methods for k in ("static","dynamic","dynamic_ridge_residual","dynamic_rank8_residual")):
                raise ValueError("required comparators missing")
            configs.append({
                "width":int(width),
                "requested_retention":float(fraction),
                "total_micro_channels":ng,
                "actually_retained_channels":kept,
                "true_retention_fraction":theoretical,
                "idealized_saved_fraction":1-theoretical,
                "validation_locked_method":method,
                "locked_holdout_local_relative_l2":metric,
                "prompt_bootstrap_95ci":list(map(float,ci)),
                "comparators_local_relative_l2":{m:float(methods[m]["relative_l2"]) for m in ("static","dynamic","dynamic_ridge_residual","dynamic_rank8_residual")},
                "no_op_full_execution":not nontrivial,
                "local_quality_point_gate":point_pass,
                "local_quality_robust_gate":robust_pass,
                "native_end_to_end_inference_proved":False,
                "auto_promotion_allowed":False})
    # Reject degenerate 99% rows rather than celebrating perfect reconstruction.
    for r in configs:
        if r["requested_retention"]==0.99 and not r["no_op_full_execution"]:
            raise ValueError("99% is required to be a full-execution control at this grid size")
    return {"schema":SCHEMA,"status":"RESEARCH_ONLY_NO_PROMOTION",
        "input_dataset_manifest_sha256":payload.get("dataset_manifest_sha256"),
        "checkpoint_index_sha256":payload.get("checkpoint_index_sha256"),
        "quality_screen_local_ffn_relative_l2":MAX_ACCEPTABLE_LOCAL_L2,
        "all_native_model_quality_gates_blocked":True,
        "reason":"Real hidden inputs, BF16 expert FFN offline; matched qNg64 full-model logits/token accuracy, wall-time & RSS not measured",
        "eligible_for_further_shadow_validation":[{"width":r["width"],"retention":r["requested_retention"]} for r in configs if r["local_quality_robust_gate"]],
        "results":configs}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    original=args.result.read_bytes()
    report=inspect_result(json.loads(original))
    report["result_sha256"]=hashlib.sha256(original).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("HIGH_RETENTION_GATE_REPORT",json.dumps({"status":report["status"],
       "candidate_count":len(report["eligible_for_further_shadow_validation"]),
       "rows":[{"width":r["width"],"fraction":r["requested_retention"],
         "kept":r["actually_retained_channels"],"of":r["total_micro_channels"],
         "selected":r["validation_locked_method"],
         "local_l2":round(r["locked_holdout_local_relative_l2"],6),
         "upper_95ci":round(r["prompt_bootstrap_95ci"][1],6),
         "pass":r["local_quality_robust_gate"],
         "no_op":r["no_op_full_execution"]} for r in report["results"]]},sort_keys=True))
if __name__=="__main__":
    main()
