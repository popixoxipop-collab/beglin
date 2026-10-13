#!/usr/bin/env python3
"""Independent, fail-closed depth gate for true Top6/7 risk forecast evidence.

The caller's learned upstream predictor cannot authorize precision switching.
Even a significant predictive win only becomes a *research hypothesis*.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib
ARMS={"static","persistence","flat","shared_twolevel","separate_twolevel"}
BIGGER={"shared_twolevel","separate_twolevel"}
SCHEMA="beglin-hcp-p3-true-topk-independent-depth-gate-v1"

class QualityGateError(ValueError):pass

def finite(z):
    try:
        value=float(z)
    except (ValueError,TypeError) as e:
        raise QualityGateError("non-numeric metric") from e
    if not math.isfinite(value):raise QualityGateError("nonfinite quality metric")
    return value

def decide(report):
    if report.get("schema")!="beglin-hcp-p3-real-topk-boundary-forecast-v1" or report.get("status")!="TRUE_ROUTER_RISK_FORECAST_EVALUATED_RESEARCH_ONLY":
        raise QualityGateError("wrong source or run status")
    for k in ("true_K_vs_Kplus1_measured","full_64_expert_probabilities_before_renormalization"):
        if report.get(k) is not True:
            raise QualityGateError("not true pre-renorm rank6/7")
    for k in ("native_policy_rebind_allowed","production_touched","auto_live_promotion","certified_precision_trigger_training_labels"):
        if report.get(k) is not False:
            raise QualityGateError("risk proxy may not carry policy authority")
    if (report.get("train_prompts"),report.get("validation_prompts"),report.get("holdout_prompts"))!=(72,24,24):
        raise QualityGateError("independent prompt split invalid")
    counts=report.get("supervised_token_positions") or {}
    if not(0<counts.get("holdout",0)<302 and 0<counts.get("train",0)<945):
        raise QualityGateError("future-target supervised positions invalid")
    val=report.get("per_arm_validation_choice") or {}
    held=report.get("heldout") or {}
    if set(val)!=ARMS or set(held)!=ARMS:
        raise QualityGateError("missing a comparator")
    for name in ARMS:
        row=held[name]
        if row.get("prompt_count")!=24 or row.get("token_position_count")!=counts["holdout"]:
            raise QualityGateError("wrong holdout identities")
        for k in ("prompt_macro_rmse","token_rmse","mae"):
            if finite(row.get(k))<0:raise QualityGateError("negative error metric")
        auc=row.get("near_tie_auroc")
        if auc is not None and not (0<=finite(auc)<=1):
            raise QualityGateError("invalid AUROC")
    comparison=report.get("paired_vs_flat") or {}
    if not BIGGER.issubset(comparison):
        raise QualityGateError("missing hierarchy pairwise prompt uncertainty")
    flat=finite(held["flat"]["prompt_macro_rmse"])
    flat_auc=held["flat"]["near_tie_auroc"]
    decisions=[]
    for name in sorted(BIGGER):
        row=comparison[name]
        ci=row.get("paired_prompt_bootstrap_95ci")
        effect=finite(row.get("improvement_vs_flat_prompt_rmse"))
        if not(isinstance(ci,list) and len(ci)==2 and 0<=int(row.get("bootstrap_draws",0)) and
               int(row["bootstrap_draws"])>=200):
            raise QualityGateError("invalid bootstrap method")
        lo,hi=map(finite,ci)
        if lo>hi:raise QualityGateError("invalid bootstrap interval")
        prediction=held[name]
        auc=prediction["near_tie_auroc"]
        does_not_degrade_auc=(auc is not None and flat_auc is not None and float(auc)>=float(flat_auc))
        passing=bool(lo>0 and effect>=.01 and finite(prediction["prompt_macro_rmse"])<flat
                 and does_not_degrade_auc)
        decisions.append({"arm":name,"prompt_rmse_reduction":effect,
            "paired_bootstrap_95ci":[lo,hi],
            "near_tie_auc_delta":float(auc)-float(flat_auc) if auc is not None and flat_auc is not None else None,
            "predictive_hierarchy_gate_pass":passing})
    qualified=[d["arm"] for d in decisions if d["predictive_hierarchy_gate_pass"]]
    return {"schema":SCHEMA,
        "status":"PROXY_HIERARCHY_RESEARCH_CANDIDATE_ONLY" if qualified else "NO_VALIDATED_HIERARCHY_GAIN",
        "qualified_deeper_arms":qualified,
        "comparison_arms":decisions,
        "flat_heldout_prompt_rmse":flat,
        "persistence_heldout_prompt_rmse":held["persistence"]["prompt_macro_rmse"],
        "true_router_margin_observed":True,
        "future_margin_is_not_native_bit_safety_proof":True,
        "supports_real_qng64_policy_switch":False,
        "supports_full_model_latency_claim":False,
        "supports_live_promotion":False,
        "production_touched":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--result",required=True,type=pathlib.Path)
    p.add_argument("--output",required=True,type=pathlib.Path)
    args=p.parse_args()
    raw=args.result.read_bytes()
    o=decide(json.loads(raw))
    o["metric_sha256"]=hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(o,sort_keys=True,indent=2)+"\n")
    print("P3_TRUE_BOUNDARY_FAIL_CLOSED_GATE",json.dumps(o,sort_keys=True))
