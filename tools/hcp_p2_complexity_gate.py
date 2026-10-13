#!/usr/bin/env python3
"""Stop complexity creep when H-JEPA-inspired slow/fast latents fail heldout proof.

Reject any hierarchy promotion based merely on one favorable RMSE digit.
The predictive target is a selected-top6 score-entropy proxy, not an actual
precision-risk certification. This module cannot authorize native policy changes.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib

SCHEMA="beglin-hcp-p2-depth-selection-v1"
ARMS=("static","flat","shared_twolevel","separate_twolevel")

class P2GateError(ValueError):pass

def require_finite_pos(x,label):
    try:z=float(x)
    except (TypeError,ValueError) as e:raise P2GateError(label+" not numeric") from e
    if not math.isfinite(z) or z<=0:raise P2GateError(label+" not positive finite")
    return z

def decide(report,*,min_effect_fraction=.01):
    if report.get("schema")!="beglin-hcp-p2-multiscale-predictor-v1" or report.get("status")!="REAL_SURROGATE_EVALUATED_NATIVE_POLICY_BLOCKED":
        raise P2GateError("expected signed-off shadow proxy evidence schema")
    if report.get("native_policy_rebinding_allowed") is not False or report.get("production_touched") is not False or report.get("automatic_live_promotion") is not False:
        raise P2GateError("not isolated read-only learning result")
    if report.get("source_proxy_is_certified_precision_risk") is not False or report.get("exact_context_certificates_verified") is not False:
        raise P2GateError("proxy risk source may not claim certification")
    if (report.get("train_prompt_count"),report.get("validation_prompt_count"),report.get("holdout_prompt_count"))!=(72,24,24):
        raise P2GateError("holdout prompt split invalid")
    if not 0<=min_effect_fraction<=1:raise P2GateError("bad depth effect threshold")
    final=report.get("final_heldout") or {}
    chosen=report.get("validation_hyperparameter_selection") or {}
    if set(final)!=set(ARMS) or set(chosen)!=set(ARMS):
        raise P2GateError("missing arm or validation selection")
    rmse={}
    for name in ARMS:
        row=final[name]
        if row.get("prompt_count")!=24 or row.get("sample_count",0)<24:
            raise P2GateError("missing independent final holdout coverage")
        rmse[name]=require_finite_pos(row.get("prompt_macro_rmse"),name+" heldout RMSE")
    comparison=report.get("paired_improvement_vs_flat") or {}
    if set(comparison)!={"shared_twolevel","separate_twolevel"}:
        raise P2GateError("missing paired comparator interval")
    candidates=[]
    for arm in ("shared_twolevel","separate_twolevel"):
        row=comparison[arm]
        interval=row.get("paired_prompt_bootstrap_95ci")
        effect=float(row.get("improvement_vs_flat_prompt_rmse",float("nan")))
        if (not isinstance(interval,list) or len(interval)!=2 or row.get("bootstrap_draws",0)<200 or
            not math.isfinite(effect) or any(not math.isfinite(float(z)) for z in interval) or
            float(interval[0])>float(interval[1])):
            raise P2GateError("invalid paired prompt-block uncertainty")
        # Both statistical and predeclared practical minimum improvement.
        qualifies=float(interval[0])>0 and effect>=min_effect_fraction and rmse[arm]<rmse["flat"]
        candidates.append({"arm":arm,"heldout_prompt_macro_rmse":rmse[arm],
            "improvement_vs_flat":effect,"paired_bootstrap_95ci":list(map(float,interval)),
            "preregistered_min_improvement":min_effect_fraction,
            "predictive_depth_gate_pass":qualifies})
    allowed=[row for row in candidates if row["predictive_depth_gate_pass"]]
    winner=min(allowed,key=lambda row:(row["heldout_prompt_macro_rmse"],row["arm"]))["arm"] if allowed else "flat"
    return {"schema":SCHEMA,
            "status":"PROXY_DEPTH_CANDIDATE_ONLY" if allowed else "REJECT_COMPLEXITY_KEEP_FLAT_REFERENCE",
            "selected_proxy_prediction_baseline":winner,
            "source_heldout_sample_count":final["flat"]["sample_count"],
            "source_heldout_prompt_count":final["flat"]["prompt_count"],
            "holdout_rmse":rmse,
            "source_feature_sha256":report.get("source_feature_sha256"),
            "alternatives":candidates,
            "H_JEPA_joint_training_performed":False,
            "outcome_not_llm_token_accuracy":True,
            "native_token_logits_not_measured":True,
            "actual_policy_speedup_not_measured":True,
            "evidence_for_context_precision_switch":False,
            "production_touched":False,
            "native_policy_change_allowed":False,
            "automatic_live_promotion":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--result",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    a=p.parse_args()
    raw=a.result.read_bytes()
    report=decide(json.loads(raw))
    report["source_result_sha256"]=hashlib.sha256(raw).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("P2_COMPLEXITY_GATE",json.dumps({"status":report["status"],
      "selected":report["selected_proxy_prediction_baseline"],
      "holdout_rmse":report["holdout_rmse"],
      "alternatives":report["alternatives"],
      "native_policy_change_allowed":False},sort_keys=True))

if __name__=="__main__":main()
