#!/usr/bin/env python3
"""BEGLIN Qv2 isolated research promotion gate (read-only, fail closed).

No release actions or automatic promotion. Verifies the fixed original BF16
source, tiny held development PPL and measured CPU real stage vs whole model
separation. Exits status 2 under --require-ready when a candidate is NOT_READY.
Uses published aggregate evidence only; never opens private row text or model.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
FILE=ROOT/"docs/research/BEGLIN_QV2_ACTIVATION_ASYNC_6POLICY_REAL_NATIVE_20261022.json"
EXPECT_PRIVATE="1dfb72b95b11262bbde56e2e47fa52965c96a4fe6c21f089b999ba1c816c9185"
MIN_INDEPENDENT_TOKENS=1000
MAX_RELATIVE_NLL_LOSS=0.02

def evaluate(source):
    if source["schema"]!="BEGLIN-Qv2-original-BF16-activation-plus-real-async-stage-source-and-native-eval/1":
        raise ValueError("unrecognized native Qv2 evidence schema")
    if source["private_native_evaluation_summary_sha256"]!=EXPECT_PRIVATE:
        raise ValueError("private originating native model/role evidence drift")
    if source["selected_tensors"]["bits8"]!=28 or source["selected_tensors"]["bits4"]!=168:
        raise ValueError("wrong source-only hardware tensor budget")
    cases=source["controls"]
    expected={"v2_async_activation","role_balanced_random","v1_kernel_weight",
              "all_q4","role_balanced_proxy","source_f32"}
    if set(cases)!=expected:raise ValueError("incomplete matched real model controls")
    for name,x in cases.items():
        if not math.isfinite(x["PPL_public_131_tokens"]) or x["PPL_public_131_tokens"]<=0:
            raise ValueError("invalid real native quality data: "+name)
        if not x.get("evaluation_case_receipt_sha256"):
            raise ValueError("native model process proof absent")
    trial=cases["v2_async_activation"]["NLL"]
    teacher=cases["source_f32"]["NLL"]
    q4=cases["all_q4"]
    v2=source["actual_v2"]
    if teacher<=0 or not math.isfinite(teacher):raise ValueError("invalid BF16-widened teacher NLL")
    ratio=trial/teacher-1
    dominated=(trial>=q4["NLL"] and
               source["selected_tensors"]["projection_weight_bytes"]>=736985088 and
               v2["best_of_three_one_token_ms"]>=75.26)
    reasons=[]
    if source["public_teacher_forced_predictions"]<MIN_INDEPENDENT_TOKENS:
        reasons.append("insufficient_development_eval_positions")
    if ratio>MAX_RELATIVE_NLL_LOSS:
        reasons.append("NLL_increase_exceeds_declared_quality_ceiling")
    if dominated:
        reasons.append("v2_dominated_by_allQ4_on_observed_quality_weight_bytes_and_single_token_time")
    if source["concurrent_stage_truth"].find("whole-model run remained serial")<0:
        raise ValueError("full engine async truth missing")
    reasons.append("full_decoder_async_scheduling_not_integrated_or_proven")
    if not source.get("actual_IID_OOD_holdout_opened",False):
        reasons.append("independent_generalization_and_task_AUROC_not_evaluated")
    if source.get("scientific_Q_final_verdict")!="NO_VERDICT":
        raise ValueError("unsafe scientific readiness promotion in evidence")
    if source.get("production_changed") or source.get("main_changed"):
        raise ValueError("production/main changes not permitted for research")
    return dict(schema="BEGLIN-Qv2-kernel-activation-asynchronous-research-promotion-gate/1",
       status="NOT_READY" if reasons else "READY",
       source_private_sha256=source["private_native_evaluation_summary_sha256"],
       candidate="Qwen2.5-1.5B 28/196 offline BF16-activation-weighted async-stage-aware",
       hardware="Apple CPU native BEGLIN; actual full model serial",
       source_calibration_tokens=461,
       development_public_eval_token_positions=source["public_teacher_forced_predictions"],
       mean_candidate_nll=trial,reference_F32_source_nll=teacher,
       observed_relative_NLL_increase=ratio,
       max_relative_loss=MAX_RELATIVE_NLL_LOSS,
       observed_allQ4_domination=bool(dominated),
       blocking_reasons=reasons,
       no_automatic_live_promotion=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--require-ready",action="store_true")
    args=parser.parse_args()
    data=json.loads(FILE.read_text())
    result=evaluate(data)
    print(json.dumps(result,sort_keys=True))
    if args.require_ready and result["status"]!="READY":
        raise SystemExit(2)
if __name__=="__main__":main()
