#!/usr/bin/env python3
"""Post-hoc research-only choice-GGN curvature trust diagnostic (never fit policy).

The original Qv9 source-only policy is immutable, and this diagnostic cannot
alter it. Inspect signed first-order four-choice logit shifts and the
linearized-choice *exact* KL vs local quadratic GGN approximation. In large
shift regimes the second-order truncation is unreliable, even when optimizer
source score is small. Neither approximate KL here equals native full-vocab
next-token NLL or computes its Hessian.

NOT a preregistered hypothesis test: warning thresholds are engineering
sanity checks and automatically DENY promotion, never authorize it.
"""
from __future__ import annotations
import hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
SOURCE=ROOT/"docs/research/BEGLIN_QV9_GGN_QUADRATIC_TRUST_REGION_DIAGNOSTIC_20261023.json"
POL=ROOT/"docs/research/BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
MASK=ROOT/"configs/research/BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt"
OUT=ROOT/"docs/research/BEGLIN_QV9_CHOICE_GGN_TRUST_REGION_WARNING_20261023.json"
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    if OUT.exists():raise FileExistsError("post-mask GGN trust diagnostic already sealed")
    data=json.loads(SOURCE.read_text());pol=json.loads(POL.read_text())
    if data["status"]!="POST_MASK_FREEZE_OUTPUT_CHOICE_LINEARIZED_KL_CROSSCHECK" or \
       data["source_policy_sha256"]!=sha(POL.read_bytes()) or \
       pol["mask_sha256"]!=sha(MASK.read_bytes()) or \
       not data["no_held_MMLU_logit_used"] or not data["not_used_for_mask_reoptimization"]:
        raise RuntimeError("cannot claim independent post-hoc scientific source audit")
    checks={}
    for split in ("train","diagnostic"):
        measures=data["source_and_diagnostic"][split]
        for arm,vals in measures.items():
            frac=vals["fraction_abs_4choice_relative_logit_shift_gt_1"]
            upper=vals["fraction_abs_4choice_relative_logit_shift_gt_4"]
            ratio=vals["KL_divided_by_second_order_GGN_mean_ratio"]
            ggn=vals["mean_conditional_GGN"]
            kl=vals["mean_linearized_4choice_exact_KL"]
            if not all(map(math.isfinite,(frac,upper,ratio,ggn,kl))) or not 0<=upper<=frac<=1 or ggn<=0 or kl<0:
                raise RuntimeError("Qv9 original BF16 output-choice KL/GGN diagnostic has invalid source scalar")
            warn=(frac>.25 or ratio<.5 or ratio>2.0)
            checks[f"{split}/{arm}"]={
              "fraction_relative_logit_shift_above_1":frac,
              "fraction_relative_logit_shift_above_4":upper,
              "GGN_quadratic_mean":ggn,
              "linearized_four_choice_exact_KL_mean":kl,
              "KL_over_local_quadratic_ratio":ratio,
              "posthoc_sanity_warning_triggered":warn}
    candidate=checks["diagnostic/Qv9_GGN24"]
    output={
      "schema":"BEGLIN-Qv9-original-BF16-choice-conditional-GGN-postmask-trust-advisory/1",
      "status":"FAIL_CLOSED_OUTPUT_GGN_LOCAL_TAYLOR_TRUST_WARNING",
      "original_source_diagnostic_sha256":sha(SOURCE.read_bytes()),
      "original_ggn_source_only_policy_sha256":sha(POL.read_bytes()),
      "original_selected_mask_sha256":pol["mask_sha256"],
      "evaluation_48_MMLU_model_outputs_used":False,
      "post_hoc_rule_not_preregistered":True,
      "advisory_threshold_large_shift_fraction":.25,
      "advisory_threshold_approximation_ratio_range":[.5,2.0],
      "all_splits_and_policies":checks,
      "Qv9_independent_source_diagnostic_fraction_gt1":candidate["fraction_relative_logit_shift_above_1"],
      "Qv9_independent_diagnostic_KL_over_GGN":candidate["KL_over_local_quadratic_ratio"],
      "GGN_choice_space_not_fullvocab":True,
      "no_exact_native_model_KL_claim":True,
      "no_model_weights_or_original_policy_modified":True,
      "new_Qv9_promotion_allowed":False,
      "no_auto_promotion":True}
    if not candidate["posthoc_sanity_warning_triggered"]:
        raise RuntimeError("unexpected Qv9 source data does not trigger advisory; do not publish canned warning")
    doc=(json.dumps(output,sort_keys=True,indent=2)+"\n").encode()
    with OUT.open("xb") as f:f.write(doc);f.flush()
    print("QV9_LOCAL_GGN_TRUST_REGIME_WARNING_FAIL_CLOSED",
          json.dumps({"diagnostic_large_shift_fraction":candidate["fraction_relative_logit_shift_above_1"],
            "KL_over_GGN":candidate["KL_over_local_quadratic_ratio"],
            "public_SHA":sha(doc),"auto_promotion":False},sort_keys=True),flush=True)
if __name__=="__main__":main()
