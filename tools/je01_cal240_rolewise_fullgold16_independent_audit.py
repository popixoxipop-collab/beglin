#!/usr/bin/env python3
"""JE01 CAL240 rolewise original27 gold16 read-only independent paired NLL audit.

Runs on original EOE already-completed OFF/ON receipts. Never performs
model inference and never touches validation/OOD. TRAIN16 was previously
inspected: bootstrap is exploratory, not an independent generalization CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

ORIGINAL_AF_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
MODEL_SHA = "db028b252d489861349259ec7cd08c87086e617d9f331b1de4a5f660bed79f8f"
BUNDLE_SHA = "ada43c5e233aa4154d37de5988aa22662ed16496363e84183a784c4c588a63ac"
ORIGINAL_OUTPUT_SHA = "b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c"
BASELINE_RECEIPT = "7dd5da25e500007a2a35fd724d82a7c8c6966727458763160f6d4ee7b3ac54ec"
BASELINE_NLL = 2.291799702354307
OFF_STATUS = "PASS_ORIGINAL27_CAL240_ROLEWISE_BINARY_OFF_GOLD_PARITY"
ON_STATUS = "ACTUAL_REAL_ORIGINAL27_CAL240_ROLEWISE_PILOT16_MEASURED"


def fail(condition, label):
    if not condition:
        raise ValueError(label)


def canonical(d):
    return hashlib.sha256(json.dumps(
        d, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def signed_json(path, expected_sha=None):
    fail(path.is_file() and not path.is_symlink(), "JE01_RECEIPT_FILE_MISSING_OR_SYMLINK")
    doc = json.loads(path.read_bytes())
    seal = doc.get("receipt_sha256")
    fail(isinstance(seal, str) and
         (expected_sha is None or seal == expected_sha) and
         canonical({k:v for k,v in doc.items() if k!="receipt_sha256"}) == seal,
         "JE01_RECEIPT_CANONICAL_SIGNATURE_INVALID")
    return doc


def validate_run(d, mode, expected_status):
    fail(d.get("status") == expected_status and d.get("mode") == mode,
         "JE01_GOLD_MODE_OR_STATUS_MISMATCH")
    fail(d.get("actual_model_complete_27layers") is True and
         d.get("gold_train_cases") == 16 and d.get("gold_tokens") == 656 and
         d.get("actual_original_AF_source_sha256") == ORIGINAL_AF_SHA and
         d.get("actual_compiled_CAL240_rolewise_native_model_sha256") == MODEL_SHA and
         d.get("original_real_CAL240_rolewise_BF16_source_sha256") == BUNDLE_SHA and
         d.get("actual_preserved_original_affine_AF4_roles") == 12 and
         d.get("actual_q4_roles") == 6 and d.get("actual_q5_roles") == 6 and
         d.get("production_checkpoint_modified") is False and
         d.get("true_source_heldout_ood_labels_accessed") is False and
         d.get("real_OOD_and_same_source_128") == "NOT_MEASURED" and
         d.get("meets_preregistered_adoption_gate") == "NOT_EVALUATED",
         "JE01_GOLD_SOURCE_OR_SCOPE_DRIFT")
    cases = d["per_case"]
    per_target = d["per_gold_target_actual_NLL"]
    fail(len(cases)==len(per_target)==16 and
         sum(int(p["tokens"]) for p in cases)==656 and
         all(isinstance(row,list) and len(row)==cases[i]["tokens"]
             for i,row in enumerate(per_target)),
         "JE01_GOLD_CASES_OR_TARGETS_WRONG")
    for i,(case,targets) in enumerate(zip(cases,per_target)):
        fail(case["row"]==i and
             all(isinstance(x,(float,int)) and math.isfinite(x) and x>=0 for x in targets) and
             abs(math.fsum(targets)-case["measured_mode_NLL_sum"]) <= 1e-7 and
             abs(case["measured_mode_NLL_sum"]-case["original_NLL_sum"] -
                 case["candidate_minus_original_NLL_sum"]) <= 1e-7,
             "JE01_GOLD_PER_CASE_TARGET_NLL_INCONSISTENT")
    fail(abs(math.fsum(c["measured_mode_NLL_sum"] for c in cases)/656 -
             d["actual_native_mean_gold_NLL"]) < 1e-10,
         "JE01_GOLD_AGGREGATE_NLL_MISMATCH")
    fail(abs(math.fsum(c["original_NLL_sum"] for c in cases)/656 -
             d["same_EOE_original_native_mean_gold_NLL"]) < 1e-10,
         "JE01_GOLD_BASELINE_PER_CASE_NLL_MISMATCH")
    fail(abs(d["actual_native_mean_gold_NLL"]-
             d["same_EOE_original_native_mean_gold_NLL"]-
             d["paired_mean_NLL_candidate_minus_original"]) < 1e-10,
         "JE01_GOLD_DELTA_MISMATCH")
    return cases


def cluster_bootstrap(cases, draws=50000, seed=20261122):
    rng=random.Random(seed)
    n=len(cases)
    diffs=[]
    for _ in range(draws):
        ids=(rng.randrange(n) for _ in range(n))
        sampled=[cases[i] for i in ids]
        tokens=sum(int(c["tokens"]) for c in sampled)
        diffs.append(math.fsum(c["candidate_minus_original_NLL_sum"] for c in sampled)/tokens)
    diffs.sort()
    return {
        "paired_mean_per_token_delta_cluster_bootstrap_percentile95":
            [diffs[int(.025*draws)],diffs[int(.975*draws)]],
        "bootstrap_fraction_lower_NLL":sum(x<0 for x in diffs)/draws,
        "bootstrap_repetitions":draws,
        "seed":seed,
    }


def audit(baseline_file, off_file, on_file):
    base=signed_json(baseline_file, BASELINE_RECEIPT)
    fail(base.get("status")=="PASS_EOE_REAL_ORIGINAL_AF384_FULLGOLD_TRAIN16_NLL_PARITY" and
         base.get("full_gold_tokens")==656 and base.get("real_cases")==16 and
         base.get("native_output_sha256")==ORIGINAL_OUTPUT_SHA and
         abs(base.get("measured_native_EOE_mean_gold_NLL", float("inf"))-BASELINE_NLL)<1e-12,
         "JE01_GOLD_ORIGINAL_BASELINE_CHANGED")
    off=signed_json(off_file)
    oc=validate_run(off,"off",OFF_STATUS)
    fail(off.get("real_native_output_sha256")==ORIGINAL_OUTPUT_SHA and
         abs(off["actual_native_mean_gold_NLL"]-BASELINE_NLL)<1e-12 and
         max(abs(c["candidate_minus_original_NLL_sum"]) for c in oc)<1e-10,
         "JE01_GOLD_OFF_NOT_BIT_IDENTICAL_ORIGINAL_AF4")
    on=signed_json(on_file)
    cc=validate_run(on,"rolewise",ON_STATUS)
    fail(all(abs(cc[i]["original_NLL_sum"]-oc[i]["original_NLL_sum"])<1e-10
             and cc[i]["tokens"]==oc[i]["tokens"] for i in range(16)),
         "JE01_GOLD_OFF_ON_PROMPT_OR_SOURCE_MISMATCH")
    d=(math.fsum(c["candidate_minus_original_NLL_sum"] for c in cc)/656)
    fail(abs(d-on["paired_mean_NLL_candidate_minus_original"])<1e-10,
         "JE01_GOLD_INDEPENDENT_PAIRED_NLL_MISMATCH")
    return {
        "status":"JE01_REAL_CAL240_ROLEWISE_TRAIN16_EXPLORATORY_NLL_AUDIT",
        "baseline_NLL":BASELINE_NLL,
        "candidate_NLL":on["actual_native_mean_gold_NLL"],
        "paired_delta_candidate_minus_baseline":d,
        "relative_NLL_change_pct":100*d/BASELINE_NLL,
        "cases":16,"gold_tokens":656,
        "worsened_cases":sum(c["candidate_minus_original_NLL_sum"]>1e-10 for c in cc),
        "improved_cases":sum(c["candidate_minus_original_NLL_sum"]< -1e-10 for c in cc),
        "unchanged_cases":sum(abs(c["candidate_minus_original_NLL_sum"])<=1e-10 for c in cc),
        "off_full_output_SHA_parity":"PASS",
        "off_receipt_sha256":off["receipt_sha256"],
        "on_receipt_sha256":on["receipt_sha256"],
        "no_independent_holdout":True,
        "full_240_prompt_independent_NLL":"NOT_RUN",
        "Fisher_Hessian":"NOT_MEASURED",
        "production_modified":False,
        **cluster_bootstrap(cc),
    }


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--off",type=Path,required=True)
    p.add_argument("--on",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.baseline,a.off,a.on),sort_keys=True,allow_nan=False))
