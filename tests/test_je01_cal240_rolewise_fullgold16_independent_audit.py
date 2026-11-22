#!/usr/bin/env python3
"""Synthetic-only tests for JE01 signed OFF/ON gold16 independent NLL verifier."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/"tools/je01_cal240_rolewise_fullgold16_independent_audit.py"
spec=importlib.util.spec_from_file_location("je01_real_cal240_gold16_auditor",PATH)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def fixture(mode="rolewise",target=2.28):
    old=2.29
    rows=[];individual=[]
    for i in range(16):
        tokens=[target]*41
        rows.append({
            "row":i, "tokens":41,
            "original_NLL_sum":41*old,
            "measured_mode_NLL_sum":sum(tokens),
            "candidate_minus_original_NLL_sum":sum(tokens)-41*old,
        })
        individual.append(tokens)
    return {
        "status":mod.ON_STATUS if mode=="rolewise" else mod.OFF_STATUS,
        "mode":mode,
        "actual_model_complete_27layers":True,
        "gold_train_cases":16,"gold_tokens":656,
        "actual_original_AF_source_sha256":mod.ORIGINAL_AF_SHA,
        "actual_compiled_CAL240_rolewise_native_model_sha256":mod.MODEL_SHA,
        "original_real_CAL240_rolewise_BF16_source_sha256":mod.BUNDLE_SHA,
        "actual_preserved_original_affine_AF4_roles":12,
        "actual_q4_roles":6,"actual_q5_roles":6,
        "production_checkpoint_modified":False,
        "true_source_heldout_ood_labels_accessed":False,
        "real_OOD_and_same_source_128":"NOT_MEASURED",
        "meets_preregistered_adoption_gate":"NOT_EVALUATED",
        "per_case":rows,"per_gold_target_actual_NLL":individual,
        "actual_native_mean_gold_NLL":target,
        "same_EOE_original_native_mean_gold_NLL":old,
        "paired_mean_NLL_candidate_minus_original":target-old,
    }


class TestJE01NllAuditMath(unittest.TestCase):
    def test_valid_synthetic_full_16x41_tokens(self):
        d=fixture()
        self.assertEqual(len(mod.validate_run(d,"rolewise",mod.ON_STATUS)),16)

    def test_wrong_mode_rejected(self):
        d=fixture()
        with self.assertRaisesRegex(ValueError,"MODE_OR_STATUS"):
            mod.validate_run(d,"off",mod.OFF_STATUS)

    def test_bf16_provenance_mismatch_rejected(self):
        d=fixture();d["actual_original_AF_source_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"SOURCE_OR_SCOPE"):
            mod.validate_run(d,"rolewise",mod.ON_STATUS)

    def test_single_token_nll_mutation_detected(self):
        d=fixture();d["per_gold_target_actual_NLL"][0][0]+=1
        with self.assertRaisesRegex(ValueError,"PER_CASE_TARGET"):
            mod.validate_run(d,"rolewise",mod.ON_STATUS)

    def test_bootstrap_is_seeded_and_reports_lower_is_better(self):
        d=fixture(target=2.28)
        cases=mod.validate_run(d,"rolewise",mod.ON_STATUS)
        a=mod.cluster_bootstrap(cases,draws=1000)
        b=mod.cluster_bootstrap(cases,draws=1000)
        self.assertEqual(a,b)
        self.assertEqual(a["bootstrap_fraction_lower_NLL"],1.0)
        self.assertAlmostEqual(a["paired_mean_per_token_delta_cluster_bootstrap_percentile95"][0],
                               -.01,places=9)

    def test_no_false_full_train_or_holdout_claim_in_code(self):
        raw=PATH.read_text()
        self.assertIn('"full_240_prompt_independent_NLL":"NOT_RUN"',raw)
        self.assertIn('"no_independent_holdout":True',raw)
        self.assertIn('"off_full_output_SHA_parity":"PASS"',raw)


if __name__=="__main__":
    unittest.main()
