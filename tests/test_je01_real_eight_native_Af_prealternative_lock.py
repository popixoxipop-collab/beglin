#!/usr/bin/env python3
"""JE01 actual original-AF eight-slot lock predates any alternative bit forward."""
import json,pathlib,unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]/"docs/research"
FROZEN=ROOT/"JE01_ORIGINAL_AF_EIGHT_EXPERT_PREALTERNATIVE_PILOT_LOCK_20261118.json"
LAYOUT=ROOT/"JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
ROUTER=ROOT/"JE01_ACTUAL_XOX_ORIGINAL_AF_REAL256_ROUTER_FINAL_20261118.json"

class OriginalRealExpertBeforeAlterations(unittest.TestCase):
    def test_same_eight_original_routed_slots_pre_alt_and_actual_budget(self):
        lock=json.loads(FROZEN.read_text())
        source=json.loads(ROUTER.read_text())
        layout=json.loads(LAYOUT.read_text())
        observed=source["exactly_eight_pilot_slots_frozen_by_unlabeled_calibration_last_token_routing"]
        self.assertEqual(lock["selected8_pre_alternative_model_results"],observed)
        self.assertEqual([(x["layer"],x["expert"]) for x in observed],
                         [(x["layer"],x["expert"]) for x in layout["selected_slots"]])
        self.assertEqual(lock["actual_selected8_file_payload_bytes"],43253760)
        self.assertEqual(layout["real_256_selected_expert_baseline_eight_bytes"],43253760)
        self.assertEqual(lock["real_model_baseline_precision_bits"],4)
        self.assertEqual(lock["actual_selected8_ffn_gate_up_down_role_payloads"],24)
        self.assertEqual(lock["source_real_selected_expert_file_byte_sha256"],layout["receipt_sha256"])
        self.assertTrue(lock["all8_have_ge32_unique_prompt_last_token_routes"])

    def test_no_false_prelabel_selection_claim_or_fake_actual_alternative(self):
        lock=json.loads(FROZEN.read_text())
        self.assertFalse(lock["quality_reference_known_before_this_lock"]["strict_all256_prelabel_selection"])
        self.assertTrue(lock["quality_reference_known_before_this_lock"]["previous_calibration16_full_target_gold_NLL_already_viewed"])
        self.assertEqual(lock["candidate_precision_2_6_8_file_artifacts"],"NOT_CREATED")
        self.assertEqual(lock["actual_quantized_expert_alt_quality"],"NOT_MEASURED")
        self.assertTrue(lock["allowed_alternatives_only_if_real_packed_artifacts_sha_and_bytes_exist"])
        self.assertEqual(lock["independent_same_source_holdout"],128)
        self.assertEqual(lock["independent_ood_holds"],128)
        self.assertEqual(lock["training_calibration"],256)
        self.assertFalse(lock["promotion_allowed"])
        self.assertTrue(lock["preregistered_decision_thresholds"]["zero_hard_topk_router_flips"])
        self.assertEqual(lock["preregistered_decision_thresholds"]["paired_bootstrap_resamples"],20000)
        self.assertEqual(lock["preregistered_decision_thresholds"]["full_independent_same_source_mean_NLL_gain_min"],.02)

if __name__=="__main__":
    unittest.main(verbosity=2)
