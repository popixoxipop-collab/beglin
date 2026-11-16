#!/usr/bin/env python3
"""JE01 synthetic-only MoE precision-pair optimizer contracts."""
import copy
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import je01_bayesian_expert_allocator as je


def example():
    return {
        "checkpoint_sha256": "a" * 64,
        "calibration_manifest_sha256": "b" * 64,
        "backend": "metal",
        "slots": [
            {"layer": 11, "expert_id": 3, "baseline_bits": 4,
             "options": [
                 {"bits": 4, "packed_bytes": 100, "paired_delta_nll": [],
                  "router_flip_count": 0, "router_checks": 64},
                 {"bits": 6, "packed_bytes": 120, "paired_delta_nll": [-2.] * 64,
                  "router_flip_count": 0, "router_checks": 64}]},
            {"layer": 11, "expert_id": 5, "baseline_bits": 4,
             "options": [
                 {"bits": 4, "packed_bytes": 100, "paired_delta_nll": [],
                  "router_flip_count": 0, "router_checks": 64},
                 {"bits": 3, "packed_bytes": 80, "paired_delta_nll": [0.1] * 64,
                  "router_flip_count": 0, "router_checks": 64}]},
        ],
    }


class JE01Contracts(unittest.TestCase):
    def test_exact_byte_bayesian_candidate_is_shadow_only(self):
        out = je.optimize_exact_bytes(example())
        self.assertEqual(out["selected_packed_bytes"], out["baseline_packed_bytes"])
        self.assertEqual(out["selected_packed_bytes"], 200)
        self.assertTrue(out["pilot_prediction_improves"])
        self.assertEqual([s["bits"] for s in out["selected"]], [6, 3])
        self.assertLess(out["predicted_conservative_loss_upper_delta"], -0.15)
        self.assertFalse(out["production_promotion_allowed"])
        self.assertEqual(out["full_moe_forward_additivity"], "NOT_VERIFIED")

    def test_deterministic_and_no_input_mutation(self):
        p = example()
        original = copy.deepcopy(p)
        self.assertEqual(je.optimize_exact_bytes(p), je.optimize_exact_bytes(p))
        self.assertEqual(p, original)

    def test_router_flip_blocks_promotion(self):
        p = example()
        p["slots"][0]["options"][1]["router_flip_count"] = 1
        out = je.optimize_exact_bytes(p)
        self.assertFalse(out["pilot_prediction_improves"])
        self.assertEqual([s["bits"] for s in out["selected"]], [4, 4])

    def test_missing_calibration_blocks(self):
        p = example()
        p["slots"][0]["options"][1]["paired_delta_nll"] = [-2.] * 31
        out = je.optimize_exact_bytes(p)
        self.assertFalse(out["pilot_prediction_improves"])
        self.assertEqual(out["status"], "SHADOW_ONLY_NO_WEIGHT_EDIT")

    def test_unequal_byte_pair_blocks(self):
        p = example()
        p["slots"][1]["options"][1]["packed_bytes"] = 81
        out = je.optimize_exact_bytes(p)
        self.assertFalse(out["pilot_prediction_improves"])

    def test_unsupported_bits_fail(self):
        p = example()
        p["slots"][0]["options"][1]["bits"] = 17
        with self.assertRaisesRegex(ValueError, "BITS_DUPLICATE"):
            je.optimize_exact_bytes(p)

    def test_native_expert_tier_9_not_in_direct_expert_bits(self):
        p = example()
        p["slots"][0]["options"][1]["bits"] = 9
        with self.assertRaisesRegex(ValueError, "BITS_DUPLICATE_OR_UNSUPPORTED"):
            je.optimize_exact_bytes(p)

    def test_first_layer_is_dense_and_must_not_be_called_routed(self):
        p = example()
        p["slots"][0]["layer"] = 0
        with self.assertRaisesRegex(ValueError, "EXPERT_ID"):
            je.optimize_exact_bytes(p)

    def test_duplicate_expert_key_fail(self):
        p = example()
        p["slots"][1]["expert_id"] = 3
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            je.optimize_exact_bytes(p)

    def test_corrupt_source_hash_fail(self):
        p = example()
        p["checkpoint_sha256"] = "NOT-A-HASH"
        with self.assertRaisesRegex(ValueError, "SHA_MISSING"):
            je.optimize_exact_bytes(p)

    def test_missing_option_cannot_invent_tier(self):
        p = example()
        p["slots"][0]["options"] = p["slots"][0]["options"][1:]
        with self.assertRaisesRegex(ValueError, "BASELINE_TIER"):
            je.optimize_exact_bytes(p)

    def test_nonfinite_delta_denied(self):
        p = example()
        p["slots"][0]["options"][1]["paired_delta_nll"][4] = math.inf
        with self.assertRaises(ValueError):
            je.optimize_exact_bytes(p)

    def test_duplicate_bits_denied(self):
        p = example()
        p["slots"][0]["options"].append(copy.deepcopy(p["slots"][0]["options"][0]))
        with self.assertRaisesRegex(ValueError, "BITS_DUPLICATE"):
            je.optimize_exact_bytes(p)

    def test_more_than_eight_experts_denied(self):
        p = example()
        for n in range(2, 9):
            slot = copy.deepcopy(p["slots"][0])
            slot["expert_id"] = 3+n
            p["slots"].append(slot)
        with self.assertRaisesRegex(ValueError, "PILOT_EXPERT_COUNT"):
            je.optimize_exact_bytes(p)

    def test_no_cross_expert_covariance_in_state(self):
        out = je.optimize_exact_bytes(example())
        self.assertEqual(out["cross_expert_covariance"], "NOT_ESTIMATED")

    def test_student_t_nig_has_finite_positive_scale(self):
        pred = je.nig_predict([-1.2] * 64)
        self.assertEqual(pred["n_calibration"], 64)
        self.assertGreater(pred["student_t_df"], 2)
        self.assertGreater(pred["predictive_scale"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
