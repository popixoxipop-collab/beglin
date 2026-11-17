#!/usr/bin/env python3
"""JE01 native q4 vs qNg64/q8g64 PRE-SOLVE contracts, not real candidate forward."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/je01_native_exact_byte_presolve.py"
spec = importlib.util.spec_from_file_location("je01_af8_presolve", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class NativeAF8PresolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = m.read_verified_ledger(m.DEFAULT_LEDGER)

    def test_actual_ledger_sha_and_byte_projection(self):
        slots = m.validate_ledger(self.ledger)
        self.assertEqual(len(slots), 8)
        self.assertEqual(
            {b: m.projected_slot_bytes(slots[0], b) for b in m.TIERS},
            {2: 2_703_360, 4: 5_406_720, 6: 7_028_736, 8: 9_191_424},
        )
        self.assertEqual(sum(m.projected_slot_bytes(s, 4) for s in slots),
                         43_253_760)

    def test_equal_byte_catalog_is_not_a_real_weight_artifact(self):
        result = m.presolve(self.ledger)
        self.assertEqual(result["projected_equal_byte_catalog_count"], 1317)
        self.assertEqual(result["projected_equal_byte_nonbaseline_count"], 1316)
        self.assertEqual(result["projected_pattern_counts"], {
            "2:0,4:8,6:0,8:0": 1,
            "2:2,4:4,6:1,8:1": 840,
            "2:3,4:0,6:5,8:0": 56,
            "2:4,4:0,6:2,8:2": 420,
        })
        self.assertEqual(len(result["projected_catalog_sha256"]), 64)
        self.assertEqual(result["real_2_6_8_packed_weight_artifact_count"], 0)
        self.assertEqual(result["real_expert_precision_paired_NLL"], "NOT_MEASURED")
        self.assertFalse(result["bayesian_plan_eligible"])
        self.assertFalse(result["production_promotion_allowed"])

    def test_original_q4_affine_bias_not_charged_to_symmetric_tiers(self):
        r = self.ledger["selected_slots"][0]["roles"][0]
        self.assertEqual(r["total_expert_role_bytes"], 1_802_240)
        self.assertEqual(r["packed_weight_bytes"] * 8 // 4
                         + r["individual_expert_scale_f32_bytes"], 3_063_808)
        self.assertNotEqual(3_063_808,
                            r["packed_weight_bytes"] * 2
                            + r["individual_expert_scale_f32_bytes"]
                            + r["individual_expert_bias_f32_bytes"])

    def test_tampered_source_rejected(self):
        data = copy.deepcopy(self.ledger)
        data["selected_slots"][0]["roles"][0]["packed_weight_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "JE01_SOURCE_ROLE_BYTE_LAYOUT_DRIFT"):
            m.validate_ledger(data)

    def test_unknown_precision_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "JE01_UNSUPPORTED_BIT_TIER"):
            m.projected_slot_bytes(self.ledger["selected_slots"][0], 3)

    def test_repeatable_catalog_digest(self):
        self.assertEqual(m.presolve(self.ledger)["projected_catalog_sha256"],
                         m.presolve(self.ledger)["projected_catalog_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
