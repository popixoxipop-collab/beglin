#!/usr/bin/env python3
"""JE01 original 8 routed Expert native AF packed byte source ledger receipt."""
import hashlib,json,pathlib,unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]/"docs/research"
FILE=ROOT/"JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
GATE=ROOT/"JE01_ACTUAL_XOX_ORIGINAL_AF_REAL256_ROUTER_FINAL_20261118.json"
EXPECTED="f47bbccbc3878a70dccb9f4b7f5f7f5bd43087aeec58c948924ef85bb428ca4c"

def verified(path,sha):
    value=json.loads(path.read_text())
    claim=value.pop("receipt_sha256")
    actual=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                              allow_nan=False).encode()).hexdigest()
    if claim!=sha or actual!=sha:raise AssertionError("JE01_ORIGINAL_AF_BYTE_EVIDENCE_SHA_NOT_MATCH")
    return value

class OriginalExpertRealByteAudits(unittest.TestCase):
    def test_actual_8_expert_3role_original_file_payload_exact(self):
        d=verified(FILE,EXPECTED)
        self.assertEqual(d["status"],"EIGHT_SELECTED_ORIGINAL_AF_BASELINE_BYTES_AUDITED")
        self.assertEqual(d["source_original_AF_total_bytes"],9814343680)
        self.assertEqual(d["source_original_AF_layout_SHA256"],
           "ba3898f3bf366b1aec8471d9896885f101bf64b78c816828ea82bd7d58851c3e")
        self.assertEqual(d["real_256_selected_expert_baseline_eight_bytes"],43253760)
        self.assertEqual(d["actual_selected_expert_slots"],8)
        self.assertEqual(d["real_original_baseline_bit_width_per_selected_expert"],4)
        self.assertEqual(len(d["selected_slots"]),8)
        intervals=[]
        for slot in d["selected_slots"]:
            self.assertEqual(slot["actual_packed_baseline_bytes"],5406720)
            self.assertEqual(len(slot["roles"]),3)
            self.assertEqual([v["role"] for v in slot["roles"]],["gate_proj","up_proj","down_proj"])
            self.assertEqual(sum(r["total_expert_role_bytes"] for r in slot["roles"]),5406720)
            for role in slot["roles"]:
                self.assertEqual(role["individual_expert_bits"],4)
                self.assertEqual(role["packed_weight_bytes"],1441792)
                self.assertEqual(role["individual_expert_scale_f32_bytes"],180224)
                self.assertEqual(role["individual_expert_bias_f32_bytes"],180224)
                for kind,v in role["source_Af_global_byte_regions"].items():
                    a,b=v
                    self.assertTrue(0<=a<b<=9814343680)
                    intervals.append((a,b,kind))
        intervals.sort()
        for prev,nex in zip(intervals,intervals[1:]):
            self.assertLessEqual(prev[1],nex[0])
        self.assertEqual(sum(b-a for a,b,k in intervals),43253760)

    def test_expert_ids_chosen_from_actual_256_routes_not_proxy(self):
        d=verified(FILE,EXPECTED)
        src=verified(GATE,"62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb")
        a=[(v["layer"],v["expert"],v["calibration_last_token_routed_calls"])
           for v in d["selected_slots"]]
        b=[(v["layer"],v["expert"],v["real_original_calibration_routed_calls"])
           for v in src["exactly_eight_pilot_slots_frozen_by_unlabeled_calibration_last_token_routing"]]
        self.assertEqual(a,b)
        self.assertGreaterEqual(min(c for l,e,c in a),32)

    def test_no_alternate_tier_byte_or_NLL_claims(self):
        d=verified(FILE,EXPECTED)
        self.assertEqual(d["real_alternate_2_6_8_precision_byte_artifacts"],"NOT_CREATED")
        self.assertEqual(d["real_mixed_expert_2_6_8_bits_end_to_end_NLL"],"NOT_MEASURED")
        self.assertTrue(d["cannot_use_formula_to_certify_equal_real_bytes"])
        self.assertTrue(d["native_reference_model_checkpoint_unchanged"])
        self.assertFalse(d["production_promotion_allowed"])

if __name__=="__main__":
    unittest.main(verbosity=2)
