#!/usr/bin/env python3
"""JE01 16/16 original-model XOX native384 vs operating32 token parity receipts."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/"docs"/"research"/"JE01_ACTUAL_XOX_NATIVE384_VS32_16_REAL_CAL_PARITY_20261117.json"
SHA="d10f987e57d802dbd86e29f8c2576e215f38cda10abbff5f21fe6119277674fd"


def verified():
    d=json.loads(P.read_text())
    sig=d.pop("receipt_sha256")
    canonical=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                                      allow_nan=False).encode()).hexdigest()
    if sig!=SHA or canonical!=SHA:
        raise AssertionError("JE01_16_INPUT_REAL_RUNTIME_PARITY_CANONICAL_SHA")
    return d


class Original32vs384CalibrationParity(unittest.TestCase):
    def test_16_real_input_same_greedy_tokens_and_immutable_source(self):
        d=verified()
        self.assertEqual(d["status"],"PASS_16_OF_16_REAL_CALIBRATION_NEXT_TOKEN_PARITY")
        self.assertEqual(d["real_calibration_inputs"],16)
        self.assertEqual(d["num_identical_greedy_next_tokens"],16)
        self.assertEqual(d["original_32_next_tokens"],d["new_384_next_tokens"])
        self.assertEqual(d["original_32_next_tokens"],
                         [11,279,2238,31460,330,72672,185,1501,245,11,457,285,327,20171,889,68307])
        self.assertEqual(d["total_prompt_positions_per_arm"],355)
        self.assertEqual(d["input_native_file_sha256"].__len__(),16)
        self.assertEqual(len(d["frozen_calibration_selected_lengths"]),16)
        self.assertGreaterEqual(min(d["frozen_calibration_selected_lengths"]),2)
        self.assertLessEqual(max(d["frozen_calibration_selected_lengths"]),31)
        self.assertEqual(d["source_512_receipt_sha256"],
                         "2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34")

    def test_real_model_gates_and_nonpromotion(self):
        d=verified()
        self.assertEqual(d["actual_model_layers_per_run"],27)
        self.assertEqual(d["AF_tensor_count_per_run"],269)
        self.assertEqual(d["isolated_384_binary_SHA256"],
                         "60a0b559496180d29b65d5a9cb86bba4d16a12238a140cdfdd157602fcc90437")
        self.assertEqual(d["original_serving_binary_SHA256"],
                         "8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9")
        self.assertGreaterEqual(d["min_memory_free_percent"],35)
        self.assertTrue(d["source_operating_workers_unchanged"])
        self.assertTrue(d["single_test_latency_not_representative_of_stable_performance"])
        self.assertEqual(d["full_102400_way_logit_parity"],"NOT_MEASURED")
        self.assertEqual(d["gold_token_NLL"],"NOT_MEASURED")
        self.assertEqual(d["full_512_model_forward"],"NOT_RUN")
        self.assertFalse(d["production_promotion_allowed"])

if __name__=="__main__":
    unittest.main(verbosity=2)
