#!/usr/bin/env python3
"""Sealed real original compressed AF native full-target WikiText pilot 656 tokens."""
import hashlib,json,math,pathlib,unittest

FILE=pathlib.Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_REAL16_FULL_TARGET_GOLD_NLL_PILOT_20261118.json"
SIG="2f99cf865cb816989b8dcd2b06d2aaa85e2d87ccc9f35e36877e767fb51f8e69"

class PretrainedNativeGoldFullTargetPilot(unittest.TestCase):
    def test_actual_pretrained_first_full_target_pilot_656_gold_tokens(self):
        d=json.loads(FILE.read_text());sig=d.pop("receipt_sha256")
        actual=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                      allow_nan=False).encode()).hexdigest()
        self.assertEqual(sig,actual)
        self.assertEqual(actual,SIG)
        self.assertEqual(d["status"],"REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_PILOT16_PASS")
        self.assertTrue(d["actual_original_pretrained_AF_model"])
        self.assertEqual(d["n_calibration_train_cases"],16)
        self.assertEqual(d["gold_full_target_tokens_scored"],656)
        self.assertEqual(len(d["per_case_real_NLL"]),16)
        self.assertEqual(sum(x["tokens"] for x in d["per_case_real_NLL"]),656)
        self.assertAlmostEqual(d["mean_gold_token_NLL"],2.291799702354307,places=13)
        self.assertAlmostEqual(d["per_example_mean_full_target_NLL"],2.245464298497445,places=13)
        self.assertLess(d["max_abs_first_gold_NLL_disagreement_from_original_256_previous_run"],1e-10)
        self.assertEqual(d["native_model_binary_sha256"],
            "de6f1a97aef17ddae42d1729282c38e2b7924f6d841daa3576dc47edeebbbba9")

    def test_actual_original_source_pinned_and_no_quality_promotion(self):
        d=json.loads(FILE.read_text())
        self.assertEqual(d["calibration_original256_router_receipt_sha256"],
           "62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb")
        self.assertEqual(d["gold_full_target_train_source_sha256"],
           "355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7")
        self.assertEqual(d["original_512_train_holdout_input_receipt_sha256"],
           "2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34")
        self.assertEqual(d["production_before"],d["production_after"])
        self.assertEqual(set(d["production_before"].values()),{35481,35493})
        self.assertGreaterEqual(d["minimum_memory_free_percent"],35)
        self.assertTrue(d["training_only"])
        self.assertFalse(d["heldout_gold_labels_used"])
        self.assertFalse(d["bayesian_expert_quantization_changed"])
        self.assertFalse(d["production_checkpoint_or_release_modified"])
        self.assertFalse(d["production_promotion_allowed"])
        self.assertEqual(d["trained_alt_bits_comparison"],"NOT_MEASURED")
        self.assertEqual(d["same_source_128_or_OOD_128_NLL"],"NOT_MEASURED")
        chunk=d["per_chunk_native_output_sha256"]
        self.assertEqual(len(chunk),1)
        self.assertEqual(chunk[0]["gold_targets_scored"],656)
        self.assertEqual(chunk[0]["result_sha256"],
           "b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c")

if __name__=="__main__":
    unittest.main(verbosity=2)
