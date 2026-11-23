#!/usr/bin/env python3
"""JE01 original pretrained compressed-AF real calibration NLL receipts and math."""
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/"docs/research"
sys.path.insert(0,str(ROOT/"tools"))
import je01_actual_original_cal16_next_gold_nll as nll

def verified(name,sha):
    d=json.loads((DOC/name).read_text())
    actual=d.pop("receipt_sha256",None)
    canonical=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                allow_nan=False).encode()).hexdigest()
    if actual!=sha or canonical!=sha:raise AssertionError("JE01_CAL_FIRST_GOLD_CANONICAL_RECEIPT_DRIFT")
    return d

class OriginalCompressedAFRealGoldNextToken(unittest.TestCase):
    def test_all_16_calibration_gold_targets_original_public_source_sha(self):
        d=verified("JE01_REAL_CAL16_ORIGINAL_WIKITEXT_NEXT_GOLD_TOKEN_SHA_20261117.json",
            "9d81479d2772c8e76f8ba9a7e2d6ef3eb97597ccf515d12186e969e037ca3b76")
        self.assertEqual(d["schema"],"beglin-je01-original-cal16-next-gold-token-identity/1")
        self.assertEqual(d["prior_fixed_input_count"],16)
        self.assertEqual(d["valid_first_gold_token_targets"],16)
        self.assertEqual(d["prompt_prefix_boundary_mismatch_count"],0)
        self.assertTrue(d["no_heldout_or_OOD_labels_accessed"])
        self.assertTrue(d["source_target_text_not_published"])
        self.assertEqual(d["source_frozen_manifest_sha256"],nll.FROZEN_SOURCE)
        self.assertEqual(len(d["records"]),16)
        for k,row in enumerate(d["records"]):
            self.assertEqual(row["request"],k)
            self.assertTrue(row["full_sequence_tokenizer_preserves_prompt_prefix"])
            self.assertGreaterEqual(row["first_public_gold_token_id"],0)
            self.assertLess(row["first_public_gold_token_id"],102400)

    def test_real_old32_and_new384_gold_token_nll_from_full_source_logits_equal(self):
        d=verified("JE01_ACTUAL_XOX_ORIGINAL_CAL16_REAL_FIRST_GOLD_NLL_20261117.json",
                   "272a5272e7af5408e89c38366dc71906ed9bfab4af6537dd14f8eece04b2920a")
        self.assertEqual(d["schema"],nll.SCHEMA)
        self.assertEqual(d["status"],"REAL_ORIGINAL_PRETRAINED_CAL_NEXT_GOLD_NLL_PASS")
        self.assertEqual(d["target_token_alignment_valid"],16)
        self.assertEqual(d["excluded_boundary_alignment_cases"],[])
        self.assertEqual(d["full_model_vocab"],102400)
        self.assertEqual(d["original_16_full_logit_receipt_sha256"],nll.FULLLOGIT_RECEIPT)
        self.assertTrue(d["true_original_compressed_AF_model"])
        self.assertTrue(d["source_calibration_only"])
        self.assertTrue(d["frozen_public_train_calibration_only"])
        self.assertEqual(d["mean_gold_one_token_nll_old32"],2.7294751068138448)
        self.assertEqual(d["mean_gold_one_token_nll_new384"],2.7294751068138448)
        self.assertEqual(d["max_case_abs_NLL_delta"],0.0)
        self.assertEqual(d["mean_delta_new384_minus_old32"],0.0)
        self.assertEqual(len(d["scored_cases"]),16)
        gold=verified("JE01_REAL_CAL16_ORIGINAL_WIKITEXT_NEXT_GOLD_TOKEN_SHA_20261117.json",
                      "9d81479d2772c8e76f8ba9a7e2d6ef3eb97597ccf515d12186e969e037ca3b76")
        self.assertEqual(d["source_gold_manifest_receipt_sha256"],
                         "9d81479d2772c8e76f8ba9a7e2d6ef3eb97597ccf515d12186e969e037ca3b76")
        for case,ident in zip(d["scored_cases"],gold["records"]):
            self.assertEqual(case["request"],ident["request"])
            self.assertEqual(case["first_public_gold_token_id"],ident["first_public_gold_token_id"])
            self.assertEqual(case["input_id"],ident["input_id"])
            self.assertEqual(case["gold_NLL_new384_minus_old32"],0.0)
            self.assertTrue(math.isfinite(case["gold_first_token_NLL_32"]))
            self.assertGreaterEqual(case["gold_first_token_NLL_32"],0)
        self.assertEqual(d["original_512_full_target_NLL"],"NOT_MEASURED")
        self.assertEqual(d["independent_same_source_and_OOD_NLL"],"NOT_MEASURED")
        self.assertEqual(d["Bayesian_Expert_precision_improvement"],"NOT_TESTED")
        self.assertFalse(d["production_weights_changed"])
        self.assertFalse(d["production_promotion_allowed"])

    def test_one_token_NLL_logsumexp_known_reference(self):
        self.assertAlmostEqual(nll.next_token_nll([0.0,0.0,0.0],0),math.log(3),places=14)
        self.assertAlmostEqual(nll.next_token_nll([2.0,1.0],0),math.log1p(math.exp(-1)),places=14)
        with self.assertRaisesRegex(ValueError,"GOLD_TOKEN_ID_INVALID"):
            nll.next_token_nll([2.0,1.0],2)
        with self.assertRaisesRegex(ValueError,"LOGITS_NOT_FINITE"):
            nll.next_token_nll([math.inf,0.0],1)


if __name__=="__main__":
    unittest.main(verbosity=2)
