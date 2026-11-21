#!/usr/bin/env python3
"""JE01 actual native DeepSeek original AF 256 first-gold NLL receipt contracts."""
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import je01_original_AF_real256_first_gold_nll as scorer

RECEIPT=ROOT/"docs/research/JE01_ACTUAL_XOX_ORIGINAL_AF_CAL256_FIRST_GOLD_NLL_20261118.json"
SEALED="aca35d52a67c8ce5854c1b7297f80b241df2b2d66823b75a98744123fe8169b4"

def source():
    doc=json.loads(RECEIPT.read_text())
    sha=doc.pop("receipt_sha256")
    canon=hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(",",":"),
                         allow_nan=False).encode()).hexdigest()
    if sha!=SEALED or canon!=SEALED:
        raise AssertionError("JE01_REAL_256_SOURCE_SEAL_DRIFT")
    return doc

class OriginalNativeTrain256(unittest.TestCase):
    def test_real_256_first_gold_mean_and_sample_count(self):
        d=source()
        self.assertEqual(d["schema"],scorer.SCORE_SCHEMA)
        self.assertEqual(d["status"],"PASS_REAL_ORIGINAL_AF_256_CAL_NEXT_GOLD_NLL")
        self.assertEqual(d["actual_calibration_prompts"],256)
        self.assertEqual(d["actually_scored_first_gold_tokens"],256)
        self.assertEqual(d["actual_full_vocab_logits_values_scored"],256*102400)
        self.assertEqual(d["total_full_gold_continuation_tokens_reserved"],11024)
        self.assertEqual(len(d["scored_case_receipts"]),256)
        self.assertAlmostEqual(d["mean_actual_first_gold_token_NLL"],3.201974851245297,places=12)
        self.assertEqual(d["first_gold_greedy_top1_exact_cases"],92)
        self.assertEqual(d["source_real_native_256_router_receipt_sha256"],scorer.NATIVE_RECEIPT_SHA)
        self.assertEqual(d["source_original_full_target_gold_sha256"],scorer.GOLD_RECEIPT_SHA)
        self.assertEqual(len(d["actual_native_256_last_prompt_logits_chunks_sha256"]),16)
        self.assertTrue(d["real_original_native_gpu_model"])

    def test_each_real_sample_has_gold_and_no_holdout_labels(self):
        d=source()
        samples=d["scored_case_receipts"]
        self.assertEqual([x["row"] for x in samples],list(range(256)))
        self.assertEqual(len({x["source_id"] for x in samples}),256)
        self.assertEqual(sum(x["model_greedy_top1_is_gold"] for x in samples),92)
        self.assertEqual(d["mean_actual_first_gold_token_NLL"],
            math.fsum(x["actual_original_next_gold_NLL"] for x in samples)/256)
        for row in samples:
            self.assertEqual(len(row["native_input_sha256"]),64)
            self.assertEqual(len(row["full102400_logits_row_sha256"]),64)
            self.assertTrue(math.isfinite(row["actual_original_next_gold_NLL"]))
            self.assertGreaterEqual(row["actual_original_next_gold_NLL"],0)
            self.assertGreaterEqual(row["gold_target_first_token_id"],0)
            self.assertLess(row["gold_target_first_token_id"],102400)
        self.assertTrue(d["frozen_wikitext_train_calibration_only"])
        self.assertEqual(d["original_256_full_target_sequence_gold_NLL"],"NOT_MEASURED")
        self.assertEqual(d["same_source_128_holdout_gold_NLL"],"NOT_MEASURED")
        self.assertEqual(d["OOD_128_holdout_gold_NLL"],"NOT_MEASURED")
        self.assertEqual(d["Bayesian_Expert_precision_gain"],"NOT_MEASURED")
        self.assertFalse(d["expert_precision_alternatives_run"])
        self.assertFalse(d["production_promotion_allowed"])

    def test_nll_math_known_values_and_rejection(self):
        self.assertAlmostEqual(scorer.actual_logit_token_nll([0.0]*102400,50),
                               math.log(102400),places=12)
        with self.assertRaisesRegex(ValueError,"GOLD_TOKEN_INVALID"):
            scorer.actual_logit_token_nll([0.0]*102400,102400)
        with self.assertRaisesRegex(ValueError,"ORIGINAL_MODEL_FULL_LOGITS_NOT_FINITE"):
            scorer.actual_logit_token_nll([math.inf]*102400,0)

if __name__=="__main__":
    unittest.main(verbosity=2)
