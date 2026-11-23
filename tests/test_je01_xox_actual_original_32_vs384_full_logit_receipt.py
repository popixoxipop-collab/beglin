#!/usr/bin/env python3
"""JE01 XOX real original DeepSeek full-vocab output parity evidence strict tests.

The original raw 6,553,886-byte stdout-free float32 dumps stay on XOX,
both have whole-file SHA 428d2f... and were independently reread and compared.
This test checks SHA-canonical metadata, all 16 row receipts, no gold NLL
claim, and unchanged real production worker identities. Does not re-run GPU.
"""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/"docs/research/JE01_ACTUAL_XOX_ORIGINAL32_VS384_FULL102400_LOGITS_16_REAL_CAL_20261117.json"
SHA="34da06b2441bd1cfa8c8593ff20e8c2d3e40850e1c0ef6d6c8a127d70cb75d14"
LOG_SHA="428d2f985d1eca80ab299a7ab8cbf3d1330391efe46791f87d013a670f5d387a"


def load():
    d=json.loads(FILE.read_text())
    sig=d.pop("receipt_sha256")
    actual=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                        allow_nan=False).encode()).hexdigest()
    if sig!=SHA or actual!=SHA:
        raise AssertionError("JE01_CANONICAL_REAL_NATIVE_LOGITS_EVIDENCE_DRIFT")
    return d


class ActualOriginalNativeLogitParity(unittest.TestCase):
    def test_true_16x102400_float32_logit_identity(self):
        d=load()
        self.assertEqual(d["status"],"PASS_FULL_102400_LOGIT_PAIRED_COMPARISON")
        a=d["paired_full_logits"]
        self.assertEqual(a["sample_count"],16)
        self.assertEqual(a["vocab_count"],102400)
        self.assertEqual(a["all_logits_compared"],1638400)
        self.assertEqual(a["count_exact_different_float32_values"],0)
        self.assertEqual(a["max_abs_logit_difference"],0.0)
        self.assertEqual(a["mean_abs_logit_difference"],0.0)
        self.assertTrue(a["all_greedy_token_ids_match"])
        self.assertEqual(len(a["cases"]),16)
        expected=[11,279,2238,31460,330,72672,185,1501,245,11,457,285,327,20171,889,68307]
        self.assertEqual([c["argmax_old32"] for c in a["cases"]],expected)
        self.assertEqual([c["argmax_new384"] for c in a["cases"]],expected)
        self.assertEqual([c["input_positions"] for c in a["cases"]],
            [15,16,18,20,22,22,22,23,24,24,24,25,25,25,25,25])
        for index,c in enumerate(a["cases"]):
            self.assertEqual(c["request"],index)
            self.assertEqual(c["max_logit_abs_delta"],0.0)
            self.assertEqual(c["mean_abs_logit_delta"],0.0)
            self.assertEqual(c["source32_full102400_sha256"],
                             c["source384_full102400_sha256"])

    def test_both_real_XOX_abi_binaries_and_all_raw_bytes_stay_pinned(self):
        d=load()
        self.assertEqual(d["build_32"]["sha256"],
            "361eced42d72d2a4b9be342188d20d6ff2e7f25a493a633aec1c82581420e7bb")
        self.assertEqual(d["build_384"]["sha256"],
            "166e5480acef41cc9d153a9479fb3f09de98a912ca52e43b19b540e80b980c1b")
        self.assertEqual(d["run_32"]["raw_logits_sha256"],LOG_SHA)
        self.assertEqual(d["run_384"]["raw_logits_sha256"],LOG_SHA)
        self.assertEqual(d["run_32"]["log_bytes"],6553886)
        self.assertEqual(d["run_384"]["log_bytes"],6553886)
        self.assertEqual(d["run_32"]["exit_code"],0)
        self.assertEqual(d["run_384"]["exit_code"],0)
        self.assertTrue(d["run_32"]["production_pids_unchanged"])
        self.assertTrue(d["run_384"]["production_pids_unchanged"])
        self.assertGreaterEqual(min(d["run_32"]["free_min"],d["run_384"]["free_min"]),35)

    def test_production_unmodified_and_NLL_scientific_limit(self):
        d=load()
        self.assertEqual(d["production_before"],d["production_after"])
        self.assertEqual(d["production_before"]["worker_pids"],{
            "beglin-baseline-q4g64":35481,
            "beglin-candidate-shared-up-l3-n6":35493})
        self.assertEqual(d["production_before"]["route"],
                         "beglin-candidate-shared-up-l3-n6")
        self.assertTrue(d["same_original_AF_weights_no_mutation"])
        self.assertFalse(d["release_modified"])
        self.assertFalse(d["production_route_modified"])
        self.assertFalse(d["production_promotion_allowed"])
        self.assertEqual(d["native_original_512_gold_NLL"],"NOT_MEASURED")
        self.assertEqual(d["schema"],"BEGLIN-JE01-ORIGINAL-AF-FULL102400-LOGITS-32-VS384/1")


if __name__=="__main__":
    unittest.main(verbosity=2)
