#!/usr/bin/env python3
"""JE01 real 512 prompt Native 32-position infeasibility + canonical receipt tests."""
import json
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import je01_data_freeze as df
import je01_native_token_feasibility as f


class NativeContextFeasibilityTests(unittest.TestCase):
    def test_replay_original_source_text_prompt_hash(self):
        wiki="Alpha "+("research and review multiple calibrated sensors "*6)+"."
        example={"text":wiki}
        prompt=f.prompt_for(df,"calibration",example)
        self.assertTrue(prompt.startswith("Alpha"))
        self.assertLess(len(prompt),len(wiki))
        self.assertEqual(f.prompt_for(df,"ood_holdout",
                         {"ctx":"  A person  sees   the road "}),
                         "A person sees the road")

    def test_original_512_real_token_feasibility_receipt(self):
        path=ROOT/"docs"/"research"/"JE01_ACTUAL_512_NATIVE_TOKEN_FEASIBILITY_20261117.json"
        report=json.loads(path.read_text())
        seal=report.pop("receipt_sha256")
        self.assertEqual(seal,"b0da7e361afbad5faae69f86d08651a21e5c6f95ffa15e7eab3bec51a7401064")
        self.assertEqual(seal,df.digest(report))
        self.assertEqual(report["schema"],f.SCHEMA)
        self.assertEqual(report["dataset_manifest_sha256"],f.EXPECTED_MANIFEST)
        self.assertEqual(report["tokenizer_sha256"],f.EXPECTED_TOKENIZER)
        self.assertEqual(report["native_engine_max_positions"],32)
        self.assertEqual(report["engine_bos_and_special_token_parity"],"NOT_VERIFIED")
        self.assertEqual(report["full_original_model_inference"],"NOT_RUN")
        self.assertFalse(report["gold_labels_used"])
        self.assertFalse(report["production_promotion_allowed"])
        groups=report["cohorts"]
        self.assertEqual({name:(r["count"],r["raw_plus_one_bos_plus_one_gold_fits_32"],
                                r["cannot_fit_one_gold_with_bos"])
                          for name,r in groups.items()},
                         {"calibration":(256,24,232),
                          "same_source_holdout":(128,6,122),
                          "ood_holdout":(128,24,104)})
        self.assertEqual(sum(v["count"] for v in groups.values()),512)
        self.assertEqual(sum(v["raw_plus_one_bos_plus_one_gold_fits_32"]
                             for v in groups.values()),54)
        self.assertEqual(sum(v["cannot_fit_one_gold_with_bos"]
                             for v in groups.values()),458)
        self.assertTrue(all(v["count"]==v["raw_plus_one_bos_plus_one_gold_fits_32"]+
                               v["cannot_fit_one_gold_with_bos"]
                            for v in groups.values()))

if __name__=="__main__":
    unittest.main(verbosity=2)
