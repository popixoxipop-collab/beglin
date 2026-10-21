#!/usr/bin/env python3
"""Read-only independent QA of actual Apple BEGLIN 196-tensor mixed-Q receipt."""
from __future__ import annotations
import hashlib
import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_native_mixed_evidence_audit as audit

class NativeMixedQEvidenceQA(unittest.TestCase):
    def test_source_only_masks_budget_and_seeded_random(self):
        src,proof,policies=audit.role_policy_details()
        self.assertEqual(src["source_tensors"],196)
        self.assertEqual(proof["source_BEGLIN_original_sha256"],audit.EXPECTED_ORIGINAL_ENGINE_SHA)
        self.assertEqual(len(policies),2)
        for name in ("proxy","random"):
            p=policies[name]
            self.assertEqual(p["weights_8bit"],28)
            self.assertEqual(p["weights_4bit"],168)
            self.assertEqual([len(p["selected_layers"][r]) for r in audit.ROLE_ORDER],[4]*7)
        self.assertEqual(policies["proxy"]["logical_projection_bytes"],
                         policies["random"]["logical_projection_bytes"])
    def test_source_weight_proxy_decreases_by_construction(self):
        _,_,p=audit.role_policy_details()
        self.assertLess(p["proxy"]["sum_squared_normalized_weight_error_proxy"],
                        p["random"]["sum_squared_normalized_weight_error_proxy"])
    def test_native_logits_and_same_two_prompt_windows(self):
        x=audit.compare()
        self.assertEqual(set(x),set(audit.MODELS))
        for entry in x.values():
            self.assertEqual(entry["PPL_tokens"],21)
            self.assertEqual(entry["top1_id"],9707)
        self.assertGreater(x["q4"]["KL_F32_to_policy_nats"],x["n8"]["KL_F32_to_policy_nats"])
    def test_failed_proxy_quality_is_preserved_not_silently_promoted(self):
        x=audit.compare()
        self.assertGreater(x["proxy"]["NLL_nats_per_token"],x["random"]["NLL_nats_per_token"])
        self.assertGreater(x["proxy"]["NLL_nats_per_token"],x["q4"]["NLL_nats_per_token"])
        self.assertGreater(x["proxy"]["bench_best_of_3_one_token_ms"],
                           x["q4"]["bench_best_of_3_one_token_ms"])
    def test_existing_receipt_fail_closed_to_overwrite(self):
        self.assertTrue(audit.OUT.exists())
        with self.assertRaises(FileExistsError):
            audit.run()
if __name__=="__main__":
    unittest.main(verbosity=2)
