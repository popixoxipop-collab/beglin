#!/usr/bin/env python3
"""JE01 actual original27 256 WikiText TRAIN last-prompt FFN capture receipt audit.

This checks the signed, source-only EOE receipt for 682 real selected h2/SwiGLU
records. The remote 16 raw binary chunks remain on EOE D50 and are not
independently rehashed by this offline test. This is *not* a true Fisher,
Hessian, full-target NLL improvement, or independent holdout evaluation.
"""
import hashlib
import json
from pathlib import Path
import unittest

PATH = (Path(__file__).resolve().parents[1] / "docs/research" /
        "JE01_ORIGINAL27_REAL256_H2_SWIGLU_CAPTURE_20261122.json")
RECEIPT_SHA = "b911aef00007b1f9b44214996353f93c7218fe51c82daf91966e450ebba460b6"
AF_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
ROUTER_SHA = "62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb"
BINARY_SHA = "77fd53111cf8316c9d9186f21ace65bccaf3c6fb1aba54c43c698076d5dc914d"


def sealed():
    receipt = json.loads(PATH.read_text(encoding="utf-8"))
    claimed = receipt.pop("receipt_sha256", None)
    computed = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if not claimed == computed == RECEIPT_SHA:
        raise AssertionError("JE01_ACTUAL_256_CAPTURE_CANONICAL_RECEIPT_DRIFT")
    return receipt


class TrueOriginal256HiddenCapture(unittest.TestCase):
    def test_training_source_chunk_and_routing_parity(self):
        d = sealed()
        self.assertEqual(d["schema"], "beglin-je01-original27-256-train-routed-H2-and-SwiGLU-real-vectors/1")
        self.assertEqual(d["status"], "JE01_TRUE_ORIGINAL27_256TRAIN_682ROUTED_H2_SWIGLU_MODEL_FORWARD_PASS")
        self.assertEqual(d["total_unique_original_training_prompts"], 256)
        self.assertEqual(d["actual_original_routed_hidden_and_swiglu_records"], 682)
        self.assertEqual(d["true_model_inference_chunks"], 16)
        self.assertEqual(d["native_original_FFN_input_H2_dimension"], 2048)
        self.assertEqual(d["native_original_FFN_down_SwiGLU_dimension"], 1408)
        self.assertEqual(d["original_27layer_MLX_native_binary_sha256"], BINARY_SHA)
        self.assertEqual(d["original_AF4_full_checkpoint_sha256"], AF_SHA)
        self.assertEqual(d["source_original_router256_receipt_sha256"], ROUTER_SHA)
        self.assertEqual(d["real16_control_max_abs_h2"], 0)
        self.assertEqual(d["real16_control_max_abs_router_weight"], 0)
        self.assertEqual(len(d["per_chunk"]), 16)
        self.assertEqual([c["chunk"] for c in d["per_chunk"]], list(range(16)))
        self.assertEqual([c["source_case_first"] for c in d["per_chunk"]], list(range(0, 256, 16)))
        self.assertTrue(all(c["cases"] == 16 and c["excluded_heldout_and_ood_labels"]
                            and c["min_memory_free_percent"] >= 35
                            and len(c["real_binary_sha256"]) == 64
                            for c in d["per_chunk"]))
        self.assertEqual(sum(c["real_routed_selected_records"] for c in d["per_chunk"]), 682)
        actual = d["per_expert_actual_routed_last_prompt_count"]
        expected = d["per_expert_original_router256_expected_count"]
        self.assertEqual(actual, expected)
        self.assertEqual(sorted(actual), sorted(("%d:%d" % tuple(x) for x in d["selected_eight_frozen_experts"])))
        self.assertEqual(sum(actual.values()), 682)
        self.assertEqual(actual, {"12:13": 81, "13:22": 79, "13:28": 87,
                                  "14:63": 80, "20:17": 86, "21:61": 97,
                                  "24:26": 91, "24:60": 81})

    def test_no_claim_of_loss_fisher_or_heldout(self):
        d = sealed()
        self.assertFalse(d["independent_holdout_OOD_gold_labels_accessed"])
        self.assertFalse(d["production_modified"])
        self.assertEqual(d["gold_full256_NLL"], "NOT_MEASURED_THIS_RUN")
        self.assertEqual(d["gold_token_loss_gradient"], "NOT_MEASURED")
        self.assertEqual(d["true_Fisher"], "NOT_MEASURED")
        self.assertEqual(d["true_Hessian"], "NOT_MEASURED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
