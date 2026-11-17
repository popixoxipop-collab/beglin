#!/usr/bin/env python3
"""JE01 independent real BF16 Expert FFN proxy receipts; no numpy/model run."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]/"docs"/"research"
TRAIN="JE01_ACTUAL_BF16_EXPERT_8_FFN_PRECISION_PROXY_20261117.json"
TEST="JE01_REAL_BF16_EXPERT_8_EQUALBUDGET_HELDOUT_20261117.json"


def verified_json(filename,expected_sha):
    doc=json.loads((ROOT/filename).read_text())
    selfsha=doc.pop("receipt_sha256")
    realsha=hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(",",":"),
        allow_nan=False).encode()).hexdigest()
    if selfsha!=expected_sha or realsha!=expected_sha:
        raise AssertionError("JE01_EVIDENCE_CANONICAL_SHA_MISMATCH")
    return doc


class RealBF16ProxyReceipts(unittest.TestCase):
    def test_train_is_real_weights_not_native_nll(self):
        t=verified_json(TRAIN,"555e11d05ca3228a4cb8a828d6a2a3eef153ff5e4f21b4aa547f5820fdecaf2d")
        self.assertEqual(t["status"],"REAL_PRETRAINED_WEIGHTS_SYNTHETIC_ACTIVATIONS_ONLY")
        self.assertEqual(t["original_bf16_checkpoint_identity_sha256"],
            "36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc")
        self.assertEqual(t["num_synthetic_inputs"],16)
        self.assertEqual(t["layer"],11)
        self.assertEqual(len(t["results"]),8)
        self.assertFalse(t["actual_real_router_hidden_states"])
        self.assertEqual(t["full_pretrained_model_forward"],"NOT_RUN")
        self.assertEqual(t["per_prompt_gold_token_nll"],"NOT_MEASURED")
        self.assertFalse(t["production_promotion_allowed"])
        for r in t["results"]:
            vals=[r["bit_proxy_loss"][str(n)]["relative_output_mse"] for n in (4,6,8)]
            self.assertGreater(vals[0],vals[1])
            self.assertGreater(vals[1],vals[2])
            self.assertEqual(set(r["consumed_bf16_tensor_sha256"]),{"gate","up","down"})

    def test_heldout_is_new_seed_and_negative_vs_uniform6(self):
        t=verified_json(TEST,"441b08e520a512f10377fb801aab0ed1ba3d24852f8280abf524be42cca5c64a")
        self.assertEqual(t["training_observation_receipt_sha256"],
            "555e11d05ca3228a4cb8a828d6a2a3eef153ff5e4f21b4aa547f5820fdecaf2d")
        self.assertNotEqual(t["heldout_uses_new_seed"],t["training_uses_only_seed"])
        self.assertEqual(t["heldout_inputs_per_expert"],16)
        self.assertEqual(t["experts_bits4_selected_without_holdout"],[0,15,39,55])
        self.assertEqual(t["experts_bits8_selected_without_holdout"],[7,23,31,47])
        self.assertEqual(t["heldout_mix_rank_among_70"],1)
        self.assertEqual(t["proxy_total_bytes_equal_for_mix4x4_4x8_and_uniform6"],60555264)
        v=t["observed_heldout_avg_relative_mse"]
        self.assertLess(v["selected_four4_four8"],v["all70_random_equal_byte_mix_mean"])
        self.assertLess(v["uniform6"],v["selected_four4_four8"])
        self.assertGreater(t["selected_vs_uniform6_mse_ratio"],8)
        self.assertGreater(t["selected_vs_random_mix_mse_reduction_fraction"],.08)
        self.assertEqual(t["gold_token_nll"],"NOT_MEASURED")
        self.assertEqual(t["full_pretrained_model_forward"],"NOT_RUN")
        self.assertFalse(t["actual_model_router_hidden_states"])
        self.assertEqual(t["native_beglin_quantization_parity"],"NOT_TESTED")
        self.assertFalse(t["production_promotion_allowed"])
        self.assertIn("NOT_NATIVE_PACKED",t["budget_byte_formula"])
        self.assertEqual(len(t["heldout_expert_results"]),8)


    def test_all_1107_equal_bit_proxy_allocations_true_optimum(self):
        import itertools
        t=verified_json(TRAIN,"555e11d05ca3228a4cb8a828d6a2a3eef153ff5e4f21b4aa547f5820fdecaf2d")
        h=verified_json(TEST,"441b08e520a512f10377fb801aab0ed1ba3d24852f8280abf524be42cca5c64a")
        doc=json.loads((ROOT/"JE01_REAL_BF16_EXPERT8_FULLGRID_6BIT_DECISION_20261117.json").read_text())
        ids=doc["experts"]
        self.assertEqual(ids,[0,7,15,23,31,39,47,55])
        tr={r["expert_id"]:r["bit_proxy_loss"] for r in t["results"]}
        te={r["expert_id"]:r["heldout_relative_output_mse"] for r in h["heldout_expert_results"]}
        seen=0
        opt_t=(float("inf"),None)
        opt_h=(float("inf"),None)
        nonuniform_min_t=float("inf")
        nonuniform_min_h=float("inf")
        six=None
        for bits in itertools.product((4,6,8), repeat=8):
            if sum(bits)!=48:
                continue
            seen+=1
            lt=sum(tr[e][str(b)]["relative_output_mse"] for e,b in zip(ids,bits))/8
            lh=sum(te[e][str(b)] for e,b in zip(ids,bits))/8
            if lt<opt_t[0]:opt_t=(lt,bits)
            if lh<opt_h[0]:opt_h=(lh,bits)
            if all(b==6 for b in bits):six=(lt,lh)
            else:
                nonuniform_min_t=min(nonuniform_min_t,lt)
                nonuniform_min_h=min(nonuniform_min_h,lh)
        self.assertEqual(seen,1107)
        self.assertEqual(opt_t[1],(6,)*8)
        self.assertEqual(opt_h[1],(6,)*8)
        self.assertTrue(doc["selected_by_training_uniform6"])
        self.assertTrue(doc["heldout_optimal_uniform6"])
        self.assertEqual(doc["state_count_with_equal_budget"],1107)
        self.assertAlmostEqual(doc["uniform6"]["trainMse"],six[0],places=13)
        self.assertAlmostEqual(doc["uniform6"]["heldoutMse"],six[1],places=13)
        self.assertAlmostEqual(doc["min_non_uniform_heldout_mse_ratio_to_uniform6"],
                               nonuniform_min_h/six[1],places=12)
        self.assertGreater(nonuniform_min_t/six[0],2)
        self.assertGreater(nonuniform_min_h/six[1],2)
        self.assertEqual(doc["gold_token_nll"],"NOT_MEASURED")
        self.assertEqual(doc["pretrained_full_forward"],"NOT_RUN")
        self.assertFalse(doc["production_promotion_allowed"])


if __name__=="__main__":
    unittest.main(verbosity=2)
