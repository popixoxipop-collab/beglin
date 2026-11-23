#!/usr/bin/env python3
"""JE01 actual original DeepSeek AF 256 full gold target NLL evidence receipt guards.

The native raw 16 binary full-token NLL files remain on the authorized XOX
host and were independently fully reopened and checked there. Offline tests
pin the canonical original pretrained baseline and independent audit records,
without presenting baseline uncertainty as Bayesian gain.
"""
import hashlib,json,math,pathlib,unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]/"docs/research"
BASE=ROOT/"JE01_ORIGINAL_AF_REAL256_FULL_TARGET_GOLD_NLL_20261118.json"
AUDIT=ROOT/"JE01_ORIGINAL_AF_REAL256_FULL_TARGET_INDEPENDENT_AUDIT_20261118.json"
BASE_SHA="f71a4c4397cbadf938c7a9631ba8306547b31e33028dd9a54b556a1a758e3f41"
AUDIT_SHA="d00a56221a12a7456b39321f870e9d4a119f3425215a78142cd36c78b7eba2fa"

def load(source,expected):
    d=json.loads(source.read_text())
    claimed=d.pop("receipt_sha256")
    actual=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
               allow_nan=False).encode()).hexdigest()
    if claimed!=expected or actual!=expected:
        raise AssertionError("JE01_FULL_TARGET_REAL_NLL_CANONICAL_SHA_CHANGED")
    return d

class OriginalRealFullTargetTokenNLL(unittest.TestCase):
    def test_real_256_original_pretrained_teacher_forced_nll_complete(self):
        d=load(BASE,BASE_SHA)
        self.assertEqual(d["schema"],"beglin-je01-original-AF-train256-native-fullgold-teacherforce-NLL/1")
        self.assertEqual(d["status"],"REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_CAL256_PASS")
        self.assertTrue(d["actual_original_pretrained_AF_model"])
        self.assertEqual(d["n_calibration_train_cases"],256)
        self.assertEqual(d["gold_full_target_tokens_scored"],11024)
        self.assertEqual(len(d["per_case_real_NLL"]),256)
        self.assertEqual(len(d["per_chunk_native_output_sha256"]),16)
        self.assertEqual(sum(x["tokens"] for x in d["per_case_real_NLL"]),11024)
        self.assertEqual([x["row"] for x in d["per_case_real_NLL"]],list(range(256)))
        self.assertEqual(len({x["source_id"] for x in d["per_case_real_NLL"]}),256)
        self.assertAlmostEqual(d["mean_gold_token_NLL"],2.46035653719724,places=13)
        true_mean=math.fsum(x["nll_sum"] for x in d["per_case_real_NLL"])/11024
        self.assertAlmostEqual(d["mean_gold_token_NLL"],true_mean,places=12)
        per_case=math.fsum(x["nll_mean"] for x in d["per_case_real_NLL"])/256
        self.assertAlmostEqual(d["per_example_mean_full_target_NLL"],per_case,places=12)
        self.assertLess(d["max_abs_first_gold_NLL_disagreement_from_original_256_previous_run"],1e-5)
        self.assertTrue(all(math.isfinite(x["nll_sum"]) and x["nll_sum"]>=0 for x in d["per_case_real_NLL"]))

    def test_independent_all_11024_native_gold_records_shas_and_case_bootstrap(self):
        d=load(AUDIT,AUDIT_SHA)
        src=load(BASE,BASE_SHA)
        self.assertEqual(d["status"],"INDEPENDENT_256_NATIVE_REAL_FULL_TARGET_GOLD_NLL_PASS")
        self.assertEqual(d["full_source_native_result_receipt_sha256"],BASE_SHA)
        self.assertEqual(d["actually_reopened_16_native_GPU_result_files"],16)
        self.assertEqual(d["all_11024_real_gold_tokens_independently_checked"],11024)
        self.assertEqual(d["all_256_original_train_prompts_checked"],256)
        self.assertAlmostEqual(d["recomputed_weighted_mean_full_target_NLL"],src["mean_gold_token_NLL"],places=13)
        self.assertAlmostEqual(d["recomputed_mean_per_train_prompt_full_target_NLL"],src["per_example_mean_full_target_NLL"],places=13)
        self.assertLess(d["max_abs_first_gold_baseline_NLL_discrepancy"],1e-5)
        self.assertEqual(len(d["per_chunk_native_original_GPU_NLL_SHA"]),16)
        self.assertEqual([x["data_sha256"] for x in d["per_chunk_native_original_GPU_NLL_SHA"]],
                         [x["result_sha256"] for x in src["per_chunk_native_output_sha256"]])
        self.assertEqual(sum(x["actual_gold_tokens_independently_checked"] for x in d["per_chunk_native_original_GPU_NLL_SHA"]),11024)
        self.assertEqual(d["bootstrap"]["n_case_resamples"],20000)
        self.assertEqual(d["bootstrap"]["bootstrap_seed"],20261118)
        self.assertTrue(d["bootstrap"]["not_paired_precision_allocation_improvement"])
        lo,hi=d["bootstrap"]["real_source_train_mean_gold_token_NLL_CI_95"]
        self.assertAlmostEqual(lo,2.3658130403184066,places=12)
        self.assertAlmostEqual(hi,2.551722621915675,places=12)
        self.assertTrue(lo<src["mean_gold_token_NLL"]<hi)

    def test_only_training_baseline_no_promotion_no_alternative_model(self):
        d=load(BASE,BASE_SHA);a=load(AUDIT,AUDIT_SHA)
        self.assertEqual(d["calibration_original256_router_receipt_sha256"],
                         "62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb")
        self.assertEqual(d["source_gold_sha256"],
                         "355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7")
        self.assertEqual(d["native_model_binary_sha256"],
                         "de6f1a97aef17ddae42d1729282c38e2b7924f6d841daa3576dc47edeebbbba9")
        self.assertEqual(d["production_before"],d["production_after"])
        self.assertEqual(set(d["production_before"].values()),{35481,35493})
        self.assertTrue(d["training_only"])
        self.assertFalse(d["heldout_gold_labels_used"])
        self.assertFalse(d["bayesian_expert_quantization_changed"])
        self.assertFalse(d["production_checkpoint_or_release_modified"])
        self.assertFalse(d["production_promotion_allowed"])
        self.assertEqual(d["trained_alt_bits_comparison"],"NOT_MEASURED")
        self.assertEqual(d["same_source_128_or_OOD_128_NLL"],"NOT_MEASURED")
        self.assertFalse(a["production_weight_change"])
        self.assertFalse(a["production_promotion_allowed"])
        self.assertTrue(a["baseline_CI_is_not_a_precision_improvement_CI"])
        self.assertEqual(a["native_Bayesian_expert_bit_quality_alternatives"],"NOT_MEASURED")
        self.assertTrue(a["no_same_source_or_ood_gold_labels_used"])

if __name__=="__main__":
    unittest.main(verbosity=2)
