"""Portable, confidential-free BEGLIN Qv6 public contract regression."""
import hashlib,json,math,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
C=ROOT/"configs/research"
D=ROOT/"docs/research"
PRE=C/"BEGLIN_QV6_SAMPLE_SCALING_PRELOCK_20261022.json"
FREEZE=D/"BEGLIN_QV6_SPLITS_PREMODEL_FREEZE_20261022.json"
SOURCE=D/"BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json"
POLICY=D/"BEGLIN_QV6_NESTED_64_128_SOURCE_POLICY_20261022.json"
BF16=D/"BEGLIN_QV6_ORIGINAL_BF16_OPENBOOKQA48_REFERENCE_20261022.json"
PUBLIC=D/"BEGLIN_QV6_CROSSCORPUS_NATIVE_INDEPENDENT_PUBLIC_20261022.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class Qv6PortableContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock=json.loads(PRE.read_text())
        cls.freeze=json.loads(FREEZE.read_text())
        cls.scores=json.loads(SOURCE.read_text())
        cls.policy=json.loads(POLICY.read_text())

    def test_policy_preregistered_before_calibration(self):
        self.assertEqual(sha(PRE),"2e35764878191717ef7b84ed546e72e2a1c513523ecb17d24b54489bd29f319b")
        self.assertEqual(self.lock["status"],"PRELOCK_BEFORE_128_SOURCE_EXTENSION_AND_NEW_CROSS_DATASET_EVALUATION")

    def test_exact_model_and_dataset_revisions(self):
        self.assertEqual(self.lock["official_BF16_revision"],"989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
        self.assertEqual(self.lock["source_dataset_revision"],self.freeze["ARC_original_rev"])
        self.assertEqual(self.lock["cross_dataset_revision"],self.freeze["OpenBookQA_rev"])

    def test_frozen_24_64_128_and_48(self):
        self.assertEqual(self.lock["nested_source_sample_sizes"],[24,64,128])
        self.assertEqual(self.freeze["status"],"FROZEN_SOURCE_NESTED_AND_CROSSCORPUS_BEFORE_ANY_NEW_FISHER_OR_EVAL")
        self.assertTrue(self.freeze["source24_prefix_exact"])
        self.assertEqual(self.freeze["cross_prompt_length_range"][1]<=208,True)

    def test_196_fisher_source_score_per_calibration(self):
        roles={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
        for n in ("24","64","128"):
            o=self.scores["quality_upgrade_by_role_layer_per_size"][n]
            self.assertEqual(set(o),roles)
            self.assertEqual(sum(len(v) for v in o.values()),196)
            self.assertTrue(all(math.isfinite(x) for v in o.values() for x in v))

    def test_104_new_gradients_and_prior_24_reused(self):
        self.assertEqual(self.scores["actually_executed_new_BF16_backward_items"],104)
        self.assertEqual(self.scores["reused_old24"],24)
        self.assertEqual(self.scores["source_gradients"],128)
        self.assertTrue(self.scores["no_validation_label_or_logits_accessed"])

    def test_split_half_rank_improves_with_more_samples_in_this_run(self):
        r=self.scores["nested_split_stats"]
        self.assertGreater(r["128"]["odd_even_sample_Fisher_rank_spearman"],r["24"]["odd_even_sample_Fisher_rank_spearman"])
        self.assertTrue(all(-1<=r[n]["odd_even_sample_Fisher_rank_spearman"]<=1 for n in ("24","64","128")))

    def test_source_stability_and_mask_overlap_recorded(self):
        self.assertEqual(self.policy["source_nested_pairwise_rank"],self.scores["nested_pairwise_rank_stability"])
        self.assertEqual(set(self.policy["constrained_selected_overlap"]),{"24_64","24_128","64_128"})
        self.assertLessEqual(self.policy["constrained_selected_overlap"]["64_128"]["policy_28_overlap"],28)

    def test_exact_28_native_kernel_tensor_masks(self):
        for n in ("64","128"):
            p=C/f"BEGLIN_QV6_FISHER_{n}N8_SOURCE_POLICY_20261022.txt"
            rows=[tuple(x.split()) for x in p.read_text().splitlines()]
            self.assertEqual(len(rows),28)
            self.assertEqual(len(set(rows)),28)
            self.assertEqual(sha(p),self.policy["selected_policies"][n]["mask_sha256"])

    def test_source_only_milp_feasibility(self):
        self.assertEqual(self.policy["status"],"FROZEN_64_128_POLICIES_BEFORE_NEW_OPENBOOKQA_VALIDATION")
        for n in ("64","128"):
            p=self.policy["selected_policies"][n]
            self.assertLess(p["source_only_gap"],1e-7)
            self.assertLessEqual(p["estimated_native_projection_bytes"],830570496)
            self.assertLessEqual(p["predicted_single_token_serial_ms"],100)
            self.assertLessEqual(p["isolated_async_stage_total_cost_ms"],50.00001)

    def test_external_reference_is_original_bf16(self):
        data=json.loads(BF16.read_text())
        self.assertEqual(data["status"],"PASS_UNMODIFIED_ORIGINAL_BF16_MODEL_OODISH_SCIENCE_CROSSCORPUS")
        self.assertEqual(data["total"],48)
        self.assertFalse(data["weights_mutated"])

    def test_auto_promotion_disabled(self):
        self.assertFalse(self.lock["auto_promotion"])
        self.assertFalse(self.policy["production_write_allowed"])
        self.assertFalse(self.policy["auto_precision_promotion"])

    def test_native_quality_source_if_published(self):
        if not PUBLIC.exists():
            self.skipTest("Qv6 native four-policy independent audit not yet completed")
        data=json.loads(PUBLIC.read_text())
        self.assertEqual(data["status"],"PASS_4_POLICY_48_REAL_APPLE_C_NATIVE_EXECUTIONS_AND_BF16_SOURCE_AUDIT")
        self.assertEqual(set(data["native_summary"]),{"q4","v5_fisher_24","v6_fisher_64","v6_fisher_128"})
        self.assertFalse(data["generalization_quality_superiority_proven"])
        self.assertFalse(data["automated_live_promotion"])
        for v in data["native_summary"].values():
            self.assertIn("correct_answer_NLL",v)
            self.assertGreaterEqual(v["correct"],0)
            self.assertLessEqual(v["correct"],48)
            self.assertTrue(math.isfinite(v["correct_answer_NLL"]))

if __name__=="__main__":
    unittest.main(verbosity=2)
