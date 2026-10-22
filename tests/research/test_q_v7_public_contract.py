"""Standalone Qv7 public proof contracts. stdlib-only for GitHub CI."""
import hashlib,json,math,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
P=ROOT/"configs/research"
D=ROOT/"docs/research"
PRE=P/"BEGLIN_QV7_CROSSFAMILY_CALIBRATION_PRELOCK_20261022.json"
LOCK=D/"BEGLIN_QV7_CROSSFAMILY_TINYLLAMA_INPUT_FREEZE_20261022.json"
COST=D/"BEGLIN_QV7_LLAMA_NATIVE_SEVEN_ROLE_COSTS_20261022.json"
SOURCE=D/"BEGLIN_QV7_TINYLLAMA_SOURCE_154_FISHER_PUBLIC_20261022.json"
POL=D/"BEGLIN_QV7_TINYLLAMA_154_COST_FISHER_POLICY_20261022.json"
MASK=P/"BEGLIN_QV7_TINYLLAMA_22N8_FISHER_FROZEN_POLICY_20261022.txt"
REF=D/"BEGLIN_QV7_TINYLLAMA_ORIGINAL_BF16_TEST48_PUBLIC.json"
FAMILY=D/"BEGLIN_QV7_TINYLLAMA_FISHER_NATIVE48_AUDIT_PUBLIC_20261022.json"
SEEDS=D/"BEGLIN_QV7_QWEN64_SEED_REPARTITION_EVIDENCE_20261022.json"
ABBA=D/"BEGLIN_QV7_TINYLLAMA_CPU_ABBA_TRACE_PUBLIC_20261022.json"
TYPES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Qv7StandalonePublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pre=json.loads(PRE.read_text())
        cls.lock=json.loads(LOCK.read_text())
        cls.cost=json.loads(COST.read_text())
        cls.source=json.loads(SOURCE.read_text())
        cls.policy=json.loads(POL.read_text())
        cls.ref=json.loads(REF.read_text())
        cls.family=json.loads(FAMILY.read_text())
        cls.seed=json.loads(SEEDS.read_text())
    def test_model_preregistered_and_unchanged(self):
        self.assertEqual(sha(PRE),"1b5fa904cf11f25292720efc75f658fc10cedc8923236b4d5b0ae7e70d3e88c6")
        self.assertEqual(self.pre["replication_revision"],self.ref["official_revision"])
        self.assertFalse(self.pre["no_main_changes"] is False)
    def test_family_differs_from_qwen(self):
        self.assertEqual(self.pre["replication_HF_model_type"],"llama")
        self.assertEqual(self.pre["layer_count"],22)
        self.assertEqual(self.pre["total_projections"],154)
        self.assertEqual(self.pre["target_n8_count"],22)
        self.assertEqual(self.pre["target_q4_count"],132)
    def test_freeze_64_train_48_unseen_test(self):
        self.assertEqual(self.lock["status"],"FROZEN_LLAMA_MODEL_SOURCE64_AND_UNSEEN_TEST48_BEFORE_CALIBRATION")
        self.assertEqual(self.lock["source_count"],64)
        self.assertEqual(self.lock["held_count"],48)
        self.assertEqual(self.lock["source_private_sha256"],"59c1d22197ba83ac3769049c01de4ae98414c678066ef6faa48fa2a0859bbe41")
        self.assertEqual(self.lock["held_private_sha256"],"8807002e79afadcb2d150b32dffdefa204afb0d70678375a608b7900f18d0ab0")
    def test_real_native_seven_role_costs(self):
        self.assertEqual(self.cost["status"],"SEVEN_REAL_SOURCE_BF16_LLAMA_ROLE_NATIVE_KERNELS_PROFILLED")
        rows={x["role"]:x for x in self.cost["role_kernel_cost_measured"]}
        self.assertEqual(set(rows),set(TYPES))
        for r in TYPES:
            x=rows[r]
            self.assertGreater(x["q4_with_f32_scales_bytes"],0)
            self.assertGreater(x["n8_with_f32_scales_bytes"],x["q4_with_f32_scales_bytes"])
            for t in ("q4_1t","q4_6t","n8_repacked"):
                self.assertGreater(x["real_original_native_kernel_costs"][t]["median_ms"],0)
    def test_source_real_original_bf16_154x64(self):
        self.assertEqual(self.source["status"],"SOURCE_ONLY_REAL_LLAMA_ORIGINAL_BF16_64_GRAD_AND_NATIVE_C_QUANT")
        self.assertEqual(self.source["source_cases"],64)
        self.assertEqual(self.source["original_model_projection_count"],154)
        self.assertEqual(sum(len(v) for v in self.source["quality_upgrade_by_role_layer"].values()),154)
        self.assertTrue(self.source["forward_always_original_unquantized_BF16"])
        self.assertTrue(self.source["crosscorpus_labels_or_logits_unopened"])
    def test_fisher_odd_even_rank_source(self):
        x=self.source["source_split_half_odd_even_spearman"]
        self.assertTrue(math.isfinite(x))
        self.assertLessEqual(abs(x),1)
    def test_exact_new_architecture_22n8_fisher_MILP(self):
        self.assertEqual(self.policy["status"],"SOURCE_ONLY_EXACT_LLAMA_22N8_MASK_FROZEN_BEFORE_CROSSCORPUS_NATIVE_QUALITY")
        self.assertEqual(self.policy["solver_status"],0)
        self.assertLessEqual(self.policy["mip_gap"],1e-7)
        self.assertEqual(self.policy["n8_count"],22)
        self.assertEqual(self.policy["q4_count"],132)
        self.assertEqual(self.policy["source_154_public_sha256"],sha(SOURCE))
        self.assertEqual(self.policy["source_original_Llama_seven_native_costs_sha256"],sha(COST))
    def test_new_architecture_memory_and_cost_caps(self):
        self.assertGreater(self.policy["selected_projection_bytes"],self.policy["baseline_q4_projection_bytes"])
        self.assertLessEqual(self.policy["selected_projection_bytes"],self.policy["rolebalanced_22_upgrade_memory_cap_bytes"])
        self.assertLessEqual(self.policy["selected_projection_cost_ms"],self.policy["source_measured_projection_cost_budget_ms"])
    def test_mask_SHA_and_unique22(self):
        rows=[tuple(x.split()) for x in MASK.read_text().splitlines()]
        self.assertEqual(len(rows),22)
        self.assertEqual(len(set(rows)),22)
        self.assertEqual(sha(MASK),self.policy["mask_sha256"])
        for r,n in rows:
            self.assertIn(r,TYPES)
            self.assertTrue(n.isdigit())
            self.assertTrue(0<=int(n)<22)
    def test_original_true_BF16_reference_independent(self):
        self.assertEqual(self.ref["status"],"ACTUAL_ORIGINAL_LLAMA_FAMILY_BF16_UNMODIFIED_MODEL_COMPLETE")
        self.assertEqual(self.ref["case_count"],48)
        self.assertFalse(self.ref["no_original_weight_changes"] is False)
    def test_both_real_BEGLIN_native_Llama_Q4_n8(self):
        self.assertEqual(self.family["status"],"PASS_SOURCE_ONLY_154x64_FISHER_MILP_AND_TWO_NATIVE_LLAMACPU_48")
        self.assertEqual(set(self.family["actual_native_summary"]),{"q4","fisher_22n8"})
        self.assertEqual(self.family["actual_native_summary"]["q4"]["N"],48)
        self.assertEqual(self.family["actual_native_summary"]["fisher_22n8"]["N"],48)
        self.assertEqual(self.family["original_model_revision"],self.ref["official_revision"])
        self.assertFalse(self.family["cross_family_mixed_accuracy_generalization_established"])
    def test_negative_quality_preserved(self):
        self.assertFalse(self.family["fisher_quality_gain_proven"])
        self.assertGreater(self.family["Fisher_minus_Q4_mean_NLL"],0)
        self.assertEqual(self.family["Fisher_correct_delta_vs_Q4"],-1)
        self.assertTrue(self.family["single_test_case_prior_Q4_smoke_consumption_declared"])
    def test_qwen_seed_pairs_honest_about_independence(self):
        self.assertEqual(self.seed["status"],"SOURCE_ONLY_QWEN_BF16_3SEED_2DISJOINT_HALVES_EACH_6_MILP_OPTIMAL")
        self.assertEqual(len(self.seed["3_seed_pairwise"]),3)
        self.assertIn("NOT 6 statistically independent",self.seed["independence_warning"])
        for name,r in self.seed["3_seed_pairwise"].items():
            self.assertTrue(r["fold_A_B_disjoint_source_64_each"])
            self.assertLessEqual(r["selected_mask_intersection_size"],28)
    def test_no_operational_promotion(self):
        self.assertFalse(self.pre["no_automatic_GPU_or_dynamic_bit_allocation"] is False)
        self.assertFalse(self.policy["production_promotion"])
        self.assertTrue(self.family["live_operator_and_main_unchanged"])
    def test_ABBA_only_if_published(self):
        if not ABBA.is_file():self.skipTest("native paired CPU trace not complete yet")
        a=json.loads(ABBA.read_text())
        self.assertEqual(a["actual_native_timed_processes"],8)
        self.assertEqual(a["actual_timed_token_count"],1536)
        self.assertEqual(a["preregistration_sha256"],"5603f4044383ad31928291caabd0a90df6d6a9a1ba580fba5015e3f40eced175")
        self.assertTrue(a["operational_runtime_changed"] is False)
        self.assertIn(a["performance_admission_verdict"],("EXPLORATORY_CPU_ABBA_ONLY","INCONCLUSIVE_LOAD_CONTENDED"))
if __name__=="__main__":
    unittest.main(verbosity=2)
