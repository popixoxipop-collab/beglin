"""Qv8 portable public SHA / source-only / native 48 research contracts.

Pure stdlib; GitHub CI never imports private data, model, PyTorch or native
Apple C toolchain. Private full source and C log audits run separately on EOE.
"""
import hashlib,json,math,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
C=ROOT/"configs/research"
D=ROOT/"docs/research"
PRE=C/"BEGLIN_QV8_JOINT_SIGNED_SOURCE_PRELOCK_20261022.json"
LOCK=D/"BEGLIN_QV8_SOURCE_AND_HELLASWAG_PREMODEL_FREEZE_20261022.json"
SRC=D/"BEGLIN_QV8_BF16_SIGNED_CROSS_TENSOR_SOURCE_PUBLIC_20261022.json"
PLAN=D/"BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_OPTIMIZER_20261022.json"
MASK=C/"BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
ERR=D/"BEGLIN_QV8_SOURCE_START_LABEL_ERRATUM_20261022.json"
AUD=D/"BEGLIN_QV8_SOURCE_JOINT_SIGNED_INDEPENDENT_PUBLIC_20261022.json"
REF=D/"BEGLIN_QV8_ORIGINAL_BF16_HELLASWAG48_REFERENCE_20261022.json"
NATIVE=D/"BEGLIN_QV8_NARRATIVE48_NATIVE_INDEPENDENT_PUBLIC_20261022.json"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class Qv8StandalonePublic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=json.loads(PRE.read_text())
        cls.l=json.loads(LOCK.read_text())
        cls.s=json.loads(SRC.read_text())
        cls.p=json.loads(PLAN.read_text())

    def test_original_source_preregistration(self):
        self.assertEqual(sha(PRE),"9b833f5fc2729d13320c87e368baaff952c890ed068586622badfc7da63b4db1")
        self.assertEqual(self.c["status"],"PRELOCK_BEFORE_SIGNED_GRADIENT_CAPTURE_ANY_NEW_EVALUATION")
        self.assertFalse(self.c["automatic_policy_promotion"])

    def test_source96_new_HellaSwag48_data_frozen(self):
        self.assertEqual(self.l["status"],"SOURCE_AND_OUTOFDOMAIN_EVALUATION_FROZEN_BEFORE_SIGNED_GRADIENT")
        self.assertEqual(self.l["source_count"],96)
        self.assertEqual(self.l["eval_count"],48)
        self.assertEqual(self.l["HF_dataset_revision"],"218ec52e09a7e7462a5400043bb9a69a41d06b76")
        self.assertEqual(sha(LOCK),"7e4c9a014bc04799ef8701500c25174aea9449976c67ae93fba3f788bf4ec565")

    def test_native_original_Qwen_source(self):
        self.assertEqual(self.c["official_rev"],self.p["original_model_rev"])
        self.assertEqual(self.p["full_original_model_layer_count"],28)
        self.assertTrue(self.s["eval_logits_unopened"])

    def test_96x196_signed_per_tensor(self):
        self.assertEqual(self.s["status"],"SOURCE_ONLY_196x96_SIGNED_NATIVE_Q4N8_DELTA_VERIFIED")
        self.assertTrue(self.s["first_order_joint_Q4_source_rms"]>0)
        self.assertTrue(self.s["mean_cross_tensor_abs_correlation"]>0)
        self.assertEqual(self.p["source_96_signed_private_sha256"],self.s["full_source_private_SHA256"])

    def test_cross_tensor_objective_not_globally_proved(self):
        self.assertFalse(self.p["source_optimization_exact_global"])
        self.assertTrue(self.p["source_optimization_feasible_pairwise_1swap_local_only"])
        self.assertTrue(self.p["source_signed_task_gradient_cross_tensor_pairwise_covariance"])
        self.assertGreater(self.p["objective_global_loss_change_squared_empirical_mean_Q4"],self.p["selected_source_signed_loss_squared"])

    def test_all_baseline_source_scores_recorded_not_reclassified_as_nll(self):
        scores=self.p["source_old_MILP_objective_competitors"]
        self.assertEqual(set(scores),{"qv5_24","qv6_64","qv6_128"})
        self.assertTrue(all(math.isfinite(v["signed_joint_source_objective"]) for v in scores.values()))
        self.assertFalse(self.p["source_optimization_exact_global"])

    def test_selected_mask_sha_28_unique(self):
        entries=[tuple(row.split()) for row in MASK.read_text().splitlines()]
        self.assertEqual(len(entries),28)
        self.assertEqual(len(set(entries)),28)
        self.assertEqual(self.p["mask_sha256"],sha(MASK))
        self.assertEqual(len(self.p["selected_ids"]),28)
        self.assertEqual(set(entries),{(role,str(layer)) for role,layer in self.p["selected_ids"]})

    def test_native_original_kernel_limits(self):
        k=self.p["final_original_Apple_kernel_costs"]
        self.assertLessEqual(k["memory_bytes"],830570496)
        self.assertLessEqual(k["serial_estimate_ms"],100.0)
        self.assertLessEqual(k["isolated_async_stage_ms"],50.00001)
        self.assertFalse(self.p["auto_serving_promotion_forbidden"] is False)

    def test_metadata_erratum_corrected_before_held(self):
        e=json.loads(ERR.read_text())
        self.assertEqual(e["new_source_report_SHA256"],sha(PLAN))
        self.assertEqual(e["selected_mask_sha256_unchanged"],sha(MASK))
        self.assertEqual(e["status"],"CORRECTED_START_LABEL_ONLY_NO_MASK_OR_RESULT_CHANGED")
        self.assertEqual(self.p["selected_best_start"],e["correct_label"])

    def test_independent_exhaustive_source_one_swap_certificate(self):
        cert_path=D/"BEGLIN_QV8_SOURCE_LOCAL_OPTIMALITY_INDEPENDENT_20261022.json"
        cert=json.loads(cert_path.read_text())
        self.assertEqual(cert["status"],"PASS_4704_POSSIBLE_ONE_SWAP_PAIRS_NO_FEASIBLE_IMPROVEMENT")
        self.assertEqual(cert["possible_selected_to_unselected_swaps"],4704)
        self.assertEqual(cert["frozen_selected_mask_SHA256"],sha(MASK))
        self.assertEqual(cert["frozen_source_optimizer_receipt_SHA256"],sha(PLAN))
        self.assertGreater(cert["independently_measured_feasible_swaps"],0)
        self.assertGreaterEqual(cert["best_feasible_exchange_J_change"],0)
        self.assertTrue(cert["no_global_optimality_claim"])

    def test_original_BF16_separate_but_not_numeric_parity(self):
        a=json.loads(REF.read_text())
        self.assertEqual(a["status"],"REFERENCE_ORIGINAL_BF16_48_NARRATIVE_ANSWERS")
        self.assertEqual(a["case_count"],48)
        self.assertFalse(a["no_native_CPU_numerical_parity_assertion"] is False)
        self.assertTrue(math.isfinite(a["mean_correct_answer_NLL"]))

    def test_independent_source_auditor(self):
        a=json.loads(AUD.read_text())
        self.assertEqual(a["status"],"PASS_RAW_96x196_SIGNED_CROSS_TENSOR_SOURCE_AND_MASK_COST_CONTRACT")
        self.assertEqual(a["source_private_SHA"],self.s["full_source_private_SHA256"])
        self.assertEqual(a["selected_policy_SHA"],self.p["mask_sha256"])
        self.assertTrue(a["no_global_optimality_claim"])

    def test_actual_4_native_if_published(self):
        if not NATIVE.exists():
            self.skipTest("Native HellaSwag48 four conditions still executing")
        n=json.loads(NATIVE.read_text())
        self.assertEqual(n["status"],"PASS_96x196_SIGNED_SOURCE_AND_4x48_ACTUAL_QWEN_CPU_NATIVE_LOGS")
        self.assertEqual(set(n["native_summary"]),{"q4","qv6_diag64","qv6_diag128","qv8_signed96"})
        for v in n["native_summary"].values():
            self.assertEqual(v["N"],48)
            self.assertGreaterEqual(v["correct"],0)
            self.assertLessEqual(v["correct"],48)
            self.assertTrue(math.isfinite(v["mean_correct_token_NLL"]))
        self.assertFalse(n["quality_generalization_established"])
        self.assertTrue(n["no_auto_promotion"])
        self.assertTrue(n["main_and_operator_release_unchanged"])

if __name__=="__main__":
    unittest.main(verbosity=2)
