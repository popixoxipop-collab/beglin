"""Portable Qv5 public evidence contracts, no model, GPU or private data needed."""
import hashlib,json,math,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
PRE=ROOT/"configs/research/BEGLIN_QV5_FISHER_GRAD_PRELOCK_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt"
LOCK=ROOT/"docs/research/BEGLIN_QV5_ARC_FREEZE_PUBLIC_AGGREGATE_20261022.json"
SCORE=ROOT/"docs/research/BEGLIN_QV5_FISHER_SOURCE_AGGREGATE_20261022.json"
POL=ROOT/"docs/research/BEGLIN_QV5_FISHER_HIGHS_SOURCE_MILP_20261022.json"
NATIVE=ROOT/"docs/research/BEGLIN_QV5_FISHER_40ARC_NATIVE_PUBLIC_RESULT_20261022.json"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(data):return hashlib.sha256(data).hexdigest()

class Qv5PublicEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pre=json.loads(PRE.read_text())
        cls.lock=json.loads(LOCK.read_text())
        cls.score=json.loads(SCORE.read_text())
        cls.pol=json.loads(POL.read_text())

    def test_source_protocol_immutable(self):
        self.assertEqual(sha(PRE.read_bytes()),"a886418affc9f665d64ce4ca9eb0b5ea2aef4ae97bc59bf58ea1affb53bf19ef")
        self.assertEqual(self.pre["status"],"LOCKED_BEFORE_ARC_LABEL_CALIBRATION_AND_NATIVE_VALIDATION")
        self.assertFalse(self.pre["auto_promotion"])

    def test_dataset_locked_before_optimizer(self):
        self.assertEqual(sha(LOCK.read_bytes()),"43f693c8697b4e8889f807780034093c62c5813a3ad31e7a3dea7428083a3f84")
        self.assertEqual(self.lock["status"],"BOTH_SPLITS_FROZEN_BEFORE_ORIGINAL_BF16_FISHER")
        self.assertEqual(self.lock["source_items"],24)
        self.assertEqual(self.lock["development_items"],40)
        self.assertNotEqual(self.lock["selected_source_ids_sha256"],self.lock["selected_dev_ids_sha256"])

    def test_model_and_public_dataset_revision(self):
        self.assertEqual(self.pre["dataset_sha"],self.lock["dataset_commit_sha256"])
        self.assertEqual(self.pre["bf16_official_rev"],self.lock["BF16_original_HF_revision"])

    def test_source_has_all_196_gradient_tensor_scores(self):
        arr=self.score["quality_upgrade_by_role_layer"]
        self.assertEqual(set(arr),set(ROLES))
        self.assertEqual(sum(len(arr[r]) for r in ROLES),196)
        self.assertTrue(all(math.isfinite(float(x)) for r in ROLES for x in arr[r]))

    def test_fisher_source_not_relabelled_exact_loss(self):
        self.assertFalse(self.score["exact_F64_covariance_computation"] if "exact_F64_covariance_computation" in self.score else False)
        self.assertIn("EMPIRICAL_FISHER_PROXY",self.score["scientific_claim"])
        self.assertTrue(self.score["no_quantized_forward_or_per_tensor_downstream_interventions"])

    def test_source_to_policy_source_hash(self):
        self.assertEqual(self.pol["original_fisher_source_pub_sha256"],sha(SCORE.read_bytes()))
        self.assertEqual(self.pol["original_input_lock_sha256"],sha(LOCK.read_bytes()))

    def test_mask_exact_28_unique_tensors(self):
        rows=[line.split() for line in MASK.read_text().splitlines()]
        self.assertEqual(len(rows),28)
        self.assertEqual(len({tuple(row) for row in rows}),28)
        for role,n in rows:
            self.assertIn(role,ROLES)
            self.assertTrue(n.isdigit())
            self.assertTrue(0<=int(n)<28)
        self.assertEqual(self.pol["mask_sha256"],sha(MASK.read_bytes()))
        self.assertEqual(len(self.pol["optimizer"]["selected_ids"]),28)

    def test_hardware_constraints_from_existing_apple_neon(self):
        obj=self.pol["optimizer"]
        self.assertEqual(obj["n8_tensors"],28)
        self.assertEqual(obj["n4_tensors"],168)
        self.assertLessEqual(obj["estimated_projection_weight_bytes"],830570496)
        self.assertLessEqual(obj["original_fullmodel_serial_estimated_ms"],100.)
        self.assertLessEqual(obj["isolated_async_CPU_stage_total_ms"],50.00000001)
        self.assertEqual(obj["source_only_solver_status"],0)
        self.assertLess(obj["mip_gap"],1.e-7)

    def test_production_gate_closed(self):
        self.assertFalse(self.pol["production_write_allowed"])
        self.assertFalse(self.pol["native_fullmodel_async_default_enabled"])
        self.assertFalse(self.pol["model_accuracy_improvement_proved"])
        self.assertEqual(self.pol["scientific_Q_verdict"],"NO_VERDICT")

    def test_native_if_present_is_exploratory(self):
        if not NATIVE.is_file():
            self.skipTest("native 6-mode 40ARC still running")
        summary=json.loads(NATIVE.read_text())
        self.assertEqual(set(summary["native_summary"]),set(("f32","q4","role_random","v2_diag","v3_cov","v5_fisher")))
        self.assertTrue(summary["no_operational_mutation"] if "no_operational_mutation" in summary else not summary["legacy_main_and_immutable_release_modified"])
        self.assertFalse(summary["Qv5_fisher_result_generalization_established"])
        self.assertFalse(summary["Qv5_fisher_research_promotion_allowed"])
        for mode,metrics in summary["native_summary"].items():
            self.assertEqual(metrics["total"],40)
            self.assertGreaterEqual(metrics["correct"],0)
            self.assertLessEqual(metrics["correct"],40)
            self.assertTrue(math.isfinite(metrics["answer_only_mean_NLL"]))

if __name__=="__main__":
    unittest.main(verbosity=2)
