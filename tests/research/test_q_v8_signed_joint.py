"""Qv8 true source signed gradient / exact native CPU target contract tests."""
import hashlib,json,math,pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v8_signed_joint_optimizer as joint
import q_v8_native_hellaswag as native
import q_v8_independent_signed_source_audit as auditor
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(b):return hashlib.sha256(b).hexdigest()

class Qv8SourceSignedNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pre=json.loads(joint.PRE.read_text())
        cls.pub=json.loads(joint.SOURCE.read_text())
        cls.raw=json.loads(joint.FULL.read_text())
        cls.policy=json.loads(joint.OUT.read_text())
        cls.data=json.loads(joint.HELD.read_text())["held_validation48"]

    def test_prelock_not_posthoc_quality_selection(self):
        self.assertEqual(sha(joint.PRE.read_bytes()),joint.PRE_SHA)
        self.assertEqual(self.pre["status"],"PRELOCK_BEFORE_SIGNED_GRADIENT_CAPTURE_ANY_NEW_EVALUATION")
        self.assertFalse(self.pre["automatic_policy_promotion"])
        self.assertTrue(self.policy["no_held_data_labels_or_logits_loaded"])

    def test_first96_original_Qv6_source_pinned(self):
        data=json.loads((ROOT/"lab_local/qv8_joint_signed_20261022/pinned_original_ARC_train96.json").read_text())
        self.assertEqual(len(data["source_train96"]),96)
        self.assertEqual(len({x["id"] for x in data["source_train96"]}),96)
        self.assertTrue(all(x["prompt_token_count"]>=18 for x in data["source_train96"]))

    def test_frozen_new_domain_no_reselection(self):
        self.assertEqual(sha(joint.HELD.read_bytes()),joint.HELD_SHA)
        self.assertEqual(len(self.data),48)
        self.assertTrue(all(set(row["suffix_ids"])==set("ABCD") for row in self.data))
        self.assertTrue(all(18<=x["prompt_token_count"]<=208 for x in self.data))

    def test_all_96_x_196_signed_gradients_present(self):
        a=np.asarray(self.raw["original_tensor_grad_dot_q4"],dtype=np.float64)
        b=np.asarray(self.raw["original_tensor_grad_dot_n8"],dtype=np.float64)
        self.assertEqual(a.shape,(96,196))
        self.assertEqual(b.shape,(96,196))
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.isfinite(b).all())

    def test_q4_baseline_approximation(self):
        a=np.asarray(self.raw["original_tensor_grad_dot_q4"],dtype=np.float64)
        expected=float(np.mean(a.sum(axis=1)**2))
        self.assertAlmostEqual(expected,self.policy["objective_global_loss_change_squared_empirical_mean_Q4"],places=6)

    def test_genuine_global_signed_cross_interactions(self):
        a=np.asarray(self.raw["original_tensor_grad_dot_q4"],dtype=np.float64)
        b=np.asarray(self.raw["original_tensor_grad_dot_n8"],dtype=np.float64)
        d=b-a
        gram=(d.T@d)/96
        self.assertGreater(np.linalg.norm(gram-np.diag(np.diag(gram))),0)
        self.assertAlmostEqual(float(np.linalg.norm(gram-np.diag(np.diag(gram)))),
                               self.policy["pairwise_cross_term_frobenius_norm"],places=5)

    def test_model_source_baselines_and_new_mask_recompute(self):
        z=auditor.source()
        self.assertLess(z["source_new_joint_squared_loss"],z["source_Q4_full_squared_loss"])
        self.assertLess(z["source_new_joint_squared_loss"],min(z["old_mask_objectives"].values()))

    def test_exact_28_real_roles_still_research_only(self):
        rows=[tuple(t.split()) for t in joint.MASK.read_text().splitlines()]
        self.assertEqual(len(rows),28)
        self.assertEqual(len(set(rows)),28)
        self.assertEqual(sha(joint.MASK.read_bytes()),self.policy["mask_sha256"])
        self.assertFalse(self.policy["source_optimization_exact_global"])
        self.assertTrue(self.policy["source_optimization_feasible_pairwise_1swap_local_only"])

    def test_memory_kernel_proxy_feasible(self):
        self.assertLessEqual(self.policy["final_original_Apple_kernel_costs"]["memory_bytes"],830570496)
        self.assertLessEqual(self.policy["final_original_Apple_kernel_costs"]["serial_estimate_ms"],100)
        self.assertLessEqual(self.policy["final_original_Apple_kernel_costs"]["isolated_async_stage_ms"],50.00000001)
        cost, swap, admission = joint.make_engine()
        self.assertTrue(admission(cost(joint.ids_from_mask(joint.MASK,self.policy["mask_sha256"]))))

    def test_original_Qv6_strong_baseline_masks_immutable(self):
        for key,path in (("qv6_64",joint.Q6_64),("qv6_128",joint.Q6_128),("qv5_24",joint.Q5_24)):
            self.assertEqual(sha(path.read_bytes()),joint.BASELINE_MASK_SHA[key])

    def test_transparent_metadata_erratum_before_model_evaluation(self):
        e=json.loads((ROOT/"docs/research/BEGLIN_QV8_SOURCE_START_LABEL_ERRATUM_20261022.json").read_text())
        self.assertEqual(e["status"],"CORRECTED_START_LABEL_ONLY_NO_MASK_OR_RESULT_CHANGED")
        self.assertEqual(e["new_source_report_SHA256"],sha(joint.OUT.read_bytes()))
        self.assertEqual(self.policy["selected_best_start"],e["correct_label"])
        self.assertEqual(e["selected_mask_sha256_unchanged"],self.policy["mask_sha256"])

    def test_original_BF16_gradients_only_not_intervention(self):
        self.assertTrue(self.raw["eval_logits_unopened"])
        self.assertTrue(self.raw["no_model_weights_mutated"])
        self.assertTrue(self.raw["no_operational_state_changed"])
        self.assertTrue(self.pre["no_per_tensor_downstream_intervention"])

    @staticmethod
    def synthetic_log(mode="q4",target_prefix=106,drop=False,wrongcount=False):
        tiers=(0,0,196) if mode=="q4" else (0,28,168)
        out=[f"projection_role_tensors: f32={tiers[0]} n8={tiers[1]} q4={tiers[2]}"]
        for i in range(48):
            if i==16 and drop:continue
            a=i%4
            out.append(f"[qv5-choice] window {i} NLL 0.75 expected {a} predicted {a} prefix {target_prefix}")
        out.append(f"qv5_choice NLL 36 items 48 correct {47 if wrongcount else 48} mean 0.75 ppl {math.exp(.75)}")
        return "\n".join(out)

    def test_native_four_arm_original_parser_synthetic(self):
        meta={"all_48_input_sha":[{"prefix_length":106} for _ in range(48)]}
        score=native.parse(self.synthetic_log(),"q4",meta)
        self.assertEqual(score["correct"],48)
        self.assertAlmostEqual(score["mean_correct_answer_NLL"],0.75)

    def test_native_parser_blocks_wrong_roles(self):
        meta={"all_48_input_sha":[{"prefix_length":106} for _ in range(48)]}
        with self.assertRaises(RuntimeError):
            native.parse(self.synthetic_log(),"qv8_signed96",meta)

    def test_native_parser_blocks_missing_window(self):
        meta={"all_48_input_sha":[{"prefix_length":106} for _ in range(48)]}
        with self.assertRaises(RuntimeError):
            native.parse(self.synthetic_log(drop=True),"q4",meta)

    def test_native_parser_blocks_false_count(self):
        meta={"all_48_input_sha":[{"prefix_length":106} for _ in range(48)]}
        with self.assertRaises(RuntimeError):
            native.parse(self.synthetic_log(wrongcount=True),"q4",meta)

    def test_native_parser_blocks_changed_prefix_length(self):
        meta={"all_48_input_sha":[{"prefix_length":106} for _ in range(48)]}
        with self.assertRaises(RuntimeError):
            native.parse(self.synthetic_log(target_prefix=105),"q4",meta)

if __name__=="__main__":
    unittest.main(verbosity=2)
