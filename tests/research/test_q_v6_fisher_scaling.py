"""Source-only and native-admission Qv6 negative/contract tests.

Uses no live models or operation mutations. Original 24/64/128 source and
48 OpenBookQA cross-corpus evidence is frozen before new Fisher policies.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v6_native_openbookqa as native
import q_v6_independent_source_audit as sourceaudit
PRE=ROOT/"configs/research/BEGLIN_QV6_SAMPLE_SCALING_PRELOCK_20261022.json"
PUB=ROOT/"docs/research/BEGLIN_QV6_NESTED_64_128_SOURCE_POLICY_20261022.json"
LAB=ROOT/"lab_local/qv6_fisher_scaling_20261022"
def sha(data):return hashlib.sha256(data).hexdigest()

class Qv6FisherStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads(PRE.read_text())
        cls.original=json.loads(sourceaudit.secure(sourceaudit.ARC))
        cls.held=json.loads(sourceaudit.secure(sourceaudit.CROSS))
        cls.policy=json.loads(PUB.read_text())
        cls.native_manifest=json.loads(native.secure(native.RUN/"frozen_natural_science_48_manifest.json"))

    def test_prelock_before_model_execution(self):
        self.assertEqual(sha(PRE.read_bytes()),"2e35764878191717ef7b84ed546e72e2a1c513523ecb17d24b54489bd29f319b")
        self.assertEqual(self.protocol["status"],"PRELOCK_BEFORE_128_SOURCE_EXTENSION_AND_NEW_CROSS_DATASET_EVALUATION")
        self.assertFalse(self.protocol["production_write_allowed"])
        self.assertFalse(self.protocol["auto_promotion"])

    def test_study_sample_sizes_exact(self):
        self.assertEqual(self.protocol["nested_source_sample_sizes"],[24,64,128])
        self.assertEqual(len(self.original["all_source_questions"]),128)
        self.assertEqual(len(self.held["unseen_questions"]),48)
        self.assertEqual(self.protocol["cross_dataset_count"],48)

    def test_prior_24_source_id_identity(self):
        old=json.loads(sourceaudit.secure(sourceaudit.OLD))
        # Already protected by original source hashes and independent verifier.
        self.assertEqual(len(old["source_score_rows"]),196)
        self.assertTrue(self.original["original_Qv5_first_24_exactly_reproduced"])

    def test_source_openbookqa_disjoint(self):
        p={x["question_sha256"] for x in self.original["all_source_questions"]}
        q={x["question_sha256"] for x in self.held["unseen_questions"]}
        self.assertFalse(p&q)
        self.assertEqual(len(p),128)
        self.assertEqual(len(q),48)

    def test_104_new_real_gradient_private_checkpoints(self):
        self.assertEqual(len(list((LAB/"additional_original_BF16_gradients").glob("gradient_*.json"))),104)

    def test_196_real_calibrated_tensor_scores(self):
        evidence=json.loads((ROOT/"docs/research/BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json").read_text())
        for n in ("24","64","128"):
            self.assertEqual(sum(len(values) for values in evidence["quality_upgrade_by_role_layer_per_size"][n].values()),196)
            self.assertTrue(math.isfinite(evidence["nested_split_stats"][n]["odd_even_sample_Fisher_rank_spearman"]))

    def test_source_independent_fisher_audit(self):
        a=json.loads(sourceaudit.secure(sourceaudit.OUT))
        self.assertEqual(a["status"],"PASS_ALL_196_TENSOR_SCORES_128_SOURCE_INPUTS")
        self.assertEqual(a["source_104_case_receipts_independently_reparsed"],104)

    def test_exact_28_tensor_source_policies(self):
        self.assertEqual(self.policy["status"],"FROZEN_64_128_POLICIES_BEFORE_NEW_OPENBOOKQA_VALIDATION")
        for n in ("64","128"):
            path=ROOT/f"configs/research/BEGLIN_QV6_FISHER_{n}N8_SOURCE_POLICY_20261022.txt"
            x={tuple(row.split()) for row in path.read_text().splitlines()}
            self.assertEqual(len(x),28)
            self.assertEqual(sha(path.read_bytes()),self.policy["selected_policies"][n]["mask_sha256"])
            self.assertLessEqual(self.policy["selected_policies"][n]["estimated_native_projection_bytes"],830570496)
            self.assertAlmostEqual(self.policy["selected_policies"][n]["source_only_gap"],0)

    def test_no_native_eval_tuning(self):
        self.assertTrue(self.policy["evaluation_48_OpenBookQA_logits_not_opened"])
        self.assertFalse(self.policy["production_write_allowed"])
        self.assertFalse(self.policy["auto_precision_promotion"])

    def test_four_arm_input_sha_same(self):
        self.assertEqual(self.native_manifest["status"],"FROZEN_FOUR_ARMS_BEFORE_FIRST_CROSSCORPUS_NATIVE_FORWARD")
        self.assertEqual(len(self.native_manifest["all_48_prompt_choice_hashes"]),48)
        self.assertEqual(tuple(self.native_manifest["modes"]),native.MODES)

    @staticmethod
    def fake_log(mode="q4"):
        h=native.COUNT[mode]
        lines=[f"projection_role_tensors: f32={h[0]} n8={h[1]} q4={h[2]}"]
        for i in range(48):
            y=i%4
            lines.append(f"[qv5-choice] window {i} NLL 0.5 expected {y} predicted {y} prefix 80")
        lines.append("qv5_choice NLL 24 items 48 correct 48 mean 0.5 ppl "+str(math.exp(.5)))
        return "\n".join(lines)

    def test_native_scores_48_and_four_options(self):
        fake={"all_48_prompt_choice_hashes":[{"prompt_length":80} for _ in range(48)]}
        obj=native.parse_log(self.fake_log(), "q4", fake)
        self.assertEqual(obj["correct"],48)
        self.assertAlmostEqual(obj["answer_only_correct_letter_NLL"],0.5)

    def test_native_wrong_n8_count_rejected(self):
        fake={"all_48_prompt_choice_hashes":[{"prompt_length":80} for _ in range(48)]}
        with self.assertRaises(RuntimeError):
            native.parse_log(self.fake_log("q4"),"v6_fisher_128",fake)

    def test_native_incomplete_eval_rejected(self):
        fake={"all_48_prompt_choice_hashes":[{"prompt_length":80} for _ in range(48)]}
        with self.assertRaises(RuntimeError):
            native.parse_log(self.fake_log().replace("[qv5-choice] window 4", "[wrong] window 4"),"q4",fake)

    def test_native_source_prompt_sha_tamper_rejected(self):
        fake={"all_48_prompt_choice_hashes":[{"prompt_length":80} for _ in range(48)]}
        with self.assertRaises(RuntimeError):
            native.parse_log(self.fake_log().replace("window 0 NLL 0.5 expected 0 predicted 0 prefix 80",
                 "window 0 NLL 0.5 expected 0 predicted 0 prefix 79"),"q4",fake)

    def test_native_nonfinite_loss_rejected(self):
        fake={"all_48_prompt_choice_hashes":[{"prompt_length":80} for _ in range(48)]}
        with self.assertRaises(RuntimeError):
            native.parse_log(self.fake_log().replace("window 3 NLL 0.5", "window 3 NLL nan"),"q4",fake)

    def test_native_correct_aggregate_tamper_rejected(self):
        fake={"all_48_prompt_choice_hashes":[{"prompt_length":80} for _ in range(48)]}
        with self.assertRaises(RuntimeError):
            native.parse_log(self.fake_log().replace("items 48 correct 48","items 48 correct 46"),"q4",fake)

if __name__=="__main__":
    unittest.main(verbosity=2)
