"""BEGLIN Qv5 standalone source-Fisher, frozen policy and native ARC guard tests."""
import hashlib,json,math,pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v5_original_bf16_fisher_source as gradients
import q_v5_fisher_milp as alloc
import q_v5_native_arc_40eval as native
import q_v5_independent_native_audit as audit

class Qv5OriginalBF16FisherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads(gradients.PROTO.read_text())
        cls.train=json.loads(gradients.SOURCE.read_text())
        cls.policy,cls.source=audit.check_source()

    def test_prelocked_before_source_and_dev(self):
        self.assertEqual(gradients.sha(gradients.PROTO.read_bytes()),gradients.PRELOCK_SHA)
        self.assertEqual(gradients.sha(gradients.SOURCE.read_bytes()),gradients.SOURCE_SHA)
        self.assertEqual(gradients.sha(gradients.DEV.read_bytes()),gradients.DEV_SHA)
        self.assertFalse(self.config["production_write_allowed"])

    def test_train_validation_different_sha_dataset(self):
        val=json.loads(gradients.DEV.read_text())
        self.assertEqual(len(self.train["questions"]),24)
        self.assertEqual(len(val["questions"]),40)
        self.assertFalse(set(x["question_sha256"] for x in self.train["questions"])&
                         set(x["question_sha256"] for x in val["questions"]))

    def test_fisher_is_original_BF16_model(self):
        self.assertTrue(json.loads(gradients.OUT.read_text())["BF16_model_forward_unchanged"])
        self.assertTrue(json.loads(gradients.OUT.read_text())["no_quantized_forward_or_per_tensor_downstream_interventions"])
        self.assertEqual(json.loads(gradients.OUT.read_text())["source_dataset"],"ARC-Challenge train")

    def test_all_196_score_dimensions(self):
        scores=json.loads(gradients.OUT.read_text())["quality_upgrade_by_role_layer"]
        self.assertEqual(set(scores),set(gradients.ROLES))
        self.assertTrue(all(len(a)==28 for a in scores.values()))
        self.assertTrue(all(math.isfinite(x) for r in scores.values() for x in r))

    def test_split_half_rank_recalculated_independently(self):
        self.assertAlmostEqual(self.source["split_half_spearman"],
                               json.loads(gradients.OUT.read_text())["split_half_odd_even_spearman"])

    def test_original_C_quantizer_source_matches(self):
        self.assertEqual(gradients.sha((ROOT/"gguf_transcode.c").read_bytes()),
                         self.config["original_C_quant_source_hashes"]["gguf_transcode.c"])
        self.assertEqual(gradients.sha((ROOT/"qng64_dense.c").read_bytes()),
                         self.config["original_C_quant_source_hashes"]["qng64_dense.c"])

    def test_synthetic_native_encoder_4bit_vs_8bit(self):
        w=np.random.default_rng(21).standard_normal((64,64)).astype(np.float32)
        d4,d8,a=gradients.quant_errors(gradients.lib_quant(),w)
        self.assertEqual(d4.shape,w.shape)
        self.assertEqual(d8.shape,w.shape)
        self.assertLess(a["n8_residual_F32_sum_sq"],a["Q4_residual_F32_sum_sq"])

    def test_signed_gradient_direction(self):
        grad=np.array([[1.,-2.],[3.,4.]],dtype=np.float32)
        d4=np.array([[.5,-.5],[-1.,.75]],dtype=np.float16)
        d8=np.array([[.25,-.25],[-.25,.5]],dtype=np.float16)
        x,y,tr=gradients.diag_gradient_dot(grad,d4,d8)
        self.assertAlmostEqual(x,float(np.sum(grad*d4))**2)
        self.assertAlmostEqual(y,float(np.sum(grad*d8))**2)
        self.assertAlmostEqual(tr,float((grad**2).sum()))

    def test_nan_gradient_rejected(self):
        with self.assertRaises(RuntimeError):
            gradients.diag_gradient_dot(
               np.array([[float("nan")]],dtype=np.float32),
               np.ones((1,1),dtype=np.float16),
               np.ones((1,1),dtype=np.float16))

    def test_source_mask_exact_28_upgrades_and_MILP_opt(self):
        pol=self.policy
        entries=[x.split() for x in alloc.MASK.read_text().splitlines()]
        self.assertEqual(len(entries),28)
        self.assertEqual(len(set(map(tuple,entries))),28)
        self.assertEqual(pol["optimizer"]["n4_tensors"],168)
        self.assertEqual(pol["optimizer"]["n8_tensors"],28)
        self.assertLessEqual(pol["optimizer"]["estimated_projection_weight_bytes"],830570496)
        self.assertAlmostEqual(pol["optimizer"]["mip_gap"],0.,places=8)

    def test_source_no_automatic_serving_promotion(self):
        self.assertFalse(self.policy["production_write_allowed"])
        self.assertFalse(self.policy["model_accuracy_improvement_proved"])
        self.assertFalse(self.policy["native_fullmodel_async_default_enabled"])

    def synthetic_log(self,mode="q4",correct=30,loss=.9,prefix=100):
        a=native.PROJ[mode]
        lines=["projection_role_tensors: f32=%d n8=%d q4=%d"%a]
        for idx in range(40):
            ans=idx%4;pred=ans if idx<correct else (ans+1)%4
            lines.append("[qv5-choice] window %d NLL %.17g expected %d predicted %d prefix %d"%
                         (idx,loss,ans,pred,prefix))
        lines.append("qv5_choice NLL %.17g items 40 correct %d mean %.17g ppl %.17g"%
                     (40*loss,correct,loss,math.exp(loss)))
        return "\n".join(lines)

    def test_native_parser_expected_correct_nll(self):
        manifest={"validation":[{"prompt_length":100} for _ in range(40)]}
        data=native.parse(self.synthetic_log(), "q4", manifest)
        self.assertEqual(data["correct"],30)
        self.assertAlmostEqual(data["accuracy"],.75)
        self.assertAlmostEqual(data["correct_answer_mean_nll"],.9)

    def test_native_parser_precisions_must_match(self):
        manifest={"validation":[{"prompt_length":100} for _ in range(40)]}
        with self.assertRaises(RuntimeError):
            native.parse(self.synthetic_log("q4"), "v5_fisher", manifest)

    def test_native_parser_must_score_all_40_cases(self):
        manifest={"validation":[{"prompt_length":100} for _ in range(40)]}
        with self.assertRaises(RuntimeError):
            native.parse(self.synthetic_log().replace("[qv5-choice] window 3","[bad] window 3"),"q4",manifest)

    def test_native_parser_detects_label_tamper(self):
        manifest={"validation":[{"prompt_length":100} for _ in range(40)]}
        with self.assertRaises(RuntimeError):
            native.parse(self.synthetic_log().replace("items 40 correct 30","items 40 correct 31"),"q4",manifest)

    def test_native_parser_detects_nonfinite_loss(self):
        manifest={"validation":[{"prompt_length":100} for _ in range(40)]}
        with self.assertRaises(RuntimeError):
            native.parse(self.synthetic_log().replace("NLL 0.90000000000000002","NLL NaN",1),"q4",manifest)

    def test_native_patch_preserves_legacy_scoring_branch(self):
        old=native.secure(native.QV4_BIN_SOURCE["mixed"]).decode()
        new=native.patch(old)
        self.assertEqual(new.split(native.PATCH,1)[1],old.split(native.PATCH,1)[1])

    def test_native_patch_duplicate_rejected(self):
        old=native.secure(native.QV4_BIN_SOURCE["base"]).decode()
        with self.assertRaises(RuntimeError):
            native.patch(native.patch(old))

    def test_native_patch_ambiguous_source_rejected(self):
        with self.assertRaises(RuntimeError):
            native.patch(native.PATCH+native.PATCH)

if __name__=="__main__":
    unittest.main(verbosity=2)
