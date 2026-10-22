import json,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v2_native_held_eval_independent_audit as a

class QV2HeldNativeRegression(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
    cls.receipt=json.loads(a.secure(a.OUT))
    cls.j,cls.fixture,cls.source=a.load_provenance()
 def test_all_five_native_full_decoder_trials(self):
    self.assertEqual(set(self.j["mode_results"]),set(a.MODES))
    for model,part in self.j["mode_results"].items():
       self.assertEqual(set(part),{"dump","ppl","bench"})
       self.assertTrue(all(r["exit_code"]==0 for r in part.values()))
 def test_model_shas_and_eight_pinned_same_input_windows(self):
    self.assertEqual(self.fixture["eval_public_prompt_count"],8)
    data=a.parse_and_verify(self.j,self.fixture)
    self.assertEqual(len(data),5)
    self.assertTrue(all(v["shared_8_pinned_public_prompts"] for v in data.values()))
 def test_native_first_prompt_logits_are_finite(self):
    out=a.parse_and_verify(self.j,self.fixture)
    self.assertEqual(out["f32"]["KL_widened_F32_to_policy_nats"],0.)
    self.assertTrue(all(v["KL_widened_F32_to_policy_nats"]>=0 for v in out.values()))
 def test_activation_proxy_did_not_win_eight_public_NLL(self):
    out=a.parse_and_verify(self.j,self.fixture)
    self.assertLess(out["old_random"]["measured_public_eval_NLL_nats"],
                    out["async_DAG"]["measured_public_eval_NLL_nats"])
    self.assertLess(out["old_random"]["measured_public_eval_NLL_nats"],
                    out["serial_only"]["measured_public_eval_NLL_nats"])
 def test_q4_execution_speed_faster_than_selected_models(self):
    out=a.parse_and_verify(self.j,self.fixture)
    self.assertGreater(out["q4"]["measured_native_best_of_3_tokens_s"],
                       out["async_DAG"]["measured_native_best_of_3_tokens_s"])
    self.assertGreater(out["serial_only"]["measured_native_best_of_3_tokens_s"],
                       out["old_random"]["measured_native_best_of_3_tokens_s"])
 def test_native_no_full_async_executor_deployed(self):
    self.assertFalse(self.receipt["native_full_model_executes_stage_asynchronously"])
    self.assertTrue(self.receipt["stage_async_microbenchmark_is_real"])
    self.assertEqual(self.receipt["final_Q_quality_verdict"],"NO_VERDICT")
 def test_sha_frozen_policies_and_preregistered_separate_prompts(self):
    self.assertEqual(self.source["status"],"OPTIMAL_SOURCE_ONLY_POLICIES_FROZEN_BEFORE_EVAL")
    self.assertEqual(self.source["role_order"],["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    self.assertTrue(self.fixture["policy_sources_verified_before_eval"])
    self.assertTrue(self.j["frozen_policy_selected_before_first_eval"])
 def test_immutable_private_audit_no_overwrite_and_no_main_change(self):
    self.assertEqual(a.OUT.stat().st_mode&0o777,0o600)
    self.assertTrue(self.receipt["no_main_or_production_change"])
    self.assertFalse(self.receipt["original_V6_used_docs_verified_excluded"])
    with self.assertRaises(FileExistsError):a.run()
if __name__=="__main__":
 unittest.main(verbosity=2)
