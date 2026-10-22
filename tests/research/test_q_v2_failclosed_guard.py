import copy,json,pathlib,subprocess,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_failclosed_promotion_guard as guard

class FailClosedQv2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(guard.FILE.read_text())
    def test_real_fixed_native_result_is_not_ready(self):
        v=guard.evaluate(self.data)
        self.assertEqual(v["status"],"NOT_READY")
        self.assertTrue(v["no_automatic_live_promotion"])
        self.assertTrue(v["observed_allQ4_domination"])
    def test_quality_and_async_truth_are_separate_conditions(self):
        v=guard.evaluate(self.data)
        self.assertIn("NLL_increase_exceeds_declared_quality_ceiling",v["blocking_reasons"])
        self.assertIn("full_decoder_async_scheduling_not_integrated_or_proven",v["blocking_reasons"])
        self.assertGreater(v["observed_relative_NLL_increase"],.02)
    def test_mandatory_measured_conditions_and_sha(self):
        q=copy.deepcopy(self.data)
        q["private_native_evaluation_summary_sha256"]="0"*64
        with self.assertRaises(ValueError):
            guard.evaluate(q)
        q=copy.deepcopy(self.data)
        del q["controls"]["role_balanced_random"]
        with self.assertRaises(ValueError):
            guard.evaluate(q)
    def test_fails_closed_on_nonfinite_PPL_and_fake_runtime_claim(self):
        q=copy.deepcopy(self.data)
        q["controls"]["all_q4"]["PPL_public_131_tokens"]=float("nan")
        with self.assertRaises(ValueError):
            guard.evaluate(q)
        q=copy.deepcopy(self.data)
        q["concurrent_stage_truth"]="asynchronous whole model fully deployed"
        with self.assertRaises(ValueError):
            guard.evaluate(q)
    def test_require_ready_command_does_not_activate(self):
        q=subprocess.run([sys.executable,str(ROOT/"research/oml_acs/q_v2_failclosed_promotion_guard.py"),"--require-ready"],
                         capture_output=True,text=True)
        self.assertEqual(q.returncode,2)
        self.assertIn('"status": "NOT_READY"',q.stdout)
        self.assertEqual(q.stderr,"")
if __name__=="__main__":
    unittest.main(verbosity=2)
