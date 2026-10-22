"""Private Qv7 source/cost/native quality/ABBA negative and SHA controls.

Never recompute live models or mutate original native execution receipts.
"""
import hashlib,json,pathlib,math,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v7_independent_family_audit as family
import q_v7_tinyllama_native_test48 as native
import q_v7_real_cpu_ABBA as run_bench
import q_v7_independent_ABBA_audit as perf

class BEGLINQv7NativeResearchAudits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src,cls.mask,cls.rank=family.sources()
        cls.results,cls.binary=family.actual_native(cls.mask)
        cls.testman=json.loads(native.read(native.RUN/"frozen_test48_native_manifest.json"))
        cls.bench=json.loads(perf.sealed(perf.RAW))
    def test_154_original_llama_BF16_source_gradients(self):
        self.assertEqual(self.src["source_cases"],64)
        self.assertEqual(self.src["original_model_projection_count"],154)
        self.assertEqual(len(self.src["per_source_case_Fisher_Q4_by_tensor"]),154)
        self.assertTrue(self.src["forward_always_original_unquantized_BF16"])
    def test_source_64_source_only_no_heldout_logits(self):
        self.assertTrue(self.src["crosscorpus_labels_or_logits_unopened"])
        self.assertEqual(self.src["frozen_eval48_not_opened_sha256"],family.TEST_SHA)
    def test_model_and_tensor_mask_identity(self):
        self.assertEqual(self.mask["n8_count"],22)
        self.assertEqual(self.mask["q4_count"],132)
        self.assertEqual(self.mask["solver_status"],0)
        self.assertLess(self.mask["mip_gap"],1e-7)
        self.assertEqual(self.src["official_rev"],self.mask["model_revision"])
    def test_same_48_ids_binary_roles(self):
        self.assertEqual(self.testman["eval_count"],48)
        self.assertEqual(self.results["q4"]["N"],48)
        self.assertEqual(self.results["fisher_22n8"]["N"],48)
        self.assertNotEqual(self.binary["q4"],self.binary["fisher_22n8"])
    def test_native_q4_and_mixed_are_actual_C(self):
        for name,count in [("q4",(0,0,154)),("fisher_22n8",(0,22,132))]:
            raw=native.read(native.RUN/name/"native_tinyllama_48choice_log.txt").decode()
            self.assertIn(f"projection_role_tensors: f32={count[0]} n8={count[1]} q4={count[2]}",raw)
            self.assertIn("qv5_choice NLL",raw)
    def test_native_score_order_and_A_B_C_D(self):
        for mode in native.NAME:
            raw=native.read(native.RUN/mode/"native_tinyllama_48choice_log.txt").decode()
            x=native.parse(raw,mode,self.testman)
            self.assertEqual(x["correct"],self.results[mode]["correct"])
            self.assertAlmostEqual(x["correct_answer_NLL"],self.results[mode]["NLL"])
    def test_native_parser_detects_nonfinite_loss(self):
        raw=native.read(native.RUN/"q4"/"native_tinyllama_48choice_log.txt").decode()
        original=native.ROW.findall(raw)
        self.assertEqual(len(original),48)
        changed=raw.replace("NLL "+original[0][1],"NLL NaN",1)
        with self.assertRaises(RuntimeError):
            native.parse(changed,"q4",self.testman)
    def test_native_parser_detects_role_tampering(self):
        raw=native.read(native.RUN/"fisher_22n8"/"native_tinyllama_48choice_log.txt").decode()
        changed=raw.replace("n8=22 q4=132","n8=0 q4=154",1)
        with self.assertRaises(RuntimeError):
            native.parse(changed,"fisher_22n8",self.testman)
    def test_native_parser_detects_missing_position(self):
        raw=native.read(native.RUN/"q4"/"native_tinyllama_48choice_log.txt").decode()
        changed=raw.replace("[qv5-choice] window 7","[forged] window 7",1)
        with self.assertRaises(RuntimeError):
            native.parse(changed,"q4",self.testman)
    def test_native_parser_detects_wrong_prompt_length(self):
        raw=native.read(native.RUN/"q4"/"native_tinyllama_48choice_log.txt").decode()
        original=native.ROW.findall(raw)
        self.assertTrue(original)
        before="prefix "+original[0][-1]
        changed=raw.replace(before,"prefix "+str(int(original[0][-1])+1),1)
        with self.assertRaises(RuntimeError):
            native.parse(changed,"q4",self.testman)
    def test_llama_fisher_quality_development_negative(self):
        self.assertEqual(self.results["q4"]["correct"],16)
        self.assertEqual(self.results["fisher_22n8"]["correct"],15)
        self.assertGreater(self.results["fisher_22n8"]["NLL"],self.results["q4"]["NLL"])
    def test_one_held_sample_consumed_disclosed(self):
        self.assertTrue(self.testman["held_first_prompt_used_prior_for_Q4_implementation_smoke"])
        self.assertTrue(self.testman["native_eval_is_exploratory_not_final_acceptance"])
    def test_ABBA_native_protocol_and_real_8_runs(self):
        self.assertEqual(len(self.bench["process_runs"]),8)
        self.assertEqual(self.bench["summary"]["actual_native_timed_processes"],8)
        self.assertEqual(self.bench["summary"]["actual_timed_token_count"],1536)
    def test_ABBA_independent_QA_complete(self):
        att=json.loads(perf.sealed(perf.OUT))
        self.assertEqual(att["status"],"PASS_REAL_8_NATIVE_PROCESSES_1536_TIMED_TOKENS__LATENCY_ADMISSION_FAILCLOSED")
        self.assertFalse(att["quality_or_serving_speed_promotion_allowed"])
    def test_CPU_high_load_blocks_promotion(self):
        self.assertEqual(self.bench["summary"]["performance_admission_verdict"],"INCONCLUSIVE_LOAD_CONTENDED")
        self.assertTrue(self.bench["summary"]["host_high_load_any_process"])
    def test_ABBA_RSS_and_23pct_relative_speed(self):
        s=self.bench["summary"]["summary"]
        self.assertGreater(s["fisher_22n8"]["native_max_RSS_bytes_median"],s["q4"]["native_max_RSS_bytes_median"])
        self.assertLess(s["fisher_22n8"]["best_of3_tps_median"],s["q4"]["best_of3_tps_median"])
    def test_source_Qwen3_seed_identity(self):
        seed=json.loads((ROOT/"docs/research/BEGLIN_QV7_QWEN64_SEED_REPARTITION_EVIDENCE_20261022.json").read_text())
        self.assertEqual(set(seed["3_seed_pairwise"]),{"17","41","73"})
        self.assertFalse(seed["model_quality_superiority_not_assessed"] is False)
        self.assertIn("NOT 6 statistically independent",seed["independence_warning"])
    def test_live_gates_off(self):
        self.assertFalse(self.mask["production_promotion"])
        summary=json.loads((ROOT/"docs/research/BEGLIN_QV7_CROSSFAMILY_COMPLETE_PUBLIC_RECEIPT_20261022.json").read_text())
        self.assertFalse(summary["live_or_main_changed"])
if __name__=="__main__":
    unittest.main(verbosity=2)
