#!/usr/bin/env python3
"""BEGLIN Qv2 fixed evaluation independent native Apple CPU quality audit.

Rejects promotion of any precision policy whose measured quality and
latency/memory fail a pre-set engineering gate. Reports developmental
dominance as descriptive evidence only: 8 synthetic prompts (131 positions)
are not final generalization proof. No old V6/C1D files opened.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,re,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_native_policy_eval as helper

LAB=helper.TARGET
META=LAB/"fixed_input_receipt.json"
SUMMARY=LAB/"qv2_6case_native_evaluation_summary_private.json"
FROZEN=helper.FROZEN
CASES=helper.CASES
def sha(b):return hashlib.sha256(b).hexdigest()
def safe(path):
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise AssertionError("model-derived file must be private 0600 "+str(path))
    return path.read_bytes()

class QV2FixedNativeStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads(safe(META))
        cls.report=json.loads(safe(SUMMARY))
        cls.frozen=json.loads(FROZEN.read_text())
        cls.models={case:json.loads(safe(LAB/f"{case}_observed_native.json")) for case in CASES}
    def test_frozen_research_input_and_mask_before_model_eval(self):
        self.assertEqual(self.meta["status"],"PREPARED_AFTER_SOURCE_FROZEN_BEFORE_ANY_NATIVE_EVALUATION")
        self.assertEqual(self.frozen["status"],"PRE_EVALUATION_FROZEN_SOURCE_ONLY")
        self.assertEqual(self.meta["frozen_selected_policy_sha256"],self.frozen["selected_policy_sha256"])
        self.assertEqual(self.meta["total_predicted_tokens"],131)
        self.assertEqual(len(self.meta["eval_window_sha256"]),8)
        self.assertEqual(sha(safe(LAB/"v2_mask.txt")),self.frozen["selected_policy_sha256"])
    def test_six_conditions_share_identical_input_bytes_and_order(self):
        for i in range(8):
            blobs=[safe(LAB/case/f"eval/ppl_work/w{i}.i32") for case in CASES]
            self.assertTrue(all(b==blobs[0] for b in blobs),str(i))
            self.assertEqual(sha(blobs[0]),self.meta["eval_window_sha256"][i])
        self.assertEqual(len(set(sha(safe(LAB/case/"prompt_ids.i32")) for case in CASES)),1)
    def test_six_actual_native_model_binary_shas(self):
        for case,entry in self.models.items():
            self.assertEqual(entry["status"],"PASS_REAL_QWEN_NATIVE_28_LAYER")
            exe,mode,mask=helper.case_config(case)
            self.assertEqual(entry["native_binary_sha256"],sha(exe.read_bytes()))
            self.assertEqual(entry["policy_mode"],mode)
            if mask is not None:
                self.assertEqual(len(mask.read_text().splitlines()),28)
            self.assertFalse(entry["main_touched"])
            self.assertFalse(entry["production_touched"])
    def test_all_real_ppl_same_131_positions_and_logs(self):
        for case,entry in self.models.items():
            data=entry["phases"]["ppl"]
            self.assertEqual(data["prediction_positions"],131)
            self.assertTrue(math.isfinite(data["PPL"]) and data["PPL"]>0)
            self.assertAlmostEqual(math.log(data["PPL"]),data["NLL_nats_per_token"],places=11)
            self.assertGreater(data["peak_RSS_bytes"],0)
            self.assertEqual(sha(safe(LAB/f"{case}_ppl_log.txt")),data["log_SHA256"])
            self.assertIn("over 131 tokens",safe(LAB/f"{case}_ppl_log.txt").decode())
    def test_precision_mix_n8_packed_actual_and_8bit_dispatch(self):
        for case in ("v2_async_activation","v1_kernel_weight","role_balanced_random","role_balanced_proxy"):
            text=safe(LAB/f"{case}_ppl_log.txt").decode()
            self.assertIn("projection_role_tensors: f32=0 n8=28 q4=168",text)
            calls=self.models[case]["phases"]["ppl"]["native_repacked_n8_fast_kernel_calls"]
            self.assertGreater(calls,0)
    def test_v2_uses_source_only_F32_original_weights_and_frozen_mask(self):
        v=self.models["v2_async_activation"]
        self.assertIn("dump",v["phases"])
        self.assertIn("bench",v["phases"])
        self.assertEqual(v["phases"]["dump"]["top1_id"],9707)
        self.assertEqual(v["phases"]["dump"]["logits_sha256"],
            sha(safe(LAB/"v2_async_activation/results/c_logits_last.f32")))
        self.assertLess(v["phases"]["bench"]["best_of_3_one_decode_ms"],100.)
    def test_negative_output_quality_and_no_promotion(self):
        r=self.report["cases"]
        ppl={name:v["PPL"] for name,v in r.items()}
        self.assertGreater(ppl["v2_async_activation"],ppl["all_q4"])
        self.assertGreater(ppl["v2_async_activation"],ppl["role_balanced_random"])
        self.assertGreater(ppl["v2_async_activation"],ppl["v1_kernel_weight"])
        self.assertGreater(ppl["v2_async_activation"],ppl["role_balanced_proxy"])
        self.assertGreater(ppl["all_q4"],ppl["source_f32"])
        self.assertFalse(self.report["v2_beats_prior_random_NLL"])
        self.assertFalse(self.report["v2_beats_allQ4_NLL"])
    def test_no_false_async_full_model_claim_and_fail_closed(self):
        self.assertTrue(self.report["isolated_parallel_stage_speed_measured_but_not_endtoend_async"])
        self.assertFalse(self.report["final_task_AUROC_or_independent_OOD_test_not_performed"] is False)
        self.assertTrue(self.report["blocked_per_tensor_output_sensitivity_not_retried"])
        self.assertFalse(self.report["production_and_main_unchanged"] is False)
        with self.assertRaises(FileExistsError):
            helper.summarize()
if __name__=="__main__":
    unittest.main(verbosity=2)
