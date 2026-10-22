#!/usr/bin/env python3
"""BEGLIN Qv2 native CPU parallel/precision policy conservative admission QA."""
from __future__ import annotations
import json,sys,unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v2_pareto_and_async_admission as q

class NativeQv2ConservativeAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence=q.audit()

    def test_all_sha_pinned_source_receipts_private_and_verifiable(self):
        m=q.read()
        self.assertEqual(set(m),set(q.PINS))
        self.assertEqual(m["input"]["windows"],8)
        self.assertEqual(m["ppl"]["status"],"PASS_ALL_FIVE_PUBLIC_EVALUATIONS")

    def test_identical_public_eval_inputs_and_131_teacher_forced_tokens(self):
        e=self.evidence
        self.assertEqual(e["locked_evaluation_window_count"],8)
        self.assertEqual(e["total_token_positions"],131)
        self.assertTrue(e["same_input_all_five_native_ppl_runs"])
        self.assertTrue(e["full_model_sustained_latency_or_task_AUROC_measured"] is False)

    def test_actual_v2_mask_same_as_milp_and_native_concurrency(self):
        e=self.evidence
        self.assertEqual(e["source_activation_quality_policy_sha256"],q.MASK)
        self.assertTrue(e["native_actual_parallel_QKV_and_gateup_executed"])
        self.assertTrue(e["output_logits_byte_equal_async_vs_serial"])
        self.assertTrue(e["PPL_equal_async_vs_serial"])

    def test_pareto_v2_precision_policy_dominated_by_Q4(self):
        d=self.evidence["dominance_comparator_metrics"]
        self.assertTrue(q.dominates(d["all_Q4"],d["v2_activation_MILP"]))
        self.assertLess(d["all_Q4"]["ppl"],d["v2_activation_MILP"]["ppl"])
        self.assertLess(d["all_Q4"]["ms"],d["v2_activation_MILP"]["ms"])
        self.assertLess(d["all_Q4"]["rss"],d["v2_activation_MILP"]["rss"])
        self.assertLess(d["all_Q4"]["bytes"],d["v2_activation_MILP"]["bytes"])

    def test_pareto_guard_rejects_partial_improvement(self):
        faster=dict(ppl=100.,ms=10.,rss=1000.,bytes=900.)
        otherwise=dict(ppl=90.,ms=11.,rss=1100.,bytes=1000.)
        self.assertFalse(q.dominates(faster,otherwise))
        self.assertFalse(q.dominates(faster,faster))
        with self.assertRaises(ValueError):
            q.dominates(dict(ppl=-1,ms=10,rss=1000,bytes=900),otherwise)

    def test_actual_async_ABBA_samples_do_not_pass_full_model_speed_gate(self):
        r=self.evidence["paired_actual_one_token_native_ms"]
        self.assertEqual(r["samples_per_mode"],3)
        self.assertAlmostEqual(r["serial_div_async_mean_ratio"],0.9996084991280209,places=8)
        self.assertFalse(self.evidence["native_async_meets_min_speedup"])
        self.assertEqual(self.evidence["research_default_async"],"disabled")
        self.assertEqual(self.evidence["optimized_precision_policy_auto_promotion"],"denied")

    def test_wrong_source_receipt_SHA_is_fail_closed(self):
        with patch.dict(q.PINS,{"ppl":"0"*64}):
            with self.assertRaisesRegex(RuntimeError,"SHA drift"):
                q.read()

    def test_receipt_immutable_mode_and_no_production_or_final_quality_claim(self):
        rec=q.OUT/"qv2_source_locked_native_async_pareto_guard.json"
        self.assertTrue(rec.is_file())
        self.assertEqual(rec.stat().st_mode&0o777,0o600)
        record=json.loads(rec.read_text())
        self.assertTrue(record["existing_V6_used_24_provenance_not_known"])
        self.assertEqual(record["scientific_Q_quality_verdict"],"NO_VERDICT")
        self.assertFalse(record["immutable_tailnet_release_or_repo_main_modified"])
        with self.assertRaises(FileExistsError):q.write()

if __name__=="__main__":
    unittest.main(verbosity=2)
