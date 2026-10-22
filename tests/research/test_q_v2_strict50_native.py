#!/usr/bin/env python3
"""Qv2 new strict 50ms stage-DAG MILP: fail-closed source+native regression."""
from __future__ import annotations
import hashlib,json,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v2_strict50_independent_audit as audit

class Qv2Strict50NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt=json.loads(audit.private(audit.EVIDENCE))
    def test_frozen_source_milp_optimality_certificate(self):
        j=json.loads(audit.FROZEN.read_text())
        self.assertEqual(j["solver_gap"],0)
        self.assertEqual(j["binary_variables"],448) if "binary_variables" in j else self.assertEqual(len(j["stage_per_layer_modes"]),112)
        self.assertEqual(len(j["selected_n8_role_layer_pairs"]),28)
        self.assertEqual(j["policy_text_sha256"],audit.POLICY_SHA)
    def test_112_dependency_stages_and_zero_data_leakage(self):
        j,s,st,k=audit.load_inputs()
        plan=audit.recompute_source_policy(j,s,st,k)
        self.assertEqual(plan["n8"],28)
        self.assertEqual(plan["n4"],168)
        self.assertEqual(sum(plan["role_counts"].values()),28)
        self.assertEqual(plan["role_counts"],
                         {"q_proj":1,"k_proj":1,"v_proj":23,"o_proj":1,
                          "gate_proj":0,"up_proj":2,"down_proj":0})
        self.assertTrue(plan["full_model_heldout_inputs_not_part_of_source_selection"])
    def test_source_exact_memory_and_time_constraints(self):
        plan=audit.recompute_source_policy(*audit.load_inputs())
        self.assertEqual(plan["packed_projection_bytes"] if "packed_projection_bytes" in plan else plan["packed_projection_bytes"],757825536)
        self.assertLess(plan["serial_C_kernel_cost_predicted_ms"],100)
        self.assertLessEqual(plan["measured_sibling_parallel_stage_sum_ms"],50)
        self.assertFalse(plan["stage_has_true_pthread_runtime"])
    def test_true_native_cpp_binding_source_SHA(self):
        j,s,st,k=audit.load_inputs()
        actual=audit.validate_actual_native(j)
        self.assertEqual(actual["native_binary_SHA256"],audit.ORIGINAL_NATIVE_BINARY_SHA)
        self.assertFalse(actual["actual_native_execution_parallel"])
        self.assertTrue(actual["reused_previously_seen_public_development_windows"])
    def test_native_131_tokens_speed_and_memory(self):
        actual=audit.validate_actual_native(audit.load_inputs()[0])
        self.assertEqual(actual["PPL_public_131_tokens"],807.363)
        self.assertEqual(actual["actual_single_token_best_of_three_ms"],83.24)
        self.assertEqual(actual["observed_ppl_process_max_RSS_bytes"],4850778112)
        self.assertEqual(actual["native_first_prompt_top1_id"],785)
    def test_quality_failure_preserved(self):
        a=audit.validate_actual_native(audit.load_inputs()[0])
        self.assertGreater(a["PPL_public_131_tokens"],a["all_q4_PPL"])
        self.assertGreater(a["PPL_public_131_tokens"],a["frozen_role_balanced_random_PPL"])
        self.assertGreater(a["PPL_public_131_tokens"],a["prior_v2_90ms_MILP_PPL"]*0.95)
        self.assertFalse(a["previous_native_async_speed_admission_gate_pass"])
    def test_source_protection_and_durable_receipt(self):
        self.assertEqual(audit.EVIDENCE.stat().st_mode&0o777,0o600)
        j=self.receipt
        self.assertEqual(j["scientific_Q_hypothesis"],"NO_VERDICT")
        self.assertTrue(j["no_production_or_main_change"])
        self.assertTrue(j["active_full_model_parallel_default_remains_OFF"])
    def test_fail_closed_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            audit.run()
if __name__=="__main__":
    unittest.main(verbosity=2)
