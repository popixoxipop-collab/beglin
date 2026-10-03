#!/usr/bin/env python3
import json
import struct
import tempfile
from pathlib import Path
import unittest

import precision_allocator as pa


def evidence(*, fail_n9=False):
    sweeps = []
    for n, passed, events in [
        (5, True, [(0, 16), (7, 8)]),
        (6, True, [(0, 16), (7, 8)]),
        (7, False, [(0, 16)]),
        (9, not fail_n9, [(0, 16)]),
    ]:
        for req, pos in events:
            sweeps.append({
                "corpus": "c", "role": "shared_up_proj", "layer": 3,
                "req": req, "pos": pos, "n": n, "pass": passed,
            })
    preflight = [
        {"role": "shared_up_proj", "layer": 3, "n": n,
         "pass": True, "status": "PASS", "context_hash": "ctx"}
        for n in (5, 6, 9)
    ]
    validation = [
        {"role": "shared_up_proj", "layer": 3, "n": n,
         "pass": True, "status": "CANARY_PASS", "context_hash": "ctx",
         "metrics": {
             "target_replay_pass": True,
             "post_attribution_hits": 0,
             "rollback_required": False,
         }}
        for n in (5, 6, 9)
    ]
    return {"sweeps": sweeps, "preflight": preflight, "validation": validation}


class PrecisionAllocatorTests(unittest.TestCase):
    def sizes(self):
        return {("shared_up_proj", 3): {
            "role": "shared_up_proj", "layer": 3,
            "tensor_count": 1, "numel": 1000,
        }}

    def test_nonmonotonic_candidates_are_independent(self):
        rows = pa.build_candidates(evidence(), self.sizes())
        by_n = {r["n"]: r for r in rows}
        self.assertTrue(by_n[5]["feasible"])
        self.assertTrue(by_n[6]["feasible"])
        self.assertFalse(by_n[7]["feasible"])
        self.assertTrue(by_n[9]["feasible"])
        self.assertIn("REAL_FAIL_EVENT", by_n[7]["blocked_reasons"])

    def test_real_fail_blocks_even_when_g4_g6_pass(self):
        rows = pa.build_candidates(evidence(fail_n9=True), self.sizes())
        by_n = {r["n"]: r for r in rows}
        self.assertFalse(by_n[9]["feasible"])
        self.assertIn("REAL_FAIL_EVENT", by_n[9]["blocked_reasons"])

    def test_memory_only_selects_smallest_feasible_cost(self):
        rows = pa.build_candidates(evidence(), self.sizes())
        got = pa.optimize(
            rows,
            current_policy=[{"role": "shared_up_proj", "layer": 3, "n": 6}],
            memory_weight=1.0, latency_weight=0.0, rss_weight=0.0,
        )
        target = got["targets"][0]
        self.assertEqual(target["selected_n"], 5)
        self.assertEqual(target["current_n"], 6)
        self.assertEqual(target["dynamic_escalation"]["status"], "EVIDENCE_REQUIRED")
        self.assertTrue(got["pairwise_evidence_required_before_combined_production"])

    def test_measured_e2e_cost_can_override_smaller_bitwidth(self):
        rows=pa.build_candidates(evidence(),self.sizes())
        by_n={r["n"]:r for r in rows}
        by_n[5]["expected_e2e_ms"]=30.0
        by_n[5]["resident_cache_bytes_after"]=300
        by_n[5]["transition_p50_ms"]=8.0
        by_n[5]["expected_inference_passes"]=2
        by_n[6]["expected_e2e_ms"]=12.0
        by_n[6]["resident_cache_bytes_after"]=200
        by_n[6]["transition_p50_ms"]=0.0
        by_n[6]["expected_inference_passes"]=1
        by_n[9]["expected_e2e_ms"]=20.0
        by_n[9]["resident_cache_bytes_after"]=400
        by_n[9]["transition_p50_ms"]=5.0
        by_n[9]["expected_inference_passes"]=1
        chosen=pa.choose_target(
            rows,memory_weight=0.0,e2e_weight=1.0,
            transition_weight=0.0,cache_weight=0.0,
            inference_pass_weight=0.0,
        )
        self.assertEqual(chosen["n"],6)
        self.assertIn("e2e",chosen["objective_dimensions"])

    def test_requested_e2e_dimension_requires_complete_measurements(self):
        rows=pa.build_candidates(evidence(),self.sizes())
        rows[0]["expected_e2e_ms"]=10.0
        with self.assertRaises(pa.AllocatorError):
            pa.choose_target(rows,memory_weight=0.0,e2e_weight=1.0)

    def test_measured_latency_can_change_choice(self):
        bench = {
            "shared_up_proj:3:5": {"p50_engine_ms": 10.0, "rss_bytes": 100},
            "shared_up_proj:3:6": {"p50_engine_ms": 8.0, "rss_bytes": 100},
            "shared_up_proj:3:9": {"p50_engine_ms": 4.0, "rss_bytes": 100},
        }
        rows = pa.build_candidates(evidence(), self.sizes(), bench)
        chosen = pa.choose_target(
            rows, memory_weight=1.0, latency_weight=3.0, rss_weight=0.0
        )
        self.assertEqual(chosen["n"], 9)
        self.assertIn("latency", chosen["objective_dimensions"])

    def conditional_evidence(self, with_trigger=True):
        role="shared_down_proj"; layer=26
        sweeps=[
            {"corpus":"safe","role":role,"layer":layer,"req":0,"pos":8,"n":5,"pass":True},
            {"corpus":"risk","role":role,"layer":layer,"req":0,"pos":9,"n":5,"pass":False},
            {"corpus":"safe","role":role,"layer":layer,"req":0,"pos":8,"n":6,"pass":True},
            {"corpus":"risk","role":role,"layer":layer,"req":0,"pos":9,"n":6,"pass":True},
            {"corpus":"risk","role":role,"layer":layer,"req":0,"pos":9,"n":7,"pass":False},
        ]
        preflight=[
            {"role":role,"layer":layer,"n":n,"pass":True,"status":"PASS","context_hash":f"ctx{n}"}
            for n in (5,6)
        ]
        validation=[
            {
                "role":role,"layer":layer,"n":n,"pass":True,
                "status":"CANARY_PASS","context_hash":f"ctx{n}",
                "metrics":{
                    "target_replay_pass":True,
                    "post_attribution_hits":0,
                    "rollback_required":False,
                },
            }
            for n in (5,6)
        ]
        trigger=[]
        if with_trigger:
            trigger=[{
                "evidence_id":"risk-low-margin-5-to-6",
                "role":role,"layer":layer,
                "from_n":5,"to_n":6,
                "trigger_type":"low_margin",
                "signal_bucket":{"margin_max":0.02},
                "requests":40,"pass":True,"status":"PASS",
                "evidence_sha256":"e"*64,
                "metrics":{
                    "base_failures":40,
                    "target_failures":0,
                    "source_event":{"corpus":"risk","req":0,"pos":9},
                },
            }]
        return {
            "sweeps":sweeps,
            "preflight":preflight,
            "validation":validation,
            "trigger":trigger,
        }

    def conditional_sizes(self):
        return {("shared_down_proj",26):{
            "role":"shared_down_proj","layer":26,
            "tensor_count":1,"numel":1000,
        }}

    def test_conditional_base_requires_all_failure_events_covered(self):
        rows=pa.build_candidates(
            self.conditional_evidence(with_trigger=False),
            self.conditional_sizes(),
        )
        by_n={r["n"]:r for r in rows}
        self.assertFalse(by_n[5]["feasible"])
        self.assertFalse(by_n[5]["conditionally_feasible"])
        self.assertTrue(by_n[6]["feasible"])
        got=pa.optimize(rows,memory_weight=1.0,latency_weight=0.0,rss_weight=0.0)
        target=got["targets"][0]
        self.assertEqual(target["selected_n"],6)
        self.assertEqual(target["policy_mode"],"STATIC")

    def test_conditional_base_uses_low_cost_n_only_with_recovery_evidence(self):
        rows=pa.build_candidates(
            self.conditional_evidence(with_trigger=True),
            self.conditional_sizes(),
        )
        by_n={r["n"]:r for r in rows}
        self.assertFalse(by_n[5]["feasible"])
        self.assertTrue(by_n[5]["conditionally_feasible"])
        self.assertTrue(by_n[6]["feasible"])
        self.assertEqual(by_n[5]["conditional_recoveries"][0]["to_n"],6)

        got=pa.optimize(rows,memory_weight=1.0,latency_weight=0.0,rss_weight=0.0)
        target=got["targets"][0]
        self.assertEqual(target["selected_n"],5)
        self.assertEqual(target["static_safe_n"],6)
        self.assertEqual(target["policy_mode"],"CONDITIONAL")
        self.assertEqual(target["dynamic_escalation"]["status"],"EVIDENCE_READY")
        self.assertEqual(
            target["dynamic_escalation"]["candidate_alternates"],
            [{"n":6,"real_pass_events":2,"persistent_p50_ms":None}],
        )

    def test_checkpoint_shared_and_expert_numel(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard = root / "model.safetensors"
            tensors = {
                "model.layers.3.mlp.shared_experts.up_proj.weight": {
                    "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12],
                },
                "model.layers.3.mlp.experts.0.up_proj.weight": {
                    "dtype": "BF16", "shape": [2, 2], "data_offsets": [12, 20],
                },
                "model.layers.3.mlp.experts.1.up_proj.weight": {
                    "dtype": "BF16", "shape": [2, 2], "data_offsets": [20, 28],
                },
            }
            header = json.dumps(tensors).encode()
            shard.write_bytes(struct.pack("<Q", len(header)) + header)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({
                "weight_map": {name: shard.name for name in tensors}
            }))
            shared = pa.target_numel(index, "shared_up_proj", 3)
            experts = pa.target_numel(index, "expert_up_proj", 3)
            self.assertEqual(shared["numel"], 6)
            self.assertEqual(shared["tensor_count"], 1)
            self.assertEqual(experts["numel"], 8)
            self.assertEqual(experts["tensor_count"], 2)

    def test_qng64_contract_excludes_q4_and_f16(self):
        self.assertIn(9, pa.SUPPORTED_QNG64)
        self.assertNotIn(4, pa.SUPPORTED_QNG64)
        self.assertNotIn(16, pa.SUPPORTED_QNG64)


if __name__ == "__main__":
    unittest.main()
