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
