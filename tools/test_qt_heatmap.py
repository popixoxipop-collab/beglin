#!/usr/bin/env python3
from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from qt_heatmap import (
    build_local_precision_candidate,
    build_q_heatmap,
    build_selective_training_candidate,
    build_t_heatmap,
)
from verify_qt_heatmap_contracts import verify

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
BIN = "d" * 64

def ident(target: str, seq: int, window: str, backend: str = "mlx", weight_epoch: int = 0, qgroup: int = 0):
    return {
        "schema": "beglin-observation-identity-v2",
        "checkpoint_identity_sha256": HEX_A,
        "skeleton_sha256": HEX_B,
        "capability_bundle_sha256": HEX_C,
        "target_key": target,
        "model_id": "qwen2.5-0.5b-instruct",
        "layer": 0,
        "tensor_role": "q_proj",
        "expert_id": None,
        "row": 0,
        "column": None,
        "qgroup_index": qgroup,
        "group_size": 64,
        "element_index": None,
        "backend": backend,
        "source_dtype": "BF16",
        "runtime_dtype": "qNg64",
        "policy_epoch": 0,
        "weight_epoch": weight_epoch,
        "observation_window_id": window,
        "observation_seq": seq,
        "source_commit": "79d98de2",
        "binary_sha256": BIN,
    }

def qevent(target: str, n: int, err: float, seq: int, window: str, parity: float = 1e-5, restore: float = 0.0):
    return {
        "schema": "beglin-quant-perturbation-v1",
        "identity": ident(target, seq, window, qgroup=int(target.rsplit("=", 1)[1])),
        "candidate_n": n,
        "baseline_precision": "BF16",
        "quantized_precision": f"n{n}",
        "delta_weight_l1": err * 2,
        "delta_weight_l2": err,
        "relative_weight_error": err,
        "output_max_abs_error": err,
        "output_rms_error": err / 2,
        "output_relative_error": err,
        "latency_delta": -0.1,
        "memory_delta": -0.2,
        "backend_parity_error": parity,
        "restore_diff": restore,
        "finite": True,
        "result_status": "PASS",
        "sample_count": 1,
    }

def tevent(target: str, gtqe: float, seq: int, window: str, grad: float):
    return {
        "schema": "beglin-training-sensitivity-v1",
        "identity": ident(target, seq, window, backend="cpu-train", qgroup=int(target.rsplit("=", 1)[1])),
        "loss_before": 1.0,
        "loss_after": None,
        "grad_abs_mean": grad / 2,
        "grad_rms": grad,
        "grad_max_abs": grad * 2,
        "grad_norm": grad,
        "grad_variance_ema": 0.0,
        "quant_error_abs": gtqe / max(grad, 1e-9),
        "gradient_times_quant_error": gtqe,
        "fisher_diag_ema": None,
        "influence_proxy": gtqe,
        "optimizer_step": seq,
        "learning_rate": 1e-4,
        "sample_count": 1,
        "finite": True,
    }

class QtHeatmapTests(unittest.TestCase):
    def test_contract_set_verifies(self):
        result = verify()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["schema_count"], 9)

    def test_q_replay_is_order_independent_and_selects_n5(self):
        target = "qwen2.5/L0/q_proj/row=0/group64=0"
        rows = [
            qevent(target, 4, 8e-4, 0, "w0"),
            qevent(target, 4, 9e-4, 1, "w1"),
            qevent(target, 5, 1.0e-5, 2, "w0"),
            qevent(target, 5, 1.1e-5, 3, "w1"),
            qevent(target, 6, 5e-6, 4, "w0"),
            qevent(target, 6, 6e-6, 5, "w1"),
        ]
        a = build_q_heatmap(rows, {target: [4, 5, 6]}, {target: 6})
        shuffled = copy.deepcopy(rows)
        random.Random(7).shuffle(shuffled)
        b = build_q_heatmap(shuffled, {target: [4, 5, 6]}, {target: 6})
        self.assertEqual(a, b)
        self.assertEqual(a["cells"][0]["recommended_n"], 5)
        self.assertEqual(a["cells"][0]["state"], "CANDIDATE")
        self.assertEqual(a["cells"][0]["hysteresis_state"], "DOWNGRADE_ARMED")
        self.assertGreaterEqual(a["cells"][0]["confidence"], 0.75)

    def test_q_candidate_is_never_preapproved(self):
        target = "qwen2.5/L0/q_proj/row=0/group64=0"
        rows = [qevent(target, 5, 1e-5, 0, "w0"), qevent(target, 5, 1e-5, 1, "w1")]
        heatmap = build_q_heatmap(rows, {target: [5]}, {target: 6})
        policy = build_local_precision_candidate(heatmap, "q-candidate-1", {target: 0}, {target: 6})
        self.assertFalse(policy["approved_by_beval"])
        self.assertEqual(policy["cells"][0]["group_size"], 64)
        self.assertEqual(policy["cells"][0]["n"], 5)

    def test_t_replay_is_order_independent_and_separates_hot_from_cold(self):
        hot = "qwen2.5/L0/q_proj/row=0/group64=0"
        cold = "qwen2.5/L0/q_proj/row=0/group64=1"
        rows = [
            tevent(hot, 0.50, 0, "w0", 0.5),
            tevent(hot, 0.55, 1, "w1", 0.55),
            tevent(cold, 0.001, 2, "w0", 0.01),
            tevent(cold, 0.0012, 3, "w1", 0.011),
        ]
        a = build_t_heatmap(rows)
        shuffled = copy.deepcopy(rows)
        random.Random(11).shuffle(shuffled)
        b = build_t_heatmap(shuffled)
        self.assertEqual(a, b)
        by_target = {c["target_key"]: c for c in a["cells"]}
        self.assertTrue(by_target[hot]["recommended_trainable"])
        self.assertEqual(by_target[hot]["lr_scale"], 1.0)
        self.assertFalse(by_target[cold]["recommended_trainable"])
        policy = build_selective_training_candidate(a, "t-candidate-1")
        self.assertFalse(policy["approved_by_beval"])

    def test_cross_weight_epoch_is_rejected(self):
        target = "qwen2.5/L0/q_proj/row=0/group64=0"
        rows = [qevent(target, 5, 1e-5, 0, "w0"), qevent(target, 5, 1e-5, 1, "w1")]
        rows[1]["identity"]["weight_epoch"] = 1
        with self.assertRaisesRegex(ValueError, "weight_epoch"):
            build_q_heatmap(rows)

    def test_non_group64_is_rejected(self):
        target = "qwen2.5/L0/q_proj/row=0/group64=0"
        row = qevent(target, 5, 1e-5, 0, "w0")
        row["identity"]["group_size"] = 32
        with self.assertRaisesRegex(ValueError, "group_size=64"):
            build_q_heatmap([row])

if __name__ == "__main__":
    unittest.main(verbosity=2)
