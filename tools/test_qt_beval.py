#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from qt_beval import approve_q_policy, approve_t_policy
from qt_heatmap import (
    build_local_precision_candidate,
    build_q_heatmap,
    build_selective_training_candidate,
    build_t_heatmap,
)
from test_qt_heatmap import BIN, HEX_A, HEX_B, qevent, tevent

SOURCE = "79d98de2"

def evidence(kind, target, heatmap, evidence_id, status="PASS", metrics=None):
    return {
        "schema": "beglin-qt-beval-evidence-v1",
        "evidence_id": evidence_id,
        "kind": kind,
        "target_key": target,
        "checkpoint_identity_sha256": heatmap["checkpoint_identity_sha256"],
        "skeleton_sha256": heatmap["skeleton_sha256"],
        "heatmap_sha256": heatmap["heatmap_sha256"],
        "weight_epoch": heatmap["weight_epoch"],
        "status": status,
        "metrics": metrics or {},
        "source_commit": SOURCE,
        "binary_sha256": BIN,
    }

class QtBevalTests(unittest.TestCase):
    def test_q_policy_only_approves_with_matching_passing_evidence(self):
        target = "qwen2.5/L0/q_proj/row=0/group64=0"
        rows = [qevent(target, 5, 1e-5, 0, "w0"), qevent(target, 5, 1.1e-5, 1, "w1")]
        heatmap = build_q_heatmap(rows, {target: [5]}, {target: 6})
        candidate = build_local_precision_candidate(heatmap, "q1", {target: 0}, {target: 6})
        ev = evidence("Q", target, heatmap, "qe1", metrics={
            "n": 5,
            "output_max_abs_error": 1.1e-5,
            "backend_parity_error": 1.0e-5,
            "restore_max_abs_diff": 0.0,
            "task_metric_delta": 0.0,
            "finite": True,
        })
        approved = approve_q_policy(candidate, heatmap, [ev])
        self.assertTrue(approved["approved_by_beval"])
        self.assertEqual(approved["beval_evidence_ids"], ["qe1"])

    def test_q_policy_rejects_heatmap_mismatch(self):
        target = "qwen2.5/L0/q_proj/row=0/group64=0"
        heatmap = build_q_heatmap([qevent(target, 5, 1e-5, 0, "w0"), qevent(target, 5, 1e-5, 1, "w1")])
        candidate = build_local_precision_candidate(heatmap, "q1", {target: 0}, {target: 6})
        candidate["heatmap_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "heatmap identity mismatch"):
            approve_q_policy(candidate, heatmap, [])

    def test_t_policy_only_approves_after_mask_holdout_lineage_and_post_q(self):
        target = "qwen2.5/L0/q_proj/row=0/group64=0"
        heatmap = build_t_heatmap([tevent(target, 0.5, 0, "w0", 0.5), tevent(target, 0.55, 1, "w1", 0.55)])
        candidate = build_selective_training_candidate(heatmap, "t1")
        ev = evidence("T", target, heatmap, "te1", metrics={
            "frozen_leakage_abs": 0.0,
            "holdout_metric_delta": 0.001,
            "selected_update_abs": 0.01,
            "post_q_pass": True,
            "lineage_complete": True,
            "finite": True,
        })
        approved = approve_t_policy(candidate, heatmap, [ev])
        self.assertTrue(approved["approved_by_beval"])
        self.assertEqual(approved["beval_evidence_ids"], ["te1"])

    def test_t_policy_rejects_mask_leak(self):
        target = "qwen2.5/L0/q_proj/row=0/group64=0"
        heatmap = build_t_heatmap([tevent(target, 0.5, 0, "w0", 0.5), tevent(target, 0.55, 1, "w1", 0.55)])
        candidate = build_selective_training_candidate(heatmap, "t1")
        ev = evidence("T", target, heatmap, "te1", metrics={
            "frozen_leakage_abs": 1e-9,
            "holdout_metric_delta": 0.0,
            "selected_update_abs": 0.01,
            "post_q_pass": True,
            "lineage_complete": True,
            "finite": True,
        })
        with self.assertRaisesRegex(ValueError, "frozen-mask leakage"):
            approve_t_policy(candidate, heatmap, [ev])

if __name__ == "__main__":
    unittest.main(verbosity=2)
