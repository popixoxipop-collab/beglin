#!/usr/bin/env python3
"""Synthetic tests of full 64-score router auditing (no model or GPU)."""
import copy
import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import je01_router_trace_audit as rt

IDS = [hashlib.sha256(("synthetic-prompt-%d" % n).encode()).hexdigest()
       for n in range(256)]
CK = "a" * 64
COHORT = "b" * 64


def trace(i, *, layer=11):
    scores = [0.01] * 64
    for expert in range(8):
        scores[expert] = float(8 - ((expert - i % 8) % 8))
    total = sum(scores)
    probs = [x / total for x in scores]
    selected = sorted(range(64), key=lambda k: (-probs[k], k))[:6]
    return {"sample_sha256": IDS[i], "layer": layer, "token_pos": 0,
            "router_probs64": probs, "selected_expert_ids": selected}


def traces():
    return [trace(i) for i in range(len(IDS))]


class RouterAudit(unittest.TestCase):
    def freeze(self, rows=None):
        return rt.freeze_pilot_slots(traces() if rows is None else rows,
            checkpoint_sha256=CK, dataset_sha256=COHORT,
            expected_sample_sha256=IDS)

    def test_freeze_8_slots_from_256_unique_prompts(self):
        state = self.freeze()
        self.assertEqual(state["total_unique_calibration_prompts"], 256)
        self.assertEqual(state["total_router_events"], 256)
        self.assertEqual(len(state["selected_slots"]), 8)
        self.assertEqual(set(x["expert_id"] for x in state["selected_slots"]), set(range(8)))
        self.assertFalse(state["external_original_native_provenance_verified"])
        self.assertFalse(state["gold_token_nll_or_labels_used"])
        self.assertGreater(state["minimum_topk_boundary_gap"], 0)

    def test_order_invariant_freeze(self):
        self.assertEqual(self.freeze(), self.freeze(list(reversed(traces()))))

    def test_label_leakage_denied(self):
        rows = traces()
        rows[0]["gold_nll"] = 3
        with self.assertRaisesRegex(ValueError, "OUTCOME_LEAKAGE"):
            self.freeze(rows)

    def test_duplicate_router_events_denied(self):
        rows = traces()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "DUPLICATE_ROUTER_EVENT"):
            self.freeze(rows)

    def test_missing_prompt_denied(self):
        with self.assertRaisesRegex(ValueError, "COVERAGE_INCOMPLETE"):
            self.freeze(traces()[:-1])

    def test_topk_wrong_denied(self):
        rows = traces()
        rows[5]["selected_expert_ids"] = [30,31,32,33,34,35]
        with self.assertRaisesRegex(ValueError, "TOPK_MISMATCH"):
            self.freeze(rows)

    def test_softmax_invalid_denied(self):
        rows = traces()
        rows[0]["router_probs64"][0] = -1
        with self.assertRaisesRegex(ValueError, "SOFTMAX_INVALID"):
            self.freeze(rows)

    def test_frozen_ids_required(self):
        with self.assertRaisesRegex(ValueError, "CALIBRATION_256"):
            rt.freeze_pilot_slots(traces(),
                checkpoint_sha256=CK,dataset_sha256=COHORT,
                expected_sample_sha256=IDS[:255])

    def test_topk_boundary_tie_denied(self):
        rows = traces()
        rows[0]["router_probs64"] = [1/64] * 64
        rows[0]["selected_expert_ids"] = list(range(6))
        with self.assertRaisesRegex(ValueError, "BOUNDARY_TIE"):
            self.freeze(rows)

    def test_same_frozen_routes_zero_flip(self):
        a = traces()
        report = rt.compare_frozen_router_traces(a, copy.deepcopy(a),
                                                  expected_sample_sha256=IDS)
        self.assertEqual(report["event_pairs"], 256)
        self.assertTrue(report["no_observed_flip"])
        self.assertFalse(report["production_promotion_authorized"])

    def test_one_cross_model_expert_flip_is_detected(self):
        baseline = traces()
        cand = copy.deepcopy(baseline)
        old = cand[25]["selected_expert_ids"]
        cand[25]["selected_expert_ids"] = old[:-1] + [18]
        report = rt.compare_frozen_router_traces(baseline, cand,
                                                  expected_sample_sha256=IDS)
        self.assertEqual(report["topk_flip_count"], 1)
        self.assertFalse(report["no_observed_flip"])

    def test_router_events_must_align_exactly(self):
        baseline = traces()
        candidate = copy.deepcopy(baseline[:-1])
        with self.assertRaisesRegex(ValueError, "ALIGNMENT_FAILED"):
            rt.compare_frozen_router_traces(baseline, candidate,
                                             expected_sample_sha256=IDS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
