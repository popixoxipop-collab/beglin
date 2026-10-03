#!/usr/bin/env python3
import unittest

import precision_dynamic_selector as ds


def allocator_decision():
    return {
        "targets": [{
            "role": "shared_up_proj",
            "layer": 3,
            "status": "PROPOSED",
            "selected_n": 5,
            "current_n": 6,
            "dynamic_escalation": {
                "candidate_alternates": [
                    {"n": 6, "real_pass_events": 3, "persistent_p50_ms": None},
                    {"n": 9, "real_pass_events": 1, "persistent_p50_ms": None},
                ]
            },
        }]
    }


def trigger_rows(to_n, trigger="near_tie", count=3, requests=12, bucket=None):
    bucket = {"trigger_only": True} if bucket is None else bucket
    return [
        {
            "evidence_id": f"{trigger}-{to_n}-{i}",
            "role": "shared_up_proj",
            "layer": 3,
            "from_n": 5,
            "to_n": to_n,
            "trigger_type": trigger,
            "status": "PASS",
            "pass": True,
            "requests": requests,
            "signal_bucket": dict(bucket),
            "metrics": {"event_id": f"event-{trigger}-{to_n}-{i}"},
        }
        for i in range(count)
    ]


class DynamicSelectorTests(unittest.TestCase):
    def test_no_trigger_uses_low_cost_base(self):
        got = ds.select(allocator_decision(), {}, [])
        row = got["targets"][0]
        self.assertEqual(row["status"], "BASE_LOW_COST")
        self.assertEqual(row["selected_n"], 5)

    def test_trigger_without_context_evidence_holds_base(self):
        got = ds.select(allocator_decision(), {"near_tie": True}, [])
        row = got["targets"][0]
        self.assertEqual(row["status"], "HOLD_BASE_NO_TRIGGER_EVIDENCE")
        self.assertEqual(row["selected_n"], 5)
        self.assertTrue(row["required_calibrations"])

    def test_one_distinct_event_is_insufficient(self):
        got = ds.select(
            allocator_decision(),
            {"near_tie": True},
            trigger_rows(9, count=1, requests=36),
        )
        self.assertEqual(
            got["targets"][0]["status"],
            "HOLD_BASE_NO_TRIGGER_EVIDENCE",
        )
        self.assertEqual(got["targets"][0]["selected_n"], 5)

    def test_three_distinct_events_enable_exact_alternate(self):
        got = ds.select(
            allocator_decision(),
            {"near_tie": True},
            trigger_rows(9, count=3, requests=12),
        )
        row = got["targets"][0]
        self.assertEqual(row["status"], "TRIGGER_CONDITIONED_ALTERNATE")
        self.assertEqual(row["selected_n"], 9)

    def test_multiple_valid_alternates_choose_lower_cost_n(self):
        evidence = trigger_rows(6) + trigger_rows(9)
        got = ds.select(allocator_decision(), {"near_tie": True}, evidence)
        self.assertEqual(got["targets"][0]["selected_n"], 6)

    def test_empty_signal_bucket_is_not_reused(self):
        got = ds.select(
            allocator_decision(),
            {"near_tie": True},
            trigger_rows(6, bucket={}),
        )
        self.assertEqual(
            got["targets"][0]["status"],
            "HOLD_BASE_NO_TRIGGER_EVIDENCE",
        )

    def test_margin_bucket_must_match_current_signal(self):
        evidence = trigger_rows(
            6,
            trigger="low_margin",
            bucket={"margin_max": 0.02},
        )
        matched = ds.select(
            allocator_decision(),
            {"low_margin": True, "margin": 0.01},
            evidence,
        )
        self.assertEqual(
            matched["targets"][0]["status"],
            "TRIGGER_CONDITIONED_ALTERNATE",
        )
        blocked = ds.select(
            allocator_decision(),
            {"low_margin": True, "margin": 0.05},
            evidence,
        )
        self.assertEqual(
            blocked["targets"][0]["status"],
            "HOLD_BASE_NO_TRIGGER_EVIDENCE",
        )

    def test_all_active_triggers_need_coverage(self):
        evidence = trigger_rows(6, "near_tie")
        got = ds.select(
            allocator_decision(),
            {"near_tie": True, "routing_ambiguity": True},
            evidence,
        )
        self.assertEqual(
            got["targets"][0]["status"],
            "HOLD_BASE_NO_TRIGGER_EVIDENCE",
        )


if __name__ == "__main__":
    unittest.main()
