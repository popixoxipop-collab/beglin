#!/usr/bin/env python3
import unittest

import precision_closed_loop as pcl
import precision_context as pc


CURRENT = [
    {"role": "shared_down_proj", "layer": 26, "n": 5},
    {"role": "shared_up_proj", "layer": 3, "n": 6},
]
TARGET = [
    {"role": "shared_down_proj", "layer": 26, "n": 6},
    {"role": "shared_up_proj", "layer": 3, "n": 5},
]


def candidate(role, layer, n, *, feasible, conditional=False, recoveries=None):
    return {
        "role": role,
        "layer": layer,
        "n": n,
        "feasible": feasible,
        "conditionally_feasible": conditional,
        "conditional_recoveries": list(recoveries or []),
        "blocked_reasons": [] if feasible else ["REAL_FAIL_EVENT"],
        "real_pass_events": 4,
        "real_fail_events": 1 if conditional else 0,
        "real_event_count": 5,
        "real_pass_event_keys": [],
        "real_fail_event_keys": [],
        "g4_context_hash": f"g4-{role}-{layer}-{n}",
        "g6_context_hash": f"g6-{role}-{layer}-{n}",
        "tensor_count": 1,
        "numel": 1000,
        "effective_bpw": float(n),
        "estimated_bytes": float(n * 1000),
        "persistent_p50_ms": None,
        "persistent_p95_ms": None,
        "persistent_rss_bytes": None,
        "benchmark_pid": None,
    }


def candidates():
    return [
        candidate("shared_up_proj", 3, 5, feasible=True),
        candidate("shared_up_proj", 3, 6, feasible=True),
        candidate(
            "shared_down_proj", 26, 5,
            feasible=False,
            conditional=True,
            recoveries=[{
                "to_n": 6,
                "trigger_type": "low_margin",
                "signal_bucket": {"margin_max": 0.02},
            }],
        ),
        candidate("shared_down_proj", 26, 6, feasible=True),
    ]


TRIGGER = [{
    "role": "shared_down_proj",
    "layer": 26,
    "from_n": 5,
    "to_n": 6,
    "trigger_type": "low_margin",
    "status": "PASS",
    "pass": True,
    "requests": 12,
    "signal_bucket": {"margin_max": 0.02},
    "metrics": {"base_failures": 12, "target_failures": 0},
}]


def combined(policy=TARGET):
    return [{
        "policy_hash": pc.policy_hash(policy),
        "status": "PASS",
        "pass": True,
        "production_touched": False,
        "evidence_sha256": "a" * 64,
    }]


class ClosedLoopTests(unittest.TestCase):
    def engine(self, combined_rows=None):
        return pcl.PrecisionClosedLoopEngine(
            candidates=candidates(),
            trigger_evidence=TRIGGER,
            combined_policy_evidence=(
                combined() if combined_rows is None else combined_rows
            ),
        )

    def test_low_margin_builds_exact_multi_target_policy(self):
        got = self.engine().decide(
            current_policy=CURRENT,
            signal={"low_margin": True, "margin": 0.010715},
        )
        self.assertEqual(got["status"], "READY_FOR_SCHEDULER")
        self.assertEqual(got["selected_policy"], pc.normalize_policy(TARGET))
        self.assertEqual(len(got["changes"]), 2)
        by_key = {
            (r["role"], r["layer"]): r
            for r in got["selection"]["targets"]
        }
        self.assertEqual(
            by_key[("shared_down_proj", 26)]["status"],
            "TRIGGER_CONDITIONED_ALTERNATE",
        )
        self.assertEqual(
            by_key[("shared_up_proj", 3)]["status"],
            "HOLD_BASE_NO_TRIGGER_EVIDENCE",
        )

    def test_missing_combined_policy_acceptance_fails_closed(self):
        with self.assertRaises(pcl.PrecisionClosedLoopError):
            self.engine([]).decide(
                current_policy=CURRENT,
                signal={"low_margin": True, "margin": 0.010715},
            )

    def test_out_of_bucket_signal_cannot_use_trigger_evidence(self):
        # Allocator would still lower L3 to n5. Because the resulting combined
        # low-cost policy lacks exact combined acceptance, it must fail closed.
        with self.assertRaises(pcl.PrecisionClosedLoopError):
            self.engine().decide(
                current_policy=CURRENT,
                signal={"low_margin": True, "margin": 0.03},
            )

    def test_selector_cannot_escape_allocator_candidate_set(self):
        engine = self.engine()
        allocation = __import__("precision_allocator").optimize(
            candidates(), current_policy=CURRENT
        )
        selection = __import__("precision_dynamic_selector").select(
            allocation,
            {"low_margin": True, "margin": 0.010715},
            TRIGGER,
        )
        selection["targets"][0]["selected_n"] = 15
        with self.assertRaises(pcl.PrecisionClosedLoopError):
            pcl.materialize_selected_policy(
                current_policy=CURRENT,
                allocation=allocation,
                selection=selection,
                combined_policy_evidence=combined(),
            )


if __name__ == "__main__":
    unittest.main()
