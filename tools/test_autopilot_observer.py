#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import autopilot_observer as obs


def attr(req, role="shared_up_proj", layer=3, margin=None):
    return {
        "kind": "attribution", "ts_unix": 1, "req": req, "pos": 8,
        "role": role, "layer": layer, "corrected_argmax": 2,
        "orig_argmax": 1, "threshold": 0.1,
        "model": "m", "corpus": "c", "manifest": "/tmp/m",
    }


def event(req, margin):
    return {
        "kind": "event", "ts_unix": 2, "req": req, "pos": 8,
        "predicted_token": 2, "competing_token": 1, "margin": margin,
        "model": "m", "corpus": "c",
    }


def write_rows(path, rows, mode="w"):
    with open(path, mode) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.tmp.name, "events.jsonl")
        self.plan = os.path.join(self.tmp.name, "plan.json")
        self.state = os.path.join(self.tmp.name, "state.json")
        self.audit = os.path.join(self.tmp.name, "audit.jsonl")
        self.demote = os.path.join(self.tmp.name, "demote.txt")
        os.environ.pop(obs.P4_ENABLE_ENV, None)

    def tearDown(self):
        self.tmp.cleanup()
    def make_plan(self):
        plan = {
            "version": 1, "phase": "P3-lowrisk-auto",
            "status": "prepared", "model": "m", "ssh_host": "fake",
            "promotion_file": "/tmp/promotion.txt",
            "before": [], "after": [
                {"role": "shared_up_proj", "layer": 3, "n": 5},
            ],
            "before_sha256": obs.guarded._mapping_hash({}),
            "after_sha256": obs.guarded._mapping_hash(
                {("shared_up_proj", 3): 5}
            ),
            "changes": [{
                "action": "ADD", "role": "shared_up_proj", "layer": 3,
                "old_n": None, "new_n": 5, "event_count": 2,
            }],
            "decisions": [],
        }
        obs.guarded._atomic_json(self.plan, plan)
        return plan

    def test_parse_pairs_attribution_before_event(self):
        blob = (
            json.dumps(attr(0)) + "\n" +
            json.dumps(event(0, 0.02)) + "\n"
        ).encode()
        events = obs.parse_events(blob, model="m")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["attrs"][0]["role"], "shared_up_proj")
        self.assertEqual(events[0]["margin"], 0.02)

    def test_baseline_suffix_is_shortest_recent_window_covering_target(self):
        events = obs.parse_events((
            json.dumps(event(0, 0.08)) + "\n" +
            json.dumps(attr(1)) + "\n" +
            json.dumps(event(1, 0.02)) + "\n" +
            json.dumps(event(2, 0.07)) + "\n"
        ).encode(), model="m")
        base = obs.baseline_suffix(
            events, [{"role": "shared_up_proj", "layer": 3}]
        )
        self.assertEqual(len(base), 2)
        self.assertEqual(
            obs.target_metrics(base, "shared_up_proj", 3)["attributed_events"],
            1,
        )
    def test_arm_captures_data_derived_window(self):
        self.make_plan()
        write_rows(self.log, [
            event(0, 0.08), attr(1), event(1, 0.02), event(2, 0.07),
        ])
        state = obs.arm(
            self.plan, "local", self.log, self.state,
            self.demote, self.audit, apply_after=False,
        )
        self.assertEqual(state["baseline_event_count"], 2)
        m = state["baseline_metrics"]["shared_up_proj:3"]
        self.assertEqual(m["attributed_events"], 1)
        self.assertEqual(m["median_margin"], 0.02)

    def _armed(self):
        self.make_plan()
        write_rows(self.log, [
            attr(0), event(0, 0.02), event(1, 0.07),
        ])
        obs.arm(
            self.plan, "local", self.log, self.state,
            self.demote, self.audit, apply_after=False,
        )
        with open(self.plan) as f:
            plan = json.load(f)
        plan["status"] = "applied"
        obs.guarded._atomic_json(self.plan, plan)
        return obs.begin_observation(self.state, self.audit)

    def test_check_waits_for_equal_post_sample_size(self):
        self._armed()
        write_rows(self.log, [event(2, 0.08)], mode="a")
        state = obs.check(self.state, self.audit)
        self.assertEqual(state["status"], "collecting")
        self.assertEqual(state["post_events"], 1)

    def test_begin_excludes_arm_apply_race_events(self):
        self.make_plan()
        write_rows(self.log, [
            attr(0), event(0, 0.02), event(1, 0.07),
        ])
        armed = obs.arm(
            self.plan, "local", self.log, self.state,
            self.demote, self.audit, apply_after=False,
        )
        old_offset = armed["log_offset"]
        write_rows(self.log, [event(99, 0.05)], mode="a")
        with open(self.plan) as f:
            plan = json.load(f)
        plan["status"] = "applied"
        obs.guarded._atomic_json(self.plan, plan)
        begun = obs.begin_observation(self.state, self.audit)
        self.assertGreater(begun["log_offset"], old_offset)
        write_rows(self.log, [event(2, 0.08)], mode="a")
        state = obs.check(self.state, self.audit)
        self.assertEqual(state["status"], "collecting")
        self.assertEqual(state["post_events"], 1)

    def test_zero_post_attributions_is_healthy(self):
        self._armed()
        write_rows(self.log, [
            event(2, 0.08), event(3, 0.09),
        ], mode="a")
        state = obs.check(self.state, self.audit)
        self.assertEqual(state["status"], "healthy")
        result = state["results"]["shared_up_proj:3"]
        self.assertTrue(result["healthy"])
        self.assertEqual(result["post"]["attributed_events"], 0)
    def test_no_count_improvement_is_regression_with_killswitch_off(self):
        self._armed()
        write_rows(self.log, [
            attr(2), event(2, 0.03), event(3, 0.09),
        ], mode="a")
        state = obs.check(self.state, self.audit)
        self.assertEqual(
            state["status"], "regression_detected_killswitch_off"
        )
        result = state["results"]["shared_up_proj:3"]
        self.assertFalse(result["healthy"])
        self.assertFalse(
            result["criteria"]["attribution_count_strictly_lower"]
        )

    @patch.object(obs.pwb, "write_remote_promotion_file_atomic")
    @patch.object(obs.pwb, "read_remote_promotion_file")
    @patch.object(obs, "_write_demotion_targets_atomic")
    @patch.object(obs, "_read_demotion_targets")
    def test_demotion_preserves_unrelated_promotions(
        self, read_demote, write_demote, read_promote, write_promote
    ):
        read_promote.return_value = {
            ("shared_up_proj", 3): 5,
            ("kv_a_proj_with_mqa", 13): 7,
        }
        read_demote.return_value = {("shared_gate_proj", 14)}
        state = {
            "ssh_host": "fake",
            "promotion_file": "/tmp/promotion.txt",
            "demotion_file": "/tmp/demote.txt",
        }
        failed = [{
            "role": "shared_up_proj", "layer": 3, "new_n": 5,
        }]
        demoted = obs._demote_failed(state, failed)
        self.assertEqual(demoted, [("shared_up_proj", 3)])
        write_promote.assert_called_once_with(
            "fake", "/tmp/promotion.txt",
            {("kv_a_proj_with_mqa", 13): 7},
        )
        written = write_demote.call_args.args[2]
        self.assertEqual(
            written,
            {("shared_gate_proj", 14), ("shared_up_proj", 3)},
        )
    def test_local_demotion_file_round_trip(self):
        obs._write_demotion_targets_atomic(
            "local", self.demote,
            {("shared_up_proj", 3), ("shared_gate_proj", 14)},
        )
        got = obs._read_demotion_targets("local", self.demote)
        self.assertEqual(
            got,
            {("shared_up_proj", 3), ("shared_gate_proj", 14)},
        )


if __name__ == "__main__":
    unittest.main()
