#!/usr/bin/env python3
import os
import tempfile
import unittest
from unittest.mock import patch

import autopilot_full as p5


class FullAutopilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan = os.path.join(self.tmp.name, "plan.json")
        self.audit = os.path.join(self.tmp.name, "audit.jsonl")
        self.state = os.path.join(self.tmp.name, "state.json")
        os.environ.pop(p5.lowrisk.P5_ENABLE_ENV, None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_merge_candidate_aggregates_corpus_rows(self):
        rows = {}
        p5._merge_candidate(rows, {
            "role": "kv_a_proj_with_mqa", "layer": 4,
            "event_count": 10, "current_bits": 8,
        })
        p5._merge_candidate(rows, {
            "role": "kv_a_proj_with_mqa", "layer": 4,
            "event_count": 3, "current_bits": 8,
        })
        got = rows[("kv_a_proj_with_mqa", 4)]
        self.assertEqual(got["event_count"], 13)
        self.assertEqual(got["source_rows"], 2)
    @patch.object(p5.shadow, "fetch_candidates")
    @patch.object(p5.pwb, "target_safe_n")
    @patch.object(p5.pwb, "read_remote_promotion_file", return_value={})
    def test_plan_admits_safe_attention_and_excludes_expert(
        self, _read, safe_n, fetch
    ):
        fetch.return_value = [
            {
                "role": "kv_a_proj_with_mqa", "layer": 4,
                "event_count": 10, "current_bits": 8,
            },
            {
                "role": "expert_up_proj", "layer": 4,
                "event_count": 99, "current_bits": 4,
            },
        ]
        safe_n.return_value = (5, {"per_corpus": {"c": {}}})
        plan = p5.build_plan(
            "m", 100, "h", "/promotion", None, self.plan
        )
        self.assertEqual(len(plan["changes"]), 1)
        self.assertEqual(plan["changes"][0]["role"], "kv_a_proj_with_mqa")
        self.assertEqual(plan["changes"][0]["new_n"], 5)
        self.assertEqual(plan["phase"], "P5-full-auto")

    @patch.object(
        p5.lowrisk, "_read_quarantine",
        return_value={("kv_a_proj_with_mqa", 4)},
    )
    @patch.object(p5.shadow, "fetch_candidates")
    @patch.object(p5.pwb, "target_safe_n")
    @patch.object(p5.pwb, "read_remote_promotion_file", return_value={})
    def test_attention_quarantine_blocks_repromotion(
        self, _read, safe_n, fetch, _quarantine
    ):
        fetch.return_value = [{
            "role": "kv_a_proj_with_mqa", "layer": 4,
            "event_count": 10, "current_bits": 8,
        }]
        plan = p5.build_plan(
            "m", 100, "h", "/promotion", "/quarantine", self.plan
        )
        self.assertEqual(plan["decisions"][0]["action"], "P4_QUARANTINED")
        self.assertEqual(plan["changes"], [])
        safe_n.assert_not_called()
    @patch.object(p5.shadow, "fetch_candidates", return_value=[])
    @patch.object(p5.pwb, "target_safe_n", return_value=(7, {"ok": True}))
    @patch.object(
        p5.pwb, "read_remote_promotion_file",
        return_value={("kv_a_proj_with_mqa", 13): 7},
    )
    def test_existing_attention_is_reaudited_as_noop(
        self, _read, _safe, _fetch
    ):
        plan = p5.build_plan(
            "m", 100, "h", "/promotion", None, self.plan
        )
        self.assertEqual(plan["decisions"][0]["action"], "NOOP")
        self.assertEqual(plan["changes"], [])

    @patch.object(p5, "prepare")
    @patch.object(p5.observer, "arm")
    def test_kill_switch_off_never_calls_observer_apply(self, arm, prepare):
        prepare.return_value = {
            "phase": "P5-full-auto",
            "status": "prepared",
            "changes": [{
                "action": "ADD", "role": "kv_a_proj_with_mqa",
                "layer": 4, "new_n": 5,
            }],
        }
        result = p5.arm_apply(
            "m", 100, "h", "/promotion", "/quarantine",
            self.plan, self.audit, "local", "/events",
            self.state, self.audit,
        )
        self.assertEqual(result["status"], "disabled")
        arm.assert_not_called()
    @patch.object(p5, "prepare")
    @patch.object(p5.observer, "arm")
    def test_enabled_arm_apply_delegates_to_observer(self, arm, prepare):
        os.environ[p5.lowrisk.P5_ENABLE_ENV] = "1"
        prepare.return_value = {
            "phase": "P5-full-auto",
            "status": "prepared",
            "changes": [{
                "action": "ADD", "role": "kv_a_proj_with_mqa",
                "layer": 4, "new_n": 5,
            }],
        }
        arm.return_value = {"status": "observing"}
        result = p5.arm_apply(
            "m", 100, "h", "/promotion", "/quarantine",
            self.plan, self.audit, "local", "/events",
            self.state, self.audit,
        )
        self.assertEqual(result["status"], "observing")
        arm.assert_called_once()
        self.assertTrue(arm.call_args.kwargs["apply_after"])

    @patch.object(p5, "prepare")
    @patch.object(p5.observer, "arm")
    def test_no_changes_never_calls_observer(self, arm, prepare):
        prepare.return_value = {
            "phase": "P5-full-auto",
            "status": "prepared",
            "changes": [],
        }
        result = p5.arm_apply(
            "m", 100, "h", "/promotion", "/quarantine",
            self.plan, self.audit, "local", "/events",
            self.state, self.audit,
        )
        self.assertEqual(result["status"], "no_changes")
        arm.assert_not_called()

if __name__ == "__main__":
    unittest.main()
