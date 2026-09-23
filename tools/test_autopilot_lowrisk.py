#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import autopilot_lowrisk as p3


class LowRiskAutopilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan = os.path.join(self.tmp.name, "plan.json")
        self.audit = os.path.join(self.tmp.name, "audit.jsonl")
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop(p3.P3_ENABLE_ENV, None)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    @patch.object(p3.shadow, "fetch_candidates")
    @patch.object(p3.pwb, "target_safe_n")
    @patch.object(p3.pwb, "read_remote_promotion_file", return_value={})
    def test_plan_filters_non_ffn_roles(self, _read, safe_n, fetch):
        fetch.return_value = [
            {"role": "q_proj", "layer": 1, "event_count": 9, "current_bits": 4},
            {"role": "shared_gate_proj", "layer": 2, "event_count": 8, "current_bits": 4},
        ]
        safe_n.return_value = (5, {"per_corpus": {}})
        plan = p3.build_plan("m", 10, "h", "/x", self.plan)
        self.assertEqual([(d["role"], d["layer"]) for d in plan["decisions"]],
                         [("shared_gate_proj", 2)])
        self.assertEqual(plan["changes"][0]["action"], "ADD")

    @patch.object(p3.shadow, "fetch_candidates", return_value=[])
    @patch.object(
        p3.pwb, "target_safe_n",
        return_value=(None, {"reason": "regressed"}),
    )
    @patch.object(
        p3.pwb, "read_remote_promotion_file",
        return_value={("shared_up_proj", 3): 5},
    )
    def test_unsafe_existing_is_review_not_auto_remove(
        self, _read, _safe, _fetch
    ):
        plan = p3.build_plan("m", 10, "h", "/x", self.plan)
        self.assertEqual(
            plan["decisions"][0]["action"], "P2_REVIEW_UNSAFE"
        )
        self.assertEqual(plan["changes"], [])
        self.assertEqual(
            p3.guarded._mapping(plan["after"]),
            {("shared_up_proj", 3): 5},
        )

    @patch.object(p3, "build_plan")
    @patch.object(p3.guarded, "apply_plan")
    def test_kill_switch_off_never_applies(self, apply, build):
        build.return_value = {
            "phase": "P3-lowrisk-auto",
            "status": "prepared",
            "model": "m",
            "decisions": [],
            "changes": [{"action": "ADD"}],
        }
        result = p3.run_once("m", 10, "h", "/x", self.plan, self.audit)
        self.assertEqual(result["status"], "disabled")
        apply.assert_not_called()
    @patch.object(p3, "build_plan")
    @patch.object(p3.guarded, "apply_plan")
    def test_enabled_run_delegates_to_guarded_transaction(self, apply, build):
        os.environ[p3.P3_ENABLE_ENV] = "1"
        build.return_value = {
            "phase": "P3-lowrisk-auto",
            "status": "prepared",
            "model": "m",
            "decisions": [],
            "changes": [{
                "action": "ADD", "role": "shared_gate_proj",
                "layer": 14, "old_n": None, "new_n": 5,
                "event_count": 1,
            }],
        }
        apply.return_value = {
            **build.return_value,
            "status": "applied",
            "remote_backup": "/backup",
        }
        result = p3.run_once("m", 10, "h", "/x", self.plan, self.audit)
        self.assertEqual(result["status"], "applied")
        apply.assert_called_once_with(self.plan)
        with open(self.audit) as f:
            row = json.loads(f.readline())
        self.assertEqual(row["status"], "applied")

    def test_safe_n_below_existing_requires_p2_review(self):
        d = p3._decision(
            "shared_down_proj", 4, 7, 5, 2, {"per_corpus": {}}
        )
        self.assertEqual(d["action"], "P2_REVIEW_DOWNGRADE")
    @patch.object(p3, "_read_quarantine", return_value={("shared_up_proj", 3)})
    @patch.object(p3.shadow, "fetch_candidates")
    @patch.object(p3.pwb, "target_safe_n")
    @patch.object(p3.pwb, "read_remote_promotion_file", return_value={})
    def test_p4_quarantine_blocks_repromotion(
        self, _read, safe_n, fetch, _quarantine
    ):
        fetch.return_value = [{
            "role": "shared_up_proj", "layer": 3,
            "event_count": 14, "current_bits": 4,
        }]
        plan = p3.build_plan(
            "m", 10, "h", "/x", self.plan, quarantine_file="/q"
        )
        self.assertEqual(plan["decisions"][0]["action"], "P4_QUARANTINED")
        self.assertEqual(plan["changes"], [])
        safe_n.assert_not_called()

    @patch.object(p3, "build_plan")
    @patch.object(p3.guarded, "apply_plan")
    def test_prepare_only_keeps_plan_for_p4(self, apply, build):
        build.return_value = {
            "phase": "P3-lowrisk-auto",
            "status": "prepared",
            "model": "m",
            "decisions": [],
            "changes": [{"action": "ADD"}],
        }
        result = p3.run_once(
            "m", 10, "h", "/x", self.plan, self.audit,
            prepare_only=True,
        )
        self.assertEqual(result["status"], "prepared")
        apply.assert_not_called()

    @patch.object(p3, "build_plan")
    @patch.object(p3.guarded, "apply_plan")
    def test_enabled_no_changes_does_not_mutate(self, apply, build):
        os.environ[p3.P3_ENABLE_ENV] = "1"
        build.return_value = {
            "phase": "P3-lowrisk-auto",
            "status": "prepared",
            "model": "m",
            "decisions": [],
            "changes": [],
        }
        result = p3.run_once("m", 10, "h", "/x", self.plan, self.audit)
        self.assertEqual(result["status"], "no_changes")
        apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
