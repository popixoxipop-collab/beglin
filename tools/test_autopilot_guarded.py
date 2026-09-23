#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import autopilot_guarded as ag


class GuardedAutopilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_plan(self, before, after, changes, status="prepared"):
        plan = {
            "version": ag.PLAN_VERSION,
            "phase": "P2-approval-gate",
            "status": status,
            "model": "m",
            "ssh_host": "fake",
            "promotion_file": "/tmp/live.txt",
            "before": ag._rows(before),
            "after": ag._rows(after),
            "before_sha256": ag._mapping_hash(before),
            "after_sha256": ag._mapping_hash(after),
            "changes": changes,
            "decisions": [],
        }
        ag._atomic_json(self.plan_path, plan)
        return plan

    @patch.object(ag.shadow, "fetch_candidates")
    @patch.object(ag.pwb, "target_safe_n")
    @patch.object(ag.pwb, "read_remote_promotion_file")
    def test_prepare_enforces_limit_and_audits_existing(
        self, read_live, safe_n, fetch
    ):
        read_live.return_value = {("live_role", 9): 7}
        fetch.return_value = [
            {
                "role": f"r{i}", "layer": i, "event_count": 100 - i,
                "current_bits": 4,
            }
            for i in range(12)
        ]
        safe_n.side_effect = lambda model, role, layer: (
            (7, {"ok": True}) if role == "live_role"
            else (None, {"reason": "unsafe"})
        )
        plan = ag.prepare_plan(
            "m", 10, "fake", "/tmp/live.txt", self.plan_path
        )
        queried = [(d["role"], d["layer"]) for d in plan["decisions"]]
        self.assertEqual(len(queried), 11)
        self.assertNotIn(("r10", 10), queried)
        self.assertNotIn(("r11", 11), queried)
        self.assertIn(("live_role", 9), queried)
        self.assertEqual(plan["changes"], [])

    @patch.object(ag.shadow, "fetch_candidates", return_value=[])
    @patch.object(
        ag.pwb, "target_safe_n",
        return_value=(None, {"reason": "evidence regressed"}),
    )
    @patch.object(ag.pwb, "read_remote_promotion_file")
    def test_prepare_surfaces_unsafe_live_removal(
        self, read_live, _safe, _fetch
    ):
        read_live.return_value = {("r", 1): 7}
        plan = ag.prepare_plan(
            "m", 10, "fake", "/tmp/live.txt", self.plan_path
        )
        self.assertEqual(plan["after"], [])
        self.assertEqual(plan["changes"][0]["action"], "REMOVE_UNSAFE")
        self.assertIsNone(plan["changes"][0]["new_n"])
    def test_apply_success_is_exact_and_audited(self):
        before = {("r", 1): 5}
        after = {("r", 1): 6}
        changes = [{
            "action": "UPGRADE", "role": "r", "layer": 1,
            "old_n": 5, "new_n": 6, "event_count": 3,
        }]
        self._write_plan(before, after, changes)
        state = dict(before)

        def read_live(_host, _path):
            return dict(state)

        def write_live(_host, _path, mapping):
            state.clear()
            state.update(mapping)
            return len(mapping)

        with patch.object(ag, "_acquire_lock", return_value="/lock"),              patch.object(ag, "_release_lock"),              patch.object(ag, "_snapshot_remote_file", return_value="/backup"),              patch.object(ag.pwb, "read_remote_promotion_file", side_effect=read_live),              patch.object(ag.pwb, "write_remote_promotion_file_atomic", side_effect=write_live),              patch.object(ag.pwb, "target_safe_n", return_value=(6, {"ok": True})):
            out = ag.apply_plan(self.plan_path)

        self.assertEqual(state, after)
        self.assertEqual(out["status"], "applied")
        self.assertEqual(out["remote_backup"], "/backup")

    def test_postwrite_evidence_change_rolls_back(self):
        before = {("r", 1): 5}
        after = {("r", 1): 6}
        changes = [{
            "action": "UPGRADE", "role": "r", "layer": 1,
            "old_n": 5, "new_n": 6, "event_count": 3,
        }]
        self._write_plan(before, after, changes)
        state = dict(before)

        def read_live(_host, _path):
            return dict(state)

        def write_live(_host, _path, mapping):
            state.clear()
            state.update(mapping)
            return len(mapping)

        evidence = [
            (6, {"ok": True}),
            (None, {"reason": "new contradiction"}),
        ]
        with patch.object(ag, "_acquire_lock", return_value="/lock"),              patch.object(ag, "_release_lock"),              patch.object(ag, "_snapshot_remote_file", return_value="/backup"),              patch.object(ag.pwb, "read_remote_promotion_file", side_effect=read_live),              patch.object(ag.pwb, "write_remote_promotion_file_atomic", side_effect=write_live),              patch.object(ag.pwb, "target_safe_n", side_effect=evidence):
            with self.assertRaises(RuntimeError):
                ag.apply_plan(self.plan_path)

        self.assertEqual(state, before)
        with open(self.plan_path) as f:
            saved = json.load(f)
        self.assertEqual(saved["status"], "apply_failed_rolled_back")

    def test_stale_plan_never_writes(self):
        before = {("r", 1): 5}
        after = {("r", 1): 6}
        changes = [{
            "action": "UPGRADE", "role": "r", "layer": 1,
            "old_n": 5, "new_n": 6, "event_count": 3,
        }]
        self._write_plan(before, after, changes)

        with patch.object(ag, "_acquire_lock", return_value="/lock"),              patch.object(ag, "_release_lock"),              patch.object(
                 ag.pwb, "read_remote_promotion_file",
                 return_value={("someone_else", 2): 7},
             ),              patch.object(ag.pwb, "write_remote_promotion_file_atomic") as write:
            with self.assertRaisesRegex(RuntimeError, "stale approval plan"):
                ag.apply_plan(self.plan_path)

        write.assert_not_called()
    def test_rollback_refuses_changed_postimage(self):
        before = {("r", 1): 5}
        after = {("r", 1): 6}
        self._write_plan(before, after, [], status="applied")

        with patch.object(ag, "_acquire_lock", return_value="/lock"),              patch.object(ag, "_release_lock"),              patch.object(
                 ag.pwb, "read_remote_promotion_file",
                 return_value={("r", 1): 7},
             ),              patch.object(ag.pwb, "write_remote_promotion_file_atomic") as write:
            with self.assertRaisesRegex(RuntimeError, "refusing rollback"):
                ag.rollback_plan(self.plan_path)

        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
