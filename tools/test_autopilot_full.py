#!/usr/bin/env python3
import os
import tempfile
import unittest
from unittest.mock import patch

import autopilot_full as p5


def candidate():
    return {
        "action": "ADD",
        "role": "kv_a_proj_with_mqa",
        "layer": 4,
        "old_n": None,
        "new_n": 5,
        "event_count": 10,
    }


def prepared(changes=None, preflight=None, decisions=None):
    return {
        "phase": "P5-full-auto",
        "status": "prepared",
        "model": "m",
        "before": [],
        "changes": list(changes or []),
        "preflight_candidates": list(preflight or []),
        "decisions": list(decisions or []),
    }


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

    @patch.object(p5, "_live_evidence_gate")
    @patch.object(p5.shadow, "fetch_candidates")
    @patch.object(p5.pwb, "target_safe_n")
    @patch.object(p5.pwb, "read_remote_promotion_file", return_value={})
    def test_durable_pass_admits_attention(
        self, _read, safe_n, fetch, gate
    ):
        fetch.return_value = [{
            "role": "kv_a_proj_with_mqa", "layer": 4,
            "event_count": 10, "current_bits": 8,
        }]
        safe_n.return_value = (5, {"per_corpus": {"c": {}}})
        gate.return_value = (None, {
            "reason": "pass",
            "evidence": {"status": "passed", "pass": True},
        })
        plan = p5.build_plan("m", 100, "h", "/promotion", None, self.plan)
        self.assertEqual(plan["evidence_contract"], "p5-v2")
        self.assertEqual(len(plan["changes"]), 1)
        self.assertEqual(plan["changes"][0]["new_n"], 5)
        self.assertEqual(plan["preflight_candidates"], [])

    @patch.object(p5, "_live_evidence_gate")
    @patch.object(p5.shadow, "fetch_candidates")
    @patch.object(p5.pwb, "target_safe_n")
    @patch.object(p5.pwb, "read_remote_promotion_file", return_value={})
    def test_missing_evidence_becomes_preflight_candidate(
        self, _read, safe_n, fetch, gate
    ):
        fetch.return_value = [{
            "role": "kv_a_proj_with_mqa", "layer": 4,
            "event_count": 10, "current_bits": 8,
        }]
        safe_n.return_value = (5, {"per_corpus": {"c": {}}})
        gate.return_value = ("NEEDS_LIVE_PREFLIGHT", {
            "reason": "missing", "evidence": None,
        })
        plan = p5.build_plan("m", 100, "h", "/promotion", None, self.plan)
        self.assertEqual(plan["changes"], [])
        self.assertEqual(len(plan["preflight_candidates"]), 1)
        self.assertEqual(plan["decisions"][0]["action"], "NEEDS_LIVE_PREFLIGHT")

    @patch.object(p5, "_live_evidence_gate")
    @patch.object(p5.shadow, "fetch_candidates")
    @patch.object(p5.pwb, "target_safe_n")
    @patch.object(p5.pwb, "read_remote_promotion_file", return_value={})
    def test_failed_evidence_is_not_repreflighted_automatically(
        self, _read, safe_n, fetch, gate
    ):
        fetch.return_value = [{
            "role": "kv_a_proj_with_mqa", "layer": 4,
            "event_count": 10, "current_bits": 8,
        }]
        safe_n.return_value = (5, {})
        gate.return_value = ("LIVE_PREFLIGHT_FAILED", {
            "reason": "REAL FLIP still required",
            "evidence": {"status": "failed", "pass": False},
        })
        plan = p5.build_plan("m", 100, "h", "/promotion", None, self.plan)
        self.assertEqual(plan["changes"], [])
        self.assertEqual(plan["preflight_candidates"], [])
        self.assertEqual(plan["decisions"][0]["action"], "LIVE_PREFLIGHT_FAILED")

    @patch.object(p5, "_live_evidence_gate")
    @patch.object(p5.shadow, "fetch_candidates")
    @patch.object(p5.pwb, "target_safe_n")
    @patch.object(p5.pwb, "read_remote_promotion_file", return_value={})
    def test_no_current_signal_is_distinct_state(
        self, _read, safe_n, fetch, gate
    ):
        fetch.return_value = [{
            "role": "shared_gate_proj", "layer": 14,
            "event_count": 6, "current_bits": 8,
        }]
        safe_n.return_value = (5, {})
        gate.return_value = ("NO_CURRENT_SIGNAL", {
            "reason": "50-request scan: 0 attribution rows",
            "evidence": {"status": "no_current_signal"},
        })
        plan = p5.build_plan("m", 100, "h", "/promotion", None, self.plan)
        self.assertEqual(plan["changes"], [])
        self.assertEqual(plan["preflight_candidates"], [])
        self.assertEqual(plan["decisions"][0]["action"], "NO_CURRENT_SIGNAL")

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
        plan = p5.build_plan("m", 100, "h", "/promotion", None, self.plan)
        self.assertEqual(plan["decisions"][0]["action"], "NOOP")
        self.assertEqual(plan["changes"], [])

    @patch.object(p5, "prepare")
    @patch.object(p5.observer, "arm")
    def test_kill_switch_off_never_applies(self, arm, prepare):
        prepare.return_value = prepared(preflight=[candidate()])
        result = p5.arm_apply(
            "m", 100, "h", "/promotion", "/quarantine",
            self.plan, self.audit, "local", "/events",
            self.state, self.audit, preflight_bin="/tmp/bin",
        )
        self.assertEqual(result["status"], "disabled")
        arm.assert_not_called()

    @patch.object(p5, "prepare")
    @patch.object(p5, "_bootstrap_preflight_plan")
    @patch.object(p5.live_preflight, "run_preflight")
    @patch.object(p5.observer, "arm")
    def test_bootstrap_preflight_then_replan_then_observe(
        self, arm, preflight, bootstrap, prepare
    ):
        os.environ[p5.lowrisk.P5_ENABLE_ENV] = "1"
        first = prepared(preflight=[candidate()])
        second = prepared(changes=[candidate()])
        prepare.side_effect = [first, second]
        bootstrap.return_value = {"status": "prepared"}
        preflight.return_value = {"status": "passed"}
        arm.return_value = {"status": "observing"}
        result = p5.arm_apply(
            "m", 100, "h", "/promotion", "/quarantine",
            self.plan, self.audit, "local", "/events",
            self.state, self.audit, preflight_bin="/tmp/current-bin",
        )
        self.assertEqual(result["status"], "observing")
        self.assertEqual(prepare.call_count, 2)
        bootstrap.assert_called_once()
        preflight.assert_called_once()
        arm.assert_called_once()

    @patch.object(p5, "prepare")
    @patch.object(p5.live_preflight, "run_preflight")
    @patch.object(p5.observer, "arm")
    def test_existing_durable_pass_skips_repreflight(
        self, arm, preflight, prepare
    ):
        os.environ[p5.lowrisk.P5_ENABLE_ENV] = "1"
        prepare.return_value = prepared(changes=[candidate()])
        arm.return_value = {"status": "observing"}
        result = p5.arm_apply(
            "m", 100, "h", "/promotion", "/quarantine",
            self.plan, self.audit, "local", "/events",
            self.state, self.audit,
        )
        self.assertEqual(result["status"], "observing")
        preflight.assert_not_called()
        arm.assert_called_once()

    @patch.object(p5, "prepare")
    @patch.object(p5.live_preflight, "run_preflight")
    @patch.object(p5.observer, "arm")
    def test_missing_preflight_bin_blocks_missing_evidence(
        self, arm, preflight, prepare
    ):
        os.environ[p5.lowrisk.P5_ENABLE_ENV] = "1"
        prepare.return_value = prepared(preflight=[candidate()])
        with self.assertRaisesRegex(RuntimeError, "requires --preflight-bin"):
            p5.arm_apply(
                "m", 100, "h", "/promotion", "/quarantine",
                self.plan, self.audit, "local", "/events",
                self.state, self.audit,
            )
        preflight.assert_not_called()
        arm.assert_not_called()

    @patch.object(p5, "prepare")
    @patch.object(p5.observer, "arm")
    def test_evidence_store_unavailable_fails_closed(self, arm, prepare):
        os.environ[p5.lowrisk.P5_ENABLE_ENV] = "1"
        prepare.return_value = prepared(
            decisions=[{"action": "EVIDENCE_STORE_UNAVAILABLE"}]
        )
        with self.assertRaisesRegex(RuntimeError, "evidence store unavailable"):
            p5.arm_apply(
                "m", 100, "h", "/promotion", "/quarantine",
                self.plan, self.audit, "local", "/events",
                self.state, self.audit,
            )
        arm.assert_not_called()

    def test_serializes_multiple_passed_targets_by_event_count(self):
        existing = {}
        decisions = [
            {"role": "shared_gate_proj", "layer": 14, "action": "ADD"},
            {"role": "kv_a_proj_with_mqa", "layer": 4, "action": "ADD"},
        ]
        changes = [
            {
                "action": "ADD", "role": "shared_gate_proj", "layer": 14,
                "old_n": None, "new_n": 5, "event_count": 6,
            },
            {
                "action": "ADD", "role": "kv_a_proj_with_mqa", "layer": 4,
                "old_n": None, "new_n": 5, "event_count": 10,
            },
        ]
        after, kept, pending = p5._serialize_new_work(
            existing, decisions, changes, []
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["role"], "kv_a_proj_with_mqa")
        self.assertEqual(pending, [])
        self.assertEqual(after, {("kv_a_proj_with_mqa", 4): 5})
        deferred = [
            d for d in decisions
            if d["role"] == "shared_gate_proj" and d["layer"] == 14
        ][0]
        self.assertEqual(deferred["action"], "DEFERRED_SERIAL_PREIMAGE")

    def test_serializes_missing_evidence_candidates(self):
        decisions = [
            {"role": "shared_gate_proj", "layer": 14, "action": "NEEDS_LIVE_PREFLIGHT"},
            {"role": "kv_a_proj_with_mqa", "layer": 4, "action": "NEEDS_LIVE_PREFLIGHT"},
        ]
        pending = [
            {
                "action": "ADD", "role": "shared_gate_proj", "layer": 14,
                "old_n": None, "new_n": 5, "event_count": 6,
            },
            {
                "action": "ADD", "role": "kv_a_proj_with_mqa", "layer": 4,
                "old_n": None, "new_n": 5, "event_count": 10,
            },
        ]
        after, changes, kept = p5._serialize_new_work(
            {}, decisions, [], pending
        )
        self.assertEqual(after, {})
        self.assertEqual(changes, [])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["role"], "kv_a_proj_with_mqa")
        deferred = [
            d for d in decisions
            if d["role"] == "shared_gate_proj" and d["layer"] == 14
        ][0]
        self.assertEqual(deferred["action"], "DEFERRED_SERIAL_PREIMAGE")

    @patch.object(p5.live_preflight, "fetch_latest_evidence")
    def test_live_ladder_selects_higher_passing_n(self, fetch):
        rows = {
            5: {"status": "failed", "pass": False, "reason": "n5 failed"},
            6: {"status": "passed", "pass": True, "reason": "n6 passed"},
        }
        fetch.side_effect = lambda model, role, layer, n, pre: rows[n]
        action, detail = p5._live_evidence_gate(
            "m", "shared_gate_proj", 14, 5, "pre"
        )
        self.assertIsNone(action)
        self.assertEqual(detail["selected_n"], 6)
        self.assertEqual(detail["qng64_safe_n"], 5)

    @patch.object(p5.live_preflight, "fetch_latest_evidence")
    def test_live_ladder_requests_first_unmeasured_higher_n(self, fetch):
        rows = {
            5: {"status": "failed", "pass": False, "reason": "n5 failed"},
            6: None,
        }
        fetch.side_effect = lambda model, role, layer, n, pre: rows[n]
        action, detail = p5._live_evidence_gate(
            "m", "shared_gate_proj", 14, 5, "pre"
        )
        self.assertEqual(action, "NEEDS_LIVE_PREFLIGHT")
        self.assertEqual(detail["next_n"], 6)

    @patch.object(p5.live_preflight, "fetch_latest_evidence")
    def test_live_ladder_all_failed_is_unsafe(self, fetch):
        fetch.return_value = {
            "status": "failed", "pass": False,
            "reason": "correction REAL FLIP still required",
        }
        action, detail = p5._live_evidence_gate(
            "m", "shared_gate_proj", 14, 5, "pre"
        )
        self.assertEqual(action, "LIVE_LADDER_UNSAFE")
        self.assertEqual(detail["tested_ns"], [5, 6, 7])
        self.assertEqual(fetch.call_count, 3)

    @patch.object(p5, "_live_evidence_gate")
    @patch.object(p5.shadow, "fetch_candidates")
    @patch.object(p5.pwb, "target_safe_n")
    @patch.object(p5.pwb, "read_remote_promotion_file", return_value={})
    def test_planner_uses_live_selected_higher_n(
        self, _read, safe_n, fetch, gate
    ):
        fetch.return_value = [{
            "role": "shared_gate_proj", "layer": 14,
            "event_count": 6, "current_bits": 4,
        }]
        safe_n.return_value = (5, {})
        gate.return_value = (None, {
            "qng64_safe_n": 5,
            "selected_n": 6,
            "reason": "n6 live pass",
            "evidence": {"status": "passed", "pass": True},
        })
        plan = p5.build_plan("m", 100, "h", "/promotion", None, self.plan)
        self.assertEqual(plan["changes"][0]["new_n"], 6)
        self.assertEqual(plan["decisions"][0]["qng64_safe_n"], 5)
        self.assertEqual(plan["decisions"][0]["live_selected_n"], 6)

    @patch.object(p5, "prepare")
    @patch.object(p5.observer, "arm")
    def test_no_changes_never_calls_observer(self, arm, prepare):
        prepare.return_value = prepared()
        result = p5.arm_apply(
            "m", 100, "h", "/promotion", "/quarantine",
            self.plan, self.audit, "local", "/events",
            self.state, self.audit,
        )
        self.assertEqual(result["status"], "no_changes")
        arm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
