#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpu_shadow_status_xox as gs


class ShadowStatusTests(unittest.TestCase):
    def test_not_started(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with (
                patch.object(gs, "ROOT", root),
                patch.object(gs, "STATUS", root / "launcher_status.json"),
                patch.object(gs, "LAST", root / "last_cycle.json"),
                patch("builtins.print") as printer,
            ):
                self.assertEqual(gs.main(), 0)
            payload = json.loads(printer.call_args.args[0])
            self.assertEqual(payload["status"], "NOT_STARTED")

    def test_current_cycle_relation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "launcher_status.json"
            last = root / "last_cycle.json"
            status.write_text(json.dumps({
                "status": "COMPLETE",
                "launch_id": "launch-1",
                "cycle_id": "cycle-1",
                "shadow_run_id": "run-1",
                "returncode": 0,
                "pid": 4242,
            }))
            last.write_text(json.dumps({
                "status": "SHADOW_CYCLE_COMPLETE",
                "launch_id": "launch-1",
                "cycle_id": "cycle-1",
                "shadow_run_id": "run-1",
                "ready_count": 1,
                "selected_candidate_id": "c1",
                "production_write_allowed": False,
                "shadow_run": {"shadow_status": "SHADOW_ADMITTED"},
            }))
            with (
                patch.object(gs, "ROOT", root),
                patch.object(gs, "STATUS", status),
                patch.object(gs, "LAST", last),
                patch.object(gs, "_pid_alive", return_value=False),
                patch("builtins.print") as printer,
            ):
                self.assertEqual(gs.main(), 0)
            payload = json.loads(printer.call_args.args[0])
            self.assertEqual(payload["last_cycle"]["relation"], "CURRENT")
            self.assertEqual(payload["last_cycle"]["cycle_id"], "cycle-1")
            self.assertEqual(payload["last_cycle"]["shadow_run_id"], "run-1")

    def test_previous_cycle_relation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "launcher_status.json"
            last = root / "last_cycle.json"
            status.write_text(json.dumps({
                "status": "RUNNING",
                "launch_id": "launch-new",
                "pid": 4242,
            }))
            last.write_text(json.dumps({
                "status": "SHADOW_CYCLE_COMPLETE",
                "launch_id": "launch-old",
                "cycle_id": "cycle-old",
                "shadow_run_id": "run-old",
                "production_write_allowed": False,
            }))
            with (
                patch.object(gs, "ROOT", root),
                patch.object(gs, "STATUS", status),
                patch.object(gs, "LAST", last),
                patch.object(gs, "_pid_alive", return_value=True),
                patch("builtins.print") as printer,
            ):
                self.assertEqual(gs.main(), 0)
            payload = json.loads(printer.call_args.args[0])
            self.assertEqual(payload["last_cycle"]["relation"], "PREVIOUS")

    def test_legacy_unlinked_relation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "launcher_status.json"
            last = root / "last_cycle.json"
            status.write_text(json.dumps({
                "status": "COMPLETE",
                "launch_id": "launch-new",
                "pid": 4242,
            }))
            last.write_text(json.dumps({
                "status": "NO_NEW_READY_CANDIDATE",
                "ready_count": 1,
                "production_write_allowed": False,
            }))
            with (
                patch.object(gs, "ROOT", root),
                patch.object(gs, "STATUS", status),
                patch.object(gs, "LAST", last),
                patch.object(gs, "_pid_alive", return_value=False),
                patch("builtins.print") as printer,
            ):
                self.assertEqual(gs.main(), 0)
            payload = json.loads(printer.call_args.args[0])
            self.assertEqual(payload["last_cycle"]["relation"], "LEGACY_UNLINKED")


if __name__ == "__main__":
    unittest.main()
