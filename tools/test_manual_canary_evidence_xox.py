#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import manual_canary_evidence_xox as ev


class EvidenceExporterTests(unittest.TestCase):
    def build_root(self, td):
        root=Path(td)/"g_autopilot_test"
        for name in ev.SUBROOTS:
            (root/name).mkdir(parents=True)
        (root/"g4"/"verdict.json").write_text(json.dumps({
            "status":"PASS",
            "candidate_emitted_token":1224,
            "reference_emitted_token":1224,
            "secret":"must-not-emit",
        }))
        (root/"g4"/"inputs.json").write_text(json.dumps({"context_hash":"a"*64}))
        (root/"g6_pre"/"applied_ack.json").write_text(json.dumps({
            "status":"STARTUP_STATE","active_policy_hash":"b"*64,"weight_epoch":0
        }))
        (root/"g6_post"/"applied_ack.json").write_text(json.dumps({
            "status":"PROMOTION_APPLIED","active_policy_hash":"c"*64,"weight_epoch":1
        }))
        (root/"g6_post"/"worker.log").write_text("raw log content must not be emitted")
        return root

    def test_collect_hashes_fixed_groups_and_projects_json(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.build_root(td)
            got=ev.collect(root)
            self.assertEqual(got["schema"],"manual-canary-raw-evidence-export-v1")
            self.assertFalse(got["production_write_allowed"])
            self.assertEqual(got["file_count"],5)
            self.assertEqual(len(got["g4_bundle_sha256"]),64)
            self.assertEqual(len(got["g6_bundle_sha256"]),64)
            verdict=next(x for x in got["groups"]["g4"]["files"] if x["path"]=="g4/verdict.json")
            self.assertEqual(verdict["json_projection"]["status"],"PASS")
            self.assertNotIn("secret",verdict["json_projection"])

    def test_raw_log_contents_are_not_exported(self):
        with tempfile.TemporaryDirectory() as td:
            got=ev.collect(self.build_root(td))
            text=json.dumps(got)
            self.assertNotIn("raw log content",text)

    def test_file_change_changes_bundle_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.build_root(td)
            first=ev.collect(root)["g6_bundle_sha256"]
            (root/"g6_post"/"worker.log").write_text("changed")
            second=ev.collect(root)["g6_bundle_sha256"]
            self.assertNotEqual(first,second)

    def test_symlink_file_is_denied(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.build_root(td)
            target=Path(td)/"outside"
            target.write_text("x")
            (root/"g4"/"link").symlink_to(target)
            with self.assertRaises(ev.EvidenceExportError):
                ev.collect(root)

    def test_symlink_root_is_denied(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.build_root(td)
            link=Path(td)/"linked"
            link.symlink_to(root)
            with self.assertRaises(ev.EvidenceExportError):
                ev.collect(link)

    def test_missing_group_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            root=self.build_root(td)
            for p in (root/"state").iterdir():
                p.unlink()
            (root/"state").rmdir()
            got=ev.collect(root)
            self.assertEqual(got["groups"]["state"]["status"],"MISSING")


if __name__=="__main__":
    unittest.main()
