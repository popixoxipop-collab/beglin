#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import precision_trigger_calibration as cal


class TriggerCalibrationTests(unittest.TestCase):
    def test_manifest_materialization_preserves_one_unique_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            src=root/"source.txt"
            src.write_text("/tmp/prompt.i32 10\n/tmp/prompt.i32 10\n")
            got=cal.materialize_manifest(src,root/"out.txt",12)
            self.assertEqual(got["unique_source_rows"],1)
            self.assertEqual(got["requests"],12)
            self.assertEqual(len((root/"out.txt").read_text().splitlines()),12)

    def test_parse_requests(self):
        text=(
            "[moe gpu cb online] req 0 prompt 9 slot 0 arrive 0 "
            "admit_step 0 ttft_ms 1.0 nout 2 tokens: 100 1224\n"
            "[moe gpu cb online] req 1 prompt 9 slot 1 arrive 0 "
            "admit_step 0 ttft_ms 1.0 nout 2 tokens: 101 1224\n"
        )
        got=cal.parse_requests(text)
        self.assertEqual(got[0],[100,1224])
        self.assertEqual(got[1],[101,1224])

    def test_policy_target_match_is_exact(self):
        policy=[{"role":"shared_up_proj","layer":3,"n":6}]
        self.assertTrue(cal.policy_has_target(policy,"shared_up_proj",3,6))
        self.assertFalse(cal.policy_has_target(policy,"shared_up_proj",3,9))
        self.assertFalse(cal.policy_has_target(policy,"shared_down_proj",3,6))

    def test_jsonl_parser_ignores_invalid_lines(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"events.jsonl"
            p.write_text('{"kind":"event","req":0}\nnot-json\n{"kind":"attribution","req":0}\n')
            rows=cal.parse_jsonl(p)
            self.assertEqual([r["kind"] for r in rows],["event","attribution"])

    def test_event_identity_hash_is_deterministic(self):
        value={"req":0,"pos":16,"role":"shared_up_proj","layer":3}
        self.assertEqual(cal.sha256_json(value),cal.sha256_json(dict(reversed(list(value.items())))))


if __name__=="__main__":
    unittest.main()
