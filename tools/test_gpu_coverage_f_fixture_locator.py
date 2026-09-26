#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpu_coverage_f_fixture_locator as fl


class FixtureLocatorTests(unittest.TestCase):
    def test_event_match_exact(self):
        self.assertTrue(fl._event_match({
            "role":"kv_a_proj_with_mqa",
            "layer":11,
            "pos":14,
            "orig_argmax":8713,
            "corrected_argmax":4794,
        }))
        self.assertFalse(fl._event_match({
            "role":"kv_a_proj_with_mqa",
            "layer":13,
            "pos":14,
            "orig_argmax":8713,
            "corrected_argmax":4794,
        }))

    def test_manifest_parser_int32(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            token=root/"prompt.i32"
            token.write_bytes(b"\0"*36)
            manifest=root/"m.txt"
            manifest.write_text(f"{token} 10\n")
            old=fl.MIRROR
            try:
                fl.MIRROR=root
                rows=fl._parse_manifest(manifest)
            finally:
                fl.MIRROR=old
            self.assertEqual(rows[0]["prompt_len"],9)

    def test_manifest_rejects_outside_token(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            mirror=root/"mirror"; mirror.mkdir()
            outside=root/"outside.i32"; outside.write_bytes(b"\0"*4)
            manifest=mirror/"m.txt"; manifest.write_text(f"{outside} 1\n")
            old=fl.MIRROR
            try:
                fl.MIRROR=mirror
                with self.assertRaises(fl.FixtureError):
                    fl._parse_manifest(manifest)
            finally:
                fl.MIRROR=old

    def test_resolve_maps_historical_vdsp_p5_pre_prefix_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            target=root/"kv_a_l11_discovery"/"manifest.txt"
            target.parent.mkdir(); target.write_text("x")
            got=fl._resolve_manifest(
                "/Users/bob/vdsp_p5_pre/kv_a_l11_discovery/manifest.txt",
                root/"source.jsonl",
                root,
            )
            self.assertEqual(got,target.resolve())
            bad=fl._resolve_manifest(
                "/Users/bob/other/manifest.txt",
                root/"source.jsonl",
                root,
            )
            self.assertIsNone(bad)


if __name__ == "__main__":
    unittest.main()
