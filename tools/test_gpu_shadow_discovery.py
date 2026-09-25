#!/usr/bin/env python3
import json
import tempfile
import unittest
from unittest.mock import patch

import gpu_shadow_discovery as gd


class FakeResp:
    def __init__(self, payload):
        self.payload = payload
    def read(self):
        return json.dumps(self.payload).encode()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


class ShadowDiscoveryTests(unittest.TestCase):
    def test_fetch_candidates_is_get_only_and_does_not_put_credentials_in_url(self):
        captured = {}
        def opener(req, timeout=30):
            captured["method"] = req.get_method()
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            return FakeResp([{
                "model": "deepseek-v2-lite",
                "role": "shared_down_proj",
                "layer": 4,
                "event_count": 9,
                "current_bits": 4,
                "min_margin_observed": 0.01,
            }])
        rows = gd.fetch_candidates(
            "deepseek-v2-lite",
            10,
            opener=opener,
            env={
                "QWEN_SUPABASE_URL": "https://example.supabase.co",
                "QWEN_SUPABASE_KEY": "secret-token",
            },
        )
        self.assertEqual(captured["method"], "GET")
        self.assertNotIn("secret-token", captured["url"])
        self.assertEqual(len(rows), 1)

    def test_discover_builds_ready_candidate_from_safe_n_and_provenance(self):
        candidates = [{
            "model": "deepseek-v2-lite",
            "role": "shared_down_proj",
            "layer": 4,
            "event_count": 12,
            "current_bits": 4,
            "min_margin_observed": 0.002,
        }]
        prov = {
            "id": 101,
            "corpus": "wikitext-103",
            "role": "shared_down_proj",
            "layer": 4,
            "manifest": "/readonly/discovery/manifest.txt",
            "req": 3,
            "pos": 16,
            "orig_argmax": 3268,
            "corrected_argmax": 1224,
            "source_jsonl": "/readonly/discovery/events.jsonl",
        }
        got = gd.discover(
            "deepseek-v2-lite",
            fetch_candidates_fn=lambda model, limit: candidates,
            target_safe_n_fn=lambda model, role, layer: (
                6, {"per_corpus": {"wikitext-103": {"safe_n": 6}}}
            ),
            fetch_best_provenance_fn=lambda model, role, layer: prov,
        )
        self.assertFalse(got["production_write_allowed"])
        self.assertEqual(got["mode"], "read_only")
        self.assertEqual(got["ready_count"], 1)
        row = got["ready"][0]
        self.assertEqual(row["n"], 6)
        self.assertEqual(row["event"]["orig_token"], 3268)
        self.assertEqual(row["event"]["corrected_token"], 1224)
        self.assertEqual(row["reference"]["emitted_token"], 1224)
        self.assertEqual(len(row["candidate_id"]), 24)
        self.assertEqual(len(got["payload_sha256"]), 64)

    def test_no_safe_n_is_not_ready(self):
        got = gd.discover(
            "deepseek-v2-lite",
            fetch_candidates_fn=lambda model, limit: [{
                "role": "shared_gate_proj",
                "layer": 14,
                "event_count": 8,
                "current_bits": 4,
            }],
            target_safe_n_fn=lambda *args: (
                None, {"reason": "real-kernel ladder unsafe"}
            ),
            fetch_best_provenance_fn=lambda *args: self.fail(
                "provenance must not be queried for unsafe target"
            ),
        )
        self.assertEqual(got["ready_count"], 0)
        self.assertEqual(got["decisions"][0]["status"], "SKIP_NO_SAFE_N")

    def test_safe_target_without_provenance_is_not_ready(self):
        got = gd.discover(
            "deepseek-v2-lite",
            fetch_candidates_fn=lambda model, limit: [{
                "role": "kv_a_proj_with_mqa",
                "layer": 13,
                "event_count": 2,
                "current_bits": 4,
            }],
            target_safe_n_fn=lambda *args: (7, {"final_n": 7}),
            fetch_best_provenance_fn=lambda *args: None,
        )
        self.assertEqual(got["ready_count"], 0)
        self.assertEqual(
            got["decisions"][0]["status"],
            "NEEDS_ATTRIBUTION_PROVENANCE",
        )

    def test_provenance_without_actual_flip_is_rejected(self):
        with self.assertRaises(gd.ShadowDiscoveryError):
            gd._ready_row(
                "deepseek-v2-lite",
                {
                    "role": "shared_down_proj",
                    "layer": 4,
                    "event_count": 1,
                },
                6,
                {"final_n": 6},
                {
                    "id": 1,
                    "req": 0,
                    "pos": 1,
                    "orig_argmax": 42,
                    "corrected_argmax": 42,
                },
            )

    def test_atomic_output_contains_no_credentials(self):
        with tempfile.TemporaryDirectory() as td:
            out = f"{td}/queue.json"
            value = {
                "schema": gd.SCHEMA,
                "mode": "read_only",
                "production_write_allowed": False,
                "ready": [],
            }
            gd._atomic_json(out, value)
            text = Path(out).read_text()
            self.assertNotIn("SUPABASE_KEY", text)
            self.assertIn('"production_write_allowed": false', text)


if __name__ == "__main__":
    unittest.main()
