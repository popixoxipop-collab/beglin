#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import d4_supabase_push as p


def event(req=0, pos=8):
    return {
        "kind": "event",
        "model": "m",
        "corpus": "c",
        "req": req,
        "pos": pos,
        "predicted_token": 1,
        "competing_token": 2,
        "margin": 0.1,
    }


def attribution(req=0, pos=8):
    return {
        "kind": "attribution",
        "model": "m",
        "corpus": "c",
        "req": req,
        "pos": pos,
        "role": "shared_down_proj",
        "layer": 4,
        "manifest": "/tmp/m.txt",
        "orig_argmax": 1,
        "corrected_argmax": 2,
        "threshold": 0.1,
    }


class PushCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "events.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text):
        with open(self.path, "wb") as f:
            f.write(text)

    def offset(self):
        op = self.path + ".pushed_offset"
        return int(open(op).read()) if os.path.exists(op) else 0

    @patch.object(p, "post", return_value=True)
    def test_unterminated_tail_is_not_checkpointed(self, post):
        first = (json.dumps(event()) + "\n").encode()
        partial = json.dumps(event(1, 9)).encode()[:20]
        self.write(first + partial)
        self.assertTrue(p.run_once(self.path, "https://x", "secret"))
        self.assertEqual(self.offset(), len(first))
        self.assertEqual(post.call_count, 1)

    @patch.object(p, "post", return_value=False)
    def test_failed_event_post_keeps_offset(self, post):
        data = (json.dumps(event()) + "\n").encode()
        self.write(data)
        self.assertFalse(p.run_once(self.path, "https://x", "secret"))
        self.assertEqual(self.offset(), 0)
        self.assertTrue(os.path.exists(self.path + ".push_outbox.json"))

    def test_resume_skips_completed_event_batch(self):
        rows = (
            json.dumps(event()) + "\n" +
            json.dumps(attribution()) + "\n"
        ).encode()
        self.write(rows)
        with patch.object(p, "post", return_value=True) as post, \
             patch.object(p, "rpc_increment", return_value=False), \
             patch.object(p, "upsert_provenance", return_value=True):
            self.assertFalse(p.run_once(self.path, "https://x", "secret"))
            self.assertEqual(post.call_count, 1)
            self.assertEqual(self.offset(), 0)

        with patch.object(p, "post", return_value=True) as post, \
             patch.object(p, "rpc_increment", return_value=True), \
             patch.object(p, "upsert_provenance", return_value=True):
            self.assertTrue(p.run_once(self.path, "https://x", "secret"))
            post.assert_not_called()
            self.assertEqual(self.offset(), len(rows))
            self.assertFalse(os.path.exists(self.path + ".push_outbox.json"))

    def test_event_key_never_replaces_api_key(self):
        rows = (json.dumps(event()) + "\n").encode()
        self.write(rows)
        seen = []

        def capture(url, api_key, path, payload):
            seen.append(api_key)
            return True

        with patch.object(p, "post", side_effect=capture):
            self.assertTrue(p.run_once(self.path, "https://x", "secret-key"))
        self.assertEqual(seen, ["secret-key"])


if __name__ == "__main__":
    unittest.main()
