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

    @patch.object(p, "upsert_events", return_value=True)
    def test_unterminated_tail_is_not_checkpointed(self, events):
        first = (json.dumps(event()) + "\n").encode()
        partial = json.dumps(event(1, 9)).encode()[:20]
        self.write(first + partial)
        self.assertTrue(p.run_once(self.path, "https://x", "secret"))
        self.assertEqual(self.offset(), len(first))
        self.assertEqual(events.call_count, 1)

    @patch.object(p, "upsert_events", return_value=False)
    def test_failed_event_post_keeps_offset(self, events):
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
        with patch.object(p, "upsert_events", return_value=True) as events, \
             patch.object(p, "rpc_increment", return_value=False), \
             patch.object(p, "upsert_provenance", return_value=True):
            self.assertFalse(p.run_once(self.path, "https://x", "secret"))
            self.assertEqual(events.call_count, 1)
            self.assertEqual(self.offset(), 0)

        with patch.object(p, "upsert_events", return_value=True) as events, \
             patch.object(p, "rpc_increment", return_value=True), \
             patch.object(p, "upsert_provenance", return_value=True):
            self.assertTrue(p.run_once(self.path, "https://x", "secret"))
            events.assert_not_called()
            self.assertEqual(self.offset(), len(rows))
            self.assertFalse(os.path.exists(self.path + ".push_outbox.json"))

    def test_event_key_never_replaces_api_key(self):
        rows = (json.dumps(event()) + "\n").encode()
        self.write(rows)
        seen = []

        def capture(url, api_key, payload):
            seen.append(api_key)
            return True

        with patch.object(p, "upsert_events", side_effect=capture):
            self.assertTrue(p.run_once(self.path, "https://x", "secret-key"))
        self.assertEqual(seen, ["secret-key"])

    def test_ingest_id_is_stable_for_same_file_offset_and_bytes(self):
        rows = (
            json.dumps(event()) + "\n" +
            json.dumps(attribution()) + "\n"
        ).encode()
        self.write(rows)
        first, _, _ = p._read_complete_records(self.path, 0)
        second, _, _ = p._read_complete_records(self.path, 0)
        self.assertEqual(
            [x["_ingest_id"] for x in first],
            [x["_ingest_id"] for x in second],
        )
        self.assertNotEqual(first[0]["_ingest_id"], first[1]["_ingest_id"])

    def test_attribution_increment_receives_deterministic_ingest_id(self):
        rows = (
            json.dumps(event()) + "\n" +
            json.dumps(attribution()) + "\n"
        ).encode()
        self.write(rows)
        seen = []
        with patch.object(p, "upsert_events", return_value=True), \
             patch.object(p, "upsert_provenance", return_value=True), \
             patch.object(
                 p, "rpc_increment",
                 side_effect=lambda url, key, ingest_id, *args, **kwargs:
                     (seen.append(ingest_id) or True),
             ):
            self.assertTrue(p.run_once(self.path, "https://x", "secret"))
        parsed, _, _ = p._read_complete_records(self.path, 0)
        self.assertEqual(seen, [parsed[1]["_ingest_id"]])

    def test_outbox_replay_reuses_same_ingest_ids(self):
        rows = (
            json.dumps(event()) + "\n" +
            json.dumps(attribution()) + "\n"
        ).encode()
        self.write(rows)
        event_ids = []
        increment_ids = []

        def events_capture(url, key, payload):
            event_ids.extend(x["ingest_id"] for x in payload)
            return True

        def increment_capture(url, key, ingest_id, *args, **kwargs):
            increment_ids.append(ingest_id)
            return False

        with patch.object(p, "upsert_events", side_effect=events_capture), \
             patch.object(p, "rpc_increment", side_effect=increment_capture), \
             patch.object(p, "upsert_provenance", return_value=True):
            self.assertFalse(p.run_once(self.path, "https://x", "secret"))

        outbox = json.load(open(self.path + ".push_outbox.json"))
        stored_event_id = outbox["operations"][0]["rows"][0]["ingest_id"]
        stored_increment_id = outbox["operations"][1]["args"]["ingest_id"]
        self.assertEqual(event_ids, [stored_event_id])
        self.assertEqual(increment_ids, [stored_increment_id])


if __name__ == "__main__":
    unittest.main()
