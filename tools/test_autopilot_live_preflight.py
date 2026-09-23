#!/usr/bin/env python3
import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

import autopilot_live_preflight as pf


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body if isinstance(body, bytes) else body.encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class LivePreflightTests(unittest.TestCase):
    def test_candidate_passes_only_without_flip_and_correct_token(self):
        out = (
            "[moe cb4b] req 0 prompt 0 slot 0 arrive 0 admit_step 0 "
            "ttft_ms 1.0 nout 1 tokens: 3078\n"
        )
        v = pf.classify_candidate(out, pos=8, prompt_len=9, corrected=3078)
        self.assertTrue(v["pass"])
        self.assertEqual(v["emitted_token"], 3078)
        self.assertEqual(v["gen_idx"], 0)

    def test_candidate_fails_when_correction_flip_was_needed(self):
        out = (
            "[moe neartie] correct req=0 pos=8 REAL FLIP "
            "orig=2449 corrected=3078 -- running attribution\n"
            "[moe cb4b] req 0 prompt 0 slot 0 arrive 0 admit_step 0 "
            "ttft_ms 1.0 nout 1 tokens: 3078\n"
        )
        v = pf.classify_candidate(out, pos=8, prompt_len=9, corrected=3078)
        self.assertFalse(v["pass"])
        self.assertIn("REAL FLIP", v["reason"])

    def test_candidate_fails_when_emitted_token_is_wrong(self):
        out = (
            "[moe cb4b] req 0 prompt 0 slot 0 arrive 0 admit_step 0 "
            "ttft_ms 1.0 nout 1 tokens: 2449\n"
        )
        v = pf.classify_candidate(out, pos=8, prompt_len=9, corrected=3078)
        self.assertFalse(v["pass"])
        self.assertEqual(v["emitted_token"], 2449)

    def test_generated_index_after_prompt(self):
        out = (
            "[moe cb4b] req 0 prompt 0 slot 0 arrive 0 admit_step 0 "
            "ttft_ms 1.0 nout 3 tokens: 111 222 333\n"
        )
        v = pf.classify_candidate(out, pos=10, prompt_len=9, corrected=333)
        self.assertTrue(v["pass"])
        self.assertEqual(v["gen_idx"], 2)

    def test_baseline_requires_exact_recorded_flip_and_position(self):
        good = (
            "[moe neartie] correct req=0 pos=8 n_scalar=9\n"
            "[moe neartie] correct req=0 pos=8 REAL FLIP "
            "orig=2449 corrected=3078 -- running attribution\n"
        )
        self.assertTrue(pf._baseline_ok(good, 2449, 3078, 8))
        self.assertFalse(pf._baseline_ok(good, 2449, 9999, 8))
        self.assertFalse(pf._baseline_ok(good, 2449, 3078, 9))

    @patch.object(pf, "_rest_credentials", return_value=("https://example.test", "k"))
    @patch.object(pf.urllib.request, "urlopen")
    def test_persist_evidence_requires_returned_row(self, urlopen, _creds):
        row = {
            "model": "m", "role": "r", "layer": 1, "n": 5,
            "promotion_preimage_sha256": "pre",
            "promotion_postimage_sha256": "post",
            "pass": True, "status": "passed",
        }
        urlopen.return_value = FakeResponse(json.dumps([{**row, "id": 7}]))
        got = pf.persist_evidence(row)
        self.assertEqual(got["id"], 7)
        req = urlopen.call_args.args[0]
        self.assertEqual(req.get_method(), "POST")
        payload = json.loads(req.data)
        self.assertEqual(payload["source"], pf.EVIDENCE_SOURCE)
        self.assertEqual(payload["promotion_preimage_sha256"], "pre")

    @patch.object(pf, "_rest_credentials", return_value=("https://example.test", "k"))
    @patch.object(pf.urllib.request, "urlopen")
    def test_fetch_latest_evidence_is_exact_preimage(self, urlopen, _creds):
        row = {
            "id": 9, "status": "passed", "pass": True,
            "promotion_preimage_sha256": "abc",
            "promotion_postimage_sha256": "def",
        }
        urlopen.return_value = FakeResponse(json.dumps([row]))
        got = pf.fetch_latest_evidence("m", "r", 2, 5, "abc")
        self.assertEqual(got["id"], 9)
        req = urlopen.call_args.args[0]
        self.assertIn("promotion_preimage_sha256=eq.abc", req.full_url)
        self.assertIn("order=tested_at.desc", req.full_url)

    @patch.object(pf, "_rest_credentials", return_value=("https://example.test", "k"))
    @patch.object(pf.urllib.request, "urlopen")
    def test_missing_evidence_table_is_fail_closed(self, urlopen, _creds):
        body = json.dumps({
            "message": "Could not find the table 'public.moe_live_preflight_results'"
        }).encode()
        urlopen.side_effect = urllib.error.HTTPError(
            "https://example.test", 404, "not found", None, io.BytesIO(body)
        )
        with self.assertRaises(pf.EvidenceStoreUnavailable):
            pf.fetch_latest_evidence("m", "r", 2, 5, "abc")

    @patch.object(pf, "persist_evidence")
    def test_no_current_signal_is_explicit_nonpass(self, persist):
        persist.return_value = {"id": 3}
        pf.persist_no_current_signal(
            "m", "shared_gate_proj", 14, 5, "pre", "post",
            corpus="c", evidence_path="/tmp/evidence",
        )
        row = persist.call_args.args[0]
        self.assertEqual(row["status"], "no_current_signal")
        self.assertFalse(row["pass"])
        self.assertIsNone(row["correction_required"])


if __name__ == "__main__":
    unittest.main()
