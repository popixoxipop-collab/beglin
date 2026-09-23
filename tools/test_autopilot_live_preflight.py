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

    def test_wrong_position_flip_does_not_poison_candidate(self):
        out = (
            "[moe neartie] correct req=0 pos=9 REAL FLIP "
            "orig=1 corrected=999 -- running attribution\n"
            "[moe cb4b] req 0 prompt 0 slot 0 arrive 0 admit_step 0 "
            "ttft_ms 1.0 nout 1 tokens: 3078\n"
        )
        v = pf.classify_candidate(out, pos=8, prompt_len=9, corrected=3078)
        self.assertTrue(v["pass"])
        self.assertEqual(v["real_flips"], [])

    def test_baseline_rejects_flip_from_wrong_position(self):
        out = (
            "[moe neartie] correct req=0 pos=8 n_scalar=9\n"
            "[moe neartie] correct req=0 pos=9 REAL FLIP "
            "orig=2449 corrected=3078 -- running attribution\n"
        )
        self.assertFalse(pf._baseline_ok(out, 2449, 3078, 8))

    def test_gpu_backend_is_fail_closed_in_cpu_runner(self):
        with self.assertRaisesRegex(RuntimeError, "dedicated MLX/Metal adapter"):
            pf._validate_backend("mlx_metal")

    @patch.object(
        pf, "_ssh",
        return_value=(
            "0123456789abcdef0123456789abcdef"
            "0123456789abcdef0123456789abcdef  /tmp/bin\n"
            "12345\nBOB.local\narm64\n"
        ),
    )
    def test_worker_identity_uses_remote_binary_sha(self, _ssh):
        got = pf._worker_identity("bob", "/tmp/bin", "cpu")
        self.assertEqual(got["binary_size"], 12345)
        self.assertEqual(got["host"], "BOB.local")
        self.assertEqual(got["arch"], "arm64")
        self.assertIsNone(got["engine_commit"])

    def test_evidence_never_uses_controller_commit_as_worker_commit(self):
        row = pf._evidence_row(
            {"model": "m", "before_sha256": "pre"},
            {"role": "r", "layer": 1, "new_n": 5},
            {"req": 0, "pos": 8, "orig_argmax": 1, "corrected_argmax": 2},
            {"pass": True, "real_flips": [], "reason": "ok"},
            "post", "passed",
            worker_identity={"binary_sha256": "abc", "engine_commit": None},
        )
        self.assertIsNone(row["engine_commit"])

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
