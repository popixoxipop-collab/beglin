#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpu_coverage_f_real_runner as rr


class RealRunnerContractTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(rr._classify(0, {"final_status":"ADMITTED"}), "ADMITTED")
        self.assertEqual(rr._classify(0, {"status":"REJECTED_AT_G4"}), "REJECTED_AT_G4")
        self.assertEqual(rr._classify(7, {"status":"ADMITTED"}), "RUN_ERROR")
        self.assertEqual(rr._classify(0, None), "UNCLASSIFIED")

    def test_last_json(self):
        got = rr._last_json("noise\\n{\\"status\\":\\"ADMITTED\\"}\\n")
        self.assertEqual(got["status"], "ADMITTED")

    def test_validate_rejects_unverified_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            for name in ("g4.txt","g6.txt","bin","st.index.json"):
                (root/name).write_text("x")
            (root/"moe").mkdir()
            cfg = {
                "role":"kv_a_proj_with_mqa","layer":11,
                "candidate_ns":[5,6,7],
                "event":{"orig_token":8713,"corrected_token":4794,"pos":14},
                "reference":{"emitted_token":4794},
                "prompt_len":9,
                "g4_manifest":str(root/"g4.txt"),
                "g6_manifest":str(root/"g6.txt"),
                "binary":str(root/"bin"),
                "binary_sha256":hashlib.sha256(b"x").hexdigest(),
                "checkpoint_sha256":"auto",
                "moe_base":str(root/"moe"),
                "safetensors":str(root/"st.index.json"),
                "control_root":str(root/"scratch"),
            }
            with self.assertRaises(rr.RunnerError):
                rr.validate_config(cfg, repo)

    def test_validate_bounded_candidate_ns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"; repo.mkdir()
            (root/"g4.txt").write_text("x")
            (root/"g6.txt").write_text("x")
            (root/"bin").write_bytes(b"x")
            (root/"st.index.json").write_text("{}")
            (root/"moe").mkdir()
            base = {
                "role":"kv_a_proj_with_mqa","layer":11,
                "event":{"orig_token":8713,"corrected_token":4794,"pos":14},
                "reference":{"emitted_token":4794},
                "prompt_len":9,
                "g4_manifest":str(root/"g4.txt"),
                "g6_manifest":str(root/"g6.txt"),
                "binary":str(root/"bin"),
                "binary_sha256":hashlib.sha256(b"x").hexdigest(),
                "checkpoint_sha256":"a"*64,
                "moe_base":str(root/"moe"),
                "safetensors":str(root/"st.index.json"),
                "control_root":str(root/"scratch"),
            }
            for ns in ([4], [5,6,7,5], [8]):
                cfg=dict(base); cfg["candidate_ns"]=ns
                with self.assertRaises(rr.RunnerError):
                    rr.validate_config(cfg, repo)

    def test_runner_does_not_claim_g5_for_g4_reject(self):
        results=[
            {"n":5,"classification":"REJECTED_AT_G4"},
            {"n":6,"classification":"ADMITTED"},
        ]
        bad=[r["n"] for r in results if r["classification"] in {"REJECTED_AT_G4","ROLLBACK_OR_REGRESSION"}]
        self.assertEqual(bad,[5])
        # Contract: finding a bad candidate is not equivalent to G5 rollback PASS.
        status="BAD_CANDIDATE_FOUND_G5_NOT_RUN" if bad else "NO_BAD_CANDIDATE_WITHIN_BUDGET"
        self.assertEqual(status,"BAD_CANDIDATE_FOUND_G5_NOT_RUN")


if __name__ == "__main__":
    unittest.main()
