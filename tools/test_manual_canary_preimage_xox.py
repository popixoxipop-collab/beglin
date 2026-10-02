#!/usr/bin/env python3
import os
import unittest
from unittest.mock import patch

import manual_canary_preimage_xox as p


class PreimageHelperTests(unittest.TestCase):
    def test_minimal_env_scrubs_credentials(self):
        with patch.dict(
            os.environ,
            {
                "PATH": "/bin",
                "HOME": "/tmp/home",
                "QWEN_SUPABASE_KEY": "secret",
                "SUPABASE_MANAGEMENT_PAT": "secret",
                "OTHER_SECRET": "secret",
            },
            clear=True,
        ):
            got = p._minimal_env()
        self.assertEqual(got["PATH"], "/bin")
        self.assertNotIn("QWEN_SUPABASE_KEY", got)
        self.assertNotIn("SUPABASE_MANAGEMENT_PAT", got)
        self.assertNotIn("OTHER_SECRET", got)

    def test_target_counts(self):
        reqs = {i: [100] * 8 + [3268, 200] for i in range(12)}
        got = p._target_counts(reqs)
        self.assertEqual(got["gen_idx"], 8)
        self.assertEqual(got["eligible_requests"], 12)
        self.assertEqual(got["orig_hits"], 12)
        self.assertEqual(got["reference_hits"], 0)

    def test_rss_units(self):
        self.assertEqual(p._rss_to_bytes(1234, "darwin"), 1234)
        self.assertEqual(p._rss_to_bytes(1234, "linux"), 1234 * 1024)

    def test_budget_has_measured_headroom(self):
        got = p.recommend_budget(
            requests=12,
            tokens=120,
            duration_ms=1800,
            peak_rss_bytes=4 * 1024**3,
        )
        self.assertGreater(got["max_requests"], 12)
        self.assertGreater(got["max_tokens"], 120)
        self.assertGreater(got["max_duration_ms"], 1800)
        self.assertGreater(got["max_memory_bytes"], 4 * 1024**3)
        self.assertTrue(got["requires_human_review"])

    def test_invalid_budget_measurement_fails_closed(self):
        with self.assertRaises(p.PreimageCaptureError):
            p.recommend_budget(
                requests=0,
                tokens=120,
                duration_ms=1000,
                peak_rss_bytes=1024,
            )


if __name__ == "__main__":
    unittest.main()
