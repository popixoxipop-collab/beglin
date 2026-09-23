#!/usr/bin/env python3
import unittest

import autopilot_live_preflight as pf


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


if __name__ == "__main__":
    unittest.main()
