#!/usr/bin/env python3
import unittest

import quant_search_n as q


def results(p5=True, p6=True, p7=True):
    return {
        5: ("pass" if p5 else "fail_wrong", {}),
        6: ("pass" if p6 else "fail_wrong", {}),
        7: ("pass" if p7 else "fail_wrong", {}),
    }


class PushVerificationTests(unittest.TestCase):
    def test_accepts_single_complete_ladder(self):
        landed = [
            {"n": 5, "pass": True},
            {"n": 6, "pass": True},
            {"n": 7, "pass": True},
        ]
        self.assertEqual(q._verify_real_ladder_rows(landed, results()), (True, None))

    def test_accepts_concurrent_identical_duplicates(self):
        landed = [
            {"n": 5, "pass": True}, {"n": 5, "pass": True},
            {"n": 6, "pass": True}, {"n": 6, "pass": True},
            {"n": 7, "pass": True}, {"n": 7, "pass": True},
        ]
        self.assertEqual(q._verify_real_ladder_rows(landed, results()), (True, None))

    def test_rejects_contradictory_duplicate(self):
        landed = [
            {"n": 5, "pass": True}, {"n": 5, "pass": False},
            {"n": 6, "pass": True},
            {"n": 7, "pass": True},
        ]
        ok, reason = q._verify_real_ladder_rows(landed, results())
        self.assertFalse(ok)
        self.assertIn("n=5", reason)

    def test_rejects_missing_n(self):
        landed = [{"n": 5, "pass": True}, {"n": 7, "pass": True}]
        ok, reason = q._verify_real_ladder_rows(landed, results())
        self.assertFalse(ok)
        self.assertIn("expected n", reason)

    def test_rejects_value_mismatch(self):
        landed = [
            {"n": 5, "pass": False},
            {"n": 6, "pass": True},
            {"n": 7, "pass": True},
        ]
        ok, reason = q._verify_real_ladder_rows(landed, results())
        self.assertFalse(ok)
        self.assertIn("expected pass=True", reason)

    def test_accepts_expected_failures_with_duplicates(self):
        landed = [
            {"n": 5, "pass": False}, {"n": 5, "pass": False},
            {"n": 6, "pass": True},
            {"n": 7, "pass": True}, {"n": 7, "pass": True},
        ]
        self.assertEqual(
            q._verify_real_ladder_rows(landed, results(False, True, True)),
            (True, None),
        )


if __name__ == "__main__":
    unittest.main()
