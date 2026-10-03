#!/usr/bin/env python3
import unittest

import quant_search_n as q


def results(overrides=None):
    overrides = overrides or {}
    return {
        n: ("pass" if overrides.get(n, True) else "fail_wrong", {})
        for n in q.REAL_LADDER
    }


def landed(overrides=None, *, duplicates=False):
    overrides = overrides or {}
    rows = []
    for n in q.REAL_LADDER:
        row = {"n": n, "pass": overrides.get(n, True)}
        rows.append(row)
        if duplicates:
            rows.append(dict(row))
    return rows


class PushVerificationTests(unittest.TestCase):
    def test_real_ladder_matches_current_qng64_runtime_contract(self):
        self.assertEqual(
            q.REAL_LADDER,
            (2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        )
        self.assertNotIn(4, q.REAL_LADDER)
        self.assertNotIn(16, q.REAL_LADDER)

    def test_accepts_single_complete_ladder(self):
        self.assertEqual(
            q._verify_real_ladder_rows(landed(), results()),
            (True, None),
        )

    def test_accepts_concurrent_identical_duplicates(self):
        self.assertEqual(
            q._verify_real_ladder_rows(landed(duplicates=True), results()),
            (True, None),
        )

    def test_rejects_contradictory_duplicate(self):
        rows = landed()
        rows.append({"n": 9, "pass": False})
        ok, reason = q._verify_real_ladder_rows(rows, results())
        self.assertFalse(ok)
        self.assertIn("n=9", reason)

    def test_rejects_missing_n(self):
        rows = [r for r in landed() if r["n"] != 15]
        ok, reason = q._verify_real_ladder_rows(rows, results())
        self.assertFalse(ok)
        self.assertIn("expected n", reason)

    def test_rejects_value_mismatch(self):
        rows = landed({10: False})
        ok, reason = q._verify_real_ladder_rows(rows, results())
        self.assertFalse(ok)
        self.assertIn("n=10", reason)
        self.assertIn("expected pass=True", reason)

    def test_accepts_expected_nonmonotonic_failures_with_duplicates(self):
        failures = {7: False, 8: False, 10: False, 11: False, 12: False, 13: False, 14: False, 15: False}
        self.assertEqual(
            q._verify_real_ladder_rows(
                landed(failures, duplicates=True),
                results(failures),
            ),
            (True, None),
        )


if __name__ == "__main__":
    unittest.main()
