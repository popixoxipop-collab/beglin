#!/usr/bin/env python3
import unittest

import persistent_precision_benchmark as pb
import precision_context as pc


class PersistentPrecisionBenchmarkTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(pb.percentile([1, 2, 3], .5), 2.0)
        self.assertAlmostEqual(pb.percentile([1, 2, 3, 4], .95), 3.85)

    def test_scratch_route_uses_independent_policy_hash(self):
        route = pb.scratch_route(9)
        self.assertEqual(route["role"], "shared_up_proj")
        self.assertEqual(route["layer"], 3)
        self.assertEqual(route["n"], 9)
        self.assertEqual(
            route["policy_hash"],
            pc.policy_hash([{"role": "shared_up_proj", "layer": 3, "n": 9}]),
        )
        self.assertTrue(route["endpoint"].startswith("scratch-persistent://"))

    def test_non_qng64_width_is_rejected(self):
        with self.assertRaises(pb.BenchmarkError):
            pb.scratch_route(4)

    def test_root_rejects_non_shadow_path(self):
        with self.assertRaises(pb.BenchmarkError):
            pb.validate_root("/tmp/not-precision-shadow")


if __name__ == "__main__":
    unittest.main()
