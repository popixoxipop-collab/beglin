#!/usr/bin/env python3
"""P7 pure protocol regressions. Does not start native/GPU workers."""
import unittest
from hcp_p7_repeated_native_cost import order_for,paired_block_ci

class P7ProtocolTests(unittest.TestCase):
    def test_order_balances_time_and_policy(self):
        for block in range(4):
            seq=[order_for(block,i) for i in range(12)]
            self.assertEqual(seq.count(("A","B","B","A")),6)
            self.assertEqual(seq.count(("B","A","A","B")),6)
            self.assertTrue(all(x.count("A")==x.count("B")==2 for x in seq))
    def test_bad_domain_block_denied(self):
        with self.assertRaises(ValueError):order_for(-1,0)
        with self.assertRaises(ValueError):order_for(0,12)
    def test_block_bootstrap_exact_pair_difference(self):
        rows=[]
        for b in range(4):
            for d in range(12):
                rows.append({"block":b,"A":{"roundtrip_ms":[100.,100.]},
                             "B":{"roundtrip_ms":[103.,103.]}})
        result=paired_block_ci(rows,"roundtrip_ms",1000)
        self.assertEqual(result["paired_B_minus_A_mean_ms"],3.)
        self.assertEqual(result["block_bootstrap_ci95_ms"],[3.,3.])
        self.assertEqual(result["pairs_per_block"],12)
        self.assertEqual(result["temporal_blocks"],4)
    def test_mismatched_and_missing_blocks_rejected(self):
        rows=[{"block":0,"A":{"roundtrip_ms":[100.,100.]},
               "B":{"roundtrip_ms":[100.,100.]}}]
        with self.assertRaisesRegex(ValueError,"block"):paired_block_ci(rows)
        rows.append({"block":1,"A":{"roundtrip_ms":[100.,100.]},
                     "B":{"roundtrip_ms":[100.,100.]}})
        self.assertEqual(paired_block_ci(rows,iterations=1000)["temporal_blocks"],2)
        rows[1]["B"]["roundtrip_ms"]=[100.]
        with self.assertRaises(ValueError):paired_block_ci(rows,iterations=1000)

if __name__=="__main__":unittest.main()
