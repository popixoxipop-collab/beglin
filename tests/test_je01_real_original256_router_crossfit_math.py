#!/usr/bin/env python3
"""JE01 real original route crossfit exact Fisher-Holm mathematics contracts."""
import math
import pathlib
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"tools"))
import je01_xox_original256_router_crossfit_audit as je

class Router256ExplorationMath(unittest.TestCase):
    def test_exact_two_sided_hypergeometric_fisher(self):
        self.assertAlmostEqual(je.exact_fisher(10,5,5,5),2/math.comb(10,5),places=12)
        self.assertAlmostEqual(je.exact_fisher(10,0,5,0),1.,places=12)
        self.assertAlmostEqual(je.exact_fisher(10,5,5,2),1.,places=12)
        with self.assertRaisesRegex(ValueError,"INVALID_MARGIN"):
            je.exact_fisher(10,5,5,6)

    def test_no_labels_frozen_top8_ranking_is_deterministic(self):
        rows=[[[0,1,2,3,4,5] for l in range(26)] for n in range(128)]
        top,counts=je.ranks(rows)
        self.assertEqual([(x["layer"],x["expert"],x["calls"]) for x in top[:8]],
                         [(1,i,128) for i in range(6)]+[(2,0,128),(2,1,128)])
        self.assertEqual(sum(counts.values()),128*26*6)

    def test_crossfit_coactivation_with_perfect_association(self):
        train=[]
        for k in range(128):
            if k<64:train.append([[0,1,2,3,4,5] for _ in range(26)])
            else:train.append([[6,7,8,9,10,11] for _ in range(26)])
        slots=[{"layer":i+1,"expert":0,"calls":64} for i in range(8)]
        pairs=je.coactivation(train,slots)
        self.assertEqual(len(pairs),28)
        self.assertTrue(all(p["n"]==128 for p in pairs))
        self.assertTrue(all(p["joint"]==64 for p in pairs))
        self.assertTrue(all(p["phi"]==1.0 for p in pairs))
        self.assertTrue(all(p["Holm_supported_at_0_05"] for p in pairs))

    def test_model_precision_error_covariance_is_not_measured_by_coactivation(self):
        path=pathlib.Path(__file__).resolve().parents[1]/"tools"/"je01_xox_original256_router_crossfit_audit.py"
        src=path.read_text()
        self.assertIn('"real_NLL_precision_effect_covariance":"NOT_MEASURED"',src)
        self.assertIn('"entire_256_train_split_strictly_unseen_gold_before_router":False',src)
        self.assertIn('"production_promotion_allowed":False',src)

if __name__=="__main__":
    unittest.main(verbosity=2)
