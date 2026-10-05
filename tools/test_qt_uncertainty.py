#!/usr/bin/env python3
import unittest
from qt_uncertainty import classify
class U(unittest.TestCase):
 def c(self,e): return {"target_key":"x","supported_n":[4,5,6],"recommended_n":5,"sample_count":3,"error_by_n":{"4":e[0],"5":e[1],"6":e[2]}}
 def test_unsatisfied(self): self.assertEqual(classify(self.c([.001,.0008,.0007]),5e-4)["classification"],"UNSATISFIED")
 def test_nonmonotonic(self): self.assertEqual(classify(self.c([.0004,.0002,.0003]),5e-4)["classification"],"NON_MONOTONIC")
 def test_safe(self): self.assertEqual(classify(self.c([.0008,.0003,.0002]),5e-4)["classification"],"SAFE")
if __name__=="__main__":unittest.main()
