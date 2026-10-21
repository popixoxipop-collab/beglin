import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
from evaluate_real_behavior import auc,bootstrap,load,run
class Ops(unittest.TestCase):
 def test_perfect(self):self.assertEqual(auc([3,4],[0,1]),1.)
 def test_inverted(self):self.assertEqual(auc([0,1],[3,4]),0.)
 def test_tied(self):self.assertEqual(auc([1,1],[1,1]),.5)
 def test_ci_perfect(self):self.assertEqual(bootstrap([3,4],[0,1],n=30),[1.,1.])
 def test_invalid_scores(self):
  for p,n in [([], [0]),([1],[]),([float("nan")],[0])]:
   with self.assertRaises(ValueError):auc(p,n)
 def test_unknown_model(self):
  with self.assertRaises(ValueError):load("unknown")
 def test_same_model_rejected(self):
  with self.assertRaises(ValueError):run("qwen","qwen")
if __name__=="__main__":unittest.main(verbosity=2)
