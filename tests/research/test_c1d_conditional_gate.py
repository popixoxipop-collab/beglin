import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
from c1d_conditional_gate import closed_form_weight,compute_gate_weights,score_condition,paired_ci,unit_rows

class GateTests(unittest.TestCase):
 def test_diagonal_prefer_lower_error(self):
  w=closed_form_weight([[.1,0],[0,.4]])
  self.assertGreater(w[0],w[1])
 def test_perfect_correlation_returns_simplex(self):
  w=closed_form_weight([[.3,.3],[.3,.3]])
  self.assertTrue(np.isfinite(w).all())
  self.assertAlmostEqual(w.sum(),1.)
 def test_input_condition_changes_weights_without_labels(self):
  e=np.zeros((64,2),dtype=int)
  e[:32,0]=1
  e[32:,1]=1
  # Two predictable dense groups on each model's input coordinates
  old=np.repeat(np.array([[1.,0.],[-1.,0.]]),32,axis=0)
  queries=np.array([[1.,0.],[-1.,0.]])
  weights,static,R,n=compute_gate_weights(old,old,queries,queries,e)
  self.assertEqual(weights.shape,(2,2))
  self.assertEqual(n.shape,(2,32))
  self.assertLess(weights[0,0],weights[1,0])
  self.assertAlmostEqual(static[0],.5)
 def test_no_target_labels_argument(self):
  import inspect
  self.assertNotIn("target_labels",inspect.signature(compute_gate_weights).parameters)
 def test_scores_gate_equal_vs_fixed(self):
  points=np.array([[1.,0.],[0.,1.]])
  axis=[np.array([1.,0.]),np.array([0.,1.])]
  a=score_condition(points,axis,np.tile([.5,.5],(2,1)))
  b=score_condition(points,axis,np.tile([.5,.5],(2,1)))
  self.assertTrue(np.allclose(a,b))
 def test_bad_weights_rejected(self):
  with self.assertRaises(ValueError):
   score_condition(np.ones((2,2)),[np.array([1.,0.]),np.array([0.,1.])],np.array([[2.,-1.],[.5,.5]]))
 def test_bootstrap_zero_for_identical_conditions(self):
  y=np.array([1]*50+[0]*50)
  values=np.linspace(-1,1,100)
  self.assertEqual(paired_ci({"dynamic":values,"static":values},y,n=30),[0.,0.])
 def test_nonfinite_points_rejected(self):
  with self.assertRaises(ValueError):
   unit_rows([[float('nan'),1.]])
 def test_empty_canonical_points_rejected(self):
  with self.assertRaises(ValueError):
   unit_rows([[0.,0.]])
 def test_error_rows_misaligned_rejected(self):
  with self.assertRaises(ValueError):
   compute_gate_weights(np.ones((32,2)),np.ones((33,2)),np.ones((2,2)),np.ones((2,2)),np.zeros((32,2)))
if __name__=="__main__":unittest.main(verbosity=2)
