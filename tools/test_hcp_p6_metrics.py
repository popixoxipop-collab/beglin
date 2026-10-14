#!/usr/bin/env python3
import pathlib,struct,tempfile,unittest
import numpy as np
from hcp_p6_metrics import MAGIC,read_dump,logit_metrics,teacher_forced,crossed_ci,distribution,summarize_speed
from hcp_p6_probe_build import patch_capture
class TestP6(unittest.TestCase):
 def fixture(self,rows,headers=None,tail=b''):
  p=pathlib.Path(self.tmp.name)/'dump.bin'
  with p.open('wb') as f:
   for i,row in enumerate(rows):
    f.write(struct.pack('<4I',*(headers[i] if headers else (MAGIC,0,i,len(row)))))
    f.write(np.asarray(row,dtype='<f4').tobytes())
   f.write(tail)
  return p
 def setUp(self):self.tmp=tempfile.TemporaryDirectory()
 def tearDown(self):self.tmp.cleanup()
 def test_dump_valid(self):np.testing.assert_equal(read_dump(self.fixture([[1,2],[3,4]]),positions=2,vocab=2),[[1,2],[3,4]])
 def test_dump_missing(self):
  with self.assertRaises(ValueError):read_dump(self.fixture([[1,2]]),positions=2,vocab=2)
 def test_dump_extra(self):
  with self.assertRaises(ValueError):read_dump(self.fixture([[1,2]],tail=b'x'),positions=1,vocab=2)
 def test_dump_position(self):
  with self.assertRaises(ValueError):read_dump(self.fixture([[1,2]],[(MAGIC,0,1,2)]),positions=1,vocab=2)
 def test_dump_magic(self):
  with self.assertRaises(ValueError):read_dump(self.fixture([[1,2]],[(0,0,0,2)]),positions=1,vocab=2)
 def test_dump_slot(self):
  with self.assertRaises(ValueError):read_dump(self.fixture([[1,2]],[(MAGIC,1,0,2)]),positions=1,vocab=2)
 def test_dump_nan(self):
  with self.assertRaises(ValueError):read_dump(self.fixture([[float('nan'),2]]),positions=1,vocab=2)
 def test_identity(self):
  x=np.array([[1,2,3],[3,4,2]]);v=logit_metrics(x,x);self.assertEqual(v['rmse'],0);self.assertEqual(v['top1_agreement'],1);self.assertEqual(v['kl_af_to_candidate_mean'],0)
 def test_offset_invariant_kl(self):
  x=np.array([[1,2,3],[3,4,2]]);v=logit_metrics(x,x+7);self.assertAlmostEqual(v['rmse'],7);self.assertAlmostEqual(v['kl_af_to_candidate_mean'],0,places=12)
 def test_different_argmax(self):self.assertEqual(logit_metrics([[1,2]],[[2,1]])['top1_agreement'],0)
 def test_shape_mismatch(self):
  with self.assertRaises(ValueError):logit_metrics([[1,2]],[[1,2,3]])
 def test_nonfinite_metrics(self):
  with self.assertRaises(ValueError):logit_metrics([[1,float('inf')]],[[1,2]])
 def test_uniform_nll(self):
  v=teacher_forced(np.zeros((3,4)),[0,1,2]);self.assertAlmostEqual(v['perplexity'],4);self.assertEqual(v['predicted_input_tokens'],2)
 def test_teacher_shape(self):
  with self.assertRaises(ValueError):teacher_forced(np.zeros((2,4)),[0,1,2])
 def test_ci_zero(self):self.assertEqual(crossed_ci(np.zeros((6,12)),draws=500)['crossed_prompt_block_bootstrap_95ci_ms'],[0,0])
 def test_ci_positive(self):self.assertFalse(crossed_ci(np.ones((6,12)),draws=500)['upper_ci_below_zero'])
 def test_ci_negative(self):self.assertTrue(crossed_ci(-np.ones((6,12)),draws=500)['upper_ci_below_zero'])
 def test_incomplete_speed(self):
  with self.assertRaises(ValueError):summarize_speed([])
 def test_capture_anchor_guard(self):
  with self.assertRaises(ValueError):patch_capture('not the native source')
 def test_percentile(self):self.assertEqual(distribution([1,2,3])['p50'],2)
if __name__=='__main__':unittest.main()
