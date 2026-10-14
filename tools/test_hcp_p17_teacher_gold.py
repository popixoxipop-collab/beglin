#!/usr/bin/env python3
"""P17 actual original-DeepSeek-code, 3-layer mini causal span-vs-prefix test."""
from __future__ import annotations
import pathlib,unittest,torch
from hcp_p17_teacher_gold import GoldWindowTeacher
from hcp_p15_streamed_teacher import StreamedTeacher,TensorStore
from test_hcp_p15_streamed_teacher import P15StreamTests

class P17GoldSpanNumerics(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  P15StreamTests.setUpClass()
 def setUp(self):
  self.mini=P15StreamTests(methodName="test_01_original_source_all_18_definition_pins")
  self.mini.setUp()
 def tearDown(self):self.mini.tearDown()
 def example(self,dtype,gold=3):
  f=self.mini
  complete=f.ids[:,:7]
  length=complete.shape[1]-gold
  source=f.store()
  teacher=GoldWindowTeacher(source,f.classes,dtype=dtype)
  actual=teacher.forward_gold(complete,length,gold)
  self.assertEqual(list(actual.shape),[gold,128])
  self.assertTrue(bool(torch.isfinite(actual).all()))
  expected=[]
  for k in range(gold):
   t=StreamedTeacher(f.store(),f.classes,dtype=dtype)
   one=t.forward(complete[:,:length+k])
   expected.append(one.squeeze(0))
  baseline=torch.stack(expected)
  torch.testing.assert_close(actual,baseline,rtol=5e-3 if dtype==torch.bfloat16 else 2e-5,
    atol=5e-3 if dtype==torch.bfloat16 else 2e-5)
  self.assertEqual(len(teacher.layer_records),3)
  self.assertEqual(len(source.value_hashes),len(f.state))
 def test_01_FP32_all_three_gold_prefixes(self):self.example(torch.float32,3)
 def test_02_BF16_all_three_gold_prefixes(self):self.example(torch.bfloat16,3)
 def test_03_FP32_one_gold_prefix(self):self.example(torch.float32,1)
 def test_04_BF16_two_gold_prefixes(self):self.example(torch.bfloat16,2)
 def test_05_reject_unsupported_zero_gold_length(self):
  with self.assertRaises(ValueError):GoldWindowTeacher(self.mini.store(),self.mini.classes).forward_gold(self.mini.ids,7,0)
 def test_06_reject_mismatched_full_context(self):
  with self.assertRaises(ValueError):GoldWindowTeacher(self.mini.store(),self.mini.classes).forward_gold(self.mini.ids,5,1)
 def test_07_reject_oversized_KV128_context(self):
  with self.assertRaises(ValueError):
   GoldWindowTeacher(self.mini.store(),self.mini.classes).forward_gold(torch.ones(1,129,dtype=torch.int64),127,2)
 def test_08_reject_out_of_vocab_token(self):
  ids=self.mini.ids.clone();ids[0,0]=10000
  with self.assertRaises(ValueError):GoldWindowTeacher(self.mini.store(),self.mini.classes).forward_gold(ids,5,2)

if __name__=="__main__":unittest.main(verbosity=2)
