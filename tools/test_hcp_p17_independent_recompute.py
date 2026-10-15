#!/usr/bin/env python3
"""P17 strict independent full-logits mathematics and provenance guard fixtures."""
from __future__ import annotations
import math,pathlib,tempfile,unittest
import numpy as np
from hcp_p17_independent_recompute import probs,mean_gold_nll,kl,js,P17_CAPTURE_LINE,validate

class P17IndependentNumericsTests(unittest.TestCase):
    def setUp(self):self.z=np.zeros(102400,dtype=np.float32)
    def test_01_uniform_full_vocab_gold_nll(self):
        self.assertAlmostEqual(mean_gold_nll(self.z,102399),math.log(102400),places=10)
    def test_02_uniform_kl_and_js_zero(self):
        p=probs(self.z)
        self.assertAlmostEqual(kl(p,p),0,places=12)
        self.assertAlmostEqual(js(p,p),0,places=12)
    def test_03_distinct_real_logit_distributions_not_equal(self):
        a=self.z.copy();b=self.z.copy();a[13]=3.;b[19]=3.
        self.assertGreater(js(probs(a),probs(b)),0)
        self.assertGreater(kl(probs(a),probs(b)),0)
    def test_04_max_shift_invariance(self):
        a=self.z.copy();a[4]=11.0
        b=a+100
        np.testing.assert_allclose(probs(a),probs(b),rtol=0,atol=1e-15)
    def test_05_zero_prefix_logit_vector_fails(self):
        with self.assertRaises(ValueError):probs(np.empty((0,),dtype=np.float32))
    def test_06_wrong_vocabulary_fails(self):
        with self.assertRaises(ValueError):mean_gold_nll(np.zeros(102399),1)
    def test_07_bad_gold_target_index_fails(self):
        with self.assertRaises(ValueError):mean_gold_nll(self.z,102400)
    def test_08_nonfinite_full_original_logits_fail(self):
        a=self.z.copy();a[0]=np.nan
        with self.assertRaises(ValueError):mean_gold_nll(a,5)
    def test_09_native_C_trace_has_real_32byte_hex_id(self):
        sample="P17_NATIVE_GOLD_PREFIX_LOGITS rid="+("a"*32)+" inputsha="+("b"*64)+" plen=41 pos=40 next=15092 vocab=102400 bytes=409600 floatsha="+("c"*64)
        self.assertIsNotNone(P17_CAPTURE_LINE.fullmatch(sample))
    def test_10_native_C_trace_rejects_extra_argmax_suffix(self):
        sample="P17_NATIVE_GOLD_PREFIX_LOGITS rid="+("a"*32)+" inputsha="+("b"*64)+" plen=41 pos=40 next=15092 vocab=102400 bytes=409600 floatsha="+("c"*64)+" forged"
        self.assertIsNone(P17_CAPTURE_LINE.fullmatch(sample))
    def test_11_missing_full_real_cohort_fails_not_promoted(self):
        with self.assertRaises(ValueError):validate("/tmp/not-a-run-owned-p17-project")
    def test_12_positive_finite_NLL(self):
        a=self.z.copy();a[8]=50
        self.assertLess(mean_gold_nll(a,8),1e-10)
        self.assertGreater(mean_gold_nll(a,9),49)

if __name__=="__main__":unittest.main(verbosity=2)
