import argparse
import pathlib
import tempfile
import unittest
import numpy as np
from permutation_alignment_probe import (
    _cosine_rows, align_descriptors, permute_sign_rows, similarity_holdout, execute
)
from hierarchical_moe_probe import assemble_swiglu
from test_real_checkpoint_reader import make_fixture

class AlignmentTests(unittest.TestCase):
    def test_permutation_and_joint_sign_preserve_swiglu(self):
        rng=np.random.default_rng(3)
        rows=rng.normal(0,.025,(64,3*64)).astype(np.float32)
        h=rng.normal(size=64)
        p=rng.permutation(64)
        signs=rng.choice([-1.,1.],size=64).astype(np.float32)
        baseline=assemble_swiglu(rows,h,16)
        transformed=assemble_swiglu(permute_sign_rows(rows,p,signs),h,16)
        self.assertLess(np.linalg.norm(baseline-transformed)/
                        (np.linalg.norm(baseline)+1e-30),1e-13)

    def test_hungarian_recovers_known_permutation_and_sign(self):
        rng=np.random.default_rng(9)
        n=32;dim=128
        signature=tuple(_cosine_rows(rng.normal(size=(n,dim)).astype(np.float32)) for _ in range(3))
        shuffle=rng.permutation(n)
        signs=rng.choice([-1.,1.],size=n)
        candidate=(signature[0][shuffle],signature[1][shuffle]*signs[:,None],
                   signature[2][shuffle]*signs[:,None])
        perm,decoded_sign,_=align_descriptors(signature,candidate)
        np.testing.assert_array_equal(shuffle[perm],np.arange(n))
        np.testing.assert_array_equal(decoded_sign,signs[perm])
        self.assertGreater(similarity_holdout(signature,candidate,perm,decoded_sign),.99)

    def test_independent_hidden_signature_is_not_shared(self):
        rng=np.random.default_rng(43)
        a=tuple(_cosine_rows(rng.normal(size=(40,60)).astype(np.float32)) for _ in range(3))
        b=tuple(_cosine_rows(rng.normal(size=(40,60)).astype(np.float32)) for _ in range(3))
        perm,sgn,_=align_descriptors(a,b)
        a2=tuple(_cosine_rows(rng.normal(size=(40,80)).astype(np.float32)) for _ in range(3))
        b2=tuple(_cosine_rows(rng.normal(size=(40,80)).astype(np.float32)) for _ in range(3))
        self.assertLess(abs(similarity_holdout(a2,b2,perm,sgn)),.13)

    def test_full_reader_and_coordinate_split(self):
        with tempfile.TemporaryDirectory() as td:
            p,_=make_fixture(pathlib.Path(td),expert_count=8)
            args=argparse.Namespace(checkpoint_index=p,layer=1,experts=8,
               train_experts=4,rank=2,hidden=64,intermediate=128,widths=[16,64],
               signature_width=12,validation_width=12,seed=20261013,
               output=pathlib.Path(td)/"out.json")
            result=execute(args)
            self.assertTrue(result["function_equivalence"]["pass"])
            self.assertLess(result["function_equivalence"]["max_relative_l2"],1e-8)
            self.assertEqual(set(result["heldout_coordinate_factorization"]),
                             {"unaligned","aligned","sham_random"})
            self.assertGreater(result["heldout_coordinate_factorization"]["aligned"]["16"]
                               ["independent_evaluation_parameter_count"],0)
if __name__=="__main__":
    unittest.main()
