import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hierarchical_moe_probe import (
    tupper_encode,tupper_decode,assemble_swiglu,permute_channels,
    byte_budget,role_quant_sq_error,probe,generate_tile,quant_sq_error,
)


class HierarchicalMoETests(unittest.TestCase):
    def test_arbitrary_tupper(self):
        rng=np.random.default_rng(73)
        for shape in [(1,1),(7,3),(22,64),(88,64),(4,256)]:
            a=rng.integers(0,2,shape,dtype=np.uint8)
            packed=tupper_encode(a)
            self.assertEqual(packed['k']%shape[0],0)
            self.assertTrue(np.array_equal(a,tupper_decode(packed['k'],*shape)))
    def test_tupper_invalid(self):
        for a in [np.ones((3,4))*0.5, np.zeros((0,4))]:
            with self.assertRaises(ValueError):tupper_encode(a)
        with self.assertRaises(ValueError):tupper_decode(7,3,10)
    def test_swiglu_multi_width_and_permutation(self):
        rng=np.random.default_rng(17)
        a=rng.normal(0,.03,(128,3*64)).astype(np.float32)
        h=rng.normal(0,1,64)
        base=assemble_swiglu(a,h,128)
        for w in [16,32,64]:
            self.assertTrue(np.allclose(base,assemble_swiglu(a,h,w),rtol=1e-12,atol=1e-12))
        shuffled=permute_channels(a,rng.permutation(128))
        self.assertTrue(np.allclose(base,assemble_swiglu(shuffled,h,16),rtol=1e-12,atol=1e-12))
    def test_q_storage_formula(self):
        b=byte_budget(64,2048,1408,2,16)
        self.assertEqual(b['parameters'],64*3*2048*1408)
        self.assertLess(b['hybrid_over_q4'],1.)
        self.assertGreater(b['dictionary_fp16_bytes'],0)
        self.assertGreater(byte_budget(64,2048,1408,2,16)['coeff_fp16_bytes'],
                           byte_budget(64,2048,1408,2,64)['coeff_fp16_bytes'])
    def test_quant_role_alignment(self):
        rng=np.random.default_rng(18)
        x=rng.normal(0,.07,(2,128,3*64)).astype(np.float32)
        err=role_quant_sq_error(x,4,64)
        err2=sum([quant_sq_error(x[:,:,:64],4),quant_sq_error(x[:,:,64:128],4),
                  quant_sq_error(x[:,:,128:].transpose(0,2,1).copy(),4)])
        self.assertAlmostEqual(err,err2)
    def test_structured_tiny(self):
        x=probe(mode='structured',seed=5,experts=8,hidden=64,intermediate=128,
                groups=[16,32,64,128],lane=16,train_experts=4,rank=2,
                calibration_fraction=.2,tiles=1,top_k=3)
        self.assertTrue(x['all_rows_processed'])
        self.assertEqual(x['model_equivalent_shape']['total_physical_swiglu_parameters_per_layer'],8*3*64*128)
        for g,row in x['widths'].items():
            self.assertTrue(row['routing']['0.5']['tupper_roundtrip'])
            self.assertLess(row['routing']['1.0']['static_calibration_relative_l2'],1e-12)
    def test_independent_control(self):
        x=probe(mode='independent',seed=5,experts=8,hidden=64,intermediate=128,
                groups=[16],lane=16,train_experts=4,rank=2,
                calibration_fraction=.2,tiles=1,top_k=3)
        self.assertLess(x['widths']['16']['metric_explained_vs_shared_mean_mse'],.25)
    def test_structured_generation_is_deterministic(self):
        a=generate_tile(np.random.default_rng(42),'structured',5,128,64,16,.035,.001)
        b=generate_tile(np.random.default_rng(42),'structured',5,128,64,16,.035,.001)
        self.assertTrue(np.array_equal(a,b))

if __name__=='__main__':unittest.main()
