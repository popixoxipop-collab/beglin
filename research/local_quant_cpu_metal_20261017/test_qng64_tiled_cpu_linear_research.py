#!/usr/bin/env python3
"""Hardware C qNg64 tiled BF16 linear tests, independent mathematical bitplane oracle.

Uses no model or prompt files. Requires the isolated dylib compiled from
BEGLIN qng64_dense.c + qng64_tiled_accelerate_portable_poc.c.
"""
import hashlib
import math
import os
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from qng64_tiled_cpu_linear_research import PackedQNg64TiledLinear

ROOT=Path(__file__).resolve().parent
LIB=Path(os.environ.get("BEGLIN_QNG64_RESEARCH_DYLIB",str(ROOT/"libbeglin_qng64_tiled_portable_poc.dylib")))
LIB_SHA=hashlib.sha256(LIB.read_bytes()).hexdigest()  # the SAME compiled lib used by this test
SOURCE_SHA="538a972765d6f1f3f9e3bd0e5cc5f4471be958db22a5814d76a034ed792e9aee"

def decode_independent(module):
    n=module.n
    out=module.out_features
    inp=module.in_features
    ng=out*(inp//64)
    code=np.zeros((ng,64),dtype=np.int32)
    p=module._codes.reshape(ng,n,8)
    for bit in range(n):
        for j in range(64):
            code[:,j]|=((p[:,bit,j//8]>>(j%8))&1).astype(np.int32)<<bit
    code=code-(1<<(n-1))
    return (code.astype(np.float32)*module._scales[:,None]).reshape(out,inp)

class TestPackedTiledCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert hashlib.sha256((ROOT/"qng64_tiled_cpu_linear_research.py").read_bytes()).hexdigest()==SOURCE_SHA
        assert hashlib.sha256(LIB.read_bytes()).hexdigest()==LIB_SHA
        torch.set_num_threads(2)

    def test_exact_storage_code_scale_and_native_f32_oracle(self):
        rng=np.random.default_rng(2026101801)
        for n in (2,3,5,6,7,9,12,15):
            W=np.ascontiguousarray(rng.normal(0,.08,(67,128)).astype(np.float32))
            input_mat=np.ascontiguousarray(rng.normal(0,.04,(3,128)).astype(np.float32))
            src=torch.from_numpy(W).to(torch.bfloat16)
            x=torch.from_numpy(input_mat).to(torch.bfloat16)
            quant=PackedQNg64TiledLinear(src,n,LIB,expected_library_sha256=LIB_SHA)
            output=quant(x).float().numpy()
            dense=decode_independent(quant)
            expected=(x.float().numpy()@dense.T).astype(np.float32)
            # Both outputs BF16 rounded. Small FP32 accumulation-order diffs
            # can cross BF16 rounding bin boundaries, so compare numeric error.
            rel=float(np.linalg.norm(output.astype(np.float64)-expected.astype(np.float64))/max(np.linalg.norm(expected.astype(np.float64)),1e-25))
            self.assertLess(rel,0.004,(n,rel))
            self.assertEqual(quant.packed_bytes(),(67*128//64)*(8*n+4))
            self.assertEqual(len(list(quant.parameters())),0)
            self.assertFalse(hasattr(quant,"weight"))
            self.assertEqual(quant.evidence()["packed_codes_sha256"],hashlib.sha256(quant._codes.tobytes()).hexdigest())

    def test_bf16_vs_high_precision_n15_better_than_n5(self):
        rng=np.random.default_rng(85)
        W=torch.from_numpy(np.ascontiguousarray(rng.normal(0,.11,(128,256)),dtype=np.float32)).to(torch.bfloat16)
        X=torch.from_numpy(np.ascontiguousarray(rng.normal(0,.07,(2,5,256)),dtype=np.float32)).to(torch.bfloat16)
        y_ref=F.linear(X,W).float()
        ys={}
        for n in (5,7,15):
            m=PackedQNg64TiledLinear(W,n,LIB,expected_library_sha256=LIB_SHA)
            ys[n]=m(X).float()
            self.assertEqual(ys[n].shape,(2,5,128))
        err={n:float(torch.linalg.vector_norm((ys[n]-y_ref).reshape(-1)).item()) for n in ys}
        self.assertLess(err[15],err[5])

    def test_invalid_bits_and_group_alignment(self):
        rng=np.random.default_rng(1)
        W=torch.tensor(rng.normal(0,.1,(4,128)).astype(np.float32)).to(torch.bfloat16)
        for n in (0,1,4,16,True,7.0,"7"):
            with self.assertRaises(ValueError):PackedQNg64TiledLinear(W,n,LIB)
        with self.assertRaises(ValueError):PackedQNg64TiledLinear(W[:,:127],7,LIB)
        with self.assertRaises(ValueError):PackedQNg64TiledLinear(torch.full((4,128),float("nan"),dtype=torch.float32),7,LIB)
        with self.assertRaises(ValueError):PackedQNg64TiledLinear(W,7,LIB,tile_rows=65)
        with self.assertRaises(ValueError):PackedQNg64TiledLinear(W,7,LIB,expected_library_sha256="0"*64)

    def test_invalid_runtime_tensor_and_unchanged_code(self):
        W=torch.ones((8,128),dtype=torch.bfloat16)
        t=PackedQNg64TiledLinear(W,7,LIB)
        before=t._codes.tobytes()
        with self.assertRaises(TypeError):t(torch.ones((2,128),dtype=torch.float32))
        with self.assertRaises(TypeError):t(torch.ones((2,127),dtype=torch.bfloat16))
        with self.assertRaises(ValueError):t(torch.empty((0,128),dtype=torch.bfloat16))
        self.assertEqual(before,t._codes.tobytes())

    def test_packed_mutation_and_restore_n9(self):
        W=torch.tensor(np.tile(np.linspace(-.09,.09,128,dtype=np.float32),(4,1)),dtype=torch.float32).to(torch.bfloat16)
        t=PackedQNg64TiledLinear(W,9,LIB)
        X=torch.zeros((1,128),dtype=torch.bfloat16);X[0,0]=1
        original=t(X).float().numpy().copy()
        # Flip sign plane, first weight in row 0 group0.
        offset=(9-1)*8
        original_byte=int(t._codes[offset])
        t._codes[offset]^=np.uint8(1)
        bad=t(X).float().numpy().copy()
        self.assertGreater(float(np.max(np.abs(bad-original))),1e-6)
        t._codes[offset]=original_byte
        recovered=t(X).float().numpy()
        self.assertTrue(np.array_equal(original,recovered))

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestPackedTiledCPU)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
