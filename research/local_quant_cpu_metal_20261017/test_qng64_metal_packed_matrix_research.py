#!/usr/bin/env python3
"""Standalone GPU tests for qng64_metal_packed_matrix_research.py, no models.

Real MLX Metal required. Random but deterministic packed codes independently
constructed in Python and decoded with a mathematical FP32 dense oracle.
n6 is an explicitly opt-in experimental Metal path, NOT a live C++ n6 route.
"""
import hashlib
import json
import math
import pathlib
import unittest

import numpy as np
import mlx.core as mx
from qng64_metal_packed_matrix_research import QNg64MetalPackedMatrix

ROOT=pathlib.Path(__file__).resolve().parent

def make_codes(rows, inp, n, rng):
    ng=inp//64
    bias=1<<(n-1)
    codes=rng.integers(-bias,bias,size=(rows,ng,64),dtype=np.int32)
    scales=np.ascontiguousarray(np.abs(rng.standard_normal((rows,ng))).astype(np.float32)*.03+.001)
    unsigned=codes+bias
    planes=np.zeros((rows,ng,n,8),dtype=np.uint8)
    for bit in range(n):
        for p in range(64):
            planes[:,:,bit,p//8]|=((unsigned[:,:,p]>>bit)&1).astype(np.uint8) << (p%8)
    packed=np.ascontiguousarray(planes.reshape(-1))
    flat_scale=np.ascontiguousarray(scales.reshape(-1))
    W=(codes.astype(np.float32)*scales[:,:,None]).reshape(rows,inp)
    return packed,flat_scale,W

class MetalPackedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "gpu" not in str(mx.default_device()):
            raise unittest.SkipTest("no Apple MLX Metal GPU")

    def test_actual_metal_numeric_heterogeneous_shapes(self):
        rng=np.random.default_rng(20261017)
        results=[]
        for n,out,inp,m in [(6,67,128,3),(7,64,512,1),(9,83,2048,7),(12,31,128,4),(15,16,512,2)]:
            bits,scales,W=make_codes(out,inp,n,rng)
            X=np.ascontiguousarray(rng.standard_normal((m,inp)),dtype=np.float32)
            expected=X@W.T
            gpu=QNg64MetalPackedMatrix(bits,scales,n,out,inp,experimental_n6=(n==6))
            result=gpu.matmul(X)
            mx.eval(result)
            actual=np.asarray(result)
            # Also exercise the on-device activation path: no CPU -> GPU
            # activation construction inside matmul() on this branch.
            resident_input=mx.array(X)
            result_resident=gpu.matmul(resident_input)
            mx.eval(result_resident)
            self.assertTrue(np.array_equal(actual,np.asarray(result_resident)))
            self.assertEqual(actual.shape,expected.shape)
            rel=float(np.linalg.norm((actual-expected).astype(np.float64))/max(np.linalg.norm(expected.astype(np.float64)),1e-15))
            self.assertLess(rel,2e-5)
            self.assertEqual(gpu.serialized_bytes,bits.nbytes+scales.nbytes)
            results.append({"n":n,"rows":out,"columns":inp,"M":m,"rel_L2":rel})
        self.results=results

    def test_invalid_formats_and_dtypes_fail_closed(self):
        rng=np.random.default_rng(2)
        bits,scales,_=make_codes(4,128,7,rng)
        for n,experimental in [(4,False),(8,False),(6,False),(16,False)]:
            with self.assertRaises(ValueError):
                QNg64MetalPackedMatrix(bits,scales,n,4,128,experimental_n6=experimental)
        with self.assertRaises(ValueError):
            QNg64MetalPackedMatrix(bits[:-1],scales,7,4,128)
        with self.assertRaises(TypeError):
            QNg64MetalPackedMatrix(bits.astype(np.int8),scales,7,4,128)
        with self.assertRaises(TypeError):
            QNg64MetalPackedMatrix(bits,scales.astype(np.float64),7,4,128)
        with self.assertRaises(ValueError):
            QNg64MetalPackedMatrix(bits,scales,7,4,127)
        m=QNg64MetalPackedMatrix(bits,scales,7,4,128)
        with self.assertRaises(TypeError):
            m.matmul(np.ones((3,128),dtype=np.float64))
        with self.assertRaises(TypeError):
            m.matmul(np.ones((3,129),dtype=np.float32))
        with self.assertRaises(ValueError):
            m.matmul(np.empty((0,128),dtype=np.float32))

    def test_packed_bitflip_changes_output_and_restore(self):
        rng=np.random.default_rng(3)
        packed,scales,W=make_codes(3,128,9,rng)
        X=np.ones((2,128),dtype=np.float32)
        source=QNg64MetalPackedMatrix(packed,scales,9,3,128)
        v0=source.matmul(X);mx.eval(v0)
        flipped=packed.copy()
        flipped[8*9-8]^=np.uint8(1)
        swapped=QNg64MetalPackedMatrix(flipped,scales,9,3,128)
        v1=swapped.matmul(X);mx.eval(v1)
        self.assertGreater(float(np.max(np.abs(np.asarray(v0)-np.asarray(v1)))),1e-6)
        flipped[8*9-8]^=np.uint8(1)
        restore=QNg64MetalPackedMatrix(flipped,scales,9,3,128)
        v2=restore.matmul(X);mx.eval(v2)
        self.assertTrue(np.array_equal(np.asarray(v0),np.asarray(v2)))

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MetalPackedTest)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        doc={"schema":"BEGLIN-MLX-GPU-QNG64-REUSABLE-RESEARCH-PROBE/1",
             "status":"GPU_STANDALONE_PACKED_BITPLANE_TESTS_PASS",
             "synthetic_only":True,"tests":result.testsRun,
             "not_full_model":True,"gpu_executed":True,
             "source_sha256":hashlib.sha256((ROOT/"qng64_metal_packed_matrix_research.py").read_bytes()).hexdigest()}
        p=ROOT/"qng64_metal_reusable_probe_tests_v2.json"
        with p.open("x",encoding="utf-8") as f:json.dump(doc,f,indent=2,sort_keys=True);f.write("\n")
        print(json.dumps({"status":doc["status"],"test_count":result.testsRun,
                          "receipt_sha256":hashlib.sha256(p.read_bytes()).hexdigest()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
