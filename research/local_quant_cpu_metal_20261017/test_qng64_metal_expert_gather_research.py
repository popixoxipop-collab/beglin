#!/usr/bin/env python3
"""Real Apple GPU unit tests for the standalone BEGLIN packed expert-gather API.

Uses only deterministic synthetic code/scale/activation buffers; validates
unrestricted expert-ID order, one GPU gather vs independent CPU dense oracle,
an independent per-expert Metal packed path, input type/range checks, and bitflip
sensitivity. Runs on Apple Silicon MLX + NumPy, no checkpoint or EOE service.
"""
import hashlib,json,pathlib,unittest
import numpy as np
import mlx.core as mx
from qng64_metal_expert_gather_research import QNg64MetalExpertBank
from qng64_metal_packed_matrix_research import QNg64MetalPackedMatrix
from test_qng64_metal_packed_matrix_research import make_codes

ROOT=pathlib.Path(__file__).resolve().parent

def prepare(n,E,out,inp,rng,experimental=False):
    c=[];s=[];dense=[]
    for _ in range(E):
        p,k,w=make_codes(out,inp,n,rng)
        c.append(p);s.append(k);dense.append(w)
    return (QNg64MetalExpertBank(np.concatenate(c),np.concatenate(s),n,E,out,inp,
                    experimental_n6=experimental),c,s,np.stack(dense))

class TestMetalMoEGather(unittest.TestCase):
    def test_gpu_vs_independent_math_and_separate_metal(self):
        rng=np.random.default_rng(20261017)
        for n,E,out,inp,N in ((7,3,47,128,17),(9,7,65,512,26),(12,8,31,128,39),(6,4,33,128,12)):
            bank,c,s,w=prepare(n,E,out,inp,rng,n==6)
            X=np.ascontiguousarray(rng.standard_normal((N,inp)).astype(np.float32))
            ids=np.ascontiguousarray(rng.integers(0,E,size=N,dtype=np.int32))
            ids[:E]=np.arange(E,dtype=np.int32)
            a=bank.gather(X,ids);mx.eval(a)
            actual=np.asarray(a)
            cpu=np.stack([X[i]@w[e].T for i,e in enumerate(ids)])
            relative=np.linalg.norm((actual-cpu).astype(np.float64))/max(np.linalg.norm(cpu.astype(np.float64)),1e-15)
            self.assertLess(relative,2e-5)
            device_input=mx.array(X)
            b=bank.gather(device_input,ids);mx.eval(b)
            self.assertTrue(np.array_equal(actual,np.asarray(b)))
            ref=np.empty_like(actual)
            for e in range(E):
                select=np.where(ids==e)[0]
                m=QNg64MetalPackedMatrix(c[e],s[e],n,out,inp,experimental_n6=(n==6))
                b=m.matmul(np.ascontiguousarray(X[select]))
                mx.eval(b);ref[select]=np.asarray(b)
            self.assertTrue(np.array_equal(ref,actual))
            self.assertEqual(bank.serialized_bytes,sum(a.nbytes+b.nbytes for a,b in zip(c,s)))

    def test_indices_invalid_fail_closed(self):
        rng=np.random.default_rng(3)
        bank,*_=prepare(7,3,5,128,rng)
        X=np.ones((2,128),dtype=np.float32)
        for ids in (np.array([0,3],np.int32),np.array([-1,1],np.int32)):
            with self.assertRaises(ValueError):bank.gather(X,ids)
        for ids in (np.array([0,1],np.int64),np.array([[0,1]],np.int32)):
            with self.assertRaises(TypeError):bank.gather(X,ids)
        with self.assertRaises(TypeError):bank.gather(np.ones((3,128),np.float32),np.array([0,1],np.int32))
        with self.assertRaises(TypeError):bank.gather(np.ones((2,128),np.float64),np.array([0,1],np.int32))

    def test_invalid_format_and_group_scale(self):
        rng=np.random.default_rng(7)
        _,parts,groups,_=prepare(7,3,5,128,rng)
        pp=np.concatenate(parts);ss=np.concatenate(groups)
        with self.assertRaises(ValueError):QNg64MetalExpertBank(pp[:-1],ss,7,3,5,128)
        with self.assertRaises(ValueError):QNg64MetalExpertBank(pp,ss,7,3,5,127)
        with self.assertRaises(ValueError):QNg64MetalExpertBank(pp,ss,6,3,5,128)
        with self.assertRaises(ValueError):QNg64MetalExpertBank(pp,ss,4,3,5,128)
        with self.assertRaises(TypeError):QNg64MetalExpertBank(pp.astype(np.int8),ss,7,3,5,128)
        with self.assertRaises(ValueError):QNg64MetalExpertBank(pp,np.zeros_like(ss),7,3,5,128)

    def test_packed_sign_bit_mutation_and_restore(self):
        rng=np.random.default_rng(9)
        bank,parts,groups,_=prepare(9,2,5,128,rng)
        X=np.zeros((2,128),np.float32);X[:,0]=1
        ids=np.array([0,1],dtype=np.int32)
        y=bank.gather(X,ids);mx.eval(y);expected=np.asarray(y).copy()
        p=np.concatenate(parts)
        s=np.concatenate(groups)
        offset=5*(128//64)*9*8+(9-1)*8
        copy=p.copy();copy[offset]^=np.uint8(1)
        altered=QNg64MetalExpertBank(copy,s,9,2,5,128)
        a=altered.gather(X,ids);mx.eval(a)
        self.assertTrue(np.array_equal(np.asarray(a)[0],expected[0]))
        self.assertGreater(float(np.max(np.abs(np.asarray(a)[1]-expected[1]))),1e-6)
        copy[offset]^=np.uint8(1)
        restored=QNg64MetalExpertBank(copy,s,9,2,5,128)
        z=restored.gather(X,ids);mx.eval(z)
        self.assertTrue(np.array_equal(np.asarray(z),expected))

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestMetalMoEGather)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        doc={"schema":"BEGLIN-QNG64-GATHER-SYNTHETIC-HARDWARE/1","tests":result.testsRun,
             "status":"PASS_APPLE_MLX_METAL_PACKED_EXPERT_GATHER",
             "source_sha256":hashlib.sha256((ROOT/"qng64_metal_expert_gather_research.py").read_bytes()).hexdigest(),
             "models_or_checkpoints_used":False,
             "engine_serving_modified":False}
        result_path=ROOT/"qng64_metal_gather_public_test_v1.json"
        with result_path.open("x",encoding="utf-8") as f:
            json.dump(doc,f,sort_keys=True,indent=2);f.write("\n")
        print(json.dumps({**doc,"receipt_sha256":hashlib.sha256(result_path.read_bytes()).hexdigest()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
