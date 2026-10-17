#!/usr/bin/env python3
"""Real Apple MLX GPU tests of the all-device packed routed qNg64 FFN pipeline.

Pure synthetic deterministic bitplanes, no original model data. Compare with
independent 171-style per-expert GPU call + torch BF16 SiLU/route aggregation.
Check with and without shared FFN/residual, different topK, device-resident
input, negative index and format validation, and packed mutation sensitivity.
"""
import hashlib,json,pathlib,unittest
import numpy as np
import mlx.core as mx
import torch
import torch.nn.functional as F
from qng64_metal_expert_gather_research import QNg64MetalExpertBank
from qng64_metal_packed_matrix_research import QNg64MetalPackedMatrix
from qng64_metal_routed_ffn_research import QNg64MetalRoutedFFN
from test_qng64_metal_packed_matrix_research import make_codes

ROOT=pathlib.Path(__file__).resolve().parent
SOURCE_SHA="448144e792bbcf3d6d630bdab52331e3ac5cbaa94ba213c074524925dd115a3e"

def build(n=7,E=4,hidden=128,intermediate=64,seed=20261017,shared=False):
    rng=np.random.default_rng(seed)
    banks=[]
    indiv=[]
    original=[]
    for role in range(3):
        out=intermediate if role in (0,1) else hidden
        inp=hidden if role in (0,1) else intermediate
        pieces=[];scales=[];mats=[]
        for i in range(E):
            codes,scale,_=make_codes(out,inp,n,rng)
            pieces.append(codes);scales.append(scale)
            mats.append(QNg64MetalPackedMatrix(codes,scale,n,out,inp))
        banks.append(QNg64MetalExpertBank(np.concatenate(pieces),np.concatenate(scales),
                      n,E,out,inp))
        indiv.append(mats)
        original.append((pieces,scales))
    shared_mod=None
    if shared:
        shared_mod=[]
        for role in range(3):
            out=intermediate if role in (0,1) else hidden
            inp=hidden if role in (0,1) else intermediate
            p,s,_=make_codes(out,inp,n,rng)
            shared_mod.append(QNg64MetalPackedMatrix(p,s,n,out,inp))
    return QNg64MetalRoutedFFN(*banks,shared=shared_mod),indiv,shared_mod,original

def input_model(T,K,E,H,seed):
    rng=np.random.default_rng(seed)
    X=np.ascontiguousarray(torch.tensor(
      rng.standard_normal((T,H)).astype(np.float32),dtype=torch.bfloat16).float().numpy())
    ids=np.ascontiguousarray(rng.integers(0,E,size=(T,K),dtype=np.int32))
    weights=np.ascontiguousarray(rng.random((T,K)).astype(np.float32))
    weights/=np.sum(weights,axis=1,keepdims=True)
    residual=np.ascontiguousarray(torch.tensor(
      rng.standard_normal((T,H)).astype(np.float32),dtype=torch.bfloat16).float().numpy())
    return X,ids,weights,residual

def packed_independent(X,ids,weights,indiv,shared=None,residual=None):
    N,K=ids.shape
    hidden=X.shape[1]
    actual=[]
    for i in range(N):
        x=mx.array(X[i:i+1])
        for k in range(K):
            e=int(ids[i,k])
            gate=indiv[0][e].matmul(x)
            up=indiv[1][e].matmul(x)
            mx.eval(gate,up)
            g=torch.from_numpy(np.asarray(gate).copy()).to(torch.bfloat16)
            u=torch.from_numpy(np.asarray(up).copy()).to(torch.bfloat16)
            intermediate=(F.silu(g)*u).to(torch.bfloat16)
            d=indiv[2][e].matmul(mx.array(intermediate.float().numpy()))
            mx.eval(d)
            actual.append(torch.from_numpy(np.asarray(d).copy()).to(torch.bfloat16))
    routed=(torch.cat(actual,axis=0).reshape(N,K,hidden).float()*
            torch.from_numpy(weights).unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)
    if shared is not None:
        xx=mx.array(X)
        sg=shared[0].matmul(xx);su=shared[1].matmul(xx)
        mx.eval(sg,su)
        g=torch.from_numpy(np.asarray(sg).copy()).to(torch.bfloat16)
        u=torch.from_numpy(np.asarray(su).copy()).to(torch.bfloat16)
        h=F.silu(g)*u
        sd=shared[2].matmul(mx.array(h.float().numpy()))
        mx.eval(sd)
        routed=(routed+torch.from_numpy(np.asarray(sd).copy()).to(torch.bfloat16))
    if residual is not None:
        routed=(torch.from_numpy(residual).to(torch.bfloat16)+routed).to(torch.bfloat16)
    return routed.float().numpy()

class TestAllDeviceMetalFFN(unittest.TestCase):
    def test_packed_ffn_matches_legacy_per_expert_with_shared_and_residual(self):
        cases=((3,2,7,False),(8,3,7,True),(9,4,9,True))
        for T,K,n,shared in cases:
            E=5; H=128; I=64
            ff,individual,shared_matrices,_=build(n,E,H,I,20261017+n,shared)
            X,ids,w,res=input_model(T,K,E,H,seed=n+7)
            got=ff.forward(X,ids,w,residual=res)
            mx.eval(got)
            actual=np.asarray(got.astype(mx.float32))
            expected=packed_independent(X,ids,w,individual,shared_matrices,residual=res)
            diff=actual.astype(np.float64)-expected.astype(np.float64)
            rel=float(np.linalg.norm(diff)/max(np.linalg.norm(expected.astype(np.float64)),1e-20))
            self.assertLess(rel,0.007,(n,T,K,rel))
            self.assertEqual(got.dtype,mx.bfloat16)
            # Device-resident activation is numerically identical to host path.
            on_device=ff.forward(mx.array(X),ids,w,residual=mx.array(res))
            mx.eval(on_device)
            self.assertTrue(np.array_equal(actual,np.asarray(on_device.astype(mx.float32))))
        self.__class__.last_error=rel

    def test_exact_independent_no_shared_simple(self):
        f,modules,_,_=build(7,3,128,64,888,False)
        X,ids,w,res=input_model(2,2,3,128,155)
        out=f.forward(X,ids,w)
        mx.eval(out)
        reference=packed_independent(X,ids,w,modules)
        a=np.asarray(out.astype(mx.float32))
        ratio=np.linalg.norm((a-reference).astype(np.float64))/np.linalg.norm(reference.astype(np.float64))
        self.assertLess(float(ratio),0.008)

    def test_reject_invalid_indices_and_route_weights(self):
        f,*_=build(7,3,128,64,91,False)
        X,ids,w,res=input_model(4,2,3,128,101)
        bad=ids.copy();bad[0,0]=3
        with self.assertRaises(ValueError):f.forward(X,bad,w)
        bad=ids.copy();bad[0,1]=-1
        with self.assertRaises(ValueError):f.forward(X,bad,w)
        with self.assertRaises(TypeError):f.forward(X,ids.astype(np.int64),w)
        with self.assertRaises(TypeError):f.forward(X,ids,w.astype(np.float64))
        with self.assertRaises(ValueError):f.forward(X,ids,np.full_like(w,np.nan))
        with self.assertRaises(TypeError):f.forward(X[:,1:],ids,w)
        with self.assertRaises(TypeError):f.forward(X,ids,w,residual=np.ones((4,127),dtype=np.float32))

    def test_reject_invalid_banks(self):
        a, *_=build(7,3,128,64,1,False)
        with self.assertRaises(TypeError):QNg64MetalRoutedFFN(None,a.up,a.down)
        with self.assertRaises(ValueError):QNg64MetalRoutedFFN(a.gate,a.up,a.gate)
        with self.assertRaises(TypeError):QNg64MetalRoutedFFN(a.gate,a.up,a.down,shared=(None,None,None))

if __name__=="__main__":
    assert hashlib.sha256((ROOT/"qng64_metal_routed_ffn_research.py").read_bytes()).hexdigest()==SOURCE_SHA
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestAllDeviceMetalFFN)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        doc={"schema":"BEGLIN-QNG64-ROUTED-FFN-ALL-DEVICE-METAL-TESTS/1",
             "status":"PASS_REAL_APPLE_GPU_FFN_ALL_DEVICE_COMPRESSED",
             "cases":result.testsRun,"research_only":True,
             "source_sha256":SOURCE_SHA,
             "not_full_model":True,"no_checkpoint_data":True,
             "automatic_live_promotion":False}
        path=ROOT/"qng64_metal_routed_ffn_synthetic_tests_v1.json"
        with path.open("x",encoding="utf-8") as f:json.dump(doc,f,sort_keys=True,indent=2);f.write("\n")
        print(json.dumps({**doc,"receipt_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
