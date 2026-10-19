#!/usr/bin/env python3
"""Independent Apple Metal hardware test: fused qNg64 routed FFN BF16 semantics.

Deterministic synthetic packed weights only. Compares fused BF16/SILU/router
Metal math to already verified device-resident and individual packed-expert
Metal paths. Do NOT infer speedup or full LLM quality from test passes.
"""
import hashlib,json,pathlib,unittest
import numpy as np,torch
import mlx.core as mx
from qng64_metal_fused_routed_ffn_research import QNg64MetalFusedRoutedFFN
from test_qng64_metal_routed_ffn_research import build,input_model,packed_independent

ROOT=pathlib.Path(__file__).resolve().parent
SOURCE_SHA="8a2aad4eefc33dcf53d11ce04c37a2ddc55114dc738dbb288a7780784f963d79"

class FusedDeviceTests(unittest.TestCase):
    def test_silu_route_shared_real_gpu_matches_unfused_and_per_expert(self):
        cases=[(7,4,2,5,128,64),(9,8,3,6,128,128),(7,5,4,4,256,128)]
        max_rel=0.
        for n,T,K,E,H,I in cases:
            prior,separate,shared,_=build(n,E,H,I,100+n+T,True)
            fused=QNg64MetalFusedRoutedFFN(prior.gate,prior.up,prior.down,shared=shared)
            X,idx,w,res=input_model(T,K,E,H,500+n+T)
            a=fused.forward(mx.array(X),idx,w,mx.array(res))
            b=prior.forward(mx.array(X),idx,w,residual=mx.array(res))
            mx.eval(a,b)
            aa=np.asarray(a)
            bb=np.asarray(b.astype(mx.float32))
            relative=float(np.linalg.norm((aa-bb).astype(np.float64))/max(np.linalg.norm(bb.astype(np.float64)),1e-20))
            self.assertLess(relative,0.002,(n,T,K,relative))
            legacy=packed_independent(X,idx,w,separate,shared,residual=res)
            expected=float(np.linalg.norm((aa-legacy).astype(np.float64))/max(np.linalg.norm(legacy.astype(np.float64)),1e-20))
            self.assertLess(expected,0.007,(n,T,K,expected))
            c=fused.forward(X,idx,w,res)
            mx.eval(c)
            self.assertTrue(np.array_equal(aa,np.asarray(c)))
            max_rel=max(max_rel,relative)
        self.__class__.max_rel=max_rel

    def test_invalid_ids_shape_and_weights_are_fail_closed(self):
        base,_,shared,_=build(7,3,128,64,18,True)
        f=QNg64MetalFusedRoutedFFN(base.gate,base.up,base.down,shared=shared)
        X,ids,w,res=input_model(3,2,3,128,19)
        bad=ids.copy();bad[1,0]=3
        with self.assertRaises(ValueError):f.forward(X,bad,w,res)
        with self.assertRaises(TypeError):f.forward(X,ids.astype(np.int64),w,res)
        with self.assertRaises(TypeError):f.forward(X,ids,w.astype(np.float64),res)
        with self.assertRaises(ValueError):f.forward(X,ids,np.full_like(w,np.nan),res)
        with self.assertRaises(TypeError):f.forward(X[:,:127],ids,w,res)
        with self.assertRaises(TypeError):f.forward(X,ids,w[:2],res)

    def test_invalid_shared_modules(self):
        bank,_,shared,_=build(7,3,128,64,23,True)
        with self.assertRaises(ValueError):
            QNg64MetalFusedRoutedFFN(bank.gate,bank.up,bank.down,shared=None)
        with self.assertRaises(TypeError):
            QNg64MetalFusedRoutedFFN(bank.gate,bank.up,bank.down,shared=(None,None,None))

if __name__=="__main__":
    assert hashlib.sha256((ROOT/"qng64_metal_fused_routed_ffn_research.py").read_bytes()).hexdigest()==SOURCE_SHA
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FusedDeviceTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        doc={"schema":"BEGLIN-P18-MLX-METAL-QNG64-FUSED-FFN-SYNTHETIC/1",
            "status":"GPU_FUSED_METAL_BF16_ROUTER_NUMERICAL_PASS",
            "tests":result.testsRun,
            "max_rel_vs_unfused_device":FusedDeviceTests.max_rel,
            "source_sha256":SOURCE_SHA,
            "no_model_weights":True,
            "not_claiming_speedup":True,
            "automatic_live_promotion":False}
        path=ROOT/"qng64_metal_fused_ffn_synthetic_tests_v1.json"
        with path.open("x",encoding="utf-8") as f:json.dump(doc,f,sort_keys=True,indent=2);f.write("\n")
        print(json.dumps({**doc,"receipt_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
