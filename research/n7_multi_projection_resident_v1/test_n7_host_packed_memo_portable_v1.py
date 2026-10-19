#!/usr/bin/env python3
"""Standalone BEGLIN n7/g64 source-sha packed-weight memo test on original C.

No original model/checkpoint or platform account is needed. Requires the
locally compiled original qng64_dense.c reference produced by build_and_test.sh
and numpy. This probes original native-C quantization, C qng64_matmul,
source-array immutability, SHA-hit avoiding packing, bounded eviction, full
pointer lifetime and native bind restoration.
"""
import ctypes
import json
from pathlib import Path
import numpy as np

from research_n7_source_pack_memo_v1 import SourcePackedN7Memo

ROOT=Path(__file__).resolve().parent
DLL=ROOT/"build"/"libbeglin_qng64_reference.dylib"
F=ctypes.POINTER(ctypes.c_float)
U=ctypes.POINTER(ctypes.c_uint8)

def check(ok,msg):
    if not ok:raise RuntimeError("BEGLIN_N7_MEMO_SYNTHETIC_FAIL_CLOSED: "+msg)
def ptr(a,t):return a.ctypes.data_as(t)

class OriginalNativeC:
    def __init__(self):
        self.lib=ctypes.CDLL(str(DLL))
        self.lib.qng64_packed_bytes.argtypes=(ctypes.c_int,ctypes.c_int,ctypes.c_int)
        self.lib.qng64_packed_bytes.restype=ctypes.c_size_t
        self.lib.qng64_quantize_f32.argtypes=(F,ctypes.c_int,ctypes.c_int,ctypes.c_int,U,F)
        self.lib.qng64_quantize_f32.restype=ctypes.c_int
        self.lib.qng64_matmul_f32.argtypes=(U,F,ctypes.c_int,F,F,F,
            ctypes.c_int,ctypes.c_int,ctypes.c_int)
        self.lib.qng64_matmul_f32.restype=None

    def bind(self,weights,n):
        out,inp=map(int,weights.shape)
        p=np.empty(self.lib.qng64_packed_bytes(out,inp,n),dtype=np.uint8)
        s=np.empty(out*(inp//64),dtype=np.float32)
        check(self.lib.qng64_quantize_f32(ptr(weights,F),out,inp,n,
             ptr(p,U),ptr(s,F))==0,"native BEGLIN C code failed")
        return p,s

def random(seed,shape):
    return np.ascontiguousarray(
        np.random.default_rng(seed).normal(0,.013,shape),dtype=np.float32)

def main():
    check(DLL.exists(),"original BEGLIN C qNg64 reference not built")
    native=OriginalNativeC()
    original=native.bind
    memo=SourcePackedN7Memo(native,max_bytes=4*1024**2)
    memo.install()
    try:
        a=random(1,(128,512))
        b=random(2,(4096,512))
        c=random(3,(2048,512))
        pa,sa=native.bind(a,7)
        pa_again,sa_again=native.bind(a.copy(),7)
        check(pa_again is pa and sa_again is sa and memo.hits==1
              and memo.original_quantization_calls==1,
              "identical source SHA should reuse same immutable packed source")
        check(not pa.flags.writeable and not sa.flags.writeable,
              "persistent compact CPU memory must be immutable")
        xa=random(4,(2,512))
        y=np.empty((2,128),np.float32)
        native.lib.qng64_matmul_f32(ptr(pa,U),ptr(sa,F),7,ptr(xa,F),None,
            ptr(y,F),128,512,2)
        check(np.isfinite(y).all(),"original C qNg64 matmul from memo arrays")
        native.bind(b,7)
        native.bind(c,7)
        changed=b.copy()
        changed[0,0]+=np.float32(.15)
        native.bind(changed,7)
        native.bind(random(9,(4096,512)),7)
        check(memo.evictions>=1,"enforced LRU must evict source packed arrays")
        original_pa=pa.copy()
        pa2,sa2=native.bind(a,7)
        check(np.array_equal(pa2,original_pa) and
              np.array_equal(sa2,sa),
              "repacking after LRU eviction must reproduce identical source planes and scales")
        for value,n,label in ((a,9,"wrong_bits"),
                              (np.ones((128,512),dtype=np.float64),7,"dtype"),
                              (np.ones((128,63),dtype=np.float32),7,"group64"),
                              (np.full((128,512),np.nan,dtype=np.float32),7,"nan")):
            try:native.bind(value,n)
            except RuntimeError as e:
                check("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED" in str(e),
                      label+" error gate")
            else:raise RuntimeError("BEGLIN_N7_MEMO_SYNTHETIC_FAIL_CLOSED:"+label)
        stats=memo.stats()
        check(stats["source_quantization_avoided_by_content_hash_hit"]>=1
              and stats["host_compact_LRU_evictions"]>=1
              and stats["resident_host_compact_n7_bytes"]<=4*1024**2,
              "source content memo bounded state")
        print(json.dumps({"status":"PASS_REAL_NATIVE_C_N7_SHA_HOST_PACKED_MEMO_SYNTHETIC",
              "sha_memo_hits":memo.hits,
              "native_n7_quantization_calls":memo.original_quantization_calls,
              "evictions":memo.evictions,
              "resident_host_compact_bytes":memo.bytes,
              "original_checkpoint_read":False,
              "production_modified":False},sort_keys=True))
    finally:
        memo.close()
        check(native.bind==original,
              "original Native.bind source implementation not restored")
if __name__=="__main__":main()
