#!/usr/bin/env python3
"""Source-attested BEGLIN n7/g64 compact F32->qNg64 memo (research-only).

Intercepts only an explicitly selected Native.bind instance after original
BF16 source data is read. On repeated shape/content SHA-256, returns previously
verified immutable qNg64 uint8 planes and F32 scales INSTEAD OF re-quantizing.
Never changes policy, precision bits, data loader or model source. This cannot
eliminate BF16 source *reads*; it eliminates repeated native C packing for
matching source-weight content while the entry remains in the bounded RAM
cache. Persistent CPU memory footprint, not GPU-only or full-model serving.
"""
from collections import OrderedDict
from hashlib import sha256
import time
import numpy as np

class SourcePackedN7Memo:
    def __init__(self,native,max_bytes=2*1024**3):
        if type(max_bytes) is not int or not (4*1024**2<=max_bytes<=8*1024**3):
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: bounded host packed byte capacity")
        if not hasattr(native,"bind") or not callable(native.bind):
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: original Native.bind missing")
        self.native=native
        self.origin=native.bind
        self.original_id=id(self.origin.__func__) if hasattr(self.origin,"__func__") else None
        self.maximum=max_bytes
        self.lru=OrderedDict()
        self.bytes=0
        self.hits=0
        self.misses=0
        self.evictions=0
        self.original_quantization_calls=0
        self.source_digest_checks=0
        self.evict_bytes=0
        self.original_quantization_wall_s=0.0
        self.source_sha256_hash_wall_s=0.0
        self.peak_host_packed_bytes=0
        self.closed=False

    def bind(self,weight,n):
        if self.closed:
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: used after close")
        if type(n) is not int or n!=7:
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: only pinned native n7 accepted")
        if (not isinstance(weight,np.ndarray) or weight.ndim!=2 or
            weight.dtype!=np.float32 or not weight.flags["C_CONTIGUOUS"]):
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: source contiguous rank2 F32 required")
        out,inp=map(int,weight.shape)
        if out<=0 or out>16384 or inp<64 or inp>16384 or inp%64!=0:
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: unexpected source tensor geometry")
        if not np.isfinite(weight).all():
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: source dense input nonfinite")
        count=out*inp*7//8+(out*inp//64)*4
        if count>self.maximum:
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: source matrix larger than total host cache")
        t=time.perf_counter()
        hash_=sha256(memoryview(weight).cast("B")).hexdigest()
        self.source_sha256_hash_wall_s+=time.perf_counter()-t
        self.source_digest_checks+=1
        key=(out,inp,n,hash_)
        if key in self.lru:
            p,s=self.lru.pop(key)
            if p.nbytes+s.nbytes!=count or not np.isfinite(s).all() or not np.all(s>0):
                raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: compact cached memory corrupt")
            self.lru[key]=(p,s)
            self.hits+=1
            return p,s
        self.misses+=1
        t=time.perf_counter()
        p,s=self.origin(weight,n)
        self.original_quantization_wall_s+=time.perf_counter()-t
        self.original_quantization_calls+=1
        if (p.dtype!=np.uint8 or s.dtype!=np.float32 or
            not p.flags["C_CONTIGUOUS"] or not s.flags["C_CONTIGUOUS"] or
            p.nbytes+s.nbytes!=count or
            not np.isfinite(s).all() or not np.all(s>0)):
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: original native C packed output invalid")
        # Since the exact original source packed arrays may also be referenced
        # by live nn.Module instances, never modify cached plane/scale buffers.
        p.flags.writeable=False
        s.flags.writeable=False
        while self.lru and self.bytes+count>self.maximum:
            _,(oldp,olds)=self.lru.popitem(last=False)
            removed=oldp.nbytes+olds.nbytes
            self.bytes-=removed
            self.evict_bytes+=removed
            self.evictions+=1
        if self.bytes+count>self.maximum:
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: host LRU admission bound")
        self.lru[key]=(p,s)
        self.bytes+=count
        self.peak_host_packed_bytes=max(self.bytes,self.peak_host_packed_bytes)
        return p,s

    def install(self):
        candidate=self.native.bind
        if (getattr(candidate,"__func__",None) is not
                getattr(self.origin,"__func__",None) or
            getattr(candidate,"__self__",None) is not
                getattr(self.origin,"__self__",None)):
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: already wrapped Native.bind")
        self.native.bind=self.bind
        return self

    def stats(self):
        if not (self.bytes<=self.maximum and
                self.hits+self.misses==self.source_digest_checks and
                self.misses==self.original_quantization_calls):
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: accounting mismatch")
        return {
          "research_host_compact_n7_cache_budget_bytes":self.maximum,
          "resident_host_compact_n7_bytes":self.bytes,
          "peak_resident_host_compact_n7_bytes":self.peak_host_packed_bytes,
          "original_native_C_quantize_executions":self.original_quantization_calls,
          "source_dense_matrix_SHA256_validations":self.source_digest_checks,
          "source_quantization_avoided_by_content_hash_hit":self.hits,
          "source_weight_not_found_or_evicted_repacking_required":self.misses,
          "original_quantization_wall_seconds":self.original_quantization_wall_s,
          "source_dense_SHA256_hash_seconds":self.source_sha256_hash_wall_s,
          "host_compact_LRU_evictions":self.evictions,
          "host_compact_LRU_evicted_bytes":self.evict_bytes,
          "resident_distinct_matrix_content_hash_count":len(self.lru),
          "source_BF16_safetensors_weight_reload_still_required":True,
          "actual_original_n7_quantization_skipped_on_content_hit":True,
          "original_precision_or_operation_policy_changed":False,
        }

    def close(self):
        if self.closed:return
        candidate=self.native.bind
        # Bound method objects are reconstructed on access, compare function
        # and bound instance separately instead of Python object identity.
        if (getattr(candidate,"__func__",None) is not self.bind.__func__ or
            getattr(candidate,"__self__",None) is not self):
            raise RuntimeError("BEGLIN_N7_PACK_MEMO_FAIL_CLOSED: native binding ownership was replaced")
        self.native.bind=self.origin
        self.lru.clear()
        self.bytes=0
        self.closed=True
