#!/usr/bin/env python3
"""BEGLIN research-only source-attested qNg64 n7 Metal weight cache.

Bounded CPU+GPU compact-weight registry. Native MLX C++ GPU handles are
registered once per quantized projection and reused for genuine model forward
calls. Explicit LRU evicts handles before admitting new matrices; origin
qNg64 module + CPU authoritative packed source retain identical C-plane/scale
bytes. The GPU bridge itself also keeps immutable CPU buffers alive due to
observed unified-memory borrowed-source lifetime hazards.

NOT a production model loader, quant policy change, isolated GPU-only
allocation claim, or engine/serving speed guarantee.
"""
from __future__ import annotations

from collections import OrderedDict
import ctypes
import hashlib
import threading
import time

import numpy as np
import torch
from torch import nn

F=ctypes.POINTER(ctypes.c_float)
U=ctypes.POINTER(ctypes.c_uint8)
HARD_LIB_BUDGET=256*1024*1024
HARD_LIB_HANDLES=128


def need(ok,desc):
    if not ok:
        raise RuntimeError("BEGLIN_DEVICE_LRU_QNG64_FAIL_CLOSED: "+desc)


def digest(planes,scales):
    d=hashlib.sha256()
    d.update(b"BEGLIN/qng64/n7/group64/immutable-source/1\0")
    d.update(planes.tobytes())
    d.update(scales.tobytes())
    return d.hexdigest()


class ResidentN7Cache:
    def __init__(self,library,bytes_budget=96*1024*1024,max_handles=12):
        need(type(bytes_budget) is int and 64*1024<=bytes_budget<=HARD_LIB_BUDGET,
             "bounded cache byte budget")
        need(type(max_handles) is int and 1<=max_handles<=HARD_LIB_HANDLES,
             "bounded GPU weight handle count")
        self.lib=library
        self.register_fn=library.beglin_multi_n7_register_v1
        self.register_fn.argtypes=(U,F,ctypes.c_int,ctypes.c_int,ctypes.c_int)
        self.register_fn.restype=ctypes.c_uint64
        self.forward_fn=library.beglin_multi_n7_forward_v1
        self.forward_fn.argtypes=(ctypes.c_uint64,F,ctypes.c_int,F)
        self.forward_fn.restype=ctypes.c_int
        self.unregister_fn=library.beglin_multi_n7_unregister_v1
        self.unregister_fn.argtypes=(ctypes.c_uint64,)
        self.unregister_fn.restype=ctypes.c_int
        self.active_fn=library.beglin_multi_n7_active_count_v1
        self.active_fn.restype=ctypes.c_size_t
        self.bytes_fn=library.beglin_multi_n7_payload_bytes_v1
        self.bytes_fn.restype=ctypes.c_uint64
        self.memory_fn=library.beglin_multi_n7_allocator_active_v1
        self.memory_fn.restype=ctypes.c_uint64
        self.budget=bytes_budget
        self.max_handles=max_handles
        need(self.active_fn()==0 and self.bytes_fn()==0,
             "requires empty isolated device registry at startup")
        self.lru=OrderedDict()
        self.source_ident={}
        self.local_bytes=0
        self.created=0
        self.evicted=0
        self.hits=0
        self.misses=0
        self.forward_calls=0
        self.forward_seconds=0.0
        self.source_sha_rechecks=0
        self.closed=False
        self.lock=threading.RLock()
        self.initial_device_memory=int(self.memory_fn())
        self.device_memory_peak=self.initial_device_memory

    @staticmethod
    def validate(module):
        need(getattr(module,"n",None)==7 and
             isinstance(getattr(module,"packed",None),np.ndarray) and
             isinstance(getattr(module,"scales",None),np.ndarray),
             "only original already quantized source n7 modules")
        p,s=module.packed,module.scales
        o=int(module.out);i=int(module.in_features)
        need(p.dtype==np.uint8 and s.dtype==np.float32 and
             p.flags["C_CONTIGUOUS"] and s.flags["C_CONTIGUOUS"]
             and o>0 and o<=16384 and i>=64 and i<=16384 and i%64==0,
             "packed GPU n7 role geometry/format")
        count=o*(i//64)*7*8
        nsc=o*(i//64)
        need(p.size==count and s.size==nsc
             and np.isfinite(s).all() and np.all(s>0),
             "canonical original qNg64 group layout/scales")
        bytes_=count+nsc*4
        need(bytes_==module.serialized_bytes,"original module packed byte accounting")
        return p,s,o,i,bytes_

    def _evict(self):
        key,entry=self.lru.popitem(last=False)
        rc=self.unregister_fn(entry["handle"])
        need(rc==1,"native qNg64 GPU source handle eviction failed")
        self.local_bytes-=entry["bytes"]
        self.evicted+=1

    def _get(self,key,module):
        need(not self.closed and type(key) is str and 0<len(key)<=180,
             "invalid device-model role or already closed")
        p,s,o,i,bytes_=self.validate(module)
        signature=digest(p,s)
        self.source_sha_rechecks+=1
        prior=self.source_ident.get(key)
        if prior is None:self.source_ident[key]=(signature,o,i)
        else:need(prior==(signature,o,i),"source n7 weight mutated or role rekeyed")
        if key in self.lru:
            entry=self.lru.pop(key)
            need(entry["sha"]==signature and entry["out"]==o and entry["in"]==i,
                 "GPU cache handle identity drift")
            self.lru[key]=entry
            self.hits+=1
            return entry
        self.misses+=1
        need(bytes_<=self.budget,"individual source matrix exceeds dedicated GPU budget")
        while self.lru and (
            self.local_bytes+bytes_>self.budget or
            len(self.lru)>=self.max_handles):
            self._evict()
        h=int(self.register_fn(p.ctypes.data_as(U),s.ctypes.data_as(F),o,i,7))
        need(h>0,"source qNg64 GPU weight registration denied")
        record={"handle":h,"sha":signature,"bytes":bytes_,"out":o,"in":i}
        self.lru[key]=record
        self.local_bytes+=bytes_
        self.created+=1
        self.device_memory_peak=max(self.device_memory_peak,int(self.memory_fn()))
        need(self.local_bytes<=self.budget and
             len(self.lru)<=self.max_handles and
             self.active_fn()==len(self.lru) and
             self.bytes_fn()==self.local_bytes,
             "bounded GPU registry actual source bytes diverged")
        return record

    def call(self,key,module,x):
        with self.lock:
            need(isinstance(x,torch.Tensor) and x.dtype==torch.bfloat16 and
                 x.device.type=="cpu" and x.shape[-1]==module.in_features,
                 "source BF16 model activation")
            input_=np.ascontiguousarray(
                x.detach().float().numpy().reshape(-1,module.in_features),
                dtype=np.float32)
            b=int(input_.shape[0])
            need(1<=b<=16 and np.isfinite(input_).all(),
                 "bounded finite N7 source GEMV minibatch")
            entry=self._get(key,module)
            y=np.empty((b,entry["out"]),dtype=np.float32)
            start=time.perf_counter()
            rc=self.forward_fn(entry["handle"],input_.ctypes.data_as(F),
                               b,y.ctypes.data_as(F))
            self.forward_seconds+=time.perf_counter()-start
            need(rc==1 and np.isfinite(y).all(),"cached device qNg64 source GEMV invalid")
            self.forward_calls+=1
            return torch.from_numpy(
                np.ascontiguousarray(y).reshape(*x.shape[:-1],entry["out"])).to(torch.bfloat16)

    def get_state(self):
        with self.lock:
            need(self.active_fn()==len(self.lru) and
                 self.bytes_fn()==self.local_bytes,
                 "native source qNg64 registry unexpected ownership")
            return {
               "GPU_budget_B":self.budget,
               "GPU_handle_cap":self.max_handles,
               "active_GPU_n7_role_handles":len(self.lru),
               "active_GPU_packed_planes_scales_bytes":self.local_bytes,
               "GPU_source_roles_ever_registered":len(self.source_ident),
               "GPU_registration_count":self.created,
               "GPU_LRU_evictions":self.evicted,
               "GPU_reuse_hits":self.hits,
               "GPU_misses":self.misses,
               "actual_GPU_forward_calls":self.forward_calls,
               "actual_GPU_forward_wall_seconds":self.forward_seconds,
               "SHA256_original_source_buffer_rechecks":self.source_sha_rechecks,
               "active_MLX_allocator_bytes":int(self.memory_fn()),
               "max_observed_MLX_allocator_bytes":self.device_memory_peak,
               "CPU_source_buffer_retained_by_GPU_implementation":True
            }

    def close(self):
        with self.lock:
            if self.closed:return
            while self.lru:self._evict()
            need(self.active_fn()==0 and self.bytes_fn()==0,
                 "new generic n7 GPU registry leak after closing")
            self.closed=True


class DeviceN7Module(nn.Module):
    def __init__(self,role,source_packed_module,cache):
        super().__init__()
        need(isinstance(cache,ResidentN7Cache),"GPU model cache type")
        cache.validate(source_packed_module)
        self.source=source_packed_module
        self.role=role
        self.cache=cache
        self.out=self.source.out
        self.in_features=self.source.in_features
        self.serialized_bytes=self.source.serialized_bytes

    def forward(self,x):
        return self.cache.call(self.role,self.source,x)
