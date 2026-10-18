#!/usr/bin/env python3
"""EOE-only GPU-resident n7 kv_b for genuine DeepSeek fused-MLA cache decoding.

Own original qNg64 packed source weight buffers are uploaded ONCE per decoder
layer into C++ GPU MLX-owned uint8+F32 arrays, then reused through prefill,
route-fork previews, and two model-selected token steps. Native fused Metal
has no expanded K/V; original PyTorch source uncached reference is separately
evaluated on repacked source tensors in every comparison pass.

Important limitations: all OTHER linear roles are still reloaded/packed every
step; device-resident kv_b alone is not whole-model resident throughput.
"""
from __future__ import annotations

import atexit
import ctypes
from hashlib import sha256
from pathlib import Path
import threading

import numpy as np

from latent_mla_kv_source_cache_v3 import need
from native_mla_fused_decode_adapter_v1 import (
    FusedMetalSourceAttention,
    FusedSourceMLACache,
)

ROOT=Path(__file__).resolve().parent
RESIDENT_CPP=ROOT/"mla_qng64_resident_weight_handle_v1.cpp"
RESIDENT_DYLIB=ROOT/"libbeglin_mla_qng64_resident_weight_v1.dylib"
CPP_SHA="4128d4d469c0312d0927d190f8971da157780bee3ffc966882e72e479b1891ac"
DYLIB_SHA="bd5f940227482d39d21db1dd14c501c17daa62e3f73a8446903e11b8f640bd3e"
SOURCE_PLANE_BYTES=1835008
SOURCE_SCALE_BYTES=131072
SOURCE_KVB_BYTES=SOURCE_PLANE_BYTES+SOURCE_SCALE_BYTES


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


class ResidentFusedMetalSourceAttention(FusedMetalSourceAttention):
    def __init__(self,baseline_fused_dylib,packed_class,source_ns):
        need(digest(RESIDENT_CPP)==CPP_SHA and
             digest(RESIDENT_DYLIB)==DYLIB_SHA,
             "GPU resident C++ source/library SHA mismatch")
        super().__init__(baseline_fused_dylib,packed_class,source_ns)
        self.resident=ctypes.CDLL(str(RESIDENT_DYLIB))
        c_u8=ctypes.POINTER(ctypes.c_uint8)
        c_f32=ctypes.POINTER(ctypes.c_float)
        self._reg=self.resident.beglin_mla_qng64_resident_register_v1
        self._reg.argtypes=(c_u8,c_f32,ctypes.c_int)
        self._reg.restype=ctypes.c_uint64
        self._forward=self.resident.beglin_mla_qng64_resident_forward_v1
        self._forward.argtypes=(ctypes.c_uint64,c_f32,c_f32,c_f32,c_f32,ctypes.c_int,ctypes.c_float,c_f32)
        self._forward.restype=ctypes.c_int
        self._unreg=self.resident.beglin_mla_qng64_resident_unregister_v1
        self._unreg.argtypes=(ctypes.c_uint64,)
        self._unreg.restype=ctypes.c_int
        self._count=self.resident.beglin_mla_qng64_resident_active_weights_v1
        self._count.restype=ctypes.c_size_t
        self._bytes=self.resident.beglin_mla_qng64_resident_payload_bytes_v1
        self._bytes.argtypes=(ctypes.c_uint64,)
        self._bytes.restype=ctypes.c_size_t
        self._active=self.resident.beglin_mla_qng64_resident_mlx_active_memory_v1
        self._active.restype=ctypes.c_size_t
        self._available=self.resident.beglin_mla_qng64_resident_gpu_available_v1
        self._available.restype=ctypes.c_int
        need(self._available()==1 and self._count()==0,
             "real isolated GPU resident ABI available and registry must be empty")
        self.handles={}                # source layer integer -> uint64 GPU handle
        self.weight_authority={}       # source layer integer -> SHA-256 packed operator
        self.initial_gpu_active_bytes=int(self._active())
        self.after_first_full_layer_prefill_bytes=None
        self.max_gpu_active_bytes=int(self._active())
        self.gpu_registrations=0
        self.gpu_resident_calls=0
        self._current_id=0
        self._lock=threading.RLock()
        self.closed=False
        # Swaps only the GPU wrapper's actual attention call. The wrapper's
        # independent CPU reference still calls source original attention.
        self.fn=self._resident_call
        atexit.register(self.close)

    def _resident_call(self,packed,scales,bits,qn,qr,latent,rope,tokens,scale,out):
        need(self._current_id>0 and bits==7,
             "resident packed GPU lookup without valid source layer")
        answer=self._forward(self._current_id,qn,qr,latent,rope,tokens,scale,out)
        if answer==1:
            self.gpu_resident_calls+=1
        return answer

    def _register_or_lookup(self,attn,cache):
        layer=int(cache.layer)
        need(attn.layer_idx==layer and
             cache.kv_projection_sha256 is not None,
             "source KV layer and qNg64 operator authority mismatch")
        if layer in self.handles:
            need(self.weight_authority[layer]==cache.kv_projection_sha256 and
                 self._bytes(self.handles[layer])==SOURCE_KVB_BYTES,
                 "packed source qNg64 GPU weight changed across layer reload")
            return self.handles[layer]
        original=attn.kv_b_proj
        need(isinstance(original,self.packed_class) and original.n==7
             and original.in_features==512 and original.out==4096,
             "source kv_b must remain a canonical packed n7 qNg64 role")
        planes=original.packed
        scales=original.scales
        need(isinstance(planes,np.ndarray) and planes.dtype==np.uint8
             and planes.flags["C_CONTIGUOUS"] and planes.nbytes==SOURCE_PLANE_BYTES
             and isinstance(scales,np.ndarray) and scales.dtype==np.float32
             and scales.flags["C_CONTIGUOUS"] and scales.nbytes==SOURCE_SCALE_BYTES,
             "actual source native bitplanes/scales layout")
        pointer_planes=planes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        pointer_scales=scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        handle=int(self._reg(pointer_planes,pointer_scales,7))
        need(handle>0 and self._bytes(handle)==SOURCE_KVB_BYTES,
             "actual source qNg64 GPU device weight registration did not succeed")
        need(handle not in self.handles.values(),"GPU C++ native handle collision")
        self.handles[layer]=handle
        self.weight_authority[layer]=cache.kv_projection_sha256
        self.gpu_registrations+=1
        self.max_gpu_active_bytes=max(self.max_gpu_active_bytes,int(self._active()))
        need(int(self._count())==len(self.handles),
             "device resident source registry counter drift")
        return handle

    def run(self,attn,hidden,mask,position,cache):
        with self._lock:
            need(not self.closed and self._current_id==0,
                 "resident GPU research wrapper is not reentrant")
            handle=self._register_or_lookup(attn,cache)
            self._current_id=handle
            try:
                return super().run(attn,hidden,mask,position,cache)
            finally:
                self._current_id=0

    def describe(self):
        with self._lock:
            count=len(self.handles)
            allocated=sum(int(self._bytes(h)) for h in self.handles.values())
            need(int(self._count())==count and
                 allocated==count*SOURCE_KVB_BYTES,
                 "GPU registry resident role count / bytes mismatch")
            return {
              "actual_device_Metal_layer_count":count,
              "real_layer_indexes":sorted(self.handles),
              "resident_device_bitplanes_F32scales_total_bytes":allocated,
              "persistent_GPU_source_qNg64_registry_handle_count":count,
              "physical_fused_GPU_source_query_calls":self.Metal_fused_queries,
              "resident_GPU_forward_calls":self.gpu_resident_calls,
              "GPU_allocator_active_baseline_B":self.initial_gpu_active_bytes,
              "GPU_allocator_active_current_B":int(self._active()),
              "GPU_allocator_active_observed_peak_B":self.max_gpu_active_bytes,
              "one_time_weight_register_calls":self.gpu_registrations,
              "same_source_packed_weight_sha_across_rebinds":True,
              "GPU_quantized_resident_bytes_only_source_KVB":True,
              "GPU_all_model_weights_resident":False,
            }

    def close(self):
        with self._lock:
            if self.closed:return
            if self._current_id:raise RuntimeError("in-flight GPU operation on close")
            ids=list(self.handles.values())
            for handle in ids:
                need(self._unreg(handle)==1,"GPU owned weight handle unregister failed")
            self.handles.clear()
            self.closed=True
            need(self._count()==0,"GPU source weights not released")
