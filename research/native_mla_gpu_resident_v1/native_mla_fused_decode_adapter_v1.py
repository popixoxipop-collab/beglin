#!/usr/bin/env python3
"""EOE-only actual DeepSeek qNg64 C++ Metal fused MLA+latent KV decoder.

The ORIGINAL BF16 source attention remains available as independent uncached
reference. Only requests carrying FusedSourceMLACache run the compiled one
Metal attention kernel with packed uint8 planes and F32 scales; no expanded
head-K/V or dense qNg64 dequantized matrix exists in candidate path.
B=1, <=16 sequence tokens, frozen source n7 group64 only.
"""
from __future__ import annotations
import ctypes
from contextlib import contextmanager
import time

import numpy as np
import torch
from latent_mla_kv_source_cache_v3 import NativeSourceMLALatentCache, need

H=16
LAT=512
NOPE=128
ROPE=64
V=128
MAX_T=16

class FusedSourceMLACache(NativeSourceMLALatentCache):
    def __init__(self, attention, layer_idx, max_tokens=MAX_T):
        super().__init__(attention,layer_idx,max_tokens)
        self.fused_calls=0

    @contextmanager
    def capture(self):
        need(self.attention is not None and self.staged is None,
             "GPU fused cache requires attached original attention")
        yield self

    def fork(self):
        need(self.attention is not None and self.staged is None,
             "fork requires matching original source layer")
        child=FusedSourceMLACache(self.attention,self.layer,self.max_tokens)
        child.normalized_latent=self.normalized_latent
        child.rotated_key=self.rotated_key
        child.updates=self.updates
        return child

    def commit(self, latent, rotated):
        need(self.attention is not None and self.staged is None,
             "cannot commit without source weight identity")
        need(isinstance(latent,torch.Tensor) and isinstance(rotated,torch.Tensor)
             and latent.dtype==rotated.dtype==torch.bfloat16
             and latent.device.type==rotated.device.type=="cpu"
             and latent.ndim==3 and latent.shape[0]==1
             and latent.shape[2]==LAT,"source MLA latent format")
        t=latent.shape[1]
        need(rotated.shape==(1,1,t,ROPE),"source rotated key format")
        old=self.get_usable_length(t,self.layer)
        next_lat=latent.detach().clone()
        next_rot=rotated.detach().clone()
        if old:
            need(self.normalized_latent.shape==(1,old,LAT)
                 and self.rotated_key.shape==(1,1,old,ROPE),
                 "old cache shape corrupted")
            next_lat=torch.cat((self.normalized_latent,next_lat),dim=1)
            next_rot=torch.cat((self.rotated_key,next_rot),dim=2)
        need(torch.isfinite(next_lat).all().item()
             and torch.isfinite(next_rot).all().item(),
             "nonfinite BF16 source cache state")
        self.normalized_latent=next_lat
        self.rotated_key=next_rot
        self.updates+=1
        need(self.size_bytes()==(old+t)*(LAT+ROPE)*2,
             "persistent source MLA KV byte accounting")
        return self

    def update(self,*args,**kwargs):
        raise RuntimeError(
            "BEGLIN_GPU_FUSED_KV_FAIL_CLOSED: original expanded KV update not allowed")


class FusedMetalSourceAttention:
    def __init__(self, lib, packed_class, ns):
        self.packed_class=packed_class
        self.ns=ns
        fn=lib.beglin_mla_qng64_fused_metal_v1
        U=ctypes.POINTER(ctypes.c_uint8)
        F=ctypes.POINTER(ctypes.c_float)
        fn.argtypes=(U,F,ctypes.c_int,F,F,F,F,ctypes.c_int,ctypes.c_float,F)
        fn.restype=ctypes.c_int
        lib.beglin_mla_qng64_fused_metal_available_v1.restype=ctypes.c_int
        need(lib.beglin_mla_qng64_fused_metal_available_v1()==1,
             "compiled Apple Metal fused kernel not available")
        self.fn=fn
        self.Metal_fused_queries=0
        self.prefill_queries=0
        self.incremental_queries=0
        self.Metal_kernel_wall_seconds=0.0
        self.max_KV_tokens=0
        self.source_qNg64_KVB_compressed_bytes=0
        self.original_uncached_reference_calls=0

    @staticmethod
    def ptr(a,t):
        return a.ctypes.data_as(ctypes.POINTER(t))

    def patch_source(self,attn):
        need(not hasattr(attn,"_beglin_original_attention_forward"),
             "double patched original attention prohibited")
        original=attn.forward
        bridge=self
        def wrapped(*args,**kwargs):
            cache=kwargs.get("past_key_value")
            if isinstance(cache,FusedSourceMLACache):
                hidden=kwargs.get("hidden_states") if len(args)==0 else (
                    args[0] if len(args)==1 else None)
                need(kwargs.get("use_cache") is True and
                     isinstance(hidden,torch.Tensor),
                     "source-model attention hidden_states keyword/positional contract")
                result=bridge.run(attn,hidden,kwargs.get("attention_mask"),
                                  kwargs.get("position_ids"),cache)
                return result,None,cache
            bridge.original_uncached_reference_calls+=1
            return original(*args,**kwargs)
        attn._beglin_original_attention_forward=original
        attn.forward=wrapped

    @torch.inference_mode()
    def run(self,attn,hidden,mask,position,cache):
        need(cache.attention is attn and
             cache.kv_projection_sha256 is not None,
             "source cache is not bound to same original kv_b")
        need(hidden.dtype==torch.bfloat16 and hidden.device.type=="cpu"
             and hidden.ndim==3 and hidden.shape[0]==1
             and hidden.shape[2]==2048,
             "source hidden geometry / device / dtype")
        t=int(hidden.shape[1])
        old=cache.get_usable_length(t,cache.layer)
        total=old+t
        need(t>=1 and total<=MAX_T,"source native MLA cache token limit")
        need(position is not None and
             position.shape==(1,t) and
             torch.equal(position,torch.arange(old,total,
                 dtype=position.dtype).reshape(1,t)),
             "source RoPE position / cache offset mismatch")
        need(mask is not None and mask.shape==(1,1,t,total),
             "source attention causal mask invalid shape")
        rawmask=mask.float()[0,0]
        for idx in range(t):
            keep=old+idx+1
            need(bool(torch.all(rawmask[idx,:keep]==0).item()),
                 "source causal attention visible keys incorrectly masked")
            if keep<total:
                need(bool(torch.all(rawmask[idx,keep:]<=-1000).item()),
                     "source causal mask did not suppress future keys")
        packed_linear=attn.kv_b_proj
        need(isinstance(packed_linear,self.packed_class)
             and packed_linear.n==7 and
             packed_linear.out==4096 and packed_linear.in_features==512,
             "only genuine frozen qNg64 compressed kv_b admitted")
        packed=packed_linear.packed
        scales=packed_linear.scales
        need(isinstance(packed,np.ndarray) and isinstance(scales,np.ndarray)
             and packed.dtype==np.uint8 and scales.dtype==np.float32
             and packed.flags["C_CONTIGUOUS"] and scales.flags["C_CONTIGUOUS"]
             and packed.nbytes==1835008 and scales.nbytes==131072
             and np.isfinite(scales).all() and np.all(scales>0),
             "compressed qNg64 source planes/scales unavailable")
        self.source_qNg64_KVB_compressed_bytes=max(
            self.source_qNg64_KVB_compressed_bytes,packed.nbytes+scales.nbytes)
        q=attn.q_proj(hidden).view(1,t,H,NOPE+ROPE).transpose(1,2)
        kv=attn.kv_a_proj_with_mqa(hidden)
        need(q.dtype==kv.dtype==torch.bfloat16 and kv.shape==(1,t,LAT+ROPE),
             "original compressed q/kv_a source state mismatch")
        norm=attn.kv_a_layernorm(kv[:,:,:LAT])
        kpe=kv[:,:,LAT:].view(1,t,1,ROPE).transpose(1,2)
        cos,sin=attn.rotary_emb(kpe,seq_len=total)
        qnope,qpe=q.split([NOPE,ROPE],dim=-1)
        qrot,krot=self.ns["apply_rotary_pos_emb"](qpe,kpe,cos,sin,position)
        need(norm.dtype==qrot.dtype==krot.dtype==torch.bfloat16,
             "source RoPE BF16 arithmetic contract")
        if old:
            need(cache.normalized_latent.shape==(1,old,LAT)
                 and cache.rotated_key.shape==(1,1,old,ROPE),
                 "original cache source prefix invalid")
            data=torch.cat([cache.normalized_latent,norm],dim=1)
            key=torch.cat([cache.rotated_key,krot],dim=2)
        else:
            data,key=norm,krot
        l=np.ascontiguousarray(data[0].float().numpy(),dtype=np.float32)
        kr=np.ascontiguousarray(key[0,0].float().numpy(),dtype=np.float32)
        qn=np.ascontiguousarray(qnope[0].float().numpy(),dtype=np.float32)
        qr=np.ascontiguousarray(qrot[0].float().numpy(),dtype=np.float32)
        output=np.empty((t,H,V),dtype=np.float32)
        buffer=np.empty((H,V),dtype=np.float32)
        U=ctypes.c_uint8
        F=ctypes.c_float
        pp=self.ptr(packed,U)
        sp=self.ptr(scales,F)
        for i in range(t):
            end=old+i+1
            qi=np.ascontiguousarray(qn[:,i],dtype=np.float32)
            ri=np.ascontiguousarray(qr[:,i],dtype=np.float32)
            latent_prefix=np.ascontiguousarray(l[:end],dtype=np.float32)
            key_prefix=np.ascontiguousarray(kr[:end],dtype=np.float32)
            start=time.perf_counter()
            rc=self.fn(pp,sp,7,
                self.ptr(qi,F),self.ptr(ri,F),
                self.ptr(latent_prefix,F),self.ptr(key_prefix,F),
                end,float(attn.softmax_scale),self.ptr(buffer,F))
            self.Metal_kernel_wall_seconds+=time.perf_counter()-start
            need(rc==1 and np.isfinite(buffer).all(),
                 "physical fused qNg64 Metal MLA computation failed")
            output[i]=buffer
            self.Metal_fused_queries+=1
            self.max_KV_tokens=max(self.max_KV_tokens,end)
            if old:self.incremental_queries+=1
            else:self.prefill_queries+=1
        result=attn.o_proj(torch.from_numpy(output).reshape(1,t,H*V).to(torch.bfloat16))
        need(result.shape==(1,t,2048) and torch.isfinite(result).all().item(),
             "original n7 o_proj output nonfinite")
        cache.commit(norm,krot)
        cache.fused_calls+=t
        return result
