#!/usr/bin/env python3
"""Research-only DeepSeek-V2 MLA *resident* BF16 latent cache.

The 512D source kv_a_layernorm result and the 64D *rotated* k_pe are
retained per position, i.e. 576 BF16 coordinates per token/layer. On a
subsequent token, previously cached K/V are materialized **transiently** by
the original per-layer kv_b_proj. The current original source attention's
own K/V are appended to that temporary reconstruction and returned to its
unmodified forward method.

This is deliberately **not** a fused native GPU attention/KV implementation.
The old 16-head K/V tensors are not persisted. This uses the original
per-layer frozen qNg64 C++ Metal kv_b projection for rehydration when that
projection has been swapped by the research harness. Production unaffected.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch


def fail(msg):
    raise RuntimeError("BEGLIN_MLA_LATENT_CACHE_FAIL_CLOSED: "+msg)


def need(cond, msg):
    if not cond:
        fail(msg)


class NativeSourceMLALatentCache:
    """Source-compatible cache API: get_usable_length / update.

    Only BF16 normalized latent and one-head rotated key position are resident.
    No raw expanded K/V retained after attention returns. Runtime checks are
    deliberately strict to prevent silent shape/dtype/position drift.
    """

    def __init__(self, attention, layer_idx: int, max_tokens: int = 64):
        need(type(layer_idx) is int and 0 <= layer_idx <= 26, "layer ID")
        need(type(max_tokens) is int and 1 <= max_tokens <= 64, "token cap")
        self.attention = attention
        self.layer = layer_idx
        self.max_tokens = max_tokens
        need(getattr(attention,"layer_idx",None)==layer_idx, "source attention layer index")
        need(attention.num_heads==16 and attention.kv_lora_rank==512
             and attention.qk_nope_head_dim==128
             and attention.qk_rope_head_dim==64
             and attention.v_head_dim==128,
             "original pinned MLA dimension contract")
        need(hasattr(attention,"kv_a_layernorm") and hasattr(attention,"kv_b_proj"),
             "source original MLA modules missing")
        self.normalized_latent = None  # {1,T,512} BF16 normalized source output
        self.rotated_key = None        # {1,1,T,64} BF16 source-original RoPE
        self.staged = None
        self.updates = 0
        self.rehydrate_calls = 0
        self.last_rehydrated_expanded_bytes = 0
        self.max_temporary_expanded_bytes = 0
        self.cumulative_rehydrate_wall_seconds = 0.0

    def unbind(self):
        """Release all model weights/attention refs while retaining only latent KV.

        Required between decoder layers and before the next source-weight load;
        otherwise a supposedly streaming research pass accidentally keeps full
        per-layer quantized projections resident through self.attention.
        """
        need(self.staged is None,"cannot release half-staged MLA cache")
        need(self.attention is not None,"already unbound")
        self.attention=None

    def bind(self, attention):
        """Reattach the *same source layer index* during the next decoding step."""
        need(self.attention is None,"double-bind can accidentally retain model weights")
        need(attention.layer_idx==self.layer
             and attention.num_heads==16 and attention.kv_lora_rank==512
             and attention.qk_nope_head_dim==128
             and attention.qk_rope_head_dim==64 and attention.v_head_dim==128,
             "rebinding wrong model's attention projection")
        self.attention=attention

    def get_usable_length(self, qlen, layer_idx):
        need(layer_idx==self.layer and type(qlen) is int
             and 1 <= qlen <= self.max_tokens, "source K/V query metadata")
        n=0 if self.normalized_latent is None else int(self.normalized_latent.shape[1])
        need(n+qlen<=self.max_tokens, "KV maximum source sequence length")
        return n

    def _stage_normalized(self, module, inputs, output):
        need(self.staged is None, "multiple normalization calls in one attention step")
        need(isinstance(output,torch.Tensor) and output.dtype==torch.bfloat16
             and output.device.type=="cpu" and output.dim()==3
             and output.shape[0]==1 and output.shape[2]==512
             and torch.isfinite(output).all().item(),"bad BF16 MLA latent normalization")
        self.staged=output.detach()

    @contextmanager
    def capture(self):
        """Install a temporary hook around an **original** attention call."""
        need(self.staged is None and self.attention is not None,
             "latent normalization staging contaminated or unbound")
        handle=self.attention.kv_a_layernorm.register_forward_hook(self._stage_normalized)
        try:
            yield self
        finally:
            handle.remove()
            self.staged=None

    def fork(self):
        """Read-only branch for source MoE routing preflight (no base mutation)."""
        need(self.staged is None and self.attention is not None,
             "cannot fork half-staged or detached model weights")
        c=NativeSourceMLALatentCache(self.attention,self.layer,self.max_tokens)
        c.normalized_latent=self.normalized_latent
        c.rotated_key=self.rotated_key
        c.updates=self.updates
        return c

    def _unpack_old(self, old_norm, old_rot):
        need(old_norm.dtype==torch.bfloat16 and
             tuple(old_norm.shape[:1])==(1,) and old_norm.shape[2]==512,
             "stored normalized source latent format")
        # ATTENTION: the same attn.kv_b_proj that produced the original K/V
        # reconstructs the old positions from the BF16 latent.
        full=self.attention.kv_b_proj(old_norm)
        n=int(old_norm.shape[1])
        need(full.dtype==torch.bfloat16 and
             tuple(full.shape)==(1,n,16*(128+128)),
             "latent to K/V original projection mismatch")
        expanded=full.view(1,n,16,256).transpose(1,2)
        k_nope, old_v=torch.split(expanded,[128,128],dim=-1)
        need(tuple(old_rot.shape)==(1,1,n,64),"stored RoPE key shape")
        old_k=torch.cat((k_nope,old_rot.expand(1,16,n,64)),dim=-1)
        need(tuple(old_k.shape)==(1,16,n,192)
             and tuple(old_v.shape)==(1,16,n,128),
             "rehydrated original expanded K/V shape")
        return old_k,old_v

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        need(layer_idx==self.layer and isinstance(cache_kwargs,dict)
             and all(k in cache_kwargs for k in ("cos","sin")),
             "source updater ID/RoPE metadata contract")
        need(isinstance(key_states,torch.Tensor)
             and isinstance(value_states,torch.Tensor)
             and key_states.device.type==value_states.device.type=="cpu"
             and key_states.dtype==value_states.dtype==torch.bfloat16,
             "original K/V BF16 CPU contract")
        need(key_states.dim()==value_states.dim()==4 and
             key_states.shape[:3]==value_states.shape[:3]
             and key_states.shape[0]==1 and key_states.shape[1]==16
             and key_states.shape[3]==192 and value_states.shape[3]==128,
             "source original K/V shape contract")
        b,heads,qlen,_=key_states.shape
        old=self.get_usable_length(int(qlen),layer_idx)
        pending=self.staged
        need(isinstance(pending,torch.Tensor) and
             tuple(pending.shape)==(1,qlen,512) and
             pending.dtype==torch.bfloat16,
             "missing/mismatched source normalized latent capture")
        if self.normalized_latent is None:
            need(self.rotated_key is None and old==0,"latent/key cache initialized consistently")
        else:
            need(self.rotated_key is not None and self.rotated_key.shape[2]==old,
                 "latent/kpe prior sequence integrity")
        new_rot=key_states[:,0:1,:,128:192]
        need(torch.equal(key_states[:,:,:,128:192],new_rot.expand(1,16,qlen,64)),
             "RoPE k_pe differs across original source heads")
        need(torch.isfinite(key_states).all().item() and
             torch.isfinite(value_states).all().item(),
             "nonfinite original source key/value")
        # Critical: reconstruct only past tokens, *not* duplicate current K/V.
        # The caller's original attention computed the current states already.
        if old>0:
            import time
            t=time.perf_counter()
            old_k,old_v=self._unpack_old(self.normalized_latent,self.rotated_key)
            self.cumulative_rehydrate_wall_seconds+=time.perf_counter()-t
            out_k=torch.cat([old_k,key_states],dim=2)
            out_v=torch.cat([old_v,value_states],dim=2)
            self.rehydrate_calls+=1
        else:
            out_k=key_states
            out_v=value_states
        # Clone exact BF16 narrow source record; do NOT retain expanded K/V.
        next_norm=pending.detach().clone()
        next_rot=new_rot.detach().clone()
        self.normalized_latent=(next_norm if old==0
            else torch.cat([self.normalized_latent,next_norm],dim=1))
        self.rotated_key=(next_rot if old==0
            else torch.cat([self.rotated_key,next_rot],dim=2))
        self.staged=None
        self.updates+=1
        resident=self.size_bytes()
        expanded=(out_k.numel()+out_v.numel())*2
        self.last_rehydrated_expanded_bytes=expanded
        self.max_temporary_expanded_bytes=max(self.max_temporary_expanded_bytes,expanded)
        need(tuple(out_k.shape)==(1,16,old+qlen,192) and
             tuple(out_v.shape)==(1,16,old+qlen,128),
             "reconstructed expanded attention return tensor shape")
        need(resident==1152*(old+qlen),"latent resident cache exact BF16 bytes")
        return out_k,out_v

    def size_bytes(self):
        need((self.normalized_latent is None)==(self.rotated_key is None),
             "latent/K-pe resident cache mismatch")
        if self.normalized_latent is None:
            return 0
        norm=self.normalized_latent
        rot=self.rotated_key
        n=int(norm.shape[1])
        need(norm.device.type==rot.device.type=="cpu"
             and norm.dtype==rot.dtype==torch.bfloat16
             and tuple(norm.shape)==(1,n,512)
             and tuple(rot.shape)==(1,1,n,64),
             "bad cache resident shapes")
        return norm.numel()*norm.element_size()+rot.numel()*rot.element_size()

    def size_tokens(self):
        return 0 if self.normalized_latent is None else int(self.normalized_latent.shape[1])

    def storage_shapes(self):
        return None if self.normalized_latent is None else {
            "normalized_latent":list(self.normalized_latent.shape),
            "rotated_RoPE_key":list(self.rotated_key.shape),
        }
