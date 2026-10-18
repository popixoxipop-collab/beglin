#!/usr/bin/env python3
"""BF16 synthetic 64-token source-latent KV replay parity + storage growth.

No real model checkpoint, serving or EOE D50 access. This is only long-context
boundary QA of the stand-alone resident normalized-latent + RoPE-key cache.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from latent_mla_kv_source_cache_v2 import NativeSourceMLALatentCache

class FakeSourceAttention(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(9001)
        self.layer_idx=1
        self.num_heads=16
        self.kv_lora_rank=512
        self.qk_nope_head_dim=128
        self.qk_rope_head_dim=64
        self.v_head_dim=128
        self.kv_a_layernorm=nn.Identity()
        self.kv_b_proj=nn.Linear(512,16*256,bias=False).to(torch.bfloat16).eval()

def make(att,n,offset):
    torch.manual_seed(300+offset)
    latent=torch.randn(1,n,512).to(torch.bfloat16)
    norm=att.kv_a_layernorm(latent)
    buf=att.kv_b_proj(norm).view(1,n,16,256).transpose(1,2)
    k_nope,v=buf.split([128,128],dim=-1)
    torch.manual_seed(2000+offset)
    key_rot=torch.randn(1,1,n,64).to(torch.bfloat16)
    k=torch.cat([k_nope,key_rot.expand(1,16,n,64)],dim=-1)
    return k,v

def main():
    torch.set_num_threads(2)
    att=FakeSourceAttention()
    cache=NativeSourceMLALatentCache(att,1,max_tokens=64)
    independent_K=None
    independent_V=None
    report=[]
    with torch.inference_mode():
        for i,n in enumerate((1,7,8,16,31,1)):
            previous=cache.size_tokens()
            with cache.capture():
                K,V=make(att,n,previous)
                updatedK,updatedV=cache.update(K,V,1,{"cos":None,"sin":None})
            independent_K=K if independent_K is None else torch.cat([independent_K,K],2)
            independent_V=V if independent_V is None else torch.cat([independent_V,V],2)
            assert torch.equal(updatedK,independent_K)
            assert torch.equal(updatedV,independent_V)
            length=previous+n
            assert cache.size_tokens()==length
            assert cache.size_bytes()==length*1152
            assert cache.max_temporary_expanded_bytes==length*10240
            report.append({"tokens":length,"resident_latent_bytes":cache.size_bytes(),
                           "expanded_baseline_bytes":length*10240,
                           "last_rehydrate_bytes":cache.last_rehydrated_expanded_bytes})
        assert cache.size_tokens()==64 and cache.size_bytes()==73728
        assert cache.rehydrate_calls==5 and cache.updates==6
        try:cache.get_usable_length(1,1)
        except RuntimeError as e:
            assert "maximum source sequence length" in str(e)
        else:raise AssertionError("64 token bound not enforced")
        cache.unbind()
        assert cache.attention is None and cache.size_bytes()==73728
        print(json.dumps({"status":"PASS_64TOKEN_SYNTHETIC_LATENT_MLA_BF16",
             "source_real_model":False,"max_context_tokens":64,
             "steps":len(report),"validated_exactly_equal_reconstructed_full_KV":True,
             "resident_latent_bytes_at_64":73728,
             "expanded_BF16_cache_bytes_at_64":655360,
             "KV_saving_percent":88.75,"source_projection_reconstruction_calls":cache.rehydrate_calls,
             "source_checkpoint_read":False,"no_serving_model_mutation":True,
             "snapshots":report},sort_keys=True))
if __name__=="__main__":
    main()
