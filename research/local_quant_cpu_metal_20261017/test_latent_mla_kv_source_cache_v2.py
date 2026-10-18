#!/usr/bin/env python3
"""Self-contained no-checkpoint synthetic adversarial tests of MLA latent KV."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("latent_kv",ROOT/"latent_mla_kv_source_cache_v2.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Dummy(nn.Module):
    def __init__(self,seed=33):
        super().__init__()
        torch.manual_seed(seed)
        self.layer_idx=0
        self.num_heads=16
        self.kv_lora_rank=512
        self.qk_nope_head_dim=128
        self.qk_rope_head_dim=64
        self.v_head_dim=128
        self.kv_a_layernorm=nn.Identity()
        self.kv_b_proj=nn.Linear(512,16*256,bias=False).to(torch.bfloat16).eval()

def make_kv(att,h,seed):
    source=att.kv_a_layernorm(h)
    out=att.kv_b_proj(source)
    b,t,_=out.shape
    v=out.view(b,t,16,256).transpose(1,2)
    kn,vv=v.split([128,128],dim=-1)
    torch.manual_seed(seed)
    rope=torch.randn(b,1,t,64).to(torch.bfloat16)
    ks=torch.cat([kn,rope.expand(b,16,t,64)],dim=-1)
    return ks,vv,rope

def reject(code,what):
    try:code()
    except RuntimeError as e:
        assert "BEGLIN_MLA_LATENT_CACHE_FAIL_CLOSED" in str(e), (what,str(e))
        return what
    raise AssertionError("not rejected: "+what)

def main():
    torch.set_num_threads(2)
    a=Dummy()
    c=m.NativeSourceMLALatentCache(a,0,max_tokens=8)
    labels=[]
    shape=lambda k:[*k.shape]
    torch.manual_seed(11)
    h1=torch.randn(1,3,512).to(torch.bfloat16)
    h2=torch.randn(1,1,512).to(torch.bfloat16)
    with torch.inference_mode():
        with c.capture():
            k1,v1,_=make_kv(a,h1,21)
            K,V=c.update(k1,v1,0,{"cos":None,"sin":None})
        assert torch.equal(K,k1) and torch.equal(V,v1)
        assert c.size_bytes()==3456 and c.storage_shapes()=={
            "normalized_latent":[1,3,512],"rotated_RoPE_key":[1,1,3,64]}
        labels.append("PREFILL_ACTUAL_BF16_3456_BYTES")
        fork=c.fork()
        assert fork.size_bytes()==c.size_bytes() and fork.updates==c.updates
        with fork.capture():
            k2,v2,_=make_kv(a,h2,25)
            K2,V2=fork.update(k2,v2,0,{"cos":None,"sin":None})
        assert fork.size_bytes()==4608 and c.size_bytes()==3456
        assert torch.equal(K2,torch.cat([k1,k2],dim=2))
        assert torch.equal(V2,torch.cat([v1,v2],dim=2))
        labels.append("SOURCE_EXPANDED_REFERENCE_EXACT_KV_REHYDRATION")
        labels.append("ROUTER_FORK_NO_ORIGINAL_CACHE_MUTATION")
        with c.capture():
            k2b,v2b,_=make_kv(a,h2,25)
            K3,V3=c.update(k2b,v2b,0,{"cos":None,"sin":None})
        assert torch.equal(K2,K3) and torch.equal(V2,V3)
        assert c.rehydrate_calls==1
        assert c.max_temporary_expanded_bytes==4*16*(192+128)*2
        labels.append("REAL_TWO_STEP_4608_BYTES_AND_TEMPORARY_KV_SEPARATED")
        labels.append(reject(lambda: c.get_usable_length(5,0),"REJECT_MAX_CONTEXT"))
        labels.append(reject(lambda: c.get_usable_length(1,1),"REJECT_WRONG_LAYER"))
        labels.append(reject(lambda: c.update(k2,v2,0,{"cos":None,"sin":None}),"REJECT_MISSING_NORMALIZATION_CAPTURE"))
        with c.capture():
            kbad,vbad,_=make_kv(a,h2,44)
            kbad=kbad.clone()
            kbad[:,1,:,-1]+=1.0
            labels.append(reject(lambda: c.update(kbad,vbad,0,{"cos":None,"sin":None}),
                                 "REJECT_HEAD_ROPE_MISMATCH"))
        with c.capture():
            kbad,vbad,_=make_kv(a,h2,45)
            labels.append(reject(lambda:c.update(kbad.float(),vbad,0,{"cos":None,"sin":None}),
                                 "REJECT_NONBF16_KV"))
        with c.capture():
            kbad,vbad,_=make_kv(a,h2,45)
            labels.append(reject(lambda:c.update(kbad,vbad,0,{}),
                                 "REJECT_MISSING_ROTARY_META"))
        with c.capture():
            kbad,vbad,_=make_kv(a,h2,46)
            labels.append(reject(lambda:c.update(kbad,vbad,2,{"cos":None,"sin":None}),
                                 "REJECT_UPDATER_LAYER_ID"))
        labels.append(reject(lambda: m.NativeSourceMLALatentCache(a,9,64),
                             "REJECT_WRONG_SOURCE_ATTENTION_LAYER"))
        labels.append(reject(lambda: m.NativeSourceMLALatentCache(a,0,0),
                             "REJECT_ILLEGAL_CONTEXT_CAP"))
        labels.append(reject(lambda: m.NativeSourceMLALatentCache(a,0,1024),
                             "REJECT_UNBOUNDED_CONTEXT_CAP"))
        # Streaming-weight lifecycle: resident cache must not retain the model
        # or an attention reference between actual decoder layers.
        old_bytes=c.size_bytes()
        c.unbind()
        assert c.attention is None and c.size_bytes()==old_bytes
        labels.append("UNBIND_RELEASES_MODEL_ATTENTION_REFERENCES")
        labels.append(reject(lambda: c.fork(),"REJECT_FORK_WITHOUT_BOUND_MODEL"))
        labels.append(reject(lambda: c.capture().__enter__(),"REJECT_CAPTURE_WITHOUT_BOUND_MODEL"))
        c.bind(a)
        assert c.attention is a and c.size_bytes()==old_bytes
        labels.append("REBIND_SAME_MODEL_LAYER_WITH_IDENTICAL_LATENT_STATE")
        labels.append(reject(lambda:c.bind(a),"REJECT_DOUBLE_BIND"))
        c.unbind()
        wrong=Dummy()
        wrong.layer_idx=1
        labels.append(reject(lambda:c.bind(wrong),"REJECT_WRONG_LAYER_REBIND"))
        assert len(labels)==20,labels
        print(json.dumps({"status":"PASS_MLA_LATENT_REHYDRATION_OFFLINE",
              "cases":len(labels),"coverage":labels,
              "physical_model_weights_read":False,"serving_modified":False},sort_keys=True))

if __name__=="__main__":
    main()
