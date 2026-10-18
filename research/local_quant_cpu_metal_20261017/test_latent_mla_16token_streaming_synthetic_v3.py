#!/usr/bin/env python3
"""BEGLIN latent MLA KV cache synthetic streaming / adversarial 16-token QA.

No original model, no environment keys, no serving changes. Uses a real BF16
linear kv_b transformation with per-head expansion as independent baseline.
Tests append-only latent cache, byte-for-byte K/V parity for every prefix and
rebind lifecycle without claiming actual model quality.
"""
import hashlib,importlib.util,json,pathlib,torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=ROOT/"latent_mla_kv_source_cache_v3.py"
PIN="2d96a2d6b2bbac169a76c91bbea7fed9e36e44e56429e73103adbcf8e2c35650"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==PIN
spec=importlib.util.spec_from_file_location("source_cache_v2_qacheck",SOURCE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx=0
        self.num_heads=16
        self.kv_lora_rank=512
        self.qk_nope_head_dim=128
        self.qk_rope_head_dim=64
        self.v_head_dim=128
        self.kv_a_layernorm=nn.Identity()
        self.kv_b_proj=nn.Linear(512,4096,bias=False).to(torch.bfloat16).eval()
def base(att,hidden,kpe):
    z=att.kv_b_proj(att.kv_a_layernorm(hidden))
    b,t,_=z.shape
    k,v=z.view(b,t,16,256).transpose(1,2).split([128,128],dim=-1)
    return torch.cat((k,kpe.expand(1,16,t,64)),dim=-1),v
def forbidden(fn,name):
    try:fn()
    except RuntimeError as e:
        assert "FAIL_CLOSED" in str(e)
        return
    raise AssertionError("unrejected "+name)
@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261019)
    a=Attention()
    cache=mod.NativeSourceMLALatentCache(a,0,16)
    expectedK=[];expectedV=[]
    test_steps=[]
    for pos in range(16):
        if pos in (5,12):
            cache.unbind()
            assert cache.attention is None
            cache.bind(a)
            assert cache.size_tokens()==pos
        h=torch.randn(1,1,512).to(torch.bfloat16)
        kpe=torch.randn(1,1,1,64).to(torch.bfloat16)
        original_k,original_v=base(a,h,kpe)
        with cache.capture():
            # A physical kv_a norm forward is essential to staging the raw source latent.
            a.kv_a_layernorm(h)
            current_k,current_v=cache.update(
                original_k,original_v,0,{"sin":None,"cos":None})
        expectedK.append(original_k);expectedV.append(original_v)
        trueK=torch.cat(expectedK,dim=2)
        trueV=torch.cat(expectedV,dim=2)
        assert torch.equal(current_k,trueK),(pos,"K mismatch")
        assert torch.equal(current_v,trueV),(pos,"V mismatch")
        assert cache.size_tokens()==pos+1
        assert cache.size_bytes()==(pos+1)*576*2
        assert cache.max_temporary_expanded_bytes==(pos+1)*16*320*2
        if pos in (0,4,5,15):
            test_steps.append({
             "prefix_tokens":pos+1,
             "saved_latent_bytes":cache.size_bytes(),
             "expanded_reference_bytes":(pos+1)*16*320*2,
             "rehydration_calls":cache.rehydrate_calls,
             "parity":"BITWISE"})
    forbidden(lambda:cache.get_usable_length(1,0),"overflow beyond 16 tokens")
    fork=cache.fork()
    assert fork.size_bytes()==cache.size_bytes()
    fork.unbind()
    assert fork.attention is None and cache.attention is a
    cache.unbind()
    assert cache.attention is None
    print(json.dumps({
     "status":"PASS_SYNTHETIC_STREAMING_LATENT_KV_16_TOKENS",
     "source_cache_sha256":PIN,
     "source_kv_backprojection":"actual BF16 torch.Linear, not checkpoint weights",
     "steps_verified":16,"rebind_count":2,
     "source_mathematical_reference_KV":"BITWISE_EQUAL",
     "resident_per_layer_at_16_tokens":16*576*2,
     "expanded_per_layer_at_16_tokens":16*16*320*2,
     "cache_saving_percent":88.75,
     "snapshots":test_steps,
     "model_checkpoint_accessed":False,
     "serving_modified":False
    },sort_keys=True))
if __name__=="__main__":main()
