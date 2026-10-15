"""Toy ORIGINAL-shape model tests for shared-weight two-length computation.

These tests do NOT substitute for actual original DeepSeek BF16 model validation.
"""
import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from hcp_p18_shared_bf16_core import forward_two_original_source_sequences

class TinyStore:
    def __init__(self):
        g = torch.Generator().manual_seed(751)
        self.weights = {
            "model.embed_tokens.weight": torch.randn(32, 8, generator=g).to(torch.bfloat16),
            "model.layers.0.weight": torch.randn(8, 8, generator=g).to(torch.bfloat16) / 32,
            "model.layers.1.weight": torch.randn(8, 8, generator=g).to(torch.bfloat16) / 32,
            "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(32, 8, generator=g).to(torch.bfloat16) / 16,
        }
        self.mapping = dict.fromkeys(self.weights, "tiny")
        self.loaded_names = []
    def load(self, name, dtype):
        if name in self.loaded_names:
            raise RuntimeError("loaded twice")
        self.loaded_names.append(name)
        return self.weights[name].to(dtype)
    def verify_sources(self):
        assert len(self.loaded_names) == len(self.mapping)
class TinyNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(hidden_size,dtype=torch.bfloat16),requires_grad=False)
    def forward(self,h):
        x=h.float()
        return (x*torch.rsqrt(torch.mean(x*x,dim=-1,keepdim=True)+self.eps)).to(h.dtype)*self.weight
class TinyLayer(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.weight = nn.Parameter(matrix,requires_grad=False)
    def forward(self,h,attention_mask,position_ids,past_key_value,output_attentions,use_cache):
        assert attention_mask.shape == (1,1,h.shape[1],h.shape[1])
        assert position_ids.shape == (1,h.shape[1])
        assert not output_attentions and not use_cache and past_key_value is None
        # Position and causal-mask-dependent computation, independent of padding.
        b=torch.full_like(h, 0.03)*position_ids.unsqueeze(-1).to(h.dtype)
        return (torch.tanh(F.linear(h+b,self.weight)),)
class TinyTeacher:
    def __init__(self):
        self.store=TinyStore()
        self.config=SimpleNamespace(vocab_size=32,hidden_size=8,rms_norm_eps=1e-6,num_hidden_layers=2)
        self.dtype=torch.bfloat16
        self.classes={"DeepseekV2RMSNorm":TinyNorm}
    def layer(self,i):
        return TinyLayer(self.store.load(f"model.layers.{i}.weight",self.dtype))
@torch.inference_mode()
def original_one(seq):
    t=TinyTeacher()
    ids=torch.tensor([seq],dtype=torch.int64)
    n=len(seq)
    p=torch.arange(n).unsqueeze(0)
    mask=torch.triu(torch.full((1,1,n,n),torch.finfo(t.dtype).min,dtype=t.dtype),diagonal=1)
    h=F.embedding(ids,t.store.load("model.embed_tokens.weight",t.dtype))
    for i in range(t.config.num_hidden_layers):
        h=t.layer(i)(h,attention_mask=mask,position_ids=p,past_key_value=None,output_attentions=False,use_cache=False)[0]
    norm=TinyNorm(8)
    norm.weight=nn.Parameter(t.store.load("model.norm.weight",t.dtype),requires_grad=False)
    h=norm(h)
    y=F.linear(h[:,-2:-1,:],t.store.load("lm_head.weight",t.dtype)).float().squeeze(0)
    t.store.verify_sources()
    return y

class SharedSourceTests(unittest.TestCase):
    def setUp(self):
        self.short=[1,2,3,4,5,6,7]
        self.long=[8,7,6,5,4,3,2,1,10,11,12,13,14]
    def test_bit_exact_original_unequal_lengths(self):
        t=TinyTeacher()
        pair=forward_two_original_source_sequences(t,[self.short,self.long])
        self.assertTrue(torch.equal(pair[0],original_one(self.short)))
        self.assertTrue(torch.equal(pair[1],original_one(self.long)))
        self.assertEqual(len(t.store.loaded_names),5)
        self.assertEqual(len(set(t.store.loaded_names)),5)
    def test_reversed_order_still_original(self):
        t=TinyTeacher()
        pair=forward_two_original_source_sequences(t,[self.long,self.short])
        self.assertTrue(torch.equal(pair[0],original_one(self.long)))
        self.assertTrue(torch.equal(pair[1],original_one(self.short)))
    def test_duplicate_tokens_still_one_tensor_load(self):
        t=TinyTeacher()
        pair=forward_two_original_source_sequences(t,[self.short,self.short])
        self.assertTrue(torch.equal(pair[0],pair[1]))
        self.assertEqual(len(t.store.loaded_names),5)
    def test_too_many_sequences_fail(self):
        with self.assertRaises(ValueError):forward_two_original_source_sequences(TinyTeacher(),[self.short]*3)
    def test_one_sequence_fail(self):
        with self.assertRaises(ValueError):forward_two_original_source_sequences(TinyTeacher(),[self.short])
    def test_invalid_token_bool_fail(self):
        with self.assertRaises(ValueError):forward_two_original_source_sequences(TinyTeacher(),[[True,2],self.long])
    def test_invalid_token_vocab_fail(self):
        with self.assertRaises(ValueError):forward_two_original_source_sequences(TinyTeacher(),[[31,32],self.long])
    def test_invalid_padding_zero_length_fail(self):
        with self.assertRaises(ValueError):forward_two_original_source_sequences(TinyTeacher(),[[],self.long])
    def test_overlong_fails(self):
        with self.assertRaises(ValueError):forward_two_original_source_sequences(TinyTeacher(),[self.short,self.long],max_length=12)
    def test_used_store_refuses(self):
        t=TinyTeacher();t.store.load("model.norm.weight",t.dtype)
        with self.assertRaises(ValueError):forward_two_original_source_sequences(t,[self.short,self.long])

if __name__=="__main__":unittest.main(verbosity=2)
