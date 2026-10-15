#!/usr/bin/env python3
"""P18 source-identical 27-layer original BF16 teacher with single bound 112→1024.

Original P17 GoldWindowTeacher.forward_gold code copied byte-for-byte except
the 112-position admissibility constant. Its entire P17 source is pinned by SHA
and original 5291 source tensor parity. No quantized policy or serving changes.
"""
from __future__ import annotations
import hashlib,pathlib,time,resource,gc
import torch
from torch import nn
from torch.nn import functional as F
import hcp_p17_teacher_gold as og
from hcp_p15_streamed_teacher import tensor_sha
ORIGINAL_P17_TEACHER_SHA="40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3"
ORIGINAL_P17_MODULE=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015/src/tools/hcp_p17_teacher_gold.py")
if hashlib.sha256(ORIGINAL_P17_MODULE.read_bytes()).hexdigest()!=ORIGINAL_P17_TEACHER_SHA:
    raise RuntimeError("P17 source drift: refuse long-window original BF16 research")

class P18LongGoldWindowTeacher(og.GoldWindowTeacher):
    @torch.inference_mode()
    def forward_gold(self,ids,prompt_length,gold_len):
        if (ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=torch.int64 or
            gold_len<1 or gold_len>8 or prompt_length<2 or
            ids.shape[1]!=prompt_length+gold_len or ids.shape[1]>1024):
            raise ValueError("Invalid P17 forced reference context")
        if bool((ids<0).any()) or bool((ids>=self.config.vocab_size).any()):
            raise ValueError("P17 frozen original tokenizer id out of bounds")
        self.layer_records=[]
        seen_before=len(self.store.loaded_names)
        n=ids.shape[1]
        pos=torch.arange(n).unsqueeze(0)
        mask=torch.full((1,1,n,n),torch.finfo(self.dtype).min,dtype=self.dtype)
        mask=torch.triu(mask,diagonal=1)
        embedding=self.store.load("model.embed_tokens.weight",self.dtype)
        h=F.embedding(ids,embedding)
        self.max_resident_parameter_bytes=max(self.max_resident_parameter_bytes,
            embedding.numel()*embedding.element_size())
        del embedding;gc.collect()
        for i in range(self.config.num_hidden_layers):
            started=time.monotonic()
            layer=self.layer(i)
            size=sum(p.numel()*p.element_size() for p in layer.parameters())
            self.max_resident_parameter_bytes=max(self.max_resident_parameter_bytes,size)
            h=layer(h,attention_mask=mask,position_ids=pos,past_key_value=None,
                     output_attentions=False,use_cache=False)[0]
            if not bool(torch.isfinite(h).all()):
                raise ValueError(f"Nonfinite real original model hidden state at {i}")
            r={"layer":i,"parameter_bytes":size,
               "hidden_sha256":tensor_sha(h),
               "seconds":time.monotonic()-started,
               "source_rss_peak_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            self.layer_records.append(r)
            del layer;gc.collect()
            if self.progress:self.progress(r)
        norm=self.classes["DeepseekV2RMSNorm"](self.config.hidden_size,eps=self.config.rms_norm_eps)
        norm.weight=nn.Parameter(self.store.load("model.norm.weight",self.dtype),requires_grad=False)
        h=norm(h);del norm
        head=self.store.load("lm_head.weight",self.dtype)
        # Token 0 of the gold completion is predicted at prompt_len - 1.
        # Token k of gold is predicted by prefix:prompt + gold[:k].
        logits=F.linear(h[:,prompt_length-1:prompt_length+gold_len-1,:],head).to(torch.float32)
        del head;gc.collect()
        if list(logits.shape)!=[1,gold_len,self.config.vocab_size] or not bool(torch.isfinite(logits).all()):
            raise ValueError("Original whole-model teacher logits span missing/nonfinite")
        loaded=self.store.loaded_names[seen_before:]
        if len(loaded)!=len(set(loaded)) or set(loaded)!=set(self.store.mapping):
            raise ValueError("Teacher full model did not read all 5291 original checkpoint tensors exactly once")
        self.store.verify_sources()
        return logits.squeeze(0).contiguous()

