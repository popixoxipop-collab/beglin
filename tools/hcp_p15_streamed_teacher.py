#!/usr/bin/env python3
"""P15 full DeepSeek-V2-Lite reference, one original decoder layer resident.

Uses ONLY SHA-pinned pure torch classes/functions extracted without edits from
modeling_deepseek.py shipped with the original checkpoint. No remote-code import,
no quantization, no native production engine mutation, no KV-cache approximation.
BF16 checkpoint weights with explicit CPU compute dtype (BF16 or FP32).
"""
from __future__ import annotations
import ast,contextlib,gc,hashlib,json,logging,math,os,pathlib,resource,struct,time,types,warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers.activations import ACT2FN

INDEX_SHA='d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf'
CONFIG_SHA='f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d'
SOURCE_SHA='7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c'
NAMES=('DeepseekV2RMSNorm','DeepseekV2RotaryEmbedding',
 'DeepseekV2LinearScalingRotaryEmbedding','DeepseekV2DynamicNTKScalingRotaryEmbedding',
 'yarn_find_correction_dim','yarn_find_correction_range','yarn_get_mscale',
 'yarn_linear_ramp_mask','DeepseekV2YarnRotaryEmbedding','rotate_half',
 'apply_rotary_pos_emb','DeepseekV2MLP','MoEGate','AddAuxiliaryLoss',
 'DeepseekV2MoE','repeat_kv','DeepseekV2Attention','DeepseekV2DecoderLayer')

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def tensor_sha(t):
    a=t.detach().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()

def original_classes(source):
    source=pathlib.Path(source)
    if source.is_symlink() or sha(source)!=SOURCE_SHA:
        raise ValueError('Original model implementation SHA mismatch')
    text=source.read_text();tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in NAMES]
    if len(nodes)!=len(NAMES) or {n.name for n in nodes}!=set(NAMES):
        raise ValueError('Original reference class dependency set differs')
    # Compile only inspected class/function definitions; imports, decorators on
    # HF wrapper models, remote-code registration, and top-level actions excluded.
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    selected=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
    ns={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'warnings':warnings,
        'ACT2FN':ACT2FN,'dist':torch.distributed,'logger':logging.getLogger('p15.original')}
    exec(compile(selected,str(source),'exec'),ns)
    ns['ATTENTION_CLASSES']={'eager':ns['DeepseekV2Attention']}
    code_receipt={n.name:hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest() for n in nodes}
    return ns,code_receipt

class TensorStore:
    """Original safetensors index with strict metadata and per-read value digests."""
    def __init__(self,root,*,production_checkpoint=True):
        self.root=pathlib.Path(root).resolve(strict=True)
        index=self.root/'model.safetensors.index.json'
        config=self.root/'config.json'
        if production_checkpoint and (sha(index)!=INDEX_SHA or sha(config)!=CONFIG_SHA):
            raise ValueError('P15 checkpoint or configuration is not the fixed original')
        self.config=json.loads(config.read_text())
        self.mapping=json.loads(index.read_text())['weight_map']
        self.meta={};self.snapshots={};self.loaded_names=[];self.value_hashes={}
        for shard in sorted(set(self.mapping.values())):
            p=self.root/shard
            if pathlib.Path(shard).name!=shard or p.is_symlink() or not p.is_file():
                raise ValueError('Foreign checkpoint shard')
            st=p.stat();self.snapshots[shard]=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
            with p.open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0]
                if not 0<n<8*1024**2:raise ValueError('Invalid safetensors header size')
                d=json.loads(f.read(n))
            intervals=[]
            for name,md in d.items():
                if name=='__metadata__':continue
                if name in self.meta or self.mapping.get(name)!=shard:
                    raise ValueError('Duplicate/misindexed checkpoint tensor')
                shape=md['shape'];a,b=md['data_offsets']
                if md['dtype']!='BF16' or not shape or any(type(x)!=int or x<=0 for x in shape):
                    raise ValueError('Teacher requires original BF16 tensors')
                if b-a!=math.prod(shape)*2 or a<0 or b+n+8>st.st_size:
                    raise ValueError('Invalid checkpoint tensor layout')
                self.meta[name]={'shape':shape,'source_bytes':b-a,'shard':shard}
                intervals.append((a,b))
            ranges=sorted(intervals)
            if any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):
                raise ValueError('Overlapping source tensor offsets')
        if set(self.mapping)!=set(self.meta):raise ValueError('Incomplete source tensor index')
        if production_checkpoint and (len(self.meta)!=5291 or self.config['num_hidden_layers']!=27):
            raise ValueError('Not the complete original DeepSeek-V2-Lite checkpoint')
        self.production_checkpoint=production_checkpoint
        self.max_loaded_source_bytes=0
    def verify_sources(self):
        for shard,expected in self.snapshots.items():
            s=(self.root/shard).stat()
            if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=expected:
                raise ValueError('Checkpoint modified during streaming')
    def load(self,name,dtype):
        if dtype not in (torch.float32,torch.bfloat16):raise ValueError('Unsupported reference dtype')
        md=self.meta[name];p=self.root/md['shard']
        s=p.stat()
        if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=self.snapshots[md['shard']]:
            raise ValueError('Checkpoint changed before tensor read')
        with safe_open(str(p),framework='pt',device='cpu') as handle:
            source=handle.get_tensor(name)
            if list(source.shape)!=md['shape'] or source.dtype!=torch.bfloat16:
                raise ValueError('Original tensor shape/dtype mismatch')
            # BF16 values are hashed before any numerical widening to FP32.
            digest=tensor_sha(source)
            if name in self.value_hashes and digest!=self.value_hashes[name]:
                raise ValueError('Teacher source values changed between passes')
            self.value_hashes[name]=digest
            result=source.to(dtype=dtype,device='cpu',copy=True)
            del source
        self.loaded_names.append(name)
        self.max_loaded_source_bytes=max(self.max_loaded_source_bytes,md['source_bytes'])
        return result
    def load_layer(self,layer,index,dtype):
        prefix=f'model.layers.{index}.'
        expected=set(layer.state_dict())
        actual={n[len(prefix):] for n in self.meta if n.startswith(prefix)}
        if expected!=actual:
            raise ValueError(f'Layer {index} unexpected/missing parameters: {expected^actual}')
        state={name:self.load(prefix+name,dtype) for name in sorted(expected)}
        result=layer.load_state_dict(state,strict=True,assign=True)
        if result.missing_keys or result.unexpected_keys:raise ValueError('Incomplete layer assignment')
        del state
        if any(p.is_meta or p.device.type!='cpu' or p.dtype!=dtype for p in layer.parameters()):
            raise ValueError('Unmaterialized/wrong-precision parameter in teacher')

class StreamedTeacher:
    def __init__(self,store,classes,*,dtype=torch.float32,progress=None):
        self.store=store;self.classes=classes;self.dtype=dtype;self.progress=progress
        cfg=dict(store.config);cfg['_attn_implementation']='eager';cfg['ep_size']=1
        self.config=types.SimpleNamespace(**cfg)
        if cfg.get('topk_method')!='greedy' or cfg.get('scoring_func')!='softmax':
            raise ValueError('Unvalidated original expert routing configuration')
        self.max_resident_parameter_bytes=0;self.layer_records=[]
    def layer(self,index):
        with torch.device('meta'):
            layer=self.classes['DeepseekV2DecoderLayer'](self.config,index)
        self.store.load_layer(layer,index,self.dtype)
        # These are deterministic original RoPE buffers, not checkpoint tensors.
        # Reinitialize from the ORIGINAL untouched configuration on CPU.
        layer.self_attn._init_rope()
        layer.eval()
        return layer
    @torch.inference_mode()
    def forward(self,ids,*,stop_after=None):
        if ids.ndim!=2 or ids.dtype!=torch.int64 or ids.shape[0]!=1 or not 1<=ids.shape[1]<=128:
            raise ValueError('P15 first qualification is a single unpadded prompt <=128 tokens')
        if bool((ids<0).any()) or bool((ids>=self.config.vocab_size).any()):
            raise ValueError('Token out of original vocabulary')
        self.layer_records=[];seen_before=len(self.store.loaded_names)
        length=ids.shape[1];pos=torch.arange(length).unsqueeze(0)
        mask=torch.full((1,1,length,length),torch.finfo(self.dtype).min,dtype=self.dtype)
        mask=torch.triu(mask,diagonal=1)
        embedding=self.store.load('model.embed_tokens.weight',self.dtype)
        h=F.embedding(ids,embedding)
        self.max_resident_parameter_bytes=max(self.max_resident_parameter_bytes,embedding.numel()*embedding.element_size())
        del embedding;gc.collect()
        limit=self.config.num_hidden_layers if stop_after is None else stop_after
        for i in range(limit):
            started=time.monotonic();layer=self.layer(i)
            nbytes=sum(p.numel()*p.element_size() for p in layer.parameters())
            self.max_resident_parameter_bytes=max(self.max_resident_parameter_bytes,nbytes)
            h=layer(h,attention_mask=mask,position_ids=pos,past_key_value=None,
                    output_attentions=False,use_cache=False)[0]
            if not bool(torch.isfinite(h).all()):raise ValueError(f'Nonfinite hidden state at layer {i}')
            record={'layer':i,'parameter_bytes':nbytes,'hidden_sha256':tensor_sha(h),
                    'seconds':time.monotonic()-started,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            self.layer_records.append(record)
            del layer;gc.collect()
            if self.progress:self.progress(record)
        if stop_after is not None:return h
        norm=self.classes['DeepseekV2RMSNorm'](self.config.hidden_size,eps=self.config.rms_norm_eps)
        norm.weight=nn.Parameter(self.store.load('model.norm.weight',self.dtype),requires_grad=False)
        h=norm(h);del norm
        head=self.store.load('lm_head.weight',self.dtype)
        logits=F.linear(h[:,-1,:],head).to(torch.float32)
        del head;gc.collect()
        if list(logits.shape)!=[1,self.config.vocab_size] or not bool(torch.isfinite(logits).all()):
            raise ValueError('Teacher did not produce finite full-vocabulary logits')
        loaded=self.store.loaded_names[seen_before:]
        if len(loaded)!=len(set(loaded)) or set(loaded)!=set(self.store.mapping):
            raise ValueError('Full forward failed to consume every checkpoint parameter exactly once')
        self.store.verify_sources()
        return logits
