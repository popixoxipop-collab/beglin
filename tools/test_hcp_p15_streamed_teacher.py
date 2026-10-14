#!/usr/bin/env python3
"""P15 real CPU tests of full-layer streaming versus a resident miniature model.

The same SHA-pinned original DeepSeek computational classes are used, no mock
model outputs. Reduced dimensions only for resident/streaming equivalence.
"""
from __future__ import annotations
import copy,json,pathlib,tempfile,types,unittest,hashlib
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file
from hcp_p15_streamed_teacher import TensorStore,StreamedTeacher,original_classes,SOURCE_SHA,tensor_sha

SOURCE=pathlib.Path('/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/modeling_deepseek.py')
MODEL=SOURCE.parent

class P15StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.classes,cls.code_hashes=original_classes(SOURCE)
        cfg=json.loads((MODEL/'config.json').read_text())
        cfg.update(vocab_size=128,hidden_size=32,num_hidden_layers=3,
          num_attention_heads=2,num_key_value_heads=2,intermediate_size=64,
          kv_lora_rank=16,q_lora_rank=None,qk_nope_head_dim=8,qk_rope_head_dim=8,
          v_head_dim=16,moe_intermediate_size=16,n_routed_experts=4,
          n_shared_experts=2,num_experts_per_tok=2,max_position_embeddings=128,
          _attn_implementation='eager',ep_size=1,attention_dropout=0.0)
        cls.cfg=cfg
        torch.manual_seed(15015)
        conf=types.SimpleNamespace(**cfg)
        cls.embed=nn.Embedding(128,32)
        cls.layers=nn.ModuleList([cls.classes['DeepseekV2DecoderLayer'](conf,i) for i in range(3)])
        cls.norm=cls.classes['DeepseekV2RMSNorm'](32,eps=cfg['rms_norm_eps'])
        cls.head=nn.Linear(32,128,bias=False)
        cls.layers.eval();cls.norm.eval()
        state={'model.embed_tokens.weight':cls.embed.weight.detach().to(torch.bfloat16),
               'model.norm.weight':cls.norm.weight.detach().to(torch.bfloat16),
               'lm_head.weight':cls.head.weight.detach().to(torch.bfloat16)}
        for i,layer in enumerate(cls.layers):
            for n,p in layer.state_dict().items():state[f'model.layers.{i}.{n}']=p.detach().to(torch.bfloat16).contiguous()
        cls.state=state
        cls.ids=torch.tensor([[1,13,57,22,8,47,5]],dtype=torch.int64)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)
        save_file(self.state,str(self.root/'model.safetensors'))
        (self.root/'config.json').write_text(json.dumps(self.cfg))
        self.index={'weight_map':{k:'model.safetensors' for k in self.state}}
        (self.root/'model.safetensors.index.json').write_text(json.dumps(self.index))
    def tearDown(self):self.tmp.cleanup()
    def store(self):return TensorStore(self.root,production_checkpoint=False)
    @torch.inference_mode()
    def resident(self,dtype,ids=None,stop=None):
        ids=self.ids if ids is None else ids
        cfg=types.SimpleNamespace(**self.cfg)
        h=F.embedding(ids,self.state['model.embed_tokens.weight'].to(dtype))
        n=ids.shape[1];pos=torch.arange(n).unsqueeze(0)
        mask=torch.triu(torch.full((1,1,n,n),torch.finfo(dtype).min,dtype=dtype),diagonal=1)
        layers=nn.ModuleList([self.classes['DeepseekV2DecoderLayer'](cfg,i) for i in range(3)])
        for i,layer in enumerate(layers):
            prefix=f'model.layers.{i}.'
            local={k[len(prefix):]:v.to(dtype) for k,v in self.state.items() if k.startswith(prefix)}
            layer.load_state_dict(local,strict=True,assign=True);layer.eval()
        for layer in layers[:3 if stop is None else stop]:
            h=layer(h,attention_mask=mask,position_ids=pos,use_cache=False)[0]
        if stop is not None:return h
        norm=self.classes['DeepseekV2RMSNorm'](32,eps=self.cfg['rms_norm_eps'])
        norm.weight=nn.Parameter(self.state['model.norm.weight'].to(dtype),requires_grad=False)
        return F.linear(norm(h[:,-1]),self.state['lm_head.weight'].to(dtype)).float()
    def test_01_original_source_all_18_definition_pins(self):
        self.assertEqual(len(self.code_hashes),18)
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),SOURCE_SHA)
    def test_02_fp32_stream_equals_resident_full_logits(self):
        actual=StreamedTeacher(self.store(),self.classes,dtype=torch.float32).forward(self.ids)
        expected=self.resident(torch.float32)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    def test_03_bf16_stream_equals_resident_full_logits(self):
        actual=StreamedTeacher(self.store(),self.classes,dtype=torch.bfloat16).forward(self.ids)
        expected=self.resident(torch.bfloat16)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    def test_04_all_original_weights_consumed_once(self):
        store=self.store();teacher=StreamedTeacher(store,self.classes)
        teacher.forward(self.ids)
        self.assertEqual(len(store.loaded_names),len(self.state))
        self.assertEqual(set(store.loaded_names),set(self.state))
        self.assertEqual(len(store.value_hashes),len(self.state))
    def test_05_layer0_dense_exact(self):
        h=StreamedTeacher(self.store(),self.classes).forward(self.ids,stop_after=1)
        torch.testing.assert_close(h,self.resident(torch.float32,stop=1),rtol=0,atol=0)
    def test_06_first_MoE_layer_exact(self):
        h=StreamedTeacher(self.store(),self.classes).forward(self.ids,stop_after=2)
        torch.testing.assert_close(h,self.resident(torch.float32,stop=2),rtol=0,atol=0)
    def test_07_second_run_repeats_full_logits(self):
        t=StreamedTeacher(self.store(),self.classes)
        a=t.forward(self.ids);b=t.forward(self.ids)
        self.assertEqual(tensor_sha(a),tensor_sha(b))
    def test_08_source_mutation_after_headers_denied(self):
        s=self.store();p=self.root/'model.safetensors'
        p.touch()
        with self.assertRaises(ValueError):s.load('lm_head.weight',torch.float32)
    def test_09_non_BF16_source_rejected(self):
        bad={k:v.to(torch.float32) for k,v in self.state.items()}
        save_file(bad,str(self.root/'model.safetensors'))
        with self.assertRaises(ValueError):self.store()
    def test_10_missing_parameter_index_rejected(self):
        self.index['weight_map'].pop('lm_head.weight')
        (self.root/'model.safetensors.index.json').write_text(json.dumps(self.index))
        with self.assertRaises(ValueError):self.store()
    def test_11_source_code_tamper_rejected(self):
        p=self.root/'wrong.py';p.write_bytes(SOURCE.read_bytes()+b'\n')
        with self.assertRaises(ValueError):original_classes(p)
    def test_12_out_of_vocabulary_rejected(self):
        with self.assertRaises(ValueError):StreamedTeacher(self.store(),self.classes).forward(torch.tensor([[128]]))
    def test_13_oversized_context_rejected(self):
        with self.assertRaises(ValueError):StreamedTeacher(self.store(),self.classes).forward(torch.ones(1,129,dtype=torch.int64))
    def test_14_padding_batch_not_silently_supported(self):
        with self.assertRaises(ValueError):StreamedTeacher(self.store(),self.classes).forward(torch.ones(2,4,dtype=torch.int64))
    def test_15_yarn_softmax_factor_preserved(self):
        t=StreamedTeacher(self.store(),self.classes);l=t.layer(1)
        c=self.cfg;r=c['rope_scaling'];m=self.classes['yarn_get_mscale'](r['factor'],r['mscale_all_dim'])
        self.assertAlmostEqual(l.self_attn.softmax_scale,(c['qk_nope_head_dim']+c['qk_rope_head_dim'])**-.5*m*m)
    def test_16_source_weights_not_rounded_when_widened(self):
        a=self.store().load('lm_head.weight',torch.float32)
        torch.testing.assert_close(a,self.state['lm_head.weight'].float(),atol=0,rtol=0)
    def test_17_parameter_set_shape_mismatch_rejected(self):
        t=StreamedTeacher(self.store(),self.classes)
        t.config.hidden_size=64
        with self.assertRaises((ValueError,RuntimeError)):t.layer(0)
    def test_18_shorter_sequence_exact(self):
        ids=self.ids[:,:1]
        actual=StreamedTeacher(self.store(),self.classes,dtype=torch.bfloat16).forward(ids)
        torch.testing.assert_close(actual,self.resident(torch.bfloat16,ids),rtol=0,atol=0)

if __name__=='__main__':unittest.main(verbosity=2)
