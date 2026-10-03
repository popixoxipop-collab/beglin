#!/usr/bin/env python3
import json,struct,tempfile,unittest
from pathlib import Path
import model_capability as mc


def write_safetensors(path:Path,tensors:dict):
    header={name:{'dtype':dtype,'shape':shape,'data_offsets':[0,0]} for name,(dtype,shape) in tensors.items()}
    raw=json.dumps(header,separators=(',',':')).encode()
    path.write_bytes(struct.pack('<Q',len(raw))+raw)


def qwen_tensors():
    return {
      'model.embed_tokens.weight':('F16',[64,16]),
      'model.layers.0.self_attn.q_proj.weight':('F16',[16,16]),
      'model.layers.0.self_attn.k_proj.weight':('F16',[8,16]),
      'model.layers.0.self_attn.v_proj.weight':('F16',[8,16]),
      'model.layers.0.self_attn.o_proj.weight':('F16',[16,16]),
      'model.layers.0.input_layernorm.weight':('F32',[16]),
      'model.layers.0.post_attention_layernorm.weight':('F32',[16]),
      'model.layers.0.mlp.gate_proj.weight':('F16',[32,16]),
      'model.layers.0.mlp.up_proj.weight':('F16',[32,16]),
      'model.layers.0.mlp.down_proj.weight':('F16',[16,32]),
      'model.norm.weight':('F32',[16]),
      'lm_head.weight':('F16',[64,16]),
    }


def make_model(root:Path,model_type='qwen2',tensors=None,tokenizer=True,extra=None):
    root.mkdir(parents=True,exist_ok=True)
    cfg={'_name_or_path':'synthetic/test','model_type':model_type,'hidden_size':16,'intermediate_size':32,
         'num_hidden_layers':1,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':4,
         'vocab_size':64,'max_position_embeddings':128}
    if extra: cfg.update(extra)
    (root/'config.json').write_text(json.dumps(cfg,sort_keys=True))
    if tokenizer: (root/'tokenizer.json').write_text('{}')
    write_safetensors(root/'model.safetensors',tensors or qwen_tensors())
    return root


class ModelCapabilityTests(unittest.TestCase):
    def test_identity_is_location_independent_and_repeatable(self):
        with tempfile.TemporaryDirectory() as td:
            a=make_model(Path(td)/'a'); b=make_model(Path(td)/'different-location')
            ba=mc.compile_model_capabilities(a); bb=mc.compile_model_capabilities(b)
            self.assertEqual(ba['checkpoint_identity_sha256'],bb['checkpoint_identity_sha256'])
            self.assertEqual(ba['source_manifest']['source_manifest_sha256'],bb['source_manifest']['source_manifest_sha256'])
            self.assertEqual(ba['bundle_sha256'],bb['bundle_sha256'])
            self.assertEqual(ba['bundle_sha256'],mc.compile_model_capabilities(a)['bundle_sha256'])

    def test_qwen2_builds_canonical_graph(self):
        with tempfile.TemporaryDirectory() as td:
            bundle=mc.compile_model_capabilities(make_model(Path(td)/'m'))
            self.assertEqual(bundle['architecture_descriptor']['architecture_id'],'qwen2')
            self.assertEqual(bundle['architecture_descriptor']['attention_kind'],'GQA')
            self.assertEqual(bundle['model_skeleton']['layer_count'],1)
            roles={n['role'] for n in bundle['tensor_role_graph']['nodes']}
            self.assertTrue({'EMBEDDING','Q_PROJ','K_PROJ','V_PROJ','O_PROJ','DENSE_GATE','DENSE_UP','DENSE_DOWN','LM_HEAD'} <= roles)
            self.assertEqual(bundle['tensor_role_graph']['unmapped_tensor_count'],0)
            self.assertEqual(bundle['p8_p11_eligibility']['status'],'PARTIAL')
            self.assertFalse(bundle['p8_p11_eligibility']['p11_allowed'])

    def test_architecture_registry_six_families(self):
        cases={
          'qwen2':('DENSE','GQA'), 'llama':('DENSE','GQA'), 'deepseek_v2':('MOE','MLA'),
          'qwen3_moe':('MOE','GQA'), 'olmoe':('MOE','GQA'), 'gpt_oss':('MOE','HYBRID')}
        for raw,(kind,attn) in cases.items():
            with self.subTest(raw=raw):
                source={'config':{'model_type':raw,'hidden_size':16,'num_hidden_layers':2,'num_attention_heads':4,'num_key_value_heads':2},'metadata':{}}
                d=mc.build_architecture_descriptor(source)
                self.assertEqual(d['status'],'VERIFIED_ADAPTER')
                self.assertEqual(d['dense_or_moe'],kind)
                self.assertEqual(d['attention_kind'],attn)

    def test_unknown_architecture_is_denied_without_guessing(self):
        with tempfile.TemporaryDirectory() as td:
            bundle=mc.compile_model_capabilities(make_model(Path(td)/'m',model_type='brand_new_arch'))
            self.assertEqual(bundle['architecture_descriptor']['status'],'UNKNOWN_ARCHITECTURE')
            self.assertEqual(bundle['p8_p11_eligibility']['status'],'DENIED')
            self.assertFalse(bundle['p8_p11_eligibility']['p8_allowed'])

    def test_missing_shard_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'m'; root.mkdir(); (root/'config.json').write_text(json.dumps({'model_type':'qwen2'}))
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'missing-00001.safetensors'}}))
            with self.assertRaisesRegex(mc.ModelCapabilityError,'missing safetensors shard'):
                mc.inspect_model_source(root)

    def test_cpu_gpu_transition_semantics_share_schema(self):
        with tempfile.TemporaryDirectory() as td:
            tensors={
              'model.embed_tokens.weight':('F16',[64,16]),
              'model.layers.0.self_attn.q_proj.weight':('F16',[16,16]),
              'model.layers.0.self_attn.kv_a_proj_with_mqa.weight':('F16',[16,16]),
              'model.layers.0.self_attn.kv_b_proj.weight':('F16',[16,16]),
              'model.layers.0.self_attn.o_proj.weight':('F16',[16,16]),
              'model.layers.0.mlp.shared_experts.gate_proj.weight':('F16',[32,16]),
              'model.layers.0.mlp.shared_experts.up_proj.weight':('F16',[32,16]),
              'model.layers.0.mlp.shared_experts.down_proj.weight':('F16',[16,32]),
              'model.norm.weight':('F32',[16]),'lm_head.weight':('F16',[64,16]),
            }
            root=make_model(Path(td)/'m','deepseek_v2',tensors=tensors,tokenizer=False,
                            extra={'n_routed_experts':4,'num_experts_per_tok':2,'n_shared_experts':1,'first_k_dense_replace':0})
            source=mc.inspect_model_source(root)
            desc=mc.build_architecture_descriptor(source)
            evidence={
                "schema":"beglin-verification-evidence-v1",
                "status":"VERIFIED",
                "component":"backend_runtime",
                "architecture_id":desc["architecture_id"],
                "checkpoint_identity_sha256":source["checkpoint_identity_sha256"],
                "backend":"mlx_metal",
                "evidence_sha256":"c"*64,
                "run_id":"legacy-unit-fixture",
                "kind":"UNIT_TEST_FIXTURE",
            }
            inspected=mc.compile_model_capabilities(root)
            target=next(
                n["canonical_target_key"] for n in inspected["tensor_role_graph"]["nodes"]
                if n["role"]=="Q_PROJ" and n["layer"]==0
            )
            qng={
                "schema":"beglin-verification-evidence-v1","status":"VERIFIED",
                "component":"qng64_runtime","architecture_id":desc["architecture_id"],
                "checkpoint_identity_sha256":source["checkpoint_identity_sha256"],
                "backend":"mlx_metal","evidence_sha256":"d"*64,"run_id":"qng-unit",
                "kind":"UNIT_TEST_FIXTURE","target_key":target,"supported_n":[5,6],
            }
            mutation={
                "schema":"beglin-verification-evidence-v1","status":"VERIFIED",
                "component":"mutation_runtime","architecture_id":desc["architecture_id"],
                "checkpoint_identity_sha256":source["checkpoint_identity_sha256"],
                "backend":"mlx_metal","evidence_sha256":"e"*64,"run_id":"mutation-unit",
                "kind":"UNIT_TEST_FIXTURE","target_key":target,"supported_n":[5,6],
                "mutation_mode":"HOT_REBIND_SINGLE",
            }
            bundle=mc.compile_model_capabilities(
                root,
                mlx_runtime_verified=True,
                mlx_runtime_evidence=evidence,
                mlx_qng64_evidence=[qng],
                mlx_mutation_evidence=[mutation],
            )
            rows=bundle['runtime_mutation_matrix']['rows']
            q=[r for r in rows if r['target_key'].endswith('/q_proj')]
            by={r['backend']:r for r in q}
            self.assertEqual(by['cpu']['mutation_mode'],'RESTART_REQUIRED')
            self.assertEqual(by['mlx_metal']['mutation_mode'],'HOT_REBIND_SINGLE')
            self.assertEqual(bundle['p8_p11_eligibility']['status'],'PARTIAL')
            self.assertIn('TOKENIZER_NOT_FULLY_VERIFIED',bundle['p8_p11_eligibility']['reasons'])

    def test_schema_files_are_valid_json_and_indexed(self):
        root=Path(__file__).resolve().parents[1]/'schemas'/'model'
        schemas=sorted(root.glob('*.schema.json'))
        self.assertEqual(len(schemas),14)
        for path in schemas:
            obj=json.loads(path.read_text())
            self.assertEqual(obj['$schema'],'https://json-schema.org/draft/2020-12/schema')
            self.assertEqual(obj['type'],'object')
            self.assertIn('schema',obj['required'])


if __name__=='__main__': unittest.main()
