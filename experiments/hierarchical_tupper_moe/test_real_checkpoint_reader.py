import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from real_checkpoint_reader import RealExpertTileReader,ReaderError
from hierarchical_moe_probe import probe


def make_fixture(root:Path,expert_count=8):
    rng=np.random.default_rng(12)
    payload=bytearray()
    header={}
    index={}
    expect={}
    for e in range(expert_count):
        for r,role in enumerate(['gate_proj','up_proj','down_proj']):
            shape=(128,64) if role!='down_proj' else (64,128)
            a=rng.normal(0,.02,shape).astype('<f4')
            dtype=['BF16','F16','F32'][(e+r)%3]
            if dtype=='BF16':
                b=(a.view('<u4')>>16).astype('<u2').tobytes()
                expected=((np.frombuffer(b,dtype='<u2').astype('<u4')<<16).view('<f4')).reshape(shape)
            elif dtype=='F16':
                b=a.astype('<f2').tobytes()
                expected=np.frombuffer(b,dtype='<f2').astype('<f4').reshape(shape)
            else:
                b=a.tobytes()
                expected=a
            key=f'model.layers.1.mlp.experts.{e}.{role}.weight'
            header[key]={'dtype':dtype,'shape':list(shape),'data_offsets':[len(payload),len(payload)+len(b)]}
            index[key]='model-00001-of-00001.safetensors'
            expect[(e,role)]=expected
            payload.extend(b)
    raw=json.dumps(header,separators=(',',':')).encode()
    (root/'model-00001-of-00001.safetensors').write_bytes(len(raw).to_bytes(8,'little')+raw+payload)
    p=root/'model.safetensors.index.json'
    p.write_text(json.dumps({'weight_map':index}))
    return p,expect

class ReaderTests(unittest.TestCase):
    def test_reader_f32_f16_bf16_and_down_transpose(self):
        with tempfile.TemporaryDirectory() as d:
            p,expect=make_fixture(Path(d))
            source=RealExpertTileReader(p,layer=1,expert_ids=list(range(8)),hidden=64,intermediate=128)
            actual=source.tile(0)
            self.assertEqual(actual.shape,(8,128,192))
            for e in range(8):
                cols=np.concatenate([expect[(e,'gate_proj')],expect[(e,'up_proj')],expect[(e,'down_proj')].T],axis=1)
                self.assertTrue(np.array_equal(actual[e],cols))
            self.assertEqual(source.provenance()['type'],'REAL_CHECKPOINT_PARTIAL_IDENTITY')
            self.assertEqual(source.provenance()['index_sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
            with self.assertRaises(ReaderError):source.tile(1)
    def test_reader_rejects_invalid_index(self):
        with tempfile.TemporaryDirectory() as d:
            p,_=make_fixture(Path(d))
            bad=json.loads(p.read_text())
            bad['weight_map']['model.layers.1.mlp.experts.0.gate_proj.weight']='../outside.safetensors'
            p.write_text(json.dumps(bad))
            with self.assertRaises(FileNotFoundError):
                RealExpertTileReader(p,layer=1,expert_ids=list(range(8)),hidden=64,intermediate=128)
    def test_end_to_end_real_mode_on_fake_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            p,_=make_fixture(Path(d))
            reader=RealExpertTileReader(p,layer=1,expert_ids=list(range(8)),hidden=64,intermediate=128)
            x=probe(mode='real_checkpoint',seed=11,experts=8,hidden=64,intermediate=128,
                   groups=[16,32,64,128],lane=16,train_experts=4,rank=2,
                   calibration_fraction=.2,tiles=None,top_k=3,
                   tile_source=reader.tile,source_provenance=reader.provenance())
            self.assertEqual(x['source']['type'],'REAL_CHECKPOINT_PARTIAL_IDENTITY')
            self.assertTrue(x['all_rows_processed'])
            self.assertTrue(all(v['routing']['1.0']['tupper_roundtrip'] for v in x['widths'].values()))

if __name__=='__main__':unittest.main()
