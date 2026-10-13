import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from test_real_checkpoint_reader import make_fixture
from prompt_holdout_micro_eval import evaluate, load_runs, _sha256


def fake_suite(root:Path,index:Path,n_prompts:int=18):
    out=root/'suite';out.mkdir()
    records=[]
    for ip in range(n_prompts):
        prompt_id=f'p{ip:02d}'
        split='train' if ip<9 else ('validation' if ip<13 else 'holdout')
        p=out/prompt_id;p.mkdir()
        hpath=p/'true_h2.bin';rpath=p/'moe3a_c_routing.txt'
        with hpath.open('wb') as f:
            for pos in range(8):
                h=np.random.default_rng(ip*100+pos).normal(size=64).astype('<f4')
                f.write(struct.pack('<iii',pos,1,64))
                f.write(h.tobytes())
        with rpath.open('w') as f:
            for pos in range(8):
                e0=(ip+pos)%8;e1=(ip+pos+3)%8;e2=(ip+pos+5)%8
                f.write(f'pos {pos} layer 1 experts {e0}:0.5 {e1}:0.3 {e2}:0.2\n')
        records.append({'prompt_id':prompt_id,'split':split,'position_count':8,
                        'input_ids_sha256':hashlib.sha256(f'{ip}-unique'.encode()).hexdigest(),
                        'capture_sha256':_sha256(hpath),'routing_sha256':_sha256(rpath)})
    meta={'schema':'beglin-prompt-suite-v1','checkpoint_index_sha256':_sha256(index),
          'layer':1,'macro_top_k':3,'runs':records}
    (out/'runs.json').write_text(json.dumps(meta))
    return out

class PromptHeldoutTests(unittest.TestCase):
    def test_fake_independent_prompt_train_val_test(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);index,_=make_fixture(root,expert_count=8)
            suite=fake_suite(root,index)
            o=evaluate(index,suite,hidden=64,intermediate=128,widths=(16,64),fractions=(.5,1.),bootstrap_draws=30)
            self.assertEqual(o['prompt_split_counts'],{'train':9,'validation':4,'holdout':5})
            self.assertEqual(o['position_split_counts'],{'train':72,'validation':32,'holdout':40})
            self.assertEqual(o['heldout_method_selection'],'validation_only')
            for width in ('16','64'):
                r=o['results'][width]['1.0']
                self.assertTrue(r['tupper_roundtrip'])
                self.assertLess(r['locked_holdout']['relative_l2'],1e-10)
                self.assertEqual(r['locked_holdout']['prompt_count'],5)
            self.assertFalse(o['production_touched'])
            self.assertFalse(o['automatic_live_promotion'])
    def test_tampered_capture_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);index,_=make_fixture(root,expert_count=8)
            suite=fake_suite(root,index)
            with (suite/'p10'/'true_h2.bin').open('ab') as f:f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_runs(suite,layer=1,hidden=64,expected_index_hash=_sha256(index))
    def test_incomplete_prompt_split_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);index,_=make_fixture(root,expert_count=8)
            suite=fake_suite(root,index)
            m=json.loads((suite/'runs.json').read_text())
            for r in m['runs']:
                if r['split']=='holdout':r['split']='train'
            (suite/'runs.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'at least three independent prompts'):
                load_runs(suite,layer=1,hidden=64,expected_index_hash=_sha256(index))
    def test_duplicate_prompt_identity_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);index,_=make_fixture(root,expert_count=8)
            suite=fake_suite(root,index)
            m=json.loads((suite/'runs.json').read_text())
            m['runs'][1]['input_ids_sha256']=m['runs'][0]['input_ids_sha256']
            (suite/'runs.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'duplicate tokenized prompt'):
                load_runs(suite,layer=1,hidden=64,expected_index_hash=_sha256(index))

if __name__=='__main__':unittest.main()