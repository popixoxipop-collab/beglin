#!/usr/bin/env python3
"""P15 independent full-vocabulary artifact and false-claim rejection tests."""
import hashlib,json,os,pathlib,shutil,tempfile,unittest
import numpy as np
from hcp_p15_independent_audit import audit
REPO=pathlib.Path(__file__).resolve().parents[1]
EVID=pathlib.Path(os.environ.get('BEGLIN_P15_EVIDENCE',str(REPO/'experiments/hcp_p15/evidence/native_xox_20261015')))

class P15EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)
        names=('prompt0_bf16.json','prompt0_bf16.logits.npy','prompt0_bf16.source_hashes.json',
               'SELFTEST.json','SELFTEST.log','FROZEN_INPUTS.json','SOURCE_MODEL_INDEX.json')
        for n in names:shutil.copyfile(EVID/n,self.root/n)
    def tearDown(self):self.tmp.cleanup()
    def check(self):return audit(self.root,'prompt0_bf16')
    def edit(self,change):
        p=self.root/'prompt0_bf16.json';r=json.loads(p.read_text());change(r);p.write_text(json.dumps(r))
    def reject(self):
        with self.assertRaises((ValueError,KeyError,TypeError,AssertionError)):self.check()
    def test_01_actual_full_logits_array_passes(self):
        self.assertEqual(self.check()['finite_logit_values'],102400)
    def test_02_partial_layers_cannot_claim_full(self):
        self.edit(lambda r:r.update(all_original_decoder_layers=26));self.reject()
    def test_03_full_layer_witness_order_required(self):
        self.edit(lambda r:r['per_layer'].reverse());self.reject()
    def test_04_all_tensors_required(self):
        self.edit(lambda r:r.update(all_original_checkpoint_tensors_read=100));self.reject()
    def test_05_nonfinite_even_with_updated_array_hash_rejected(self):
        p=self.root/'prompt0_bf16.logits.npy';v=np.load(p,allow_pickle=False);v[0,4]=np.nan
        with p.open('wb') as f:np.save(f,v,allow_pickle=False)
        self.edit(lambda r:r.update(logits_file_sha256=hashlib.sha256(p.read_bytes()).hexdigest()));self.reject()
    def test_06_truncated_vocab_even_with_updated_hash_rejected(self):
        p=self.root/'prompt0_bf16.logits.npy';v=np.load(p,allow_pickle=False)[:,:100]
        with p.open('wb') as f:np.save(f,v,allow_pickle=False)
        self.edit(lambda r:r.update(logits_file_sha256=hashlib.sha256(p.read_bytes()).hexdigest()));self.reject()
    def test_07_wrong_dtype_label_rejected(self):
        self.edit(lambda r:r.update(execution_arithmetic='INT4'));self.reject()
    def test_08_original_source_must_match(self):
        self.edit(lambda r:r.update(original_modeling_source_sha256='0'*64));self.reject()
    def test_09_native_AB_accuracy_claim_rejected(self):
        self.edit(lambda r:r.update(quality_noninferiority_certified=True));self.reject()
    def test_10_fake_RSS_exceeds_bound_rejected(self):
        self.edit(lambda r:r.update(process_rss_peak_bytes=10*1024**3));self.reject()
    def test_11_missing_weight_even_with_updated_hash_rejected(self):
        p=self.root/'prompt0_bf16.source_hashes.json';r=json.loads(p.read_text());r.pop(next(iter(r)))
        p.write_text(json.dumps(r));self.edit(lambda r:r.update(source_tensor_hash_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest()));self.reject()
    def test_12_false_logit_statistics_rejected(self):
        self.edit(lambda r:r.update(logits_l2=r['logits_l2']+1));self.reject()

if __name__=='__main__':unittest.main(verbosity=2)
