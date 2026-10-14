#!/usr/bin/env python3
"""Read-only independent P15 audit of full-vocabulary output and provenance.

Numerical recomputation here verifies output arrays/statistics, not a second
resident original 16B forward. Native quantized quality remains unmeasured.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib
import numpy as np

INDEX_SHA='d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf'
CONFIG_SHA='f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d'
SOURCE_SHA='7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c'
FROZEN_SHA='c85d16c113780d73692fd0bdb9b1afaacb40bda9a998876420f361899d3cbb8b'
CORE_SHA='46510526dbaf21884bca30a85ad3d1131099ccf8a1d41063064c1b1fd6683f56'
RUNNER_SHA='1dbaa7e59d4e42e059c4c88dc3c2e735d1e5b740b971aca408cb52f218e961e2'
TEST_SHA='ac29ee917315ef21e76d3af2a8a34d973b2f0c5b0805b59d500fcadd6e9f4889'
STAGE='8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9'
ROUTE='c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47'

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def close(a,b):return math.isclose(float(a),float(b),abs_tol=1e-8,rel_tol=1e-9)
def audit(root,label):
    root=pathlib.Path(root)
    if not root.is_dir() or root.is_symlink():raise ValueError('Evidence root invalid')
    report_path=root/(label+'.json');r=json.loads(report_path.read_text())
    if r.get('schema')!='beglin-p15-real-complete-layer-streamed-teacher-v1' or r.get('status')!='FULL_ORIGINAL_MODEL_FORWARD_AND_FULL_VOCAB_LOGITS_COMPLETE':
        raise ValueError('No complete real full-teacher result')
    if (r.get('full_model_forward') is not True or r.get('all_original_decoder_layers')!=27 or
        r.get('all_original_checkpoint_tensors_read')!=5291 or r.get('full_vocabulary_size')!=102400 or
        r.get('logits_shape')!=[1,102400] or r.get('logits_finite') is not True or
        r.get('checkpoint_original_storage_dtype')!='BF16' or r.get('execution_arithmetic') not in ('FP32','BF16')):
        raise ValueError('Partial/mislabeled inference may not claim full BF16 teacher')
    if (r.get('original_model_index_sha256')!=INDEX_SHA or r.get('original_config_sha256')!=CONFIG_SHA or
        r.get('original_modeling_source_sha256')!=SOURCE_SHA or
        r.get('frozen_input_file_sha256')!=FROZEN_SHA or
        r.get('implementation_sha256')!={'hcp_p15_streamed_teacher.py':CORE_SHA,
          'hcp_p15_run_teacher.py':RUNNER_SHA,'test_hcp_p15_streamed_teacher.py':TEST_SHA}):
        raise ValueError('Teacher source implementation/provenance changed')
    for forbidden in ('production_touched','automatic_live_promotion','GPU_used','quantized_native_A_B_compared','quality_noninferiority_certified'):
        if r.get(forbidden) is not False:raise ValueError('Unsupported teacher promotion/quality claim')
    if not (0<r['process_rss_peak_bytes']<=6*1024**3 and r['cpu_threads']==2):
        raise ValueError('Execution exceeded registered resource scope')
    layer_ids=[z['layer'] for z in r['per_layer']]
    if layer_ids!=list(range(27)):raise ValueError('Not every original decoder layer was executed in order')
    expected_params=1169695744*(2 if r['execution_arithmetic']=='FP32' else 1)
    if r['resident_parameter_bytes_peak']!=expected_params:raise ValueError('Unexpected resident original parameter set')
    for z in r['per_layer']:
        if len(z['hidden_sha256'])!=64 or z['parameter_bytes']<=0 or not math.isfinite(z['seconds']):
            raise ValueError('Incomplete layer witness')
    if sha(root/'FROZEN_INPUTS.json')!=FROZEN_SHA or sha(root/'SOURCE_MODEL_INDEX.json')!=INDEX_SHA:
        raise ValueError('Original frozen input/index not supplied')
    frozen=json.loads((root/'FROZEN_INPUTS.json').read_text())
    entry=next(e for e in frozen['entries'] if e['id']==r['prompt_id'])
    ids=entry['token_ids'];tokensha=hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()
    if tokensha!=r['token_sha256'] or r['input_tokens']!=len(ids) or r['prompt_text']!=entry['text']:
        raise ValueError('Input token alignment changed')
    manifest=root/(label+'.source_hashes.json')
    if sha(manifest)!=r['source_tensor_hash_manifest_sha256']:raise ValueError('Weight read digests changed')
    weights=json.loads(manifest.read_text())
    expected=json.loads((root/'SOURCE_MODEL_INDEX.json').read_text())['weight_map']
    if set(weights)!=set(expected) or len(weights)!=5291 or any(len(v)!=64 for v in weights.values()):
        raise ValueError('Not every original checkpoint tensor was read and hashed')
    lp=root/r['logits_file']
    if lp.parent!=root or lp.is_symlink() or sha(lp)!=r['logits_file_sha256']:
        raise ValueError('Logits file missing/changed')
    logits=np.load(lp,allow_pickle=False)
    if logits.shape!=(1,102400) or logits.dtype!=np.float32 or not np.isfinite(logits).all():
        raise ValueError('Invalid full-vocabulary logits')
    if not close(logits.min(),r['logits_min']) or not close(logits.max(),r['logits_max']):
        raise ValueError('Recorded logit statistics disagree with array')
    d=logits.astype(np.float64);z=d-d.max();prob=np.exp(z);prob/=prob.sum()
    if not close(np.linalg.norm(d),r['logits_l2']) or not close(prob.sum(),r['probability_sum']):
        raise ValueError('Independent full-vocabulary computation differs')
    if logits[0,r['next_token_id']]!=logits.max():raise ValueError('Reported next token is not argmax')
    live={'stage_sha256':STAGE,'route_sha256':ROUTE,'pids':[35481,35493]}
    if r['production_before']!=live or r['production_after']!=live:
        raise ValueError('Original native production state did not remain intact')
    tests=json.loads((root/'SELFTEST.json').read_text())
    if not tests['successful'] or tests['tests']!=18 or tests['errors'] or tests['failures'] or sha(root/'SELFTEST.log')!=tests['test_log_sha256']:
        raise ValueError('Resident versus streaming numerical gate not passed')
    return {'schema':'beglin-p15-independent-full-logits-and-provenance-audit-v1',
       'status':'COMPLETE_FULL_VOCAB_ARRAY_AND_27_LAYER_5291_TENSOR_PROVENANCE_VERIFIED',
       'teacher_report_sha256':sha(report_path),'logits_sha256':sha(lp),
       'arithmetic':r['execution_arithmetic'],'prompt_id':r['prompt_id'],
       'layers':27,'source_tensor_count':5291,'vocabulary_logits':102400,
       'source_BF16_weights':True,'finite_logit_values':int(np.isfinite(logits).sum()),
       'real_next_token_argmax_verified':True,'resource_RSS_peak_bytes':r['process_rss_peak_bytes'],
       'resident_toy_numerical_equivalence_tests':18,
       'independent_full_model_resident_forward_repeated':False,
       'native_quantized_AB_quality_compared':False,'promotion_allowed':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--label',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=audit(a.root,a.label)
    with pathlib.Path(a.output).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
    print('P15_INDEPENDENT_FULL_LOGITS_AUDIT_PASS',json.dumps(v,sort_keys=True))
