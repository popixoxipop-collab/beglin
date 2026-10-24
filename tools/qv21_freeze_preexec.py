#!/usr/bin/env python3
"""Qv21 create-only 96/48/48 freeze; never opens Qv20 validation or scores."""
import base64,hashlib,json,os,pathlib,struct,subprocess,sys
from datasets import Dataset
from tokenizers import Tokenizer
ROOT=pathlib.Path('/Volumes/D50/TailnetCommander/work/beglin-qv21-tailrisk-20261025')
OLD=pathlib.Path('/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024')
CACHE=pathlib.Path('/Users/eoe/.cache/huggingface/datasets')
FILES={
 'arc_train':CACHE/'allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-train.arrow',
 'ob_train':CACHE/'allenai___openbookqa/main/0.0.0/388097ea7776314e93a529163e0fea805b8a6454/openbookqa-train.arrow'}
TOKENIZER=pathlib.Path('/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/tokenizer.json')
EXPECTED_SPEC='408c661e66259784480b18eefc15f75d3bc7d034124194baaf3a218637e2f2d0'
API='repos/popixoxipop-collab/beglin/contents/qv21_spec.json?ref=research/qv21-tailrisk-prefix-selector-preregister-20261025'
def h(x):return hashlib.sha256(x).hexdigest()
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(262144),b''):d.update(part)
 return d.hexdigest()
def need(ok,s):
 if not ok:raise RuntimeError('QV21_FREEZE_FAIL_CLOSED_'+s)
def enc(x):return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
def new(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 need(not p.is_symlink(),'SYMLINK_OUTPUT')
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def fetch_spec():
 cp=subprocess.run(['/opt/homebrew/bin/gh','api',API,'--jq','.content'],capture_output=True,text=True,timeout=22)
 need(cp.returncode==0,'PREREG_GITHUB_UNAVAILABLE')
 b=base64.b64decode(''.join(cp.stdout.split()))
 need(h(b)==EXPECTED_SPEC,'PREREG_SPEC_DRIFT')
 return b,json.loads(b)
def row(corpus,raw,token,seed):
 s=(raw['question_stem'] if corpus=='ob_train' else raw['question']).strip()
 labels=raw['choices']['label'];answers=raw['choices']['text']
 if not s or len(labels)!=4 or len(answers)!=4 or len(set(labels))!=4 or raw['answerKey'] not in labels or any(not x.strip() for x in answers):return None
 question='Choose the correct answer.\nQuestion: '+s+'\n'+'\n'.join(f'{k}. {v.strip()}' for k,v in zip('ABCD',answers))+'\nAnswer using one letter A, B, C, or D.'
 prompt='<|im_start|>user\n'+question+'<|im_end|>\n<|im_start|>assistant\n'
 ids=token.encode(prompt,add_special_tokens=False).ids
 if not 18<=len(ids)<=208:return None
 suff={}
 for k in 'ABCD':
  seq=token.encode(prompt+k,add_special_tokens=False).ids
  if len(seq)!=len(ids)+1 or seq[:-1]!=ids:return None
  suff[k]=seq[-1]
 if len(set(suff.values()))!=4:return None
 qsha=h(s.encode());identifier=raw['id']
 return {'source_id':identifier,'question_sha256':qsha,'full_prompt_sha256':h(question.encode()),
         'selection_key':h((seed+'/'+corpus+'/'+identifier+'/'+qsha).encode()),
         'prefix_ids':ids,'suffix_ids':suff,'correct':'ABCD'[labels.index(raw['answerKey'])],
         'prefix_len':len(ids)}
def alternating(s,l):
 need(len(s)==len(l),'STRATUM_MISMATCH')
 return [v for pair in zip(s,l) for v in pair]
def run(dry):
 need(not ROOT.exists() and not ROOT.is_symlink(),'OUTPUT_PREEXISTS')
 bs,spec=fetch_spec()
 old=OLD/'cal_arc_train/frozen_inputs_private.json'
 need(sha(old)==spec['exclude_qv20_calibration_manifest_sha256'],'QV20_PRIOR_CAL_PIN')
 q=json.loads(old.read_text())['selected_rows']
 need(len(q)==32,'QV20_CAL_COUNT')
 ids={x['source_id'] for x in q}; qs={x['question_sha256'] for x in q}
 need(len(ids)==len(qs)==32,'QV20_CAL_UNIQUE')
 for k,p in FILES.items():need(sha(p)==spec['dataset_sha256'][k],'ARROW_'+k)
 need(sha(TOKENIZER)==spec['tokenizer_sha256'],'TOKENIZER_PIN')
 for arm,s in spec['static_masks'].items():need(sha(OLD/'masks'/(arm+'.txt'))==s,'MASK_'+arm)
 token=Tokenizer.from_file(str(TOKENIZER))
 buckets={};eligible={}
 for corpus,p in FILES.items():
  rows=[]
  for raw in Dataset.from_file(str(p)):
   x=row(corpus,raw,token,spec['selection_seed'])
   if x and x['question_sha256'] not in qs and not(corpus=='arc_train' and x['source_id'] in ids):rows.append(x)
  short=sorted((x for x in rows if x['prefix_len']<85),key=lambda x:x['selection_key'])
  long=sorted((x for x in rows if x['prefix_len']>=85),key=lambda x:x['selection_key'])
  buckets[corpus]={'short':short,'long':long};eligible[corpus]={'short':len(short),'long':len(long)}
 a=buckets['arc_train'];o=buckets['ob_train']
 need(min(len(a['short']),len(a['long']))>=72 and min(len(o['short']),len(o['long']))>=24,'INSUFFICIENT_STRATA')
 splits={'cal_arc_train96':alternating(a['short'][:48],a['long'][:48]),
         'eval_arc_train48':alternating(a['short'][48:72],a['long'][48:72]),
         'eval_ob_train48':alternating(o['short'][:24],o['long'][:24])}
 selected=[r for rows in splits.values() for r in rows]
 need(len(selected)==len({x['question_sha256'] for x in selected})==192,'NO_DUPLICATE_QUESTION')
 need(len({('ob' if key.startswith('eval_ob') else 'arc',v['source_id']) for key,rows in splits.items() for v in rows})==192,'NO_DUPLICATE_IDS')
 public={'schema':'BEGLIN-Qv21-immutable-preexec-freeze-public/1',
  'state':'FROZEN_BEFORE_ANY_QV21_NATIVE_SCORING',
  'study_spec_SHA256':h(bs),'freeze_script_SHA256':sha(pathlib.Path(__file__)),
  'source_arrow_SHA256':spec['dataset_sha256'],'tokenizer_SHA256':spec['tokenizer_sha256'],
  'selection_seed':spec['selection_seed'],'source_mask_SHA256':spec['static_masks']['static_source'],
  'random_mask_SHA256':spec['static_masks']['static_random'],
  'original_native_executable_SHA256':spec['original_native_c_sha256'],
  'qv20_calibration_exclusion_SHA256':sha(old),
  'qv20_validation_data_read':False,'no_Qv21_native_scoring_during_freeze':True,
  'sample_counts':{k:len(x) for k,x in splits.items()},'eligible_by_stratum':eligible,
  'hidden_case_ids_and_labels_disclosed':False,'projection_packed_bytes_each':765493248}
 if dry:
  print('QV21_DRY_RUN_NO_WRITES',json.dumps({'sample_counts':public['sample_counts'],'eligible_by_stratum':eligible,
    'Qv20_heldout_read':False,'study_spec_sha256':EXPECTED_SPEC},sort_keys=True));return
 os.umask(0o077)
 ROOT.mkdir(parents=True,exist_ok=False,mode=0o700)
 new(ROOT/'qv21_spec.json',bs)
 privsha={}
 for split,rows in splits.items():
  manifest={'schema':'BEGLIN-Qv21-frozen-actual-original-C-inputs-private/1','split':split,
    'selected_rows':rows,'study_spec_SHA256':EXPECTED_SPEC,
    'source_arrow_SHA256':spec['dataset_sha256']['ob_train' if split.startswith('eval_ob') else 'arc_train'],
    'no_model_scoring_performed_at_freeze':True}
  for arm in ('static_source','static_random','dynamic'):
   base=ROOT/split/arm
   for i,r in enumerate(rows):
    values=struct.pack('<'+'i'*len(r['prefix_ids']),*r['prefix_ids'])
    gold=(' '.join(str(r['suffix_ids'][k]) for k in 'ABCD')+' '+str('ABCD'.index(r['correct']))+'\n').encode()
    new(base/'eval/ppl_work'/f'w{i}.i32',values)
    new(base/'eval/ppl_work'/f'w{i}.choices',gold)
    if i==0:new(base/'prompt_ids.i32',values)
  b=enc(manifest);privsha[split]=h(b);new(ROOT/split/'frozen_inputs_private.json',b)
 for arm in ('static_source','static_random'):new(ROOT/'masks'/(arm+'.txt'),(OLD/'masks'/(arm+'.txt')).read_bytes())
 public['frozen_private_input_SHA256']=privsha
 b=enc(public);new(ROOT/'Qv21_PREEXEC_PUBLIC.json',b)
 print('QV21_PREEXEC_FROZEN',json.dumps({'public_sha256':h(b),'private_sha256':privsha,
  'n':public['sample_counts'],'qv20_heldout_read':False},sort_keys=True),flush=True)
if __name__=='__main__':
 need(len(sys.argv)==2 and sys.argv[1] in ('--dry-run','--freeze'),'FIXED_MODE')
 run(sys.argv[1]=='--dry-run')
