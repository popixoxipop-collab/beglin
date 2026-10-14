#!/usr/bin/env python3
"""P15 bounded full BF16-original teacher execution and immutable receipts."""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.metadata as md,json,os,pathlib,resource,subprocess,sys,threading,time,unittest
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import psutil
import torch
from transformers import AutoTokenizer
from hcp_p15_streamed_teacher import (TensorStore,StreamedTeacher,original_classes,sha,tensor_sha,INDEX_SHA,CONFIG_SHA,SOURCE_SHA)

ROOT=pathlib.Path('/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p15-streamed-teacher-20261015')
MODEL=pathlib.Path('/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors')
STAGE=pathlib.Path('/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu')
ROUTE=pathlib.Path('/Users/xox/vdsp_serving/active-route.json')
STAGE_SHA='8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9'
ROUTE_SHA='c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47'
INPUTS=('The result of seven times eight is','Complete the sentence: The capital of France is')
LIMIT=6*1024**3

def save_new(p,obj):
    with pathlib.Path(p).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def original_serving():
    text=subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True,timeout=8)
    return {'stage_sha256':sha(STAGE),'route_sha256':sha(ROUTE),
            'pids':sorted(int(r.split()[0]) for r in text.splitlines() if str(STAGE) in r)}

def sources():
    here=pathlib.Path(__file__).parent
    return {n:sha(here/n) for n in ('hcp_p15_streamed_teacher.py','hcp_p15_run_teacher.py','test_hcp_p15_streamed_teacher.py')}

def validate_root():
    if ROOT.is_symlink() or ROOT.resolve(strict=True)!=ROOT or ROOT.parent.name!='.tailnet-tmp':
        raise ValueError('P15 only runs in the exact new isolated root')
    if sha(MODEL/'modeling_deepseek.py')!=SOURCE_SHA or sha(MODEL/'config.json')!=CONFIG_SHA:
        raise ValueError('Original checkpoint implementation/configuration changed')
    if sha(MODEL/'model.safetensors.index.json')!=INDEX_SHA:raise ValueError('Original checkpoint index differs')
    live=original_serving()
    if live!={'stage_sha256':STAGE_SHA,'route_sha256':ROUTE_SHA,'pids':[35481,35493]}:
        raise ValueError('Original production image/PIDs changed, stop before work')
    return live

class Guard:
    def __init__(self,label):
        self.label=label;self.done=threading.Event();self.peak=0
        self.swap0=psutil.swap_memory().used;self.proc=psutil.Process()
        self.swap_peak_delta=0;self.minimum_available=psutil.virtual_memory().available
    def check(self):
        rss=self.proc.memory_info().rss;self.peak=max(self.peak,rss)
        mem=psutil.virtual_memory().available;self.minimum_available=min(self.minimum_available,mem)
        sw=max(0,psutil.swap_memory().used-self.swap0);self.swap_peak_delta=max(self.swap_peak_delta,sw)
        disk=psutil.disk_usage(str(ROOT)).free
        if rss>LIMIT:return 'OWN_PROCESS_RSS_EXCEEDED_6_GiB'
        if mem<8*1024**3:return 'HOST_AVAILABLE_MEMORY_BELOW_8_GiB'
        if sw>1024**3:return 'NEW_HOST_SWAP_EXCEEDS_1_GiB'
        if disk<20_000_000_000:return 'HOST_DISK_BELOW_20_GB'
        return None
    def watch(self):
        while not self.done.wait(.25):
            reason=self.check()
            if reason:
                failure={'schema':'beglin-p15-watchdog-stop-v1','reason':reason,'own_pid':os.getpid(),
                         'rss_peak':self.peak,'swap_delta':self.swap_peak_delta,
                         'full_teacher_completed':False,'production_touched':False}
                try:save_new(ROOT/'evidence'/f'{self.label}_WATCHDOG_STOP.json',failure)
                finally:
                    print('P15_WATCHDOG_STOP',json.dumps(failure),flush=True)
                    os._exit(91)
    def __enter__(self):
        reason=self.check()
        if reason:raise RuntimeError(reason)
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.check();self.done.set();self.thread.join(timeout=2)

def selftest():
    import test_hcp_p15_streamed_teacher as tests
    capture=ROOT/'evidence/SELFTEST.log'
    if capture.exists():raise ValueError('Tests already recorded; do not overwrite')
    start=time.monotonic()
    with capture.open('x') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    receipt={'schema':'beglin-p15-original-streaming-resident-numerical-tests-v1',
             'successful':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
             'errors':len(result.errors),'test_log_sha256':sha(capture),'source_sha256':sources(),
             'original_computational_source_sha256':SOURCE_SHA,'seconds':time.monotonic()-start}
    save_new(ROOT/'evidence/SELFTEST.json',receipt)
    print('P15_NUMERICAL_SELFTEST',json.dumps(receipt,sort_keys=True),flush=True)
    if not result.wasSuccessful():raise RuntimeError('Resident/streamed numerical gate failed')

def freeze():
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False,use_fast=True)
    entries=[]
    for i,prompt in enumerate(INPUTS):
        ids=tok.encode(prompt,add_special_tokens=True)
        entries.append({'id':f'p15-engineering-{i}','text':prompt,'token_ids':ids,
                        'token_sha256':hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()})
    save_new(ROOT/'FROZEN_INPUTS.json',{'schema':'beglin-p15-preforward-inputs-v1','entries':entries,
             'task_accuracy_benchmark':False,'source_model_index_sha256':INDEX_SHA})
    print('P15_INPUTS_FROZEN',sha(ROOT/'FROZEN_INPUTS.json'),[(e['id'],len(e['token_ids'])) for e in entries],flush=True)

def run(prompt,dtype_name):
    test=json.loads((ROOT/'evidence/SELFTEST.json').read_text())
    if test['successful'] is not True or test['source_sha256']!=sources():
        raise ValueError('P15 implementation changed after numerical selftest; refuse full teacher')
    record=json.loads((ROOT/'FROZEN_INPUTS.json').read_text())['entries'][prompt]
    if record['text']!=INPUTS[prompt]:raise ValueError('Input changed after preregistration')
    label=f'prompt{prompt}_{dtype_name}'
    if (ROOT/'evidence'/f'{label}.json').exists() or (ROOT/'evidence'/f'{label}.logits.npy').exists():
        raise ValueError('Full-forward evidence already exists, do not rerun sealed experiment')
    classes,code_receipt=original_classes(MODEL/'modeling_deepseek.py')
    store=TensorStore(MODEL)
    dtype={'bf16':torch.bfloat16,'fp32':torch.float32}[dtype_name]
    inputs=torch.tensor([record['token_ids']],dtype=torch.int64)
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False,use_fast=True)
    started=time.monotonic()
    with Guard(label) as guard:
        def progress(r):
            print('P15_LAYER',json.dumps({'layer':r['layer'],'completed':r['layer']+1,'total':27,
               'seconds':round(r['seconds'],3),'rss_GiB':round(psutil.Process().memory_info().rss/1024**3,3),
               'dtype':dtype_name}),flush=True)
        teacher=StreamedTeacher(store,classes,dtype=dtype,progress=progress)
        logits=teacher.forward(inputs)
        probs=torch.softmax(logits.double(),dim=-1)
        topv,topi=torch.topk(logits,5,dim=-1)
        nextid=int(topi[0,0]);nexttext=tok.decode([nextid])
        logits_path=ROOT/'evidence'/f'{label}.logits.npy'
        with logits_path.open('xb') as f:np.save(f,logits.numpy(),allow_pickle=False)
        source_hashes=ROOT/'evidence'/f'{label}.source_hashes.json'
        save_new(source_hashes,store.value_hashes)
        report={'schema':'beglin-p15-real-complete-layer-streamed-teacher-v1',
          'status':'FULL_ORIGINAL_MODEL_FORWARD_AND_FULL_VOCAB_LOGITS_COMPLETE',
          'prompt_id':record['id'],'prompt_text':record['text'],'input_tokens':inputs.shape[1],
          'token_sha256':record['token_sha256'],'frozen_input_file_sha256':sha(ROOT/'FROZEN_INPUTS.json'),
          'checkpoint_original_storage_dtype':'BF16','execution_arithmetic':dtype_name.upper(),
          'all_original_decoder_layers':len(teacher.layer_records),'all_original_checkpoint_tensors_read':len(store.value_hashes),
          'full_model_forward':True,'full_vocabulary_size':102400,'logits_shape':list(logits.shape),
          'logits_finite':bool(torch.isfinite(logits).all()),'logits_dtype_export':'float32',
          'logits_file':logits_path.name,'logits_file_sha256':sha(logits_path),
          'logits_min':float(logits.min()),'logits_max':float(logits.max()),
          'logits_l2':float(torch.linalg.vector_norm(logits.double())),
          'probability_sum':float(probs.sum()),'next_token':nexttext,
          'next_token_id':nextid,'top5_values':topv[0].tolist(),'top5_token_ids':topi[0].tolist(),
          'resident_parameter_bytes_peak':teacher.max_resident_parameter_bytes,
          'process_rss_peak_bytes':max(guard.peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
          'host_new_swap_peak_bytes':guard.swap_peak_delta,'host_available_memory_min_bytes':guard.minimum_available,
          'elapsed_seconds':time.monotonic()-started,'cpu_threads':torch.get_num_threads(),
          'GPU_used':False,'source_parameters_modified':False,'original_model_index_sha256':INDEX_SHA,
          'original_config_sha256':CONFIG_SHA,'original_modeling_source_sha256':SOURCE_SHA,
          'original_function_source_sha256':code_receipt,'source_tensor_hash_manifest_sha256':sha(source_hashes),
          'source_checkpoint_metadata_unchanged':True,'implementation_sha256':sources(),
          'versions':{k:md.version(k) for k in ('torch','transformers','safetensors','accelerate')},
          'per_layer':teacher.layer_records,'production_before':LIVE_BEFORE,'production_after':original_serving(),
          'quantized_native_A_B_compared':False,'quality_noninferiority_certified':False,
          'original_resident_full_model_bitwise_comparison_performed':False,
          'native_KV_cache_enabled':False,'production_touched':False,'automatic_live_promotion':False}
        if report['production_before']!=report['production_after']:raise RuntimeError('Original serving changed')
        if report['all_original_decoder_layers']!=27 or report['all_original_checkpoint_tensors_read']!=5291:
            raise RuntimeError('Partial forward is not a full-model result')
        save_new(ROOT/'evidence'/f'{label}.json',report)
    print('P15_FULL_TEACHER_COMPLETE',json.dumps({'label':label,'layers':27,'tensors':5291,
           'vocabulary':102400,'next_token':nexttext,'arithmetic':dtype_name,
           'rss_GiB':report['process_rss_peak_bytes']/1024**3,
           'seconds':report['elapsed_seconds'],'receipt_sha256':sha(ROOT/'evidence'/f'{label}.json')},sort_keys=True),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('selftest','freeze','run'),required=True)
    parser.add_argument('--prompt',type=int,choices=(0,1),default=0)
    parser.add_argument('--dtype',choices=('bf16','fp32'),default='bf16')
    a=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    try:os.nice(10)
    except PermissionError:pass
    LIVE_BEFORE=validate_root()
    with (ROOT/'P15_RUN.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='selftest':selftest()
        elif a.phase=='freeze':freeze()
        else:run(a.prompt,a.dtype)
