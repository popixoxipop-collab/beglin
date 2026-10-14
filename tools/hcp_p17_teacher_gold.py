#!/usr/bin/env python3
"""P17 original DeepSeek-V2-Lite full BF16 teacher, gold-token logits window.

Runs EVERY original 27 decoder layer using original SHA-pinned P15 torch source.
Only one layer of weights resident, full checkpoint BF16 source digested,
teacher-forced window logits for each gold token produced in *one* causal
full-sequence forward (never cached proxy, never partial model).
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,resource,subprocess,sys,time,threading,gc,types
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import psutil
from hcp_p15_streamed_teacher import TensorStore,StreamedTeacher,original_classes,sha,tensor_sha,INDEX_SHA,CONFIG_SHA,SOURCE_SHA

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015")
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
FROZEN_SHA="a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521"
MODEL_WEIGHT_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
PRODUCTION_STAGE="/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu"
PRODUCTION_ROUTE="/Users/xox/vdsp_serving/active-route.json"
LIVE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
PIDS=(35481,35493)
MAX_RSS=6*1024**3
def tokenhash(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def save(p,doc):
    p=pathlib.Path(p)
    with p.open("x") as f:json.dump(doc,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def live():
    t=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=8)
    stage=str(PRODUCTION_STAGE)
    return {"stage_sha256":sha(PRODUCTION_STAGE),"route_sha256":sha(PRODUCTION_ROUTE),
      "pids":sorted(int(line.split()[0]) for line in t.splitlines() if stage in line)}
def guard_snapshot():
    p=psutil.Process()
    ram=psutil.virtual_memory()
    return {"rss":p.memory_info().rss,"available":ram.available,
            "swap":psutil.swap_memory().used,
            "disk":psutil.disk_usage(str(ROOT)).free}
class Watch:
    def __init__(self,index):
        self.index=index;self.stop=threading.Event();self.peak=0
        self.start_swap=psutil.swap_memory().used
        self.max_swap_delta=0
    def check(self):
        z=guard_snapshot();self.peak=max(self.peak,z["rss"])
        delta=max(0,z["swap"]-self.start_swap);self.max_swap_delta=max(self.max_swap_delta,delta)
        if z["rss"]>MAX_RSS:return "RESEARCH_WORKER_RSS_ABOVE_6GIB"
        if z["available"]<8*1024**3:return "HOST_AVAILABLE_RAM_BELOW_8GIB"
        if delta>1024**3:return "NEW_SWAP_OVER_1GIB"
        if z["disk"]<20_000_000_000:return "SSD_BELOW_20GB_PROTECTED"
        return None
    def watch(self):
        while not self.stop.wait(.25):
            msg=self.check()
            if msg:
                print("P17_MEMORY_WATCHDOG_FAIL_CLOSED",json.dumps({"index":self.index,"reason":msg,"pid":os.getpid(),"peak":self.peak}),flush=True)
                os._exit(92) # exact research process only
    def __enter__(self):
        msg=self.check()
        if msg:raise RuntimeError("P17 preflight "+msg)
        self.thread=threading.Thread(target=self.watch,daemon=True)
        self.thread.start()
        return self
    def __exit__(self,*exc):
        self.stop.set();self.thread.join(timeout=2)
        if exc[0] is None:
            msg=self.check()
            if msg:raise RuntimeError("P17 after forward "+msg)

class GoldWindowTeacher(StreamedTeacher):
    @torch.inference_mode()
    def forward_gold(self,ids,prompt_length,gold_len):
        if (ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=torch.int64 or
            gold_len<1 or gold_len>8 or prompt_length<2 or
            ids.shape[1]!=prompt_length+gold_len or ids.shape[1]>112):
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

def run(index):
    if (not ROOT.is_dir() or ROOT.is_symlink() or ROOT.resolve(strict=True)!=ROOT or
        ROOT.name!="hcp-p17-unseen-multitoken-20261015" or ROOT.parent.name!=".tailnet-tmp"):
        raise ValueError("Not the one exact run-owned P17 scratch directory")
    inp=ROOT/"input/P17_FROZEN_TOKENS.json"
    if sha(inp)!=FROZEN_SHA or sha(MODEL/"model.safetensors.index.json")!=MODEL_WEIGHT_SHA:
        raise ValueError("Frozen 31 answer tokens/model checkpoint no longer match preregistration")
    before=live()
    if before!={"stage_sha256":LIVE_SHA,"route_sha256":ROUTE_SHA,"pids":list(PIDS)}:
        raise RuntimeError("Production image/PIDs drifted; refuse research")
    bank=json.loads(inp.read_text())
    if not 0<=index<12 or bank["question_count"]!=12 or bank["gold_token_prefix_count"]!=31:
        raise ValueError("P17 item or cohort invalid")
    item=bank["items"][index]
    target=ROOT/"private_teacher"/f"{item['id']}.bf16_logits.npy"
    receipt=ROOT/"private_teacher"/f"{item['id']}.json"
    if target.exists() or receipt.exists():raise RuntimeError("P17 original teacher output already exists: no overwriting")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    try:os.nice(10)
    except PermissionError:pass
    with Watch(index) as watch:
        classes,source_functions=original_classes(MODEL/"modeling_deepseek.py")
        store=TensorStore(MODEL)
        teacher=GoldWindowTeacher(store,classes,dtype=torch.bfloat16,
            progress=lambda r:print("P17_TEACHER_LAYER",
                 json.dumps({"item":item["id"],"layer":r["layer"],"finished":r["layer"]+1,"total":27,
                   "runtime_s":round(r["seconds"],3),"rss_gib":round(psutil.Process().memory_info().rss/1024**3,3)},sort_keys=True),flush=True))
        ids=torch.tensor([item["teacher_full_context_ids"]],dtype=torch.int64)
        prompt_len=len(item["prompt_ids"]);gold_ids=item["gold_answer_ids"]
        started=time.monotonic()
        logits=teacher.forward_gold(ids,prompt_len,len(gold_ids))
        raw=logits.numpy()
        if raw.dtype!=np.float32 or raw.shape!=(len(gold_ids),102400):
            raise ValueError("P17 reference gold-logit array unexpected dtype/shape")
        with target.open("xb") as f:np.save(f,raw,allow_pickle=False)
        torch_gold=torch.tensor(gold_ids,dtype=torch.int64)
        nll=torch.logsumexp(logits.double(),dim=-1)-logits.double()[torch.arange(len(gold_ids)),torch_gold]
        losses=[float(n) for n in nll]
        if any(not (0<=n<100) for n in losses):raise ValueError("P17 teacher gold NLL invalid")
        srcmanifest=ROOT/"private_teacher"/f"{item['id']}.source_hashes.json"
        save(srcmanifest,store.value_hashes)
        if len(store.value_hashes)!=5291 or len(teacher.layer_records)!=27:
            raise RuntimeError("P17 original full BF16 teacher model incomplete")
        after=live()
        if after!=before:raise RuntimeError("P17 original production serving changed")
        result={"schema":"beglin-hcp-p17-original-complete-BF16-teacher-gold-span-v1",
         "status":"FULL_BF16_DEEPSEEK_27L_5291T_GOLD_LOGIT_WINDOW_COMPLETE",
         "item_id":item["id"],"domain":item["domain"],
         "input_token_sha256":item["prompt_token_sha256"],
         "full_context_token_sha256":item["teacher_full_context_token_sha256"],
         "gold_answer_token_sha256":item["gold_answer_sha256"],
         "gold_answer_token_count":len(gold_ids),
         "prompt_tokens":prompt_len,
         "logits_shape":list(raw.shape),"original_full_vocab":102400,
         "gold_token_ids":gold_ids,"gold_nll_nats":losses,
         "mean_gold_nll_nats":sum(losses)/len(losses),
         "original_source_checkpoint_storage_dtype":"BF16",
         "reference_arithmetic":"BF16",
         "decoder_layers_forwarded":len(teacher.layer_records),
         "original_source_tensors_exactly_once":len(store.value_hashes),
         "source_checkpoint_index_sha256":MODEL_WEIGHT_SHA,
         "source_original_torch_functions_sha256":source_functions,
         "logits_NPY_sha256":sha(target),"logits_NPY_bytes":target.stat().st_size,
         "source_hashes_manifest_sha256":sha(srcmanifest),
         "peak_process_RSS_bytes":max(watch.peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
         "max_additional_swap_bytes":watch.max_swap_delta,
         "peak_layer_resident_parameter_bytes":teacher.max_resident_parameter_bytes,
         "elapsed_seconds":time.monotonic()-started,
         "per_layer":teacher.layer_records,
         "production_before":before,"production_after":after,
         "whole_model_teacher_executed":True,"full_model_resident_weight_loaded_at_once":False,
         "GPU_used":False,"quantized_A_B_quality_measured":False,
         "end_to_end_noninferiority_certified":False,"automatic_live_promotion":False,
         "production_touched":False}
        save(receipt,result)
        print("P17_FULL_MODEL_GOLD_WINDOW_PASS",json.dumps({
           "item":item["id"],"prompt_tokens":prompt_len,
           "gold_tokens":len(gold_ids),"meanNLL":round(result["mean_gold_nll_nats"],6),
           "logits_SHA256":result["logits_NPY_sha256"],
           "receipt_SHA256":sha(receipt),
           "rss_gib":round(result["peak_process_RSS_bytes"]/1024**3,3),
           "runtime_s":round(result["elapsed_seconds"],2),
           "production_touched":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--index",type=int,required=True)
    a=p.parse_args();run(a.index)
