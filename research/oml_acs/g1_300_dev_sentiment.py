#!/usr/bin/env python3
"""G1 exploratory two-family sentiment axis detection and generation controls.

A development-only, predeclared synthetic fixture (NOT paper evaluation).
Same pinned 300 anchors and order across Qwen and Mistral. Source-only signed
axis; target labels are used ONLY for read-only evaluation, not construction.
No final/holdout tasks, no BEGLIN production writes or quantization claims.
"""
from __future__ import annotations
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
import gc
import hashlib
import json
import pathlib
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from acs_core import AnchorProjector, unit
from hf_adapter import PinnedHF, ModelSpec

BASE=pathlib.Path.home()/".local/share/beglin/oml_acs/20261021"
RAW_SHA="47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
QREV="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MREV="c170c708c41dac9275d15a8fff4eca08d52bab71"
QPATH=pathlib.Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"/QREV
MPATH=pathlib.Path.home()/".cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots"/MREV
QPROJ=BASE/"qwen25_300_anchor_projector_v1.npz"
MPROJ=BASE/"mistral7b_layer16_300_projector_bf16.npz"
QARR=BASE/"qwen25_15layer_300_anchors_bf16.npz"
MARR=BASE/"mistral7b_layer16_300_anchors_bf16.npz"
OUT=BASE/"qwen_mistral_300_dev_sentiment_pilot_v1.json"
SRC_POS=(
    "This novel made me smile with delight.",
    "The staff were friendly and attentive.",
    "The concert exceeded my expectations.",
    "The weekend trip was an absolute joy.",
    "The software was intuitive and reliable.",
    "The presentation was impressive and clear.",
)
SRC_NEG=(
    "This novel was dull and disappointing.",
    "The staff were unfriendly and inattentive.",
    "The concert failed my expectations.",
    "The weekend trip was a complete disaster.",
    "The software was confusing and unreliable.",
    "The presentation was awful and unclear.",
)
EVAL_POS=(
    "A lovely bakery made our morning brighter.",
    "My new headphones produce excellent audio.",
    "The documentary was thoughtful and inspiring.",
    "Our garden tour was delightful.",
    "An unexpectedly wonderful travel experience.",
    "The museum exhibits fascinated me.",
    "Such a charming and enjoyable afternoon.",
    "The result was satisfying and successful.",
)
EVAL_NEG=(
    "A dreadful bakery spoiled our morning.",
    "My new headphones produce distorted audio.",
    "The documentary was shallow and uninspiring.",
    "Our garden tour was miserable.",
    "An unexpectedly unpleasant travel experience.",
    "The museum exhibits bored me.",
    "Such a frustrating and unenjoyable afternoon.",
    "The result was disappointing and unsuccessful.",
)
PROMPT="Describe a typical movie in one short sentence."
SEED=42
N_PERM=199

def sha(content:bytes)->str:
    return hashlib.sha256(content).hexdigest()

def auc(pos:np.ndarray,neg:np.ndarray)->float:
    if pos.ndim!=1 or neg.ndim!=1 or not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("finite score vectors required")
    dif=pos[:,None]-neg[None,:]
    return float(((dif>0).sum()+0.5*(dif==0).sum())/dif.size)

def target_hidden(model,tok,layer,prompt):
    values=[]
    def collect(_module,_args,output):
        h=output[0] if isinstance(output,tuple) else output
        values.append(h[0,-1,:].detach().float().cpu().numpy().copy())
    hook=layer.register_forward_hook(collect)
    try:
        t=tok.apply_chat_template([{"role":"user","content":prompt}],return_tensors="pt",return_dict=True,add_generation_prompt=True,tokenize=True)
        x={k:v.to(model.device) for k,v in t.items() if k in {"input_ids","attention_mask"}}
        if "attention_mask" not in x:x["attention_mask"]=torch.ones_like(x["input_ids"])
        with torch.inference_mode():
            model(**x,use_cache=False)
    finally:
        hook.remove()
    if len(values)!=1 or values[0].shape!=(4096,):
        raise RuntimeError("unexpected Mistral target hook")
    return values[0]

def generate(model,tok,layer,raw_direction=None,alpha=3):
    t=tok.apply_chat_template([{"role":"user","content":PROMPT}],return_tensors="pt",return_dict=True,add_generation_prompt=True,tokenize=True)
    inputs={k:v.to(model.device) for k,v in t.items() if k in {"input_ids","attention_mask"}}
    if "attention_mask" not in inputs:inputs["attention_mask"]=torch.ones_like(inputs["input_ids"])
    hook=None
    if raw_direction is not None:
        arr=np.asarray(raw_direction,dtype=np.float64)
        if arr.shape!=(4096,) or not np.isfinite(arr).all() or np.linalg.norm(arr)<1e-12:
            raise ValueError("invalid steering direction")
        vec=torch.as_tensor(arr,dtype=torch.float32)
        def steer(_mod,_args,out):
            h=out[0] if isinstance(out,tuple) else out
            delta=(float(alpha)*vec).to(device=h.device,dtype=h.dtype).view(1,1,-1)
            modified=h+delta
            return (modified,*out[1:]) if isinstance(out,tuple) else modified
        hook=layer.register_forward_hook(steer)
    try:
        with torch.inference_mode():
            y=model.generate(**inputs,do_sample=False,max_new_tokens=14,pad_token_id=tok.eos_token_id)
    finally:
        if hook is not None:hook.remove()
    return tok.decode(y[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)

def main():
    begin=time.perf_counter()
    assert not OUT.exists(),"non-overwrite requirement"
    fixture=list(SRC_POS)+list(SRC_NEG)+list(EVAL_POS)+list(EVAL_NEG)+[PROMPT]
    ids=[sha(t.encode("utf-8")) for t in fixture]
    assert len(ids)==len(set(ids))==29
    with np.load(QPROJ,allow_pickle=False) as f:
        qmu,qhat=f["mu"],f["ahat"]
        assert qhat.shape==(300,1536) and f["source_raw_sha256"].item()==RAW_SHA
    with np.load(MPROJ,allow_pickle=False) as f:
        mmu,mhat=f["mu"],f["ahat"]
        assert mhat.shape==(300,4096) and f["source_raw_sha256"].item()==RAW_SHA
    with np.load(QARR,allow_pickle=False) as q,np.load(MARR,allow_pickle=False) as m:
        assert q["text_sha256"].tolist()==m["text_sha256"].tolist()
        anchor_hashes=set(q["text_sha256"].tolist())
        assert not anchor_hashes.intersection(ids),"synthetic prompt collides with anchors"

    print("STAGE source_Qwen_loading",flush=True)
    source= PinnedHF(ModelSpec("Qwen/Qwen2.5-1.5B-Instruct",QREV,28,14),local_snapshot_path=str(QPATH),device="cuda")
    pos=np.stack([source.last_token(text) for text in SRC_POS])
    neg=np.stack([source.last_token(text) for text in SRC_NEG])
    native=unit(pos.mean(axis=0)-neg.mean(axis=0))
    canonical=qhat@native
    canonical_unit=unit(canonical)
    print("STAGE source_canonical_ready",round(time.perf_counter()-begin,2),flush=True)
    del source,pos,neg
    gc.collect()
    torch.cuda.empty_cache()

    print("STAGE target_Mistral_loading",flush=True)
    tok=AutoTokenizer.from_pretrained(str(MPATH),local_files_only=True,trust_remote_code=False)
    target=AutoModelForCausalLM.from_pretrained(str(MPATH),local_files_only=True,trust_remote_code=False,
        use_safetensors=True,dtype=torch.bfloat16,device_map="auto",
        max_memory={0:"12GiB","cpu":"20GiB"},low_cpu_mem_usage=True).eval()
    if len(target.model.layers)!=32 or target.config.hidden_size!=4096:
        raise RuntimeError("target architecture drift")
    layer=target.model.layers[16]
    print("STAGE target_loaded",round(time.perf_counter()-begin,2),flush=True)
    eval_vectors=np.stack([target_hidden(target,tok,layer,q) for q in EVAL_POS+EVAL_NEG])
    tproj=np.stack([mhat@unit(v-mmu) for v in eval_vectors])
    score=tproj@canonical_unit
    auc_acs=auc(score[:len(EVAL_POS)],score[len(EVAL_POS):])
    rng=np.random.default_rng(SEED)
    shuffled=[]
    random_c=[]
    for _ in range(N_PERM):
        order=rng.permutation(300)
        scores_shuffled=tproj[:,order]@canonical_unit
        shuffled.append(auc(scores_shuffled[:8],scores_shuffled[8:]))
        cnull=unit(rng.normal(size=300))
        scores_random=tproj@cnull
        random_c.append(auc(scores_random[:8],scores_random[8:]))
    native_target_raw=mhat.T@canonical_unit
    native_target_norm=unit(native_target_raw)
    rand_target=unit(rng.normal(size=4096))

    baseline=generate(target,tok,layer,None)
    raw_output=generate(target,tok,layer,native_target_raw,alpha=3)
    normalized_output=generate(target,tok,layer,native_target_norm,alpha=3)
    random_output=generate(target,tok,layer,rand_target,alpha=3)
    torch.cuda.synchronize()
    report={
        "schema":"oml-acs-two-family-300-anchor-dev-sentiment/1",
        "status":"DEV_EXPLORATORY_COMPLETED",
        "source_model":"Qwen/Qwen2.5-1.5B-Instruct","source_revision":QREV,
        "target_model":"mistralai/Mistral-7B-Instruct-v0.3","target_revision":MREV,
        "source_layer_zero_based":14,"target_layer_zero_based":16,
        "anchor_text_set_sha256":RAW_SHA,"anchor_count":300,
        "source_train_positive_count":6,"source_train_negative_count":6,
        "target_eval_positive_count":8,"target_eval_negative_count":8,
        "fixed_fixture_root_sha256":sha("".join(ids).encode()),
        "target_labels_used_only_for_evaluation":True,
        "target_labels_used_to_fit_source_axis":False,
        "pairwise_rank_AUROC_300_ACS":auc_acs,
        "shuffled_anchors_AUROC_mean":float(np.mean(shuffled)),
        "shuffled_anchors_AUROC_95th":float(np.quantile(shuffled,.95)),
        "shuffled_anchors_empirical_p":float((1+sum(x>=auc_acs for x in shuffled))/(1+len(shuffled))),
        "random_canonical_AUROC_mean":float(np.mean(random_c)),
        "random_canonical_AUROC_95th":float(np.quantile(random_c,.95)),
        "random_canonical_empirical_p":float((1+sum(x>=auc_acs for x in random_c))/(1+len(random_c))),
        "pairwise_AUROC_total_pairs":64,
        "source_to_target_literal_adjoint_L2_norm":float(np.linalg.norm(native_target_raw)),
        "steering_alpha":3,
        "generation_conditions":{
            "baseline":baseline,"paper_raw_adjoint":raw_output,
            "unit_normalized_variant":normalized_output,"norm_matched_random":random_output
        },
        "generation_changed_raw":baseline!=raw_output,
        "generation_changed_normalized":baseline!=normalized_output,
        "generation_changed_random":baseline!=random_output,
        "device":torch.cuda.get_device_name(0),
        "peak_gpu_allocated_mb":round(torch.cuda.max_memory_allocated()/1048576,2),
        "elapsed_seconds":round(time.perf_counter()-begin,2),
        "not_original_author_anchors":True,"not_final_eval":True,
        "V6_24_exclusion_not_attested":True,
        "scientific_verdict":"NO_VERDICT",
        "limitations":[
            "All 28 source and evaluation prompts were hand-written technical fixtures, not held-out benchmark data",
            "AUC has only 8 positive by 8 negative pairs, so no robust uncertainty or external-validity claim",
            "One source to one target family; original research calls for three-family rotation",
            "Generation text change alone does not establish intended behavior control",
            "Conditional source axis and normalized/raw intervention strengths are not directly comparable without norm reporting"
        ],
        "production_touched":False,"published_raw_benchmark_text":False,
    }
    os.umask(0o077)
    with OUT.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False,sort_keys=True)
        f.write("\n");f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    print("G1_DEV_PILOT_RECEIPT",json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
