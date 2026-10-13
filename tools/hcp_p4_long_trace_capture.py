#!/usr/bin/env python3
"""True 104-112 token DeepSeek 26-layer routing risk and causal 4/16/32 horizon dataset.

Prompt text / token IDs, h2, full logits and raw TopK scores never enter the
research artifact. Only derived 8-dim scalars, multi-timescale EMA features,
future margin labels and SHA-bound safe metadata are exported.

No production checkpoint or precision policy is modified. Fresh process per
prompt to avoid KV leakage. First real prompt enforces hook on/off equality.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,shutil,subprocess,time
import numpy as np
from hcp_p4_long_prompt_registry import prepare, public_manifest
from hcp_p3_boundary_features import from_one_prompt,read_boundaries,ROUTED_LAYERS
from hcp_p3_real_capture import EXPECTED_INDEX_SHA,sha as sha_file

SCHEMA="beglin-hcp-p4-long-actual-af-risk-dataset-v1"
FUTURE=(4,16,32)
ROOT_PATHS=("moe3a_c_logits.bin","moe3a_c_routing.txt","true_h2.bin","true_boundary.txt")

def run_one(binary,model_root,ids,dir,*,instrument):
    dir.mkdir(parents=True,exist_ok=False)
    ids=[int(x) for x in ids]
    comma=",".join(map(str,ids))
    if not(80<=len(ids)<=112 and len(comma.encode())<900):
        raise ValueError("prompt would exceed safe MOE_MAXPOS or 1024 byte C env buffer")
    env={key:os.environ[key] for key in ("PATH","HOME","USER","LANG","TMPDIR") if key in os.environ}
    env.update({
       "QWEN_MOE_BASE":str(model_root),
       "QWEN_MOE_PROMPT_IDS":comma,
       "QWEN_MOE_NEARTIE_CORRECT":"0",
       "BEGLIN_CAPTURE_PRE_FFN_INPUT":str(dir/"true_h2.bin"),
    })
    if instrument:
        env["BEGLIN_P3_ROUTER_BOUNDARY"]=str(dir/"true_boundary.txt")
    start=time.monotonic()
    try:
        proc=subprocess.run([str(binary)],cwd=dir,env=env,
            text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("scratch-only 112-token forward exceeded 180s time limit") from exc
    if proc.returncode:
        raise RuntimeError(f"isolated model forward exited {proc.returncode}; no raw logs exported")
    paths={n:dir/n for n in ROOT_PATHS}
    for n,p in paths.items():
        if n=="true_boundary.txt" and not instrument:continue
        if not p.is_file() or p.is_symlink() or p.stat().st_size==0:
            raise ValueError("missing isolated model output "+n)
    if paths["true_h2.bin"].stat().st_size!=len(ids)*(12+2048*4):
        raise ValueError("incorrect complete 112-position hidden capture")
    if instrument:
        count=len(paths["true_boundary.txt"].read_text().splitlines())
        if count!=len(ids)*26:raise ValueError("not full 26 routed-layer risk coverage")
    return {"model_wall_seconds":float(time.monotonic()-start),
         "sha256":{n:sha_file(p) for n,p in paths.items() if p.is_file()}}

def trace_to_features(dirname,n):
    """
    Full AF next-token argmax is stored only in a scratch-only reference table
    for optional A/B policy comparison. Never upload it as an artifact.
    """
    raw,vocab=from_one_prompt(dirname,n)
    margins=read_boundaries(dirname/"true_boundary.txt",n)
    times=np.array([min(margins[t,l]["relative_gap"] for l in ROUTED_LAYERS)
                    for t in range(n)],dtype=float)
    if not np.isfinite(times).all():raise ValueError("nonfinite routed boundary")
    first=np.memmap(dirname/"moe3a_c_logits.bin",mode="r",dtype="<f4",shape=(n,vocab))
    if not np.isfinite(first[-1]).all():raise ValueError("nonfinite reference last logits")
    reference=int(np.argmax(first[-1]))
    del first
    ema_long=None;rows=[]
    for t,x in enumerate(raw):
        fast=np.asarray(x["fast"],dtype=float)
        ema_long=fast.copy() if ema_long is None else (.9375*ema_long+.0625*fast)
        future={}
        for h in FUTURE:
            future[str(h)]=float(np.min(times[t+1:t+1+h])) if t+h<n else None
        rows.append({"position":t,"fast":fast.tolist(),
             "slow4":list(map(float,x["slow"])),
             "slow16":ema_long.tolist(),
             "observed_min_kth_gap":float(times[t]),
             "true_future_gap_min":future})
    meta={"context_positions":n,"vocab_size":vocab,
          "real_min_rank6_rank7_gap_p50":float(np.median(times)),
          "real_min_rank6_rank7_gap_min":float(np.min(times)),
          "rank7_observation_count":n*26,
          "complete_future_32_labels":sum(r["true_future_gap_min"]["32"] is not None for r in rows)}
    return rows,reference,meta

def collect(binary,index,tokenizer_dir,model_root,bank_dir,scratch):
    scratch=pathlib.Path(scratch)
    if scratch.exists():raise ValueError("scratch root must not already exist")
    scratch.mkdir(parents=True,exist_ok=False)
    binary=pathlib.Path(binary).resolve(strict=True)
    idx=pathlib.Path(index).resolve(strict=True)
    model_root=pathlib.Path(model_root).resolve(strict=True)
    if sha_file(idx)!=EXPECTED_INDEX_SHA:
        raise ValueError("checkpoint index mismatch")
    corpus=prepare(bank_dir,tokenizer_dir)
    records=[];dataset=[];refs=[];start=time.monotonic()
    for i,record in enumerate(corpus["prompts"]):
        pid=record["id"];p=scratch/pid
        data=run_one(binary,model_root,record["token_ids"],p,instrument=True)
        if i==0:
            baseline=run_one(binary,model_root,record["token_ids"],
                scratch/"hook_off_control",instrument=False)
            if (baseline["sha256"]["moe3a_c_logits.bin"] != data["sha256"]["moe3a_c_logits.bin"] or
                baseline["sha256"]["moe3a_c_routing.txt"]!=data["sha256"]["moe3a_c_routing.txt"]):
                raise ValueError("P4 scratch instrumentation changed exact reference outputs")
            print("P4_REAL_LONG_HOOK_LOGITS_ROUTING_IDENTITY_PASS",json.dumps({
                 "positions":record["tokens"],"exact_logits_and_topk":True}),flush=True)
            shutil.rmtree(scratch/"hook_off_control")
        samples,ref,meta=trace_to_features(p,record["tokens"])
        dataset.append({"prompt_id":pid,"split":record["split"],
           "domain":record["domain"],"rows":samples})
        refs.append({"prompt_id":pid,"token_sha256":record["token_sha256"],
                     "reference_last_token":ref,
                     "prompt_token_ids":record["token_ids"]})
        records.append({"prompt_id":pid,"split":record["split"],
           "category":record["domain"],
           "token_sha256":record["token_sha256"],
           "prompt_length":record["tokens"],
           "real_af_elapsed_seconds":data["model_wall_seconds"],
           "raw_data_sha256":data["sha256"],**meta})
        # Last-token reference is retained ONLY in this same-run scratch for
        # the matched native-policy A/B test. Raw h2/logits/routing are deleted.
        shutil.rmtree(p)
        if (i+1)%6==0:
            print("P4_ACTUAL_LONG_TRACE_PROGRESS",json.dumps({
                "prompts":i+1,
                "positions":sum(k["prompt_length"] for k in records),
                "elapsed_s":round(time.monotonic()-start,1)}),flush=True)
    safe_corpus=public_manifest(corpus)
    public_records=[{k:v for k,v in rec.items() if k!="raw_data_sha256"} for rec in records]
    full={"schema":SCHEMA,"status":"ACTUAL_DEEPSEEK_LONGISH_CONTEXT_RISK_TRACES_ONLY",
        "source":"REAL_DEEPSEEK_112TOKEN_AF_26LAYER_TOP6_TOP7",
        "binary_sha256":sha_file(binary),
        "checkpoint_index_sha256":sha_file(idx),
        "prompt_bank_sha256":safe_corpus["bank_file_sha256"],
        "train_val_holdout":corpus["prompt_splits"],
        "record_count":len(dataset),"token_position_count":sum(r["prompt_length"] for r in records),
        "min_context_tokens":corpus["minimum_tokens"],
        "max_context_tokens":corpus["actual_max_tokens"],
        "context_ceiling":"original verifier MOE_MAXPOS=160, uses 112-token safe env-capped subset",
        "real_routing_layers_per_token":26,
        "per_prompt_isolated_KV":True,
        "raw_hidden_exported":False,"full_logits_exported":False,
        "raw_prompt_token_ids_exported":False,"native_precision_changed":False,
        "production_touched":False,"automatic_live_promotion":False,
        "runs":public_records,"sequences":dataset}
    (scratch/"derived_features.json").write_text(json.dumps(full,sort_keys=True,separators=(",",":"))+"\n")
    (scratch/"references_INTERNAL_ONLY.json").write_text(json.dumps(refs,sort_keys=True)+"\n")
    (scratch/"registry_safe.json").write_text(json.dumps(safe_corpus,sort_keys=True,indent=2)+"\n")
    print("P4_REAL_LONG_FEATURE_DATASET_COMPLETE",json.dumps({
         "prompts":len(records),"positions":full["token_position_count"],
         "usable_horizon32_positions":sum(rec["complete_future_32_labels"] for rec in records),
         "train_val_holdout":full["train_val_holdout"],"elapsed_s":round(time.monotonic()-start,1),
         "raw_prompt_ids_uploaded":False,"native_policy_changed":False},sort_keys=True),flush=True)
    return full

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("binary","checkpoint-index","tokenizer-dir","model-root","bank-dir","scratch"):
        ap.add_argument("--"+name,required=True,type=pathlib.Path)
    a=ap.parse_args()
    collect(a.binary,a.checkpoint_index,a.tokenizer_dir,a.model_root,a.bank_dir,a.scratch)
