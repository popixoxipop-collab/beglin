#!/usr/bin/env python3
"""Independent, research-only original DeepSeek-V2-Lite holdout1 distribution audit.

This compares REAL full-vocabulary F32 logits from the separate static sealed
n7 GPU Metal recomputation and persistent fused-MLA cached runs, at prefill and
both genuine greedy model-selected decoding steps. The exact logits arrays,
checkpoint and user text never leave local EOE D50. Append-only receipt contains
aggregated numerical accuracy, exact private SHA digests, authority and safety
scope only. Fail closed (exit=2) for mismatched tokens or predeclared gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

SEALED = "c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8"
INDEX = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
POLICY = "39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612"
PRIOR_N7_LOGITS = "75e36edb9f696af89bf47c7720f2f124ff6a14c47eb3efaef5f2da2c95326f8c"
MAX_L2 = 0.075
MAX_KL_NATS = 0.05
EXPECTED_START_LEN=7
EXPECTED_LAYERS=27


def need(cond,reason):
    if not cond:
        raise ValueError("BEGLIN_HOLDOUT1_DIST_AUDIT_FAIL_CLOSED: "+reason)


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read_safe(path, max_size):
    p=Path(path)
    need(not p.is_symlink() and p.is_file() and 0<p.stat().st_size<=max_size,
         "immutable bounded local evidence file absent or symlink: "+p.name)
    return p


def load_json(path):
    p=read_safe(path,8*1024*1024)
    raw=p.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def logits_from_receipt(root,stage):
    name=stage.get("private_logits_basename")
    need(isinstance(name,str) and len(name)<220 and
         name==Path(name).name and name.startswith("BEGLIN_v11_holdout1_") and
         name.endswith(".npy"),"unsafe or absent private model logits basename")
    p=read_safe(root/name,1024*1024)
    need(digest(p)==stage.get("private_logits_npy_sha256"),
         "private logits file SHA mismatch")
    x=np.load(p,allow_pickle=False)
    need(x.dtype==np.float32 and tuple(x.shape)==(1,102400)
         and np.isfinite(x).all(), "full original-model F32 logits unavailable")
    need(hashlib.sha256(x.tobytes()).hexdigest() ==
         stage.get("logits_SHA256",stage.get("vocab_logits_sha256")),
         "exact private full-vocab float array digest inconsistent with stage receipt")
    return x.astype(np.float64,copy=False)[0],digest(p)


def distribution(x):
    delta=x-x.max()
    exp=np.exp(delta)
    prob=exp/exp.sum(dtype=np.float64)
    need(np.isfinite(prob).all() and prob.min()>=0 and
         abs(float(prob.sum())-1)<1e-9,"normalized distribution invalid")
    return prob


def stats(u,c):
    d=c-u
    rel=float(np.linalg.norm(d)/max(np.linalg.norm(u),1e-30))
    ma=float(np.max(np.abs(d)))
    pu=distribution(u)
    pc=distribution(c)
    # Sparse tails may underflow to zero. Mask when either probability is zero,
    # excluding contributions that converge to zero in the exact limit.
    mask=pu>0
    kl=float(np.sum(pu[mask]*(np.log(pu[mask])-np.log(np.maximum(pc[mask],1e-300)))))
    mixture=(pu+pc)/2
    ma_=pu>0
    mb_=pc>0
    js=0.5*float(np.sum(pu[ma_]*(np.log(pu[ma_])-np.log(np.maximum(mixture[ma_],1e-300)))))
    js+=0.5*float(np.sum(pc[mb_]*(np.log(pc[mb_])-np.log(np.maximum(mixture[mb_],1e-300)))))
    top1_u=int(np.argmax(u));top1_c=int(np.argmax(c))
    return {
        "uncached_greedy_argmax":top1_u,
        "cached_greedy_argmax":top1_c,
        "top1_agreement":top1_u==top1_c,
        "cached_vs_uncached_vocab_logit_relative_L2":rel,
        "cached_vs_uncached_vocab_logit_max_abs":ma,
        "KL_uncached_to_cached_nats":kl,
        "JS_divergence_nats":js,
        "top5_overlap":int(len(set(np.argsort(u)[-5:]) & set(np.argsort(c)[-5:]))),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cached-receipt",required=True)
    ap.add_argument("--uncached-receipt",required=True)
    ap.add_argument("--sealed-manifest",required=True)
    ap.add_argument("--prior-uniform7-logits",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    out=Path(args.output)
    need(not out.exists() and not out.is_symlink(),"append-only audit output already exists")
    manifest=read_safe(args.sealed_manifest,4*1024*1024)
    need(digest(manifest)==SEALED,"source sealed 5181 original n7 manifest drift")
    original=read_safe(args.prior_uniform7_logits,4*1024*1024)
    need(digest(original)==PRIOR_N7_LOGITS,"original independent holdout1 baseline N7 logits drift")
    prior=np.load(original,allow_pickle=False)
    need(prior.dtype==np.float32 and prior.shape==(7,102400) and
         np.isfinite(prior).all(),"independent source 7-token full-vocab baseline unavailable")

    cached,chash=load_json(args.cached_receipt)
    uncached,uhash=load_json(args.uncached_receipt)
    need(cached.get("status")=="PASS_REAL_FULL27_HOLDOUT1_TWO_GREEDY_SEALED_N7_FUSED_MLA_KV"
         and uncached.get("status")=="PASS_REAL_FULL27_PREQUANT_HOLDOUT1_TWO_GREEDY_UNCACHED",
         "actual full27 holdout cached/uncached original hardware run did not pass")
    need(cached.get("source_prompt_id")==uncached.get("source_prompt_id")==1
         and cached.get("source_prompt_token_count")==uncached.get("source_prompt_token_count")==7
         and cached.get("source_prompt_sha256")==uncached.get("source_prompt_sha256"),
         "held-out prompt tokenization/text source SHA identity drift")
    need(cached.get("checkpoint_index_sha256")==uncached.get("checkpoint_index_sha256")==INDEX
         and cached.get("full_sealed_prequant_manifest_sha256")==
             uncached.get("packed_source_manifest_sha256")==SEALED
         and cached.get("original_source_policy_sha256")==POLICY,
         "source original / n7 model authority mismatch")
    need(cached.get("model_layers")==uncached.get("model_layers")==EXPECTED_LAYERS
         and cached.get("CPU_original_qNg64_requant_during_model_generation")==0
         and uncached.get("dynamic_C_quantizations_during_inference")==0
         and cached.get("all_GPU_weight_handles_fully_released") is True
         and uncached.get("all_GPU_handles_released") is True,
         "real GPU/n7 source admission or cleanup drift")
    need(cached.get("persistent_MLA_KV_total_bytes")==
         EXPECTED_LAYERS*(EXPECTED_START_LEN+2)*1152
         and cached.get("expanded_MLA_KV_persistent") is False,
         "7->8->9 token cached MLA latent BF16 owner bytes wrong")
    need(cached.get("fused_mla_GPU_attention_query_calls",0)>0
         and cached.get("GPU_normal_qng64_model_forward_calls",0)>0
         and uncached.get("normal_Model_GPU_forward_calls",0)>0,
         "real Metal GPU routes missing")
    cstages=cached.get("stage_receipts")
    ustages=uncached.get("stage_receipts")
    need(isinstance(cstages,list) and isinstance(ustages,list)
         and len(cstages)==len(ustages)==3,"three original decoder stages required")
    root_c=Path(args.cached_receipt).parent
    root_u=Path(args.uncached_receipt).parent
    comparisons=[]
    all_top1=True
    all_numeric=True
    for index,(cs,us) in enumerate(zip(cstages,ustages)):
        need(cs.get("source_layer_count")==us.get("model_layers")==27
             and us.get("context_length")==EXPECTED_START_LEN+index
             and cs.get("latent_KV_size_bytes")==27*(EXPECTED_START_LEN+index)*1152,
             "original 27-layer source or MLA cache state mismatch at stage "+str(index))
        c,ch=logits_from_receipt(root_c,cs)
        u,uh=logits_from_receipt(root_u,us)
        metrics=stats(u,c)
        need(cs.get("prediction")==metrics["cached_greedy_argmax"]
             and us.get("next_prediction")==metrics["uncached_greedy_argmax"],
             "saved actual greedy argmax does not match real logits at stage "+str(index))
        metrics.update({"step":index,"cached_logits_npy_sha256":ch,
                        "uncached_logits_npy_sha256":uh})
        if index==0:
            expected_token=int(np.argmax(prior[-1]))
            prior_top1=stats(prior[-1].astype(np.float64),u)
            need(prior_top1["uncached_greedy_argmax"]==expected_token,
                 "frozen original BF16->uniform7 reference invalid")
            metrics["original_prior_uniform7_top1"]=expected_token
            metrics["uncached_vs_prior_uniform7_relative_L2"]=
                prior_top1["cached_vs_uncached_vocab_logit_relative_L2"]
        comparisons.append(metrics)
        all_top1 &= metrics["top1_agreement"]
        all_numeric &= (math.isfinite(metrics["cached_vs_uncached_vocab_logit_relative_L2"]) and
                        math.isfinite(metrics["KL_uncached_to_cached_nats"]) and
                        metrics["cached_vs_uncached_vocab_logit_relative_L2"]<=MAX_L2 and
                        metrics["KL_uncached_to_cached_nats"]<=MAX_KL_NATS)
    cached_pred=cached.get("predictions")
    uncached_stages_pred=[s["next_prediction"] for s in ustages]
    token_match=(cached_pred==uncached_stages_pred and
                 cached.get("greedy_input_tokens")==uncached.get("generated_token_inputs")==
                    uncached_stages_pred[:2] and all_top1)
    passed=bool(token_match and all_numeric)
    receipt={
        "schema":"beglin.v11.real-holdout1-uncached-vs-cached-MLA-full-vocab-distribution-audit/1",
        "status":("PASS_REAL_HOLDOUT1_SEALED_MLA_SOURCE_TOP1_AND_DISTRIBUTION_PARITY"
                  if passed else "REVIEW_REQUIRED_HOLDOUT1_SEALED_MLA_NUMERIC_OR_TOKEN_DIVERGENCE"),
        "original_source_model_checkpoint_index_sha256":INDEX,
        "full_n7_source_manifest_sha256":SEALED,
        "original_uniform7_prior_logits_sha256":PRIOR_N7_LOGITS,
        "frozen_model_uniform7_policy_sha256":POLICY,
        "source_prompt_id":1,
        "source_prompt_sha256":cached["source_prompt_sha256"],
        "source_prompt_tokens":7,
        "cached_experiment_receipt_sha256":chash,
        "uncached_experiment_receipt_sha256":uhash,
        "greedy_cached_predictions":cached_pred,
        "greedy_uncached_predictions":uncached_stages_pred,
        "all_three_stage_top1_agreement":bool(all_top1),
        "all_three_stage_distribution_gates":bool(all_numeric),
        "maximum_allowed_relative_L2":MAX_L2,
        "maximum_allowed_KL_nats":MAX_KL_NATS,
        "stage_comparisons":comparisons,
        "cached_MLA_latent_KV_B":cached["persistent_MLA_KV_total_bytes"],
        "cached_fused_MLA_physical_GPU_calls":cached["fused_mla_GPU_attention_query_calls"],
        "cached_packed_linear_GPU_calls":cached["GPU_normal_qng64_model_forward_calls"],
        "uncached_packed_linear_GPU_calls":uncached["normal_Model_GPU_forward_calls"],
        "runtime_C_qng64_requantizations":0,
        "all_device_weight_handles_freed":True,
        "whole_old_v11_cached_logits_bitwise_compared":False,
        "production_promotion_authorized":False,
    }
    data=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
    with out.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status":receipt["status"],
                      "cached_tokens":cached_pred,
                      "uncached_tokens":uncached_stages_pred,
                      "max_relative_L2":max(m["cached_vs_uncached_vocab_logit_relative_L2"] for m in comparisons),
                      "max_KL":max(m["KL_uncached_to_cached_nats"] for m in comparisons),
                      "receipt_SHA256":hashlib.sha256(data).hexdigest(),
                      "production_authorized":False},sort_keys=True))
    if not passed:raise SystemExit(2)


if __name__=="__main__":
    main()
