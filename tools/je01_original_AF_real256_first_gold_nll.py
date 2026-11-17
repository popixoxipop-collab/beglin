#!/usr/bin/env python3
"""JE01 original pretrained BEGLIN AF: ACTUAL 256 train first-gold-token NLL.

Only previously frozen, full256 source-calibration inputs and public WikiText
train targets; neither original heldout split nor OOD data can enter this
process. Replays the 16 SHA-attested 16-token-batch original GPU outputs
(26-layer native full model, 102400 f32 logits at each prompt end) against
the first SHA-attested original gold continuation token. Never edits model
weights, precision map, user serving or frozen input. This is NOT the
preregistered full-target NLL; no Bayesian alternate bit width is evaluated.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import struct
import statistics
import sys

SOURCE_ROOT=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original_router256_20261118_v1")
GOLD_ROOT=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_train256_fullgold_NLL_20261118_v1")
OUTPUT=GOLD_ROOT/"JE01_ACTUAL_PRETRAINED_ORIGINAL_CAL256_NEXTGOLD_NLL.json"
SCORE_SCHEMA="beglin-je01-original-pretrained-AF-cal256-first-gold-NLL/1"
NATIVE_RECEIPT_SHA="62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb"
GOLD_RECEIPT_SHA="355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7"
ORIGINAL_CAL_MANIFEST_SHA="6674cefd09124f45599ba40a33a32ebf513968aa91f562f201f4e10e9f34cb3d"
ACTUAL_SOURCE_NATIVE_BINARY_SHA="37eccd3870e4b95be493ae18773497b8c6ab1ba54fff37d317477c0c7cfc78ad"
MAGIC=b"JE01_NATIVE_FULLLOGITS_F32_V1\n"
N=256
VOCAB=102400

def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                           allow_nan=False).encode()).hexdigest()

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):h.update(part)
    return h.hexdigest()

def source_receipt(path,expected):
    d=json.loads(Path(path).read_text())
    sig=d.pop("receipt_sha256",None)
    if sig!=expected or canonical(d)!=expected:
        raise ValueError("JE01_SOURCE_RECEIPT_CANONICAL_HASH_DRIFT:"+str(path))
    return d

def actual_logit_token_nll(values,target):
    if not isinstance(target,int) or isinstance(target,bool) or target<0 or target>=VOCAB:
        raise ValueError("JE01_GOLD_TOKEN_INVALID")
    if len(values)!=VOCAB or any(not math.isfinite(v) for v in values):
        raise ValueError("JE01_ORIGINAL_MODEL_FULL_LOGITS_NOT_FINITE")
    maximum=max(values)
    total=math.fsum(math.exp(x-maximum) for x in values)
    if total<=0 or not math.isfinite(total):
        raise ValueError("JE01_LOGSUMEXP_NOT_FINITE")
    result=maximum+math.log(total)-values[target]
    if not math.isfinite(result) or result<-1e-9:
        raise ValueError("JE01_INVALID_ORIGINAL_NLL")
    return result

def score():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError("JE01_ACTUAL_REAL_NLL_OUTPUT_MUST_BE_NEW")
    native=source_receipt(SOURCE_ROOT/"JE01_ACTUAL_ORIGINAL_256_ROUTER_CAL_FINAL.json",
                          NATIVE_RECEIPT_SHA)
    if native["status"]!="PASS_256_REAL_ORIGINAL_AF_ROUTER_LAST_PROMPT_ONLY" or \
       native["calibration_prompt_count"]!=N or \
       native["source_calibration256_manifest_sha256"]!=ORIGINAL_CAL_MANIFEST_SHA or \
       native["native_model_binary_sha256"]!=ACTUAL_SOURCE_NATIVE_BINARY_SHA or \
       len(native["all_chunks"])!=16 or \
       native["original_256_full_gold_NLL"]!="NOT_MEASURED" or \
       native["production_promotion_allowed"] is not False:
        raise ValueError("JE01_ORIGINAL_GPU_CAL256_PROVENANCE_MISSING")
    filenames=list(GOLD_ROOT.glob("JE01_ORIGINAL_REAL_TRAIN256_FULL_GOLD_IDS.json"))
    if len(filenames)!=1:raise ValueError("JE01_ORIGINAL_FULL_TARGET_MANIFEST_NOT_EXACTLY_ONE")
    gold=source_receipt(filenames[0],GOLD_RECEIPT_SHA)
    if gold["schema"]!="beglin-je01-original-deepseek-wikitext256-train-full-target-tokens/1" or \
       gold["status"]!="TRAIN256_FULL_GOLD_SOURCE_READY" or \
       gold["original_256_source_rows"]!=N or \
       gold["aligned_to_original_prompt_prefix_count"]!=N or \
       gold["same_source_validation_and_ood_labels_used"] is not False or \
       gold["real_gold_full_target_NLL"]!="NOT_MEASURED" or \
       len(gold["cases"])!=N:
        raise ValueError("JE01_SOURCE_TRAIN_GOLD_NOT_FROZEN")
    if sum(g["gold_token_count"] for g in gold["cases"])!=11024:
        raise ValueError("JE01_REAL_GOLD_TARGET_COUNTS_DRIFT")
    scores=[]
    raw_shas=[]
    for block in range(16):
        chunk=SOURCE_ROOT/("chunk"+str(block).zfill(2))
        c=source_receipt(chunk/"JE01_CHUNK_RECEIPT.json",
              native["all_chunks"][block]["receipt_sha256"])
        if c["chunk_index"]!=block or c["real_records"]!=16 or \
           len(c["sample_records"])!=16 or c["gold_labels_used"] is not False or \
           c["status"]!="PASS_REAL_ORIGINAL_PRETRAINED_ROUTER_AND_LOGITS":
            raise ValueError("JE01_ORIGINAL_ROUTER_BUNCH_METADATA_DRIFT")
        p=chunk/"full102400_logits.bin"
        digest=sha_file(p)
        if digest!=c["raw_original_logits_sha256"] or \
           digest!=native["all_chunks"][block]["native_logits_sha256"]:
            raise ValueError("JE01_ORIGINAL_FULL_LOGITS_FROM_GPU_HASH_DRIFT")
        raw_shas.append(digest)
        with p.open("rb") as f:
            if f.read(len(MAGIC))!=MAGIC:
                raise ValueError("JE01_NATIVE_ORIGINAL_LOGITS_FILE_MAGIC_DRIFT")
            for local in range(16):
                i=block*16+local
                header=f.read(16)
                if len(header)!=16:raise ValueError("JE01_ORIGINAL_LOGIT_ROW_MISSING")
                r,pos,vocab,plen=struct.unpack("<4I",header)
                m=c["sample_records"][local]
                g=gold["cases"][i]
                if r!=local or vocab!=VOCAB or pos+1!=plen or \
                   g["row"]!=i or g["input_id"]!=m["source_id"] or \
                   g["native_BOS_prompt_sha256"]!=m["input_file_sha256"] or \
                   g["original_BOS_prompt_positions"]!=plen or \
                   m["positions"]!=plen:
                    raise ValueError("JE01_ORIGINAL_CASE_ID_TOKEN_OFFSET_NOT_PAIRED")
                target=g["gold_continuation_token_ids"][0]
                raw=f.read(4*VOCAB)
                if len(raw)!=4*VOCAB:raise ValueError("JE01_ORIGINAL_FULL102400_ROW_TRUNCATED")
                vec=struct.unpack("<102400f",raw)
                label_nll=actual_logit_token_nll(vec,target)
                scores.append({"row":i,"source_id":g["input_id"],
                     "native_input_sha256":m["input_file_sha256"],
                     "full102400_logits_row_sha256":hashlib.sha256(header+raw).hexdigest(),
                     "prompt_token_length":plen,
                     "gold_target_first_token_id":target,
                     "gold_target_full_token_length":g["gold_token_count"],
                     "actual_original_next_gold_NLL":label_nll,
                     "predicted_first_token":m["next_token_argmax"],
                     "model_greedy_top1_is_gold":target==m["next_token_argmax"]})
            if f.read(1):raise ValueError("JE01_NATIVE_EXTRA_MODEL_LOGIT_ROWS")
        print(json.dumps({"stage":"REAL_ORIGINAL_TRAIN256_FIRSTGOLD_CHUNK_SCORED",
                          "chunk":block,"processed":(block+1)*16,
                          "chunk_logits_sha":digest}),flush=True)
    if len(scores)!=N or len({r["source_id"] for r in scores})!=N:
        raise ValueError("JE01_ORIGINAL_TRAIN256_DUPLICATE_OR_MISSING")
    vals=[r["actual_original_next_gold_NLL"] for r in scores]
    mean=math.fsum(vals)/N
    ordered=sorted(vals)
    receipt={"schema":SCORE_SCHEMA,
       "status":"PASS_REAL_ORIGINAL_AF_256_CAL_NEXT_GOLD_NLL",
       "real_original_native_gpu_model":True,
       "full_pretrained_model_layers":27,"model_vocabulary":VOCAB,
       "frozen_wikitext_train_calibration_only":True,"actual_calibration_prompts":N,
       "source_real_native_256_router_receipt_sha256":NATIVE_RECEIPT_SHA,
       "source_original_full_target_gold_sha256":GOLD_RECEIPT_SHA,
       "actual_native_256_last_prompt_logits_chunks_sha256":raw_shas,
       "actual_full_vocab_logits_values_scored":N*VOCAB,
       "total_full_gold_continuation_tokens_reserved":11024,
       "actually_scored_first_gold_tokens":N,
       "mean_actual_first_gold_token_NLL":mean,
       "median_actual_first_gold_token_NLL":statistics.median(vals),
       "p95_actual_first_gold_token_NLL":ordered[int(.95*N)-1],
       "min_actual_first_gold_token_NLL":ordered[0],
       "max_actual_first_gold_token_NLL":ordered[-1],
       "first_gold_greedy_top1_exact_cases":sum(r["model_greedy_top1_is_gold"] for r in scores),
       "scored_case_receipts":scores,
       "original_256_full_target_sequence_gold_NLL":"NOT_MEASURED",
       "same_source_128_holdout_gold_NLL":"NOT_MEASURED",
       "OOD_128_holdout_gold_NLL":"NOT_MEASURED",
       "Bayesian_Expert_precision_gain":"NOT_MEASURED",
       "expert_precision_alternatives_run":False,
       "original_checkpoint_or_serving_modified":False,
       "production_promotion_allowed":False}
    receipt["receipt_sha256"]=canonical(receipt)
    with OUTPUT.open("x") as f:
        f.write(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":receipt["status"],"receipts":N,
         "actual_original_first_gold_mean_NLL":mean,
         "greedy_first_gold_correct":receipt["first_gold_greedy_top1_exact_cases"],
         "pilot_sha256":receipt["receipt_sha256"],
         "full_sequence_NLL":"NOT_MEASURED"},sort_keys=True),flush=True)

if __name__=="__main__":
    try:score()
    except BaseException as e:
        print(json.dumps({"status":"BLOCKED","error_type":type(e).__name__,
             "reason":str(e)[:450],"production_promotion":False}),flush=True)
        sys.exit(2)
