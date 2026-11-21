#!/usr/bin/env python3
"""Compute genuine WikiText CALIBRATION one-next-token NLL from original AF logits.

Only 16 already-fixed calibration prompts. Never use heldout/OOD labels,
candidate posteriors, or the full target continuation. Does NOT establish
pre-registered 512-case Bayesian Expert improvement, NLL of full targets,
or any source-BF16 vs compressed-AF equivalence.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct

SCHEMA="beglin-je01-original-AF-calibration16-next-gold-token-NLL/1"
FULLLOGIT_RECEIPT="34da06b2441bd1cfa8c8593ff20e8c2d3e40850e1c0ef6d6c8a127d70cb75d14"
RAWLOGIT_SHA="428d2f985d1eca80ab299a7ab8cbf3d1330391efe46791f87d013a670f5d387a"
FROZEN_SOURCE="a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"
MAGIC=b"JE01_NATIVE_FULLLOGITS_F32_V1\n"
VOCAB=102400
EXPECTED_INPUTS=16


def canon(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                   allow_nan=False).encode()).hexdigest()

def pinned(path,expected):
    d=json.loads(Path(path).read_text())
    claimed=d.pop("receipt_sha256",None)
    if claimed!=expected or canon(d)!=expected:
        raise ValueError("JE01_MODEL_OR_GOLD_EVIDENCE_SHA_MISMATCH")
    return d


def read_vectors(path,logitsha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=logitsha:
        raise ValueError("JE01_NATIVE_ACTUAL_LOGIT_BYTE_SOURCE_DRIFT")
    if not raw.startswith(MAGIC):
        raise ValueError("JE01_NATIVE_LOGIT_MAGIC_DRIFT")
    p=len(MAGIC)
    vectors=[]
    for req in range(EXPECTED_INPUTS):
        if p+16>len(raw):raise ValueError("JE01_NATIVE_LOGIT_HEADER_TRUNCATED")
        r,pos,vocab,length=struct.unpack("<4I",raw[p:p+16])
        p+=16
        if r!=req or vocab!=VOCAB or pos!=length-1:
            raise ValueError("JE01_LOGIT_REQUEST_INDEX_DRIFT")
        if p+4*VOCAB>len(raw):raise ValueError("JE01_NATIVE_LOGIT_DATA_TRUNCATED")
        values=array("f")
        values.frombytes(raw[p:p+4*VOCAB])
        p+=4*VOCAB
        if any(not math.isfinite(x) for x in values):
            raise ValueError("JE01_NATIVE_NONFINITE_FULL_LOGITS")
        vectors.append((length,values))
    if p!=len(raw):raise ValueError("JE01_NATIVE_LOGIT_EXTRA_ROWS")
    return vectors


def next_token_nll(logits,target):
    if not isinstance(target,int) or isinstance(target,bool) or not 0<=target<len(logits):
        raise ValueError("JE01_GOLD_TOKEN_ID_INVALID")
    if any(not math.isfinite(x) for x in logits):
        raise ValueError("JE01_LOGITS_NOT_FINITE")
    peak=max(logits)
    total=math.fsum(math.exp(x-peak) for x in logits)
    if total<=0 or not math.isfinite(total):
        raise ValueError("JE01_LOGSUMEXP_INVALID")
    return peak+math.log(total)-logits[target]


def run(original_logits_root,gold_path,sha_expected,output_root):
    root=Path(original_logits_root).resolve(strict=True)
    output=Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("JE01_SCORE_TARGET_MUST_BE_NEW")
    if output==root or root in output.parents:
        raise ValueError("JE01_SCORE_ROOT_MUST_NOT_OVERWRITE_RAW_LOGITS")
    gold=pinned(gold_path,sha_expected)
    if (gold.get("prior_fixed_input_count")!=16 or
        gold.get("no_heldout_or_OOD_labels_accessed") is not True or
        gold.get("source_frozen_manifest_sha256")!=FROZEN_SOURCE or
        len(gold["records"])!=16):
        raise ValueError("JE01_REAL_CAL_GOLD_IDENTITY_INVALID")
    proof=pinned(root/"JE01_SHADOW_LOGITS_32_384_FINAL.json",FULLLOGIT_RECEIPT)
    if (proof.get("status")!="PASS_FULL_102400_LOGIT_PAIRED_COMPARISON" or
        proof["paired_full_logits"]["all_logits_compared"]!=16*VOCAB or
        proof["paired_full_logits"]["count_exact_different_float32_values"]!=0 or
        proof["run_32"]["raw_logits_sha256"]!=RAWLOGIT_SHA or
        proof["run_384"]["raw_logits_sha256"]!=RAWLOGIT_SHA):
        raise ValueError("JE01_FULL_LOGIT_SOURCE_PARITY_NOT_CERTIFIED")
    old=read_vectors(root/"run32/raw_full_logits.bin",RAWLOGIT_SHA)
    new=read_vectors(root/"run384/raw_full_logits.bin",RAWLOGIT_SHA)
    scored=[];unscoreable=[]
    for k,row in enumerate(gold["records"]):
        if row["request"]!=k or row["prompt_tokens_including_bos"]!=old[k][0]:
            raise ValueError("JE01_CALIBRATION_INPUT_ORDER_OR_LENGTH_DRIFT")
        if row["prompt_tokens_including_bos"]!=new[k][0]:
            raise ValueError("JE01_CONTEXT_ARM_TOKEN_LENGTH_DRIFT")
        target=row["first_public_gold_token_id"]
        if target is None:
            if row["full_sequence_tokenizer_preserves_prompt_prefix"] is not False:
                raise ValueError("JE01_UNEXPLAINED_MISSING_TARGET")
            unscoreable.append({"request":k,"reason":"TOKENIZER_PREFIX_BOUNDARY_MISMATCH"})
            continue
        if row["full_sequence_tokenizer_preserves_prompt_prefix"] is not True:
            raise ValueError("JE01_INVALID_BOUNDARY_PARITY")
        l32=next_token_nll(old[k][1],target)
        l384=next_token_nll(new[k][1],target)
        scored.append({"request":k,"input_id":row["input_id"],
                       "first_public_gold_token_id":target,
                       "native_prompt_sha256":row["native_prompt_sha256"],
                       "gold_first_token_NLL_32":l32,
                       "gold_first_token_NLL_384":l384,
                       "gold_NLL_new384_minus_old32":l384-l32})
    if not scored:raise ValueError("JE01_NO_VALID_NATIVE_GOLD_BOUNDARY")
    avg32=math.fsum(x["gold_first_token_NLL_32"] for x in scored)/len(scored)
    avg384=math.fsum(x["gold_first_token_NLL_384"] for x in scored)/len(scored)
    report={"schema":SCHEMA,"status":"REAL_ORIGINAL_PRETRAINED_CAL_NEXT_GOLD_NLL_PASS",
            "source_calibration_only":True,"original_16_full_logit_receipt_sha256":FULLLOGIT_RECEIPT,
            "true_original_compressed_AF_model":True,"source_gold_manifest_receipt_sha256":sha_expected,
            "frozen_public_train_calibration_only":True,
            "native_logit_vectors_per_arm":16,
            "target_token_alignment_valid":len(scored),
            "excluded_boundary_alignment_cases":unscoreable,
            "full_model_vocab":VOCAB,
            "mean_gold_one_token_nll_old32":avg32,
            "mean_gold_one_token_nll_new384":avg384,
            "mean_delta_new384_minus_old32":avg384-avg32,
            "max_case_abs_NLL_delta":max(abs(x["gold_NLL_new384_minus_old32"]) for x in scored),
            "scored_cases":scored,
            "original_512_full_target_NLL":"NOT_MEASURED",
            "independent_same_source_and_OOD_NLL":"NOT_MEASURED",
            "Bayesian_Expert_precision_improvement":"NOT_TESTED",
            "production_promotion_allowed":False,
            "production_weights_changed":False}
    report["receipt_sha256"]=canon(report)
    output.mkdir(mode=0o700)
    with (output/"JE01_ORIGINAL_CAL_NEXT_GOLD_NLL_PILOT.json").open("x") as file:
        file.write(json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--real-logits-root",required=True)
    p.add_argument("--gold-manifest",required=True)
    p.add_argument("--gold-receipt-sha256",required=True)
    p.add_argument("--new-output-root",required=True)
    a=p.parse_args()
    try:
        result=run(a.real_logits_root,a.gold_manifest,a.gold_receipt_sha256,a.new_output_root)
        print(json.dumps({"status":result["status"],
                          "real_scored_tokens":result["target_token_alignment_valid"],
                          "mean_nll_old32":result["mean_gold_one_token_nll_old32"],
                          "mean_nll_new384":result["mean_gold_one_token_nll_new384"],
                          "max_abs_NLL_delta":result["max_case_abs_NLL_delta"],
                          "receipt_sha256":result["receipt_sha256"]},sort_keys=True))
    except BaseException as exc:
        print(json.dumps({"status":"BLOCKED","error":type(exc).__name__,
                          "reason":str(exc)[:270]},sort_keys=True))
        raise SystemExit(2)
