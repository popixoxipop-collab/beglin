#!/usr/bin/env python3
"""Independent raw-binary re-audit of original AF pretrained full-target NLL.

Run after the 256 TRAIN-only teacher-forced Metal shadow has completely
finished. Reopen all 16 native 32-byte-per-gold-token output files, validate
every actual gold token ID and position against the separately SHA-sealed
original WikiText TRAIN256 tokenizer source, then recompute total NLL,
per-example mean and 20,000 case-bootstrap baseline uncertainty. NEVER
interpret baseline CI as Bayesian relative gain or access heldout/OOD labels.
"""
from __future__ import annotations
import hashlib,json,math,random,statistics
from pathlib import Path
import struct,sys

ROOT=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original_fulltarget_teacherforce_20261118_v2")
SOURCES=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_train256_fullgold_NLL_20261118_v1")
GOLD=SOURCES/"JE01_ORIGINAL_REAL_TRAIN256_FULL_GOLD_IDS.json"
PREVIOUS=SOURCES/"JE01_ACTUAL_PRETRAINED_ORIGINAL_CAL256_NEXTGOLD_NLL.json"
SCHEMA="beglin-je01-native-original-AF-train256-fullgold-NLL-independent-audit/1"
MAGIC=b"JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1\n"
GOLD_SHA="355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7"
FIRST_SHA="aca35d52a67c8ce5854c1b7297f80b241df2b2d66823b75a98744123fe8169b4"
BOOT_SEED=20261118
RESAMPLES=20000

def canonical(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),
                allow_nan=False).encode()).hexdigest()

def load_seal(path,expected=None):
    raw=json.loads(Path(path).read_text())
    sig=raw.pop("receipt_sha256")
    if canonical(raw)!=sig or expected is not None and sig!=expected:
        raise ValueError("JE01_ORIGINAL_REAL_SOURCE_CANONICAL_SHA_CHANGED")
    return raw,sig

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def percentile(values,p):
    arr=sorted(values)
    return arr[min(len(arr)-1,max(0,int(p*len(arr))-1))]

def resample_ci(rows,n=RESAMPLES,seed=BOOT_SEED):
    if len(rows)!=256:raise ValueError("JE01_BOOTSTRAP_NOT_256_PROMPTS")
    rng=random.Random(seed)
    means=[]
    for _ in range(n):
        loss=0.;count=0
        for _ in range(256):
            choice=rows[rng.randrange(256)]
            loss+=choice["nll_sum"];count+=choice["tokens"]
        means.append(loss/count)
    return {"bootstrap_seed":seed,"bootstrap_independent_units":"256 TRAIN source prompts",
            "n_case_resamples":n,
            "real_source_train_mean_gold_token_NLL_CI_95":
                [percentile(means,.025),percentile(means,.975)],
            "not_paired_precision_allocation_improvement":True}

def audit():
    end=ROOT/"JE01_ORIGINAL_AF_FULL_TARGET_GOLD_NLL_ALL256.json"
    final,sig=load_seal(end)
    if final["status"]!="REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_CAL256_PASS" or \
       final["n_calibration_train_cases"]!=256 or \
       final["gold_full_target_tokens_scored"]!=11024 or \
       final["production_before"]!=final["production_after"] or \
       final["production_promotion_allowed"] is not False or \
       final["trained_alt_bits_comparison"]!="NOT_MEASURED":
        raise ValueError("JE01_ORIGINAL_FULL_GOLD_256_RESULT_NOT_FINAL")
    gold,gsha=load_seal(GOLD,GOLD_SHA)
    previous,psha=load_seal(PREVIOUS,FIRST_SHA)
    if len(gold["cases"])!=256 or \
       sum(x["gold_token_count"] for x in gold["cases"])!=11024 or \
       len(previous["scored_case_receipts"])!=256:
        raise ValueError("JE01_ORIGINAL_GOLD_SOURCE_ROWS_CHANGED")
    if len(final["per_chunk_native_output_sha256"])!=16 or \
       len(final["per_case_real_NLL"])!=256:
        raise ValueError("JE01_NATIVE_NLL_RECEIPT_MISSING_CHUNKS_OR_ROWS")
    events=[]
    count=0
    chunk_summaries=[]
    for block in range(16):
        folder=ROOT/("chunk"+str(block).zfill(2))
        c=final["per_chunk_native_output_sha256"][block]
        if c["chunk"]!=block or sha(folder/"gold_native_nll.bin")!=c["result_sha256"]:
            raise ValueError("JE01_ACTUAL_NATIVE_NLL_OUTPUT_CHUNK_SHA_DRIFT")
        raw=(folder/"gold_native_nll.bin").read_bytes()
        if not raw.startswith(MAGIC):
            raise ValueError("JE01_ORIGINAL_TEACHERFORCED_GPU_OUTPUT_MAGIC_DRIFT")
        pos=len(MAGIC);n_scored=0
        for local in range(16):
            n=block*16+local
            g=gold["cases"][n]
            ref=final["per_case_real_NLL"][n]
            prev=previous["scored_case_receipts"][n]
            if ref["row"]!=n or ref["source_id"]!=g["input_id"] or \
               prev["source_id"]!=g["input_id"] or \
               ref["tokens"]!=g["gold_token_count"]:
                raise ValueError("JE01_TRUE_GOLD_SAMPLE_UID_CHANGED")
            losses=[]
            for k,token in enumerate(g["gold_continuation_token_ids"]):
                chunk=raw[pos:pos+32]
                if len(chunk)!=32:raise ValueError("JE01_ACTUAL_MODEL_NLL_TOKEN_MISSING")
                rr,tidx,target,position=struct.unpack("<4I",chunk[:16])
                nll,greedy,goldlogit=struct.unpack("<dIf",chunk[16:])
                if rr!=local or tidx!=k or target!=token or \
                   position!=g["original_BOS_prompt_positions"]+k-1 or \
                   not math.isfinite(nll) or nll < -1e-7 or \
                   not math.isfinite(goldlogit) or greedy>=102400:
                    raise ValueError("JE01_REAL_NATIVE_TEACHER_FORCE_TOKEN_ALIGNMENT_DRIFT")
                losses.append(nll)
                pos+=32
                count+=1;n_scored+=1
            nllsum=math.fsum(losses)
            if abs(nllsum-ref["nll_sum"])>1e-8 or \
               abs(nllsum/len(losses)-ref["nll_mean"])>1e-9 or \
               abs(losses[0]-ref["first_gold_nll"])>1e-9 or \
               abs(losses[0]-prev["actual_original_next_gold_NLL"])>1e-5:
                raise ValueError("JE01_FULL_TARGET_NLL_CASE_VALUES_NOT_REPRODUCED")
            events.append({"row":n,"input_id":g["input_id"],
                           "tokens":len(losses),"nll_sum":nllsum,
                           "mean_nll":nllsum/len(losses),
                           "first_gold_nll_delta_from_prior":losses[0]-prev["actual_original_next_gold_NLL"]})
        if pos!=len(raw) or n_scored!=c["gold_targets_scored"]:
            raise ValueError("JE01_ORIGINAL_NATIVE_NLL_FILE_EXTRA_OR_MISSING_TARGETS")
        chunk_summaries.append({"chunk":block,"data_sha256":c["result_sha256"],
                                "actual_gold_tokens_independently_checked":n_scored})
    if count!=11024 or len(events)!=256:
        raise ValueError("JE01_ORIGINAL_ALL_TARGET_TOKENS_NOT_SCORED")
    weighted=math.fsum(e["nll_sum"] for e in events)/count
    unweighted=math.fsum(e["mean_nll"] for e in events)/256
    if abs(weighted-final["mean_gold_token_NLL"])>1e-9 or \
       abs(unweighted-final["per_example_mean_full_target_NLL"])>1e-9:
        raise ValueError("JE01_REAL_FULL_GOLD_NLL_GLOBAL_MEAN_NOT_REPRODUCED")
    bootstrap=resample_ci(events)
    return {"schema":SCHEMA,
        "status":"INDEPENDENT_256_NATIVE_REAL_FULL_TARGET_GOLD_NLL_PASS",
        "real_original_af_pretrained_model":True,
        "full_source_native_result_receipt_sha256":sig,
        "original_256_train_gold_source_receipt_sha256":gsha,
        "prior_real_256_first_gold_NLL_receipt_sha256":psha,
        "actually_reopened_16_native_GPU_result_files":16,
        "all_256_original_train_prompts_checked":256,
        "all_11024_real_gold_tokens_independently_checked":count,
        "recomputed_weighted_mean_full_target_NLL":weighted,
        "recomputed_mean_per_train_prompt_full_target_NLL":unweighted,
        "actual_full_target_per_prompt_NLL_p05":percentile([e["mean_nll"] for e in events],.05),
        "actual_full_target_per_prompt_NLL_median":statistics.median(e["mean_nll"] for e in events),
        "actual_full_target_per_prompt_NLL_p95":percentile([e["mean_nll"] for e in events],.95),
        "max_abs_first_gold_baseline_NLL_discrepancy":max(abs(e["first_gold_nll_delta_from_prior"]) for e in events),
        "per_chunk_native_original_GPU_NLL_SHA":chunk_summaries,
        "bootstrap":bootstrap,
        "no_same_source_or_ood_gold_labels_used":True,
        "native_Bayesian_expert_bit_quality_alternatives":"NOT_MEASURED",
        "baseline_CI_is_not_a_precision_improvement_CI":True,
        "production_weight_change":False,"production_promotion_allowed":False}

if __name__=="__main__":
    try:
        result=audit()
        result["receipt_sha256"]=canonical(result)
        p=ROOT/"JE01_REAL_NATIVE_ORIGINAL_256_FULL_GOLD_INDEPENDENT_AUDIT.json"
        with p.open("x") as f:json.dump(result,f,sort_keys=True,indent=2)
        print(json.dumps({"status":result["status"],
            "receipt_sha256":result["receipt_sha256"],
            "real_gold_tokens":result["all_11024_real_gold_tokens_independently_checked"],
            "train_NLL":result["recomputed_weighted_mean_full_target_NLL"],
            "baseline_bootstrap_95pct":result["bootstrap"]["real_source_train_mean_gold_token_NLL_CI_95"],
            "Bayes_bit_gain":"NOT_MEASURED"},sort_keys=True))
    except BaseException as err:
        print(json.dumps({"status":"BLOCKED","type":type(err).__name__,
                          "reason":str(err)[:420]},sort_keys=True))
        sys.exit(2)
