#!/usr/bin/env python3
"""JE01 original pretrained DeepSeek 512 prompts -> isolated native 384 manifest.

Reconstructs pre-registered public source rows and verifies source revision,
source page SHA, 512 selected prompt SHA, original tokenizer SHA and the three
PREVIOUSLY OBSERVED original token-id collection SHAs before writing any input
files. Never selects by gold, reads label, changes prompt, truncates, or executes
the pretrained BEGLIN model.

Generated raw int32 little-endian input files are for BEGLIN's existing
QWEN_MOE_CB_PROMPT_MANIFEST, with its 384-pos B1 standalone research binary.
This does not score gold-token NLL; native runtime currently returns greedy
tokens rather than an auditable teacher-forced likelihood. That gate remains
NOT_IMPLEMENTED/NOT_RUN, regardless of manifest creation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct

import je01_data_freeze as data
from je01_native_token_feasibility import prompt_for

SCHEMA="beglin-je01-original-tokenizer-native384-512-inputs-stage-v1"
SOURCE_SHA="a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"
TOKENIZER_SHA="41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
CONFIG_SHA="f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"
BOS=100000
CAP=384
SOURCE_TOKENS={
  "calibration":"3d6cc45aa2e64da2f137bce24bc3a2f6df9869e0e2a351bf4979f601304719b5",
  "same_source_holdout":"6ce05e5a708c8e60a13850c43589b0a0cfc00669bdfd98784c5952305fc234df",
  "ood_holdout":"9c113946bcb4c45493e919b4d3aedbf34842f753e584caeafaa65b7be024d55d",
}

def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def verify_manifest(original):
    sig=original.get("manifest_sha256")
    if sig!=SOURCE_SHA:raise ValueError("JE01_PINNED_SOURCE_IDENTITY_MISMATCH")
    stripped={k:v for k,v in original.items() if k!="manifest_sha256"}
    if digest(stripped)!=SOURCE_SHA:raise ValueError("JE01_PINNED_SOURCE_CANONICAL_SHA_MISMATCH")
    if original.get("gold_labels_used_in_sampling") is not False:
        raise ValueError("JE01_GOLD_LABEL_LEAKAGE")
    if {k:g["count"] for k,g in original["cohorts"].items()}!={
         "calibration":256,"same_source_holdout":128,"ood_holdout":128}:
        raise ValueError("JE01_INVALID_COHORT_COUNTS")

def build(tokenizer_path,config_path,manifest_path,output_root):
    from tokenizers import Tokenizer
    tokenizer_path=Path(tokenizer_path)
    config_path=Path(config_path)
    manifest_path=Path(manifest_path)
    output_root=Path(output_root)
    if not tokenizer_path.is_file() or hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()!=TOKENIZER_SHA:
        raise ValueError("JE01_ORIGINAL_TOKENIZER_SHA_MISMATCH")
    if not config_path.is_file() or hashlib.sha256(config_path.read_bytes()).hexdigest()!=CONFIG_SHA:
        raise ValueError("JE01_ORIGINAL_MODEL_CONFIG_SHA_MISMATCH")
    cfg=json.loads(config_path.read_text())
    if cfg["bos_token_id"]!=BOS or cfg["model_type"]!="deepseek_v2" or cfg["vocab_size"]!=102400:
        raise ValueError("JE01_ORIGINAL_BOS_OR_VOCAB_DRIFT")
    original=json.loads(manifest_path.read_text())
    verify_manifest(original)
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("JE01_OUTPUT_ROOT_MUST_BE_NEW")
    if CAP <= 331 or CAP!=384:raise ValueError("JE01_384_CONTEXT_REQUIRED")
    plans=[(s,off) for s in data.SOURCES for off in data.planned_offsets(s)]
    pages={s[0]:[] for s in data.SOURCES}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(data.request_rows,s,off):s[0] for s,off in plans}
        for future in as_completed(futures):
            pages[futures[future]].append(future.result())
    frozen=data.freeze(pages)
    if frozen["manifest_sha256"]!=SOURCE_SHA or frozen!=original:
        raise ValueError("JE01_PUBLIC_SOURCE_ROWS_CHANGED")
    tok=Tokenizer.from_file(str(tokenizer_path))
    groups={}
    for cohort in ("calibration","same_source_holdout","ood_holdout"):
        rows={}
        for pg in pages[cohort]:
            for item in pg["row_records"]:
                idx=item["row_idx"]
                if idx in rows and rows[idx]!=item["row"]:
                    raise ValueError("JE01_DUPLICATE_SOURCE_ROW_CONTENT_CHANGED")
                rows[idx]=item["row"]
        recs=[]
        raw_tok_ids=[]
        for item in original["cohorts"][cohort]["records"]:
            prompt=prompt_for(data,cohort,rows[item["row_idx"]])
            if data.digest(prompt)!=item["prompt_sha256"]:
                raise ValueError("JE01_SELECTED_PROMPT_SHA_CHANGED")
            ids=tok.encode(prompt,add_special_tokens=False).ids
            raw_tok_ids.append(ids)
            if not ids or len(ids)+BOS//100000>=CAP:
                raise ValueError("JE01_PROMPT_EXCEEDS_384_WITH_BOS_AND_TARGET")
            native_ids=[BOS,*ids]
            if any(v<0 or v>=cfg["vocab_size"] for v in native_ids):
                raise ValueError("JE01_TOKEN_OUT_OF_VOCAB_RANGE")
            raw=struct.pack("<"+"i"*len(native_ids),*native_ids)
            recs.append({
                "input_id":item["source_row_id_sha256"],
                "prompt_sha256":item["prompt_sha256"],
                "token_count_with_bos":len(native_ids),
                "native_int32_sha256":hashlib.sha256(raw).hexdigest(),
                "native_int32_bytes":len(raw),"raw":raw,
            })
        if data.digest(raw_tok_ids)!=SOURCE_TOKENS[cohort]:
            raise ValueError("JE01_ORIGINAL_TOKENIZER_IDS_NOT_REPRODUCED:"+cohort)
        groups[cohort]=recs
    all_ids=[r["input_id"] for group in groups.values() for r in group]
    if len(all_ids)!=512 or len(set(all_ids))!=512:
        raise ValueError("JE01_COHORT_SPLIT_OVERLAP")
    # ALL verification is complete before creating any output.
    output_root.mkdir(mode=0o700,parents=False)
    try:
        attest={"schema":SCHEMA,"source_manifest_sha256":SOURCE_SHA,
                "model_config_sha256":CONFIG_SHA,"tokenizer_sha256":TOKENIZER_SHA,
                "bos_id":BOS,"native_context_positions":CAP,
                "artifacts":{},"cohorts":{},"no_labels_used":True,
                "real_model_forward":"NOT_RUN","native_gold_nll":"NOT_IMPLEMENTED",
                "production_write_allowed":False}
        for name,rows in groups.items():
            root=output_root/name
            root.mkdir(mode=0o700)
            lines=[]
            evidence=[]
            for item in rows:
                file=root/(item["input_id"]+".bin")
                with file.open("xb") as f:
                    f.write(item["raw"])
                # whitespace-safe path for C fscanf("%1023s %d")
                if len(str(file))>=900 or any(c.isspace() for c in str(file)):
                    raise ValueError("JE01_NATIVE_MANIFEST_PATH_NOT_SAFE")
                lines.append(str(file)+" 1\n")
                evidence.append({k:v for k,v in item.items() if k!="raw"})
            manifest="".join(lines)
            (root/"manifest.txt").write_text(manifest)
            attest["cohorts"][name]={
                "count":len(rows),"max_tokens_including_bos":max(x["token_count_with_bos"] for x in rows),
                "min_tokens_including_bos":min(x["token_count_with_bos"] for x in rows),
                "source_token_ids_sha256":SOURCE_TOKENS[name],
                "native_input_index_sha256":digest(evidence),
                "native_manifest_sha256":hashlib.sha256(manifest.encode()).hexdigest(),
                "file_receipts":evidence,
            }
        attest["total_input_records"]=len(all_ids)
        attest["receipt_sha256"]=digest(attest)
        (output_root/"JE01_NATIVE_384_INPUTS_RECEIPT.json").write_text(json.dumps(attest,sort_keys=True,indent=2)+"\n")
        return attest
    except BaseException:
        # Fail closed: do NOT declare a partial written output verified.
        (output_root/"JE01_INCOMPLETE_DO_NOT_SCORE").touch(exist_ok=True)
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("tokenizer","config","frozen","output"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    try:
        output=build(a.tokenizer,a.config,a.frozen,a.output)
        print(json.dumps({"status":"REAL_512_NATIVE_INPUTS_FROZEN_NOT_MODEL_RUN",
                          "counts":{k:v["count"] for k,v in output["cohorts"].items()},
                          "receipt_sha256":output["receipt_sha256"]},sort_keys=True))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error":type(e).__name__,"reason":str(e)[:250]}))
        raise SystemExit(2)
