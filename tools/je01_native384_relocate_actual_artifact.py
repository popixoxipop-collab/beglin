#!/usr/bin/env python3
"""JE01: transport/relocate 512 native DeepSeek int32 files safely.

The original GitHub hosted native manifest embeds runner-specific absolute
paths. It cannot be used directly after moving that archive to EOE. This
tool checks the source provenance seal and every raw binary *before* creating
standalone, destination-local manifest files with correct absolute filenames.
Original native bytes are never rewritten and no BEGLIN model is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

SOURCE_RECEIPT_SHA="2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34"
SOURCE_MANIFEST_SHA="a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"
ORIG_TOKENIZER_SHA="41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
COHORTS={"calibration":256,"same_source_holdout":128,"ood_holdout":128}
SCHEMA="beglin-je01-native384-move-verified-manifests-v1"


def canonical(doc):
    return hashlib.sha256(json.dumps(doc,sort_keys=True,
                        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def verify_file(path,entry,root):
    if not isinstance(entry,dict):raise ValueError("JE01_RELOCATE_INVALID_ENTRY")
    identity=entry.get("input_id")
    if not isinstance(identity,str) or re.fullmatch(r"[0-9a-f]{64}",identity) is None:
        raise ValueError("JE01_RELOCATE_INVALID_INPUT_ID")
    if path.is_symlink() or not path.is_file() or path.parent.resolve()!=root.resolve():
        raise ValueError("JE01_RELOCATE_SOURCE_PATH_ESCAPE")
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=entry["native_int32_sha256"]:
        raise ValueError("JE01_RELOCATE_SOURCE_CONTENT_SHA_CHANGED")
    if len(content)!=entry["native_int32_bytes"] or len(content)%4:
        raise ValueError("JE01_RELOCATE_NATIVE_BYTE_LEN_DRIFT")
    ids=struct.unpack("<"+"i"*(len(content)//4),content)
    if (not 2<=len(ids)<384 or ids[0]!=100000 or
        len(ids)!=entry["token_count_with_bos"] or
        any(x<0 or x>=102400 for x in ids)):
        raise ValueError("JE01_RELOCATE_SOURCE_TOKENS_INVALID")
    return len(content)


def relocate(source,output):
    source=Path(source).resolve(strict=True)
    target=Path(output).resolve(strict=False)
    if not source.is_dir() or target.exists() or target.is_symlink() or \
       source==target or source in target.parents or target in source.parents:
        raise ValueError("JE01_RELOCATE_TARGET_MUST_BE_NEW_OUTSIDE_SOURCE")
    p=source/"JE01_NATIVE_384_INPUTS_RECEIPT.json"
    if p.is_symlink() or not p.is_file():raise ValueError("JE01_SOURCE_RECEIPT_MISSING")
    report=json.loads(p.read_text())
    sig=report.pop("receipt_sha256",None)
    if sig!=SOURCE_RECEIPT_SHA or canonical(report)!=SOURCE_RECEIPT_SHA:
        raise ValueError("JE01_SOURCE_RECEIPT_SHA_DRIFT")
    if report["source_manifest_sha256"]!=SOURCE_MANIFEST_SHA or \
       report["tokenizer_sha256"]!=ORIG_TOKENIZER_SHA or \
       report["total_input_records"]!=512 or report["bos_id"]!=100000 or \
       report["native_context_positions"]!=384 or \
       report["real_model_forward"]!="NOT_RUN" or \
       report["native_gold_nll"]!="NOT_IMPLEMENTED":
        raise ValueError("JE01_RELOCATE_FROZEN_SOURCE_CONTRACT_DRIFT")
    if {k:v["count"] for k,v in report["cohorts"].items()}!=COHORTS:
        raise ValueError("JE01_RELOCATE_SPLIT_COUNTS_DRIFT")
    manifest_outputs={}
    seen=set()
    for cohort,expected in COHORTS.items():
        group=report["cohorts"][cohort]
        records=group["file_receipts"]
        if len(records)!=expected:raise ValueError("JE01_RELOCATE_RECORD_COUNT_DRIFT")
        file_strings=[]
        canonical_entries=[]
        for entry in records:
            identity=entry["input_id"]
            if identity in seen:raise ValueError("JE01_RELOCATE_DUPLICATE_INPUT_ID")
            seen.add(identity)
            path=source/cohort/(identity+".bin")
            verify_file(path,entry,source/cohort)
            filename=str(path)
            if len(filename)>=900 or any(c.isspace() for c in filename):
                raise ValueError("JE01_RELOCATE_WHITESPACE_OR_OVERLONG_PATH")
            file_strings.append(filename+" 1\n")
            canonical_entries.append(entry)
        # Cohort identity is preserved independent of source archive path.
        import je01_data_freeze as data
        if data.digest(canonical_entries)!=group["native_input_index_sha256"]:
            raise ValueError("JE01_RELOCATE_INPUT_INDEX_HASH_DRIFT")
        manifest_outputs[cohort]="".join(file_strings)
    if len(seen)!=512:raise ValueError("JE01_RELOCATE_TOTAL_IDS_DRIFT")
    # ALL 512 raw bytes + immutable original cohort identity verified before writes.
    target.mkdir(mode=0o700,parents=False)
    receipt={
      "schema":SCHEMA,"status":"RELOCATED_512_REAL_NATIVE_FILES_MANIFESTS_PASS",
      "source_receipt_sha256":SOURCE_RECEIPT_SHA,
      "dataset_manifest_sha256":SOURCE_MANIFEST_SHA,
      "original_tokenizer_sha256":ORIG_TOKENIZER_SHA,
      "files_still_in_source_artifact_root":True,
      "source_data_directory":str(source),"manifest_directory":str(target),
      "files":512,"cohorts":{},
      "original_model_forward":"NOT_RUN","gold_token_NLL":"NOT_MEASURED",
      "original_data_modified":False,"production_promotion_allowed":False,
    }
    for cohort,manifest in manifest_outputs.items():
        with (target/(cohort+".manifest")).open("x") as f:f.write(manifest)
        receipt["cohorts"][cohort]={
          "count":COHORTS[cohort],
          "absolute_manifest_sha256":hashlib.sha256(manifest.encode()).hexdigest(),
          "unmodified_input_index_sha256":report["cohorts"][cohort]["native_input_index_sha256"],
        }
    receipt["receipt_sha256"]=canonical(receipt)
    with (target/"JE01_RELOCATION_PASS.json").open("x") as f:
        f.write(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--artifact-input-root",required=True)
    p.add_argument("--new-manifests-directory",required=True)
    a=p.parse_args()
    try:
        report=relocate(a.artifact_input_root,a.new_manifests_directory)
        print(json.dumps({"status":report["status"],"files":report["files"],
                          "receipt_sha256":report["receipt_sha256"]},sort_keys=True))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error_type":type(e).__name__,
                          "reason":str(e)[:240]},sort_keys=True))
        raise SystemExit(2)
