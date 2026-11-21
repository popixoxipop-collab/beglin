#!/usr/bin/env python3
"""JE01 EOE D50 actual original pretrained Q4/Q5 physical source stage audit.

Run AFTER approved staged-only GitHub downloads to a NEW research directory.
Never modifies original AF, BF16, actual staged binary, existing NIG evidence,
release, server or operating weights. No model forward in this stage.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path

Q45_BYTES=43253760
BUNDLE_SHA="b4bd15db43f60f5aeb3ae8d6b9af148fa38545e5289ac049cd26a57a82932e6b"
BUNDLE_RECEIPT_SHA="f82645bcff33fa2f7268c1fb7e2797cbc20cbd667a2cd8556e9adc01e20204e3"
BINARY_SHA="1eb8dd5311cc23efba11a54a2891e9ec77fc9bb5cb88cfb60bdd6183310059ab"
Q4_SLOTS={(13,22),(13,28),(14,63),(24,60)}
Q5_SLOTS={(12,13),(20,17),(21,61),(24,26)}
ROLES=("gate_proj","up_proj","down_proj")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def stage(root):
    if not root.is_dir() or root.is_symlink():raise ValueError("JE01_Q45_ROOT_MISSING")
    src_candidates=[root/"weights",root/"weights"/"je01_real_q45"]
    matches=[p for p in src_candidates if p.is_dir() and not p.is_symlink() and
        (p/"JE01_REAL_Q45_EQUALBYTE_RECEIPT.json").is_file()]
    if len(matches)!=1:raise ValueError("JE01_Q45_SOURCE_ROOT_UNKNOWN_OR_AMBIGUOUS")
    path=matches[0];rec=path/"JE01_REAL_Q45_EQUALBYTE_RECEIPT.json"
    r=json.loads(rec.read_text())
    signed=r.pop("receipt_sha256",None)
    if signed!=BUNDLE_RECEIPT_SHA or canonical(r)!=BUNDLE_RECEIPT_SHA or \
        r["status"]!="ORIGINAL_REAL_BF16_Q4G64_Q5G64_12EACH_43MB_SHA_PASS" or \
        r["exact_original_AF_eight_expert_budget_bytes"]!=Q45_BYTES or \
        r["actual_full_binary_bytes"]!=Q45_BYTES or \
        r["q4_role_count"]!=12 or r["q5_role_count"]!=12 or \
        r["actual_full_binary_sha256"]!=BUNDLE_SHA or \
        r["scientific_scope"]!="POST_NIG_16TRAIN_PILOT_EXPLORATORY_CHALLENGER_NOT_CONFIRMATORY" or \
        r["original_BF16_Chat_index_sha256"]!="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf" or \
        r["old_real_Q5_physical_source_receipt"]!="04ddca2f84367f41098357b5fae905e76c1207ef023d648a9f388aaddb3bb0ca":
        raise ValueError("JE01_Q45_ORIGINAL_SOURCE_IMMUTABLE_RECEIPT_CHANGED")
    blob=path/"JE01_SOURCE_REAL_Q4Q5_12_12_EXACT_BYTE.bin"
    tsv=path/"JE01_SOURCE_REAL_Q4Q5_24_ROLES.tsv"
    exe=root/"bin"/"je01_eoe_original384_Q45_fullgold"
    binreceipt=root/"bin"/"JE01_EOE_REAL_Q45_NATIVE_BUILD_RECEIPT.json"
    if any(f.is_symlink() or not f.is_file() for f in (blob,tsv,exe,binreceipt)) or \
       blob.stat().st_size!=Q45_BYTES or sha(blob)!=BUNDLE_SHA or \
       sha(tsv)!=r["real_native_tsv_sha256"] or sha(exe)!=BINARY_SHA:
        raise ValueError("JE01_Q45_REAL_PHYSICAL_FILE_SHA_FAIL")
    native=json.loads(binreceipt.read_text())
    if native["status"]!="REAL_ORIGINAL384_Q4SYM12_Q5SYM12_NATIVE_FULLGOLD_COMPILED" or \
       native["binary_sha256"]!=BINARY_SHA or native["operating_model_modified"] or \
       native["source_patch"]["status"]!="JE01_Q45_ORIGINAL384_B1_Q4SYMMETRIC_Q5_NATIVE_SCRATCH_ONLY_PATCH" or \
       native["real_gold_NLL"]!="NOT_MEASURED":
        raise ValueError("JE01_Q45_NATIVE_BINARY_BUILD_RECEIPT_WRONG")
    spans=[]
    for v in r["physical_regions"]:
        start,end=v["offsets"]
        if not 0<=start<end<=Q45_BYTES:
            raise ValueError("JE01_Q45_PHYSICAL_REGION_OVERFLOW")
        spans.append((start,end))
    spans.sort()
    if len(spans)!=48 or spans[0][0]!=0 or spans[-1][1]!=Q45_BYTES or \
       any(spans[i][1]!=spans[i+1][0] for i in range(47)):
        raise ValueError("JE01_Q45_REAL_48_REGION_GAP_OR_OVERLAP")
    rows=[]
    for line in tsv.read_text().splitlines():
        a=line.split("\t")
        if len(a)!=10:raise ValueError("JE01_Q45_TSV_TEN_COLUMNS")
        layer,expert,role,bits,out,inn,pb,pe,sb,se=a
        layer,expert,bits,out,inn,pb,pe,sb,se=map(int,(layer,expert,bits,out,inn,pb,pe,sb,se))
        if role not in ROLES or bits not in (4,5) or \
           ((layer,expert) in Q4_SLOTS)!=(bits==4) or \
           (layer,expert) not in Q4_SLOTS|Q5_SLOTS:
            raise ValueError("JE01_Q45_REAL_PRESELECTED_ROLE_BIT_DRIFT")
        if (out,inn)!=((2048,1408) if role=="down_proj" else (1408,2048)) or \
           pe-pb!=out*inn*bits//8 or se-sb!=out*(inn//64)*4:
            raise ValueError("JE01_Q45_REAL_ROLE_MATRIX_BYTES_INVALID")
        rows.append((layer,expert,role,bits))
    if len(rows)!=24 or len(set(x[:3] for x in rows))!=24 or \
       sum(x[3]==4 for x in rows)!=12 or sum(x[3]==5 for x in rows)!=12:
        raise ValueError("JE01_Q45_SOURCE_ROLE_COVERAGE_INCOMPLETE")
    os.chmod(exe,0o700)
    result={"schema":"beglin-je01-eoe-true-original-BF16-Q45-native-stage/1",
         "status":"JE01_EOE_REAL_Q45_12ROLE4_12ROLE5_SOURCE_STAGE_PASS",
         "source_receipt_sha256":BUNDLE_RECEIPT_SHA,
         "source_Q45_SHA256":BUNDLE_SHA,
         "binary_SHA256":BINARY_SHA,
         "full_43MB_byte_size":Q45_BYTES,
         "real_role_count":len(rows),
         "real_q4_roles":12,"real_q5_roles":12,
         "bundle_path":str(blob),
         "tsv_path":str(tsv),
         "binary_path":str(exe),
         "native_role_tsv_sha256":r["real_native_tsv_sha256"],
         "scientific_scope":"POST_NIG_16TRAIN_PILOT_EXPLORATORY_CHALLENGER_NOT_CONFIRMATORY",
         "original_AF4_real_source_sha256":"9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4",
         "full_model_NLL":"NOT_MEASURED",
         "production_modified":False}
    result["receipt_sha256"]=canonical(result)
    target=root/"JE01_REAL_Q45_EOE_SOURCE_STAGE.json"
    with target.open("x") as f:
        f.write(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source-root",required=True,type=Path)
    args=p.parse_args()
    try:
        r=stage(args.source_root)
        print(json.dumps({"status":r["status"],"receipt_sha256":r["receipt_sha256"],
                          "binary_SHA256":r["binary_SHA256"],"source_Q45_SHA256":r["source_Q45_SHA256"],
                          "real_roles":r["real_role_count"]},sort_keys=True),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:340]},sort_keys=True),flush=True)
        raise SystemExit(2)
