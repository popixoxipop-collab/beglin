#!/usr/bin/env python3
"""Reconstruct official pinned Phi-4-mini second safetensors shard by HTTP ranges.

Used only if HF Xet downloader stalls. Downloads from Hugging Face official
resolve endpoint; validates total bytes, each Content-Range and final SHA-256
against the pinned repository LFS SHA256 (Xet ETag is a different hash). Leaves already cached first shard untouched.
No model execution, credentials, production changes or extra software.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request

MODEL="microsoft/Phi-4-mini-instruct"
REV="cfbefacb99257ffa30c83adab238a50856ac3083"
NAME="model-00002-of-00002.safetensors"
BASE=Path("/mnt/g/TailnetCommander/work/oml_acs_phi_range_cache_20261021")
DIR=BASE/"parts"
DONE=BASE/NAME
SNAP=Path("/mnt/g/TailnetCommander/work/oml_acs_hf/hub/models--microsoft--Phi-4-mini-instruct/snapshots")/REV
SNAP_FILE=SNAP/NAME
EXPECTED_SIZE=2768428504
EXPECTED_SHA256="7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983"
EXPECTED_XET_HASH="24804b7b84ebd0ff3c0bca109f7fc998b0a236cd31e1cc3177ff288c5a2c8754"
BLOCK=32*1024*1024
WORKERS=8
URL=f"https://huggingface.co/{MODEL}/resolve/{REV}/{NAME}"
UA={"User-Agent":"BEGLIN-G0-pinned-research-range-download/0.1"}

def sha_file(p: Path)->str:
    dig=hashlib.sha256()
    with p.open("rb") as src:
        while True:
            chunk=src.read(4*1024*1024)
            if not chunk:break
            dig.update(chunk)
    return dig.hexdigest()

def preflight()->None:
    req=urllib.request.Request(URL,headers=UA,method="HEAD")
    with urllib.request.urlopen(req,timeout=20) as resp:
        if resp.status!=200:
            raise RuntimeError("HEAD must return HTTP 200")
        etag=str(resp.headers.get("ETag","")).strip('"').lower()
        length=int(resp.headers.get("Content-Length","-1"))
        ranges=resp.headers.get("Accept-Ranges","")
        if length!=EXPECTED_SIZE or etag!=EXPECTED_XET_HASH or ranges!="bytes":
            raise RuntimeError("pinned model shard authority mismatch")
    if not (SNAP/"model-00001-of-00002.safetensors").is_file():
        raise RuntimeError("first shard incomplete or missing")
    if not (SNAP/"model.safetensors.index.json").is_file():
        raise RuntimeError("index missing")
    index=json.loads((SNAP/"model.safetensors.index.json").read_text())
    files=set(index["weight_map"].values())
    if files!={"model-00001-of-00002.safetensors",NAME}:
        raise RuntimeError("unexpected safetensors index file set")
    print("RANGE_PREFLIGHT_OK",EXPECTED_SIZE,EXPECTED_SHA256,flush=True)

def part_path(i:int)->Path:
    return DIR/f"{i:04d}.part"

def download_part(i:int)->int:
    begin=i*BLOCK
    end=min(EXPECTED_SIZE,begin+BLOCK)-1
    expected=end-begin+1
    dest=part_path(i)
    if dest.is_file() and dest.stat().st_size==expected:
        return expected
    if dest.exists():
        dest.unlink() # incomplete public-source part, safe to redownload
    for attempt in range(5):
        temp=dest.with_name(dest.name+f".{attempt}.tmp")
        try:
            req=urllib.request.Request(URL,headers={**UA,"Range":f"bytes={begin}-{end}"})
            with urllib.request.urlopen(req,timeout=75) as resp:
                if resp.status!=206:
                    raise RuntimeError("range request not honored")
                expected_range=f"bytes {begin}-{end}/{EXPECTED_SIZE}"
                if resp.headers.get("Content-Range")!=expected_range:
                    raise RuntimeError("wrong content range or file version")
                count=0
                with temp.open("xb") as f:
                    while True:
                        block=resp.read(1024*1024)
                        if not block:break
                        count+=len(block)
                        if count>expected:
                            raise RuntimeError("range overflow")
                        f.write(block)
                    f.flush();os.fsync(f.fileno())
            if count!=expected:
                raise RuntimeError(f"short read: {count} != {expected}")
            temp.replace(dest)
            return expected
        except Exception as err:
            if temp.exists():temp.unlink()
            if attempt==4:raise
            time.sleep(0.6*(attempt+1))
    raise RuntimeError("unreachable")

def main():
    start=time.monotonic()
    preflight()
    os.umask(0o077)
    BASE.mkdir(parents=True,exist_ok=True)
    DIR.mkdir(parents=True,exist_ok=True)
    if SNAP_FILE.exists():
        if SNAP_FILE.stat().st_size!=EXPECTED_SIZE or sha_file(SNAP_FILE)!=EXPECTED_SHA256:
            raise RuntimeError("snapshot has conflicting shard")
        print("RANGE_ALREADY_FINISHED",flush=True)
        return
    total=(EXPECTED_SIZE+BLOCK-1)//BLOCK
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i,size in enumerate(pool.map(download_part,range(total)),start=1):
            if i%10==0 or i==total:
                print("RANGE_PROGRESS",i,total,round(time.monotonic()-start,1),flush=True)
    temp=DONE.with_name(DONE.name+".assembling")
    if temp.exists():
        written=temp.stat().st_size
        final_digest=sha_file(temp)
        if written!=EXPECTED_SIZE or final_digest!=EXPECTED_SHA256:
            raise RuntimeError("preexisting assembly has wrong LFS SHA256; refuse adoption")
        print("RECOVERED_VERIFIED_ASSEMBLY",written,final_digest,flush=True)
    else:
        h=hashlib.sha256()
        written=0
        with temp.open("xb") as output:
            for i in range(total):
                with part_path(i).open("rb") as f:
                    while True:
                        part=f.read(4*1024*1024)
                        if not part:break
                        written+=len(part)
                        h.update(part)
                        output.write(part)
            output.flush();os.fsync(output.fileno())
        final_digest=h.hexdigest()
        if written!=EXPECTED_SIZE or final_digest!=EXPECTED_SHA256:
            raise RuntimeError("final pinned official LFS SHA256 or length mismatch")
    temp.replace(DONE)
    if SNAP_FILE.exists():
        raise RuntimeError("snapshot concurrently changed; manual reconciliation required")
    os.link(DONE,SNAP_FILE) # Same G volume: no extra physical copy.
    if not SNAP_FILE.is_file() or SNAP_FILE.stat().st_size!=EXPECTED_SIZE or sha_file(SNAP_FILE)!=EXPECTED_SHA256:
        raise RuntimeError("snapshot linking verification failed")
    # Parts are now redundant. Remove only known-created pieces under our work root.
    for i in range(total):
        part_path(i).unlink()
    rec={
       "schema":"oml-acs-phi-pinned-range-cache/1","status":"SHA256_VERIFIED_COMPLETE",
       "model":MODEL,"revision":REV,"file":NAME,"size_bytes":written,
       "file_sha256":final_digest,"expected_xet_hash":EXPECTED_XET_HASH,"recovered_chunks":total,"chunk_bytes":BLOCK,
       "workers":WORKERS,"elapsed_seconds":round(time.monotonic()-start,2),
       "both_model_shards_present":len(list(SNAP.glob("*.safetensors")))==2,
       "operational_mutation":False,"paid_api":False,
    }
    print("PHI_RANGE_CACHE_RECEIPT",json.dumps(rec,sort_keys=True),flush=True)
if __name__=="__main__":main()
