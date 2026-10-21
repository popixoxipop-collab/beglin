#!/usr/bin/env python3
"""C1-D prospective-development source collector, train-only independent datasets.

Pinned IMDB train and GoEmotions simplified train. No public test splits,
no target-model scores observed before selection. All row contents private.
"""
from __future__ import annotations
import collections,hashlib,json,os,re,time
from pathlib import Path
import numpy as np

ROOT=Path.home()/".local/share/beglin/oml_acs/20261021"
CACHE="/mnt/g/TailnetCommander/work/oml_acs_c1d_dataset_cache"
OLD_ANCHORS=ROOT/"anchors_300_local.jsonl"
OLD_BEHAVIOR=ROOT/"behavior_sst2_carer_train_dev_v1.jsonl"
OLD_ANCHOR_SHA="47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
OLD_BEHAVIOR_SHA="b99fb5d7e8c909e6821f555ac590b36ff52393b9d4e7a01828ee936c615bd274"
FILE=ROOT/"C1D_NEW_PROSPECTIVE_DEV_200_20261021.jsonl"
RECEIPT=ROOT/"C1D_NEW_PROSPECTIVE_DEV_200_20261021_receipt.json"
N=50;SEED=73
SPEC=(
 dict(axis="sst2_positive_vs_negative",repo="stanfordnlp/imdb",
 revision="e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
 file="plain_text/train-00000-of-00001.parquet",col="text",label_col="label",
 positive=1,negative=0,limit=1000,license="other_review_pending"),
 dict(axis="carer_joy_vs_anger",repo="google-research-datasets/go_emotions",
 revision="add492243ff905527e67aeb8b80c082af02207c3",
 file="simplified/train-00000-of-00001.parquet",col="text",label_col="labels",
 positive=17,negative=2,limit=300,license="apache-2.0"),
)

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def tidy(text):
    if not isinstance(text,str):return ""
    return " ".join(re.sub(r"<br\s*/?>"," ",text,flags=re.I).split()).strip()
def normalized(text):return " ".join(tidy(text).casefold().split())
def canonical_sha(text):return sha(normalized(text).encode("utf-8"))
def stable(row):return sha(json.dumps(row,ensure_ascii=True,sort_keys=True,separators=(",",":")).encode())
def load_previous():
    for p,expected in [(OLD_ANCHORS,OLD_ANCHOR_SHA),(OLD_BEHAVIOR,OLD_BEHAVIOR_SHA)]:
        if not p.is_file() or p.stat().st_mode&0o777!=0o600 or sha(p.read_bytes())!=expected:
            raise RuntimeError("previous private source identity or mode mismatch")
    prev=[]
    for p in (OLD_ANCHORS,OLD_BEHAVIOR):
        prev.extend(canonical_sha(json.loads(z)["text"]) for z in p.read_text().splitlines() if z.strip())
    if len(prev)!=len(set(prev)): raise RuntimeError("old corpus canonical overlap must be explicitly reviewed")
    return set(prev)

def pick(spec,rows,rng,occupied):
    buckets={0:[],1:[]};attempts=0
    for idx in rng.permutation(len(rows)):
        x=rows[int(idx)];attempts+=1
        label=x[spec["label_col"]]
        if spec["repo"].endswith("go_emotions"):
            if not isinstance(label,(tuple,list)) or len(label)!=1:continue
            label=label[0]
        if label not in (spec["positive"],spec["negative"]):continue
        binary=1 if label==spec["positive"] else 0
        if len(buckets[binary])>=N:continue
        text=tidy(x[spec["col"]])
        if not 10<=len(text)<=spec["limit"]:continue
        h=canonical_sha(text)
        if h in occupied:continue
        occupied.add(h)
        record={
          "schema":"oml-acs-c1d-new-source-row/1",
          "axis":spec["axis"],"dataset":spec["repo"],"revision":spec["revision"],
          "source_split":"train","source_file":spec["file"],
          "source_id":str(x.get("id",int(idx))),"source_row_number":int(idx),
          "raw_row_digest":stable(x),"text":text,
          "text_sha256":sha(text.encode()),"canonical_sha256":h,
          "class_label":binary,"purpose":"prospective_development_only",
        }
        buckets[binary].append(record)
        if len(buckets[0])==N and len(buckets[1])==N:break
    if min(map(len,buckets.values()))!=N:
        raise RuntimeError("insufficient source train-only labels: "+spec["axis"]+" count "+str({k:len(v) for k,v in buckets.items()}))
    print("C1D_SELECTED",spec["axis"],"pos",N,"neg",N,"attempts",attempts,flush=True)
    return buckets[1]+buckets[0]

def atomic(path:Path,data:bytes):
    if path.exists():raise FileExistsError("refuse to overwrite pinned source: "+str(path))
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    tmp.replace(path);path.chmod(0o600)

def main():
    from huggingface_hub import HfApi,hf_hub_download
    import pyarrow.parquet as pq
    os.umask(0o077)
    if FILE.exists() or RECEIPT.exists():raise FileExistsError("candidate dataset already selected")
    start=time.perf_counter()
    occupied=load_previous()
    rows=[];meta=[]
    for i,sp in enumerate(SPEC):
        api=HfApi().dataset_info(sp["repo"],revision=sp["revision"])
        if api.sha!=sp["revision"] or api.gated:
            raise RuntimeError("source revision unavailable or gated")
        f=Path(hf_hub_download(repo_id=sp["repo"],repo_type="dataset",
          revision=sp["revision"],filename=sp["file"],cache_dir=CACHE))
        data=pq.read_table(f).to_pylist()
        selected=pick(sp,data,np.random.default_rng(SEED+i),occupied)
        rows.extend(selected)
        meta.append(dict(axis=sp["axis"],repo=sp["repo"],revision=sp["revision"],source_file=sp["file"],
                 source_sha256=sha(f.read_bytes()),source_size_bytes=f.stat().st_size,
                 labels=[sp["negative"],sp["positive"]],review_license=sp["license"],
                 total_source_rows=len(data),selected_rows=len(selected)))
    if len(rows)!=4*N:raise RuntimeError("wrong number of prospective dev examples")
    h=[r["canonical_sha256"] for r in rows]
    if len(set(h))!=len(h):raise RuntimeError("new exact duplicates")
    stats=collections.Counter((x["axis"],x["class_label"]) for x in rows)
    if set(stats.values())!={N}:raise RuntimeError("new source class imbalance")
    payload="".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows).encode()
    receipt=dict(schema="oml-acs-c1d-new-development-source/1",status="SELECTED_NOT_EVALUATED",
        purpose="new source-separated prospective development; no final/generalization claim",
        seed=SEED,row_count=len(rows),counts={str(k):v for k,v in stats.items()},
        source_file_sha256=sha(payload),sources=meta,
        historical_anchor_sha256=OLD_ANCHOR_SHA,old_behavior_sha256=OLD_BEHAVIOR_SHA,
        exact_canonical_overlap_with_historical=0,exact_internal_duplicate_count=0,
        old_train_dev_labels_not_revised=True,test_split_read=False,
        model_scores_observed_during_selection=False,
        semantic_near_duplicate_review_complete=False,
        licenses_review_complete=False,
        previously_used_V6_24_verified_excluded=False,
        final_eval_opened=False,stored_private_mode="0600",
        elapsed_seconds=round(time.perf_counter()-start,2))
    atomic(FILE,payload)
    atomic(RECEIPT,(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
    print("C1D_SOURCE_RECEIPT",json.dumps(receipt,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
