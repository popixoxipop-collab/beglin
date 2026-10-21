#!/usr/bin/env python3
"""Pinned SST-2 and CARER/Emotion training+development behavior data (G1).

This is an engineering/development-only acquisition, NOT a final evaluation.
Only original train+validation files are read, at exact HF git revisions.
All text, IDs and labels remain on the research worker under chmod 0600.
No test/final split is accessed; no inference or production mutation.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
import re
import time
import numpy as np

ROOT = Path.home() / ".local/share/beglin/oml_acs/20261021"
CACHE = "/mnt/g/TailnetCommander/work/oml_acs_behavior_hf_cache"
RAW_ANCHORS = ROOT / "anchors_300_local.jsonl"
EXPECTED_ANCHOR_SHA = "47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b"
OUT = ROOT / "behavior_sst2_carer_train_dev_v1.jsonl"
RECEIPT = ROOT / "behavior_sst2_carer_train_dev_v1_receipt.json"
SEED = 42

DATASETS = (
    dict(axis="sst2_positive_vs_negative", repo="stanfordnlp/sst2",
         revision="8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb",
         columns=("sentence", "label"), field="sentence",
         positive_label=1, negative_label=0, config="default",
         train_file="data/train-00000-of-00001.parquet",
         dev_file="data/validation-00000-of-00001.parquet",
         expected_labels=["negative", "positive"],
         license_status="UNKNOWN_REQUIRES_MANUAL_REVIEW"),
    dict(axis="carer_joy_vs_anger", repo="dair-ai/emotion",
         revision="cab853a1dbdf4c42c2b3ef2173804746df8825fe",
         columns=("text", "label"), field="text",
         positive_label=1, negative_label=3, config="split",
         train_file="split/train-00000-of-00001.parquet",
         dev_file="split/validation-00000-of-00001.parquet",
         expected_labels=["sadness", "joy", "love", "anger", "fear", "surprise"],
         license_status="OTHER_REQUIRES_MANUAL_REVIEW"),
)

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip().casefold()

def canonical_sha(text: str) -> str:
    return sha(canonical(text).encode("utf-8"))

def source_sha(row: dict) -> str:
    return sha(json.dumps(row, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))

def select_rows(*, records: list[dict], config: dict, split: str, n_per_class: int,
                seed: int, occupied: set[str], anchor_hashes: set[str]) -> tuple[list[dict],dict]:
    if split not in {"behavior_train", "behavior_dev"} or n_per_class < 1:
        raise ValueError("train/dev only; final evaluation is sealed")
    required=set(config["columns"])
    if not records or not required.issubset(records[0].keys()):
        raise ValueError("dataset columns mismatch")
    rng=np.random.default_rng(seed)
    selected={1:[],0:[]}
    skip=collections.Counter()
    for idx in rng.permutation(len(records)):
        row=records[int(idx)]
        label=row["label"]
        if label not in (config["positive_label"], config["negative_label"]):
            skip["other_class"]+=1
            continue
        binary = 1 if label==config["positive_label"] else 0
        if len(selected[binary])>=n_per_class:
            continue
        source_text = row[config["field"]]
        clean=" ".join(str(source_text).split())
        if not 10 <= len(clean) <= 300:
            skip["length"]+=1
            continue
        h=canonical_sha(clean)
        if h in occupied or h in anchor_hashes:
            skip["duplicate_or_anchor_overlap"]+=1
            continue
        occupied.add(h)
        chosen=dict(
            schema="oml-acs-behavior-source-row/1",
            axis=config["axis"],
            dataset=config["repo"],
            dataset_revision=config["revision"],
            source_file=config["train_file"] if split=="behavior_train" else config["dev_file"],
            source_row_index=int(idx),
            original_binary_label=int(label),
            class_label=int(binary),
            split=split,
            text=clean,
            text_sha256=sha(clean.encode()),
            canonical_text_sha256=h,
            source_row_sha256=source_sha(row),
        )
        selected[binary].append(chosen)
        if all(len(selected[c])==n_per_class for c in (0,1)):
            break
    if any(len(selected[c])!=n_per_class for c in (0,1)):
        raise RuntimeError(f"insufficient class-balanced rows for {config['axis']} {split}: "+
                           str({c:len(selected[c]) for c in (0,1)}))
    out=selected[1]+selected[0]
    return out,dict(selected_positive=len(selected[1]),
                    selected_negative=len(selected[0]),
                    source_total_rows=len(records),skips=dict(skip))

def validate_rows(rows: list[dict], anchor_hashes: set[str])->dict:
    if len(rows)!=600:
        raise AssertionError("expected exactly 600 labeled source rows")
    hashes=[r["canonical_text_sha256"] for r in rows]
    if len(set(hashes))!=600 or set(hashes)&anchor_hashes:
        raise AssertionError("train/dev duplicate or exact anchor leakage")
    counts=collections.Counter((r["axis"],r["split"],r["class_label"]) for r in rows)
    for spec in DATASETS:
        for part,needed in (("behavior_train",100),("behavior_dev",50)):
            for label in (0,1):
                if counts[(spec["axis"],part,label)]!=needed:
                    raise AssertionError("unexpected split/class quotas")
    for r in rows:
        if r["split"] not in {"behavior_train","behavior_dev"}:
            raise AssertionError("forbidden final split")
        if canonical_sha(r["text"]) != r["canonical_text_sha256"]:
            raise AssertionError("row canonical digest drift")
        if sha(r["text"].encode()) != r["text_sha256"]:
            raise AssertionError("raw text digest drift")
    return dict(record_count=len(rows),unique_text=len(set(hashes)),
                counts={"|".join(map(str,k)):v for k,v in sorted(counts.items())})

def write_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError("will not overwrite "+str(path))
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("xb") as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
    path.chmod(0o600)

def build()->None:
    from huggingface_hub import hf_hub_download, HfApi
    import pyarrow.parquet as pq
    start=time.perf_counter()
    os.umask(0o077)
    assert RAW_ANCHORS.is_file() and RAW_ANCHORS.stat().st_mode&0o777==0o600
    assert sha(RAW_ANCHORS.read_bytes())==EXPECTED_ANCHOR_SHA
    anchors=[json.loads(s) for s in RAW_ANCHORS.read_text().splitlines() if s]
    assert len(anchors)==300
    anchor_hashes={canonical_sha(x["text"]) for x in anchors}
    assert len(anchor_hashes)==300
    if OUT.exists() or RECEIPT.exists():
        raise FileExistsError("existing private behavior output; manual reconciliation required")
    repo_files=[]
    all_rows=[]
    used=set()
    for spec_index,spec in enumerate(DATASETS):
        meta=HfApi().dataset_info(spec["repo"],revision=spec["revision"])
        if meta.sha!=spec["revision"] or meta.gated:
            raise RuntimeError("model/dataset SHA or access drift")
        for split,filename,quota in [
            ("behavior_train",spec["train_file"],100),
            ("behavior_dev",spec["dev_file"],50),
        ]:
            path=Path(hf_hub_download(repo_id=spec["repo"],repo_type="dataset",
                                      revision=spec["revision"],
                                      filename=filename,cache_dir=CACHE))
            info=dict(axis=spec["axis"],repo=spec["repo"],revision=spec["revision"],
                      split=split,filename=filename,file_sha256=sha(path.read_bytes()),
                      size_bytes=path.stat().st_size,license_status=spec["license_status"])
            tab=pq.read_table(path,columns=list(spec["columns"]))
            records=tab.to_pylist()
            out,selection=select_rows(records=records,config=spec,split=split,
                                      n_per_class=quota,
                                      seed=SEED+spec_index*100+(0 if split=="behavior_train" else 1),
                                      occupied=used,anchor_hashes=anchor_hashes)
            info.update(selection)
            repo_files.append(info)
            all_rows.extend(out)
            print("DATASET_ACQUIRED",json.dumps({k:v for k,v in info.items()
                if k not in ("source_row_sha256",)},sort_keys=True),flush=True)
    qa=validate_rows(all_rows,anchor_hashes)
    payload="".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in all_rows).encode()
    rec=dict(
        schema="oml-acs-real-behavior-train-dev/1",
        status="PASS_DEV_SOURCE_ACQUISITION_NOT_FROZEN",
        seed=SEED,source_file_sha256=sha(payload),records=qa["record_count"],
        unique_canonical_texts=qa["unique_text"],counts=qa["counts"],
        source_files=repo_files,
        old_300_anchor_sha256=EXPECTED_ANCHOR_SHA,
        overlaps_with_300_anchors_exact_canonical=0,
        train_vs_dev_exact_overlap=0,
        splits_used=["train","validation"],
        original_test_split_accessed=False,
        behavior_final_split_created=False,
        original_paper_100_50_100_protocol_completed=False,
        source_license_review_complete=False,
        semantic_near_duplicate_check_complete=False,
        old_V6_consumed_24_provenance_attested=False,
        independent_final_evaluation_allowed=False,
        data_is_dev_only=True,raw_text_or_labels_published=False,
        elapsed_seconds=round(time.perf_counter()-start,2),
    )
    write_atomic(OUT,payload)
    write_atomic(RECEIPT,(json.dumps(rec,indent=2,ensure_ascii=True)+"\n").encode())
    print("REAL_BEHAVIOR_DATA_RECEIPT",json.dumps(rec,sort_keys=True),flush=True)
if __name__=="__main__":
    build()
