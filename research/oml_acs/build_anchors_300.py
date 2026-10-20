#!/usr/bin/env python3
"""Build 300 development-only ACS anchors from paper's 15 benchmark *names*.

Not author-identified HELM rows: Hugging Face mirrors/source selections are
documented substitutions. Dataset Viewer rows are not revision-addressed;
the complete raw row digest is retained locally to detect later drift.
This script refuses to declare a frozen evaluation without V6 holdout evidence.
Raw benchmark text (including sensitive safety prompts) stays locally private.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, os, re, sys, time
import urllib.parse, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

SEED = 42
MAX_CHARS = 300
NUM_EACH = 20
UA = {"User-Agent": "beglin-oml-acs-research-anchor-audit/0.1"}
SPECS = [
 ("MMLU","cais/mmlu","c30699e8356da336a370243923dbaf21066bb9fe","all","test","question","factual"),
 ("NaturalQA","google-research-datasets/natural_questions","e8103d566bef4154c2c12b17c6095ec5275840cc","default","train","question.text","factual"),
 ("NarrativeQA","deepmind/narrativeqa","2e643e7363944af1c33a652d1c87320d0871c4e4","default","train","question.text","factual"),
 ("BoolQ","google/boolq","35b264d03638db9f4ce671b711558bf7ff0f80d5","default","train","question","factual"),
 ("HellaSwag","Rowan/hellaswag","218ec52e09a7e7462a5400043bb9a69a41d06b76","default","train","ctx","commonsense"),
 ("TruthfulQA","truthfulqa/truthful_qa","741b8276f2d1982aa3d5b832d3ee81ed3b896490","generation","validation","question","truthfulness"),
 ("GSM8K","openai/gsm8k","740312add88f781978c0658806c59bc2815b9866","main","train","question","reasoning"),
 ("MATH","EleutherAI/hendrycks_math","21a5633873b6a120296cce3e2df9d5550074f4a3","algebra","train","problem","reasoning"),
 ("IMDB","stanfordnlp/imdb","e6281661ce1c48d982bc483cf8a173c1bbeb5d31","plain_text","train","text","affect"),
 ("CivilComments","google/civil_comments","f2970eb3a55777454c94069077cc8d9b5866312d","default","train","text","toxicity"),
 ("RealToxicityPrompts","allenai/real-toxicity-prompts","f21629712ffd6a3d13a54fd2807ccd521c55ef74","default","train","prompt.text","toxicity"),
 ("HarmBench","swiss-ai/harmbench","f306ca7195c286ffaf20324333e9da804bdd8987","DirectRequest","val","Behavior","safety"),
 ("SimpleSafetyTests","Bertievidgen/SimpleSafetyTests","98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5","csv","all","prompt","safety"),
 ("XSTest","Paul/XSTest","f600c994b256f12867dfa5b3eb3d545a3e62f8b5","default","train","prompt","safety"),
 ("AnthropicRedTeam","Anthropic/hh-rlhf","09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa","raw_gzip","all","transcript","safety"),
]
LEN = {"MMLU":14042,"NaturalQA":10639,"NarrativeQA":14650,"BoolQ":9427,
       "HellaSwag":39905,"TruthfulQA":817,"GSM8K":7473,"MATH":1744,
       "IMDB":25000,"CivilComments":1804874,"RealToxicityPrompts":99442,
       "HarmBench":80,"XSTest":450}
LICENSE_REVIEW = {"HellaSwag", "HarmBench", "IMDB"}
ISSUES = []

def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def stable(data)->str:
    return digest(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode())

def request(url:str,*,timeout=15,tries=3)->bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt==tries-1: raise
            time.sleep(0.4*(attempt+1))
    raise RuntimeError("unreachable")

def clean(text):
    if not isinstance(text,str): return ""
    text=re.sub(r"<br\s*/?>"," ",text,flags=re.IGNORECASE)
    text=re.sub(r"\s+"," ",text).strip()
    text=re.sub(r"\s+(?:A:|Answer:)\s*$","",text,flags=re.IGNORECASE)
    return text.strip()

def getfield(data,name):
    cur=data
    for part in name.split("."):
        if not isinstance(cur,dict):return ""
        cur=cur.get(part)
    return cur

def src_key(s):
    return {"scenario":s[0],"dataset":s[1],"revision":s[2],
            "config":s[3],"split":s[4],"field":s[5],"group":s[6]}

def check_head(s):
    data=json.loads(request("https://huggingface.co/api/datasets/"+s[1],timeout=15))
    if data.get("sha")!=s[2]:
        raise RuntimeError("dataset branch HEAD moved for %s: expected %s got %s"%(s[0],s[2],data.get("sha")))
    if data.get("gated"):
        raise RuntimeError("gated source without user authorization: "+s[0])
    return data.get("cardData",{}).get("license")

def viewer_row(s,index:int):
    q=urllib.parse.urlencode(dict(dataset=s[1],config=s[3],split=s[4],
                                   offset=index,length=1))
    data=json.loads(request("https://datasets-server.huggingface.co/rows?"+q,timeout=20))
    row=data["rows"][0]
    if int(row["row_idx"])!=index: raise ValueError("HF viewer row index changed")
    return row["row"]

def select_pinned_stream(s,rng,seen):
    """Read actual HF dataset repository at a pinned commit, NOT dataset-viewer HEAD.
    Uses bounded shuffle buffer; source-record metadata and local hashes are kept.
    This samples the dataset's streaming shard shuffle, not the paper's exact IDs.
    """
    from datasets import load_dataset
    dataset = load_dataset(s[1],s[3],split=s[4],revision=s[2],streaming=True,
                           trust_remote_code=False)
    derived_seed = int(rng.integers(0,2**32-1))
    dataset = dataset.shuffle(seed=derived_seed,buffer_size=128)
    limit=6000 if s[0]=="IMDB" else 1300
    chosen=[];attempted=0
    for row in dataset:
        attempted+=1
        text=clean(getfield(row,s[5]))
        if not 10<=len(text)<=MAX_CHARS:continue
        sha=digest(text.encode("utf-8"))
        if sha in seen:continue
        seen.add(sha)
        chosen.append(row_record(s,attempted-1,text,row))
        if len(chosen)>=NUM_EACH:break
        if attempted>=limit:break
    print("SCENARIO",s[0],"selected",len(chosen),"stream_rows",attempted,"pinned_revision",s[2],flush=True)
    return chosen

def row_record(s,i,text,source_row):
    group = ""
    if s[0]=="NarrativeQA":group=str(getfield(source_row,"document.id") or "")
    elif s[0]=="NaturalQA":group=str(getfield(source_row,"document.url") or "")
    elif s[0]=="HellaSwag":group=str(source_row.get("source_id",""))
    if not group:group="source-row-sha256:"+stable(source_row) # Independent semantic-overlap audit still needed.
    return {
        "schema":"oml-acs-anchor-row/1","scenario":s[0],"category":s[6],
        "dataset":s[1],"dataset_revision":s[2],"dataset_config":s[3],
        "source_split":s[4],"source_id":str(source_row.get("id") or source_row.get("source_id") or source_row.get("BehaviorID") or stable(source_row)),"stream_position":i,"cluster_id":digest(group.encode()),
        "text":text,"text_sha256":digest(text.encode()),
        "source_row_sha256":stable(source_row),
        "extract_field":s[5],"purpose":"development_anchor_only",
    }

def select_simple(s,rng,seen):
    url=f"https://huggingface.co/datasets/{s[1]}/resolve/{s[2]}/sst_test_cases.csv"
    data=request(url)
    rows=list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    chosen=[]
    for j in rng.permutation(len(rows)):
        row=rows[int(j)]
        txt=clean(row.get("prompt",""))
        if not 10<=len(txt)<=300:continue
        sha=digest(txt.encode())
        if sha in seen:continue
        seen.add(sha)
        chosen.append(row_record(s,int(j),txt,row))
        if len(chosen)==NUM_EACH:break
    print("SCENARIO",s[0],"selected",len(chosen),"total",len(rows),flush=True)
    return chosen

def select_anthropic(s,rng,seen):
    # Pinned red-team attempt file is a JSON *array*, one object per line,
    # not a standard JSONL file; the HF viewer's chosen/rejected rows are WRONG.
    url=f"https://huggingface.co/datasets/{s[1]}/resolve/{s[2]}/red-team-attempts/red_team_attempts.jsonl.gz"
    chosen=[];count=0;eligible=0
    with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=55) as http:
        with gzip.GzipFile(fileobj=http) as file:
            for line in file:
                line=line.strip().lstrip(b"[").rstrip(b",")
                if not line or line==b"]": continue
                try: obj=json.loads(line)
                except json.JSONDecodeError:continue
                # Human-provided prompt prior to an assistant answer, no assistant text.
                transcript=obj.get("transcript","")
                if not isinstance(transcript,str):continue
                m=re.search(r"(?:^|\n)Human:\s*(.*?)(?:\nAssistant:|\n\nAssistant:|$)",transcript,re.S)
                txt=clean(m.group(1)) if m else ""
                count+=1
                if not 10<=len(txt)<=300:continue
                eligible+=1
                sha=digest(txt.encode())
                if sha in seen:continue
                # Streaming reservoir sampling from the full file, seed reproducible.
                record=row_record(s,count-1,txt,{"transcript_text_sha256":sha})
                if len(chosen)<NUM_EACH:chosen.append(record)
                else:
                    j=int(rng.integers(0,eligible))
                    if j<NUM_EACH:chosen[j]=record
    if len(chosen)==NUM_EACH:
        for r in chosen:seen.add(r["text_sha256"])
    print("SCENARIO",s[0],"selected",len(chosen),"attempts",count,"eligible",eligible,flush=True)
    return chosen

def write_atomic(path, data:bytes, mode=0o600):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".tmp")
    fd=os.open(str(temp),os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
    with os.fdopen(fd,"wb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(str(temp),str(path))
    os.chmod(path,mode)

def build(out:Path):
    rng=np.random.default_rng(SEED)
    revisions={}
    for spec in SPECS:
        license_=check_head(spec)
        revisions[spec[0]]=dict(src_key(spec),license=license_,
            license_review_required=spec[0] in LICENSE_REVIEW,
            original_HELM_author_row_ids_unavailable=True,pinned_hf_repository_revision=True)
    rows=[];seen=set()
    for spec in SPECS:
        if spec[0]=="SimpleSafetyTests":found=select_simple(spec,rng,seen)
        elif spec[0]=="AnthropicRedTeam":found=select_anthropic(spec,rng,seen)
        else:found=select_pinned_stream(spec,rng,seen)
        rows.extend(found)
    for spec in SPECS:check_head(spec)
    counts=Counter(r["scenario"] for r in rows)
    quotas_ok=(len(rows)==300 and len(counts)==15 and all(counts.get(s[0])==20 for s in SPECS))
    duplicate_ok=len({r["text_sha256"] for r in rows})==len(rows)
    anchors_bytes=("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows)).encode()
    status="PROVISIONAL_300_NOT_FROZEN" if quotas_ok and duplicate_ok else "INCOMPLETE_NOT_FROZEN"
    summary=dict(schema="oml-acs-anchors-manifest/1",status=status,
        note="Original author anchor IDs unpublished; HF dataset read at pinned revision, but sample IDs differ. Raw text remains on local host.",
        paper="arXiv:2605.09875v1",seed=SEED,method="NumPy default_rng(42), derived per-source seed + datasets streaming shuffle buffer128, 10-300 char filter; author row IDs unavailable",
        total=len(rows),counts=dict(sorted(counts.items())),quota_ok=quotas_ok,unique_text=duplicate_ok,
        sources=list(revisions.values()),ids=[{k:v for k,v in r.items() if k not in ("text",)} for r in rows],
        raw_local_sha256=digest(anchors_bytes),data_splits_frozen=False,
        v6_consumed_holdouts_verified=False,final_eval_access_allowed=False,
        limitations=["No original 300 IDs; not exact paper replication",
        "Pinned HF dataset mirror may not be identical to HELM preprocessing",
        "License review pending for unknown/other sources",
        "V6 24 consumed holdout IDs not matched",
        "No independent near-duplicate/semantic-overlap audit",
        "Raw safety/red-team prompts kept off GitHub"])
    out.mkdir(parents=True,exist_ok=True)
    write_atomic(out/"anchors_300_local.jsonl",anchors_bytes)
    write_atomic(out/"anchor_manifest_public_hashes_only.json",(json.dumps(summary,indent=2,ensure_ascii=False)+"\n").encode())
    print("BUILD_RESULT",json.dumps({k:v for k,v in summary.items() if k not in ("ids","sources")},ensure_ascii=False),flush=True)
    if not quotas_ok or not duplicate_ok:raise SystemExit(2)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="/mnt/g/TailnetCommander/work/oml_acs_20261021_anchors")
    ns=ap.parse_args();build(Path(ns.out))
