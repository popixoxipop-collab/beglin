#!/usr/bin/env python3
"""Read-only official DeepSeek-V2-Lite commit->LFS SHA identity audit.

Do NOT select a different source revision by name/size. Each candidate must
match the EOE-attested safetensors index AND all original shard SHA/bytes.
This audit never downloads pretrained weights or executes the model.
"""
from __future__ import annotations
import hashlib,json,requests,sys,urllib.parse,time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_selected_bf16_native_artifacts import SOURCE_MODEL_ID, attestation
from je01_native_exact_byte_presolve import canonical

REPO="deepseek-ai/DeepSeek-V2-Lite"
API=f"https://huggingface.co/api/models/{REPO}/commits/main"
MAX_COMMITS=100
DEFAULT_SOURCE_REV="419ef8b6b7a816343af14bc1a4436d4e33affb5d"
SCHEMA="beglin-je01-official-revision-vs-EOE-BF16-full-SHA-readonly/1"
INDEX_NAME="model.safetensors.index.json"
SHARD_NAME="model-00001-of-000004.safetensors"

def get_bytes(sess,url,maximum=2_000_000):
    with sess.get(url,timeout=(20,60),stream=True) as resp:
        resp.raise_for_status()
        raw=bytearray()
        for chunk in resp.iter_content(1<<15):
            raw.extend(chunk)
            if len(raw)>maximum:raise ValueError("JE01_HTTP_BODY_OVERSIZE")
        return bytes(raw)

def public_blob_meta(sess,commit,shard):
    url=f"https://huggingface.co/{REPO}/resolve/{commit}/{shard}?download=true"
    with sess.head(url,allow_redirects=False,timeout=(20,45)) as resp:
        if resp.status_code not in (302,307):
            raise ValueError("JE01_REVISION_SHARD_HEAD_NOT_REDIRECT:"+str(resp.status_code))
        return {"sha256":(resp.headers.get("X-Linked-Etag") or "").strip('"').lower(),
                "size":resp.headers.get("X-Linked-Size"),
                "response_revision":resp.headers.get("X-Repo-Commit")}

def audit():
    source=attestation()
    sess=requests.Session()
    sess.headers.update({"User-Agent":"BEGLIN-JE01-source-identity-readonly/1"})
    res=get_bytes(sess,API,maximum=400000)
    commits=json.loads(res)
    if not isinstance(commits,list) or not 1<=len(commits)<=MAX_COMMITS:
        raise ValueError("JE01_HF_REVISION_HISTORY_UNEXPECTED")
    ids=[x.get("id") for x in commits]
    if not all(isinstance(c,str) and len(c)==40 and all(t in "0123456789abcdef" for t in c) for c in ids):
        raise ValueError("JE01_HF_HISTORY_COMMIT_ID_INVALID")
    if DEFAULT_SOURCE_REV not in ids: ids.append(DEFAULT_SOURCE_REV)
    rows=[];matches=[]
    ref={s["name"]:s for s in source["shards"]}
    for commit in dict.fromkeys(ids):
        url=f"https://huggingface.co/{REPO}/resolve/{commit}/{INDEX_NAME}?download=true"
        index=get_bytes(sess,url)
        index_sha=hashlib.sha256(index).hexdigest()
        index_ok=index_sha==source["index_sha256"]
        row={"commit":commit,"index_sha256":index_sha,"index_matches_original_EOE":index_ok,
             "first_shard_sha256":None,"first_shard_full_bytes":None,
             "first_shard_sha_matches_original_EOE":False,
             "all_four_shards_match_original_EOE":False}
        if index_ok:
            first=public_blob_meta(sess,commit,SHARD_NAME)
            row["first_shard_sha256"]=first["sha256"]
            row["first_shard_full_bytes"]=first["size"]
            row["first_shard_sha_matches_original_EOE"]=(
                first["sha256"]==ref[SHARD_NAME]["sha256"] and
                first["size"]==str(ref[SHARD_NAME]["size_bytes"]))
            if row["first_shard_sha_matches_original_EOE"]:
                all_match=True
                for name,spec in ref.items():
                    if name==SHARD_NAME: continue
                    meta=public_blob_meta(sess,commit,name)
                    if meta["sha256"]!=spec["sha256"] or meta["size"]!=str(spec["size_bytes"]):
                        all_match=False
                row["all_four_shards_match_original_EOE"]=all_match
                if all_match: matches.append(commit)
        rows.append(row)
        print(json.dumps({"source_commit":commit,"index_match":index_ok,
                          "first_shard_match":row["first_shard_sha_matches_original_EOE"],
                          "all_shards_match":row["all_four_shards_match_original_EOE"]}),flush=True)
        if len(rows)>MAX_COMMITS:raise ValueError("JE01_TOO_MANY_REVISIONS")
    result={"schema":SCHEMA,"status":"OFFICIAL_REVISION_IDENTITY_AUDIT_COMPLETE",
            "reference_original_bf16_full_model_identity_sha256":SOURCE_MODEL_ID,
            "reference_original_index_sha256":source["index_sha256"],
            "reference_original_shard1_sha256":ref[SHARD_NAME]["sha256"],
            "history_commit_count":len(rows),"exact_full_shard_source_revisions":matches,
            "commit_identity_evidence":rows,"pretrained_weight_downloaded":False,
            "production_modified":False,"quality_or_NLL_measured":False}
    result["receipt_sha256"]=canonical(result)
    return result

if __name__=="__main__":
    try:
        v=audit()
        print(json.dumps({"status":v["status"],"receipt_sha256":v["receipt_sha256"],
                          "exact_full_shard_source_revisions":v["exact_full_shard_source_revisions"],
                          "history_commit_count":v["history_commit_count"]},sort_keys=True))
        Path("JE01_HF_REVISION_IDENTITY_READONLY_AUDIT.json").write_text(
            json.dumps(v,indent=2,sort_keys=True)+"\n")
    except Exception as ex:
        print(json.dumps({"status":"BLOCKED","reason":str(ex)[:260]}))
        raise SystemExit(2)
