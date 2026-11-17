#!/usr/bin/env python3
"""Compare the EOE-attested BF16 model's full-shard SHA with official Chat family.

No model weights downloaded. An exact full-index-and-4-shard match is required
to identify a Chat checkpoint. A similarly named checkpoint is insufficient.
"""
import hashlib,json,requests
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_selected_bf16_native_artifacts import SOURCE_MODEL_ID,attestation
from je01_native_exact_byte_presolve import canonical
REPO="deepseek-ai/DeepSeek-V2-Lite-Chat"
SCHEMA="beglin-je01-readonly-BF16-official-chat-source-exact-shard-SHA/1"

def audit():
    session=requests.Session()
    session.headers.update({"User-Agent":"BEGLIN-JE01-immutable-model-check/1"})
    source=attestation()
    q=session.get("https://huggingface.co/api/models/"+REPO,timeout=(20,50))
    q.raise_for_status()
    rev=q.json().get("sha")
    if not isinstance(rev,str) or len(rev)!=40 or any(c not in "0123456789abcdef" for c in rev):
        raise ValueError("JE01_CHAT_IMMUTABLE_COMMIT_MISSING")
    url=f"https://huggingface.co/{REPO}/resolve/{rev}/model.safetensors.index.json?download=true"
    index_resp=session.get(url,timeout=(20,60))
    index_resp.raise_for_status()
    index_raw=index_resp.content
    index_hash=hashlib.sha256(index_raw).hexdigest()
    if len(index_raw)>2_000_000:raise ValueError("JE01_CHAT_INDEX_TOO_BIG")
    shards=[]
    for spec in source["shards"]:
        f=f"https://huggingface.co/{REPO}/resolve/{rev}/{spec['name']}?download=true"
        h=session.head(f,allow_redirects=False,timeout=(20,50))
        if h.status_code not in (302,307):
            raise ValueError("JE01_CHAT_SHARD_HEAD_FAILED:"+spec["name"])
        meta={"name":spec["name"],"expected_sha256":spec["sha256"],
              "observed_sha256":(h.headers.get("X-Linked-Etag") or "").strip('"').lower(),
              "expected_bytes":spec["size_bytes"],
              "observed_bytes":h.headers.get("X-Linked-Size"),
              "observed_repo_revision":h.headers.get("X-Repo-Commit")}
        meta["sha_match"]=meta["expected_sha256"]==meta["observed_sha256"]
        meta["bytes_match"]=str(meta["expected_bytes"])==meta["observed_bytes"]
        meta["revision_match"]=meta["observed_repo_revision"]==rev
        shards.append(meta)
        print(json.dumps({"shard":spec["name"],"sha_match":meta["sha_match"],
                          "bytes_match":meta["bytes_match"],"revision_match":meta["revision_match"],
                          "etag":meta["observed_sha256"]}),flush=True)
    exact=index_hash==source["index_sha256"] and all(
        x["sha_match"] and x["bytes_match"] and x["revision_match"] for x in shards)
    result={"schema":SCHEMA,"status":"COMPLETE",
            "EOE_original_BF16_identity_SHA256":SOURCE_MODEL_ID,
            "candidate_repo":REPO,"candidate_immutable_revision":rev,
            "original_index_sha256":source["index_sha256"],"candidate_index_sha256":index_hash,
            "index_equal":index_hash==source["index_sha256"],
            "full_shards":shards,"all_original_4_shards_exactly_same":exact,
            "original_model_forward":"NOT_RUN","production_modified":False}
    result["receipt_sha256"]=canonical(result)
    Path("JE01_CHAT_MODEL_SOURCE_IDENTITY_AUDIT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result

if __name__=="__main__":
    try:
        d=audit()
        print(json.dumps({"status":d["status"],"exact_identity":d["all_original_4_shards_exactly_same"],
                          "revision":d["candidate_immutable_revision"],"receipt_sha256":d["receipt_sha256"]},sort_keys=True))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:300]}))
        raise SystemExit(2)
