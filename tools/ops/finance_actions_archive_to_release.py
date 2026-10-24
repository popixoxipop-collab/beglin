"""Lossless, immutable GitHub Actions ZIP -> private Draft Release archival.

Reads source metadata and ZIP from the SAME Beglin repository using the
short-lived GitHub Actions token; independently re-downloads release assets.
DOES NOT DELETE ACTIONS ARTIFACTS. DOES NOT access trading, finance or secrets.
Intended as step one of off-GitHub dual backup. Release alone isn't sufficient
to authorize deletion.
"""
from __future__ import annotations
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib import request, error
import zipfile

REPO = "popixoxipop-collab/beglin"
BRANCH = "research/finance-quota-inventory-20261024"
TAG = "archive-beglin-actions-quota-20261024"
LIMIT_BATCH = 25
BATCHES = 4
MAX_BYTES = 35_000_000
MAX_ITEMS = 100
OUTPUT = "ops/finance_quota_release_backup_batch_{batch}_20261024.json"

def fail_unless(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)

def gh(*args: str, binary_output: Path | None = None, timeout=60):
    cmd = ["gh", *args]
    if binary_output is None:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False)
    else:
        with binary_output.open("xb") as output:
            proc = subprocess.run(cmd, stdout=output, stderr=subprocess.PIPE,
                                  timeout=timeout, check=False)
    if proc.returncode:
        # Never print arbitrary stderr which might include temporary URL/token.
        raise RuntimeError("gh_command_failed:" + args[0] + ":" + str(proc.returncode))
    return proc.stdout if binary_output is None else b""

def gh_json(*args: str) -> dict:
    data = json.loads(gh(*args))
    fail_unless(isinstance(data, dict), "GH_JSON_UNEXPECTED_TYPE")
    return data

def verify_zip(path: Path, *, size: int, digest: str) -> int:
    h=hashlib.sha256()
    measured=0
    with path.open("rb") as handle:
        while True:
            b=handle.read(1<<20)
            if not b:break
            measured+=len(b)
            fail_unless(measured<=MAX_BYTES,"ARTIFACT_TOO_BIG")
            h.update(b)
    fail_unless(measured==size and h.hexdigest()==digest, "SOURCE_OR_BACKUP_SHA_MISMATCH")
    with zipfile.ZipFile(path) as file:
        fail_unless(len(file.infolist())>0 and file.testzip() is None,"ZIP_CRC_FAILED")
    return measured

def artifact_backup(item: dict) -> dict:
    ident=item["id"]; size=item["size"]; digest=item["digest"]
    fail_unless(type(ident) is int and ident>0 and type(size) is int and 0<size<MAX_BYTES,"SOURCE_ID_OR_SIZE")
    fail_unless(isinstance(digest,str) and digest.startswith("sha256:") and len(digest)==71,"ORIGINAL_DIGEST")
    sha=digest.removeprefix("sha256:")
    meta=gh_json("api",f"repos/{REPO}/actions/artifacts/{ident}")
    fail_unless(meta.get("id")==ident and meta.get("expired") is False and
                meta.get("size_in_bytes")==size and meta.get("digest")==digest,
                "SOURCE_METADATA_CHANGED")
    name=f"beglin-actions-{ident}.zip"
    with TemporaryDirectory(prefix=f"beglin-archive-{ident}-") as directory:
        p=Path(directory)
        source=p/name
        gh("api",f"repos/{REPO}/actions/artifacts/{ident}/zip",
           binary_output=source,timeout=90)
        verify_zip(source,size=size,digest=sha)
        state=gh_json("release","view",TAG,"--repo",REPO,"--json","isDraft,assets")
        fail_unless(state.get("isDraft") is True,"BACKUP_RELEASE_NOT_PRIVATE_DRAFT")
        names={a["name"]:a for a in state.get("assets",[])}
        if name not in names:
            # Never clobber previous backups even when source changed.
            gh("release","upload",TAG,"--repo",REPO,str(source),timeout=90)
        target=p/"fresh"
        target.mkdir()
        gh("release","download",TAG,"--repo",REPO,"--pattern",name,"--dir",str(target),timeout=90)
        verify_zip(target/name,size=size,digest=sha)
        state=gh_json("release","view",TAG,"--repo",REPO,"--json","isDraft,assets")
        fail_unless(state.get("isDraft") is True,"RELEASE_BECAME_PUBLIC")
        assets=[a for a in state.get("assets",[]) if a.get("name")==name]
        fail_unless(len(assets)==1 and assets[0].get("size")==size,"RELEASE_ASSET_META_MISMATCH")
        report={
            "artifact_id":ident,"original_run_id":item.get("run_id"),
            "original_created_at":item.get("created_at"),
            "original_sha256":sha,"original_bytes":size,
            "private_draft_release_asset_id":assets[0].get("id"),
            "release_asset_name":name,
            "release_readback_sha_crc_verified":True,
            "original_actions_artifact_deleted":False,
            "off_github_disk_copy_verified":False,
        }
        print("PRIVATE_DRAFT_RELEASE_ORIGINAL_ARCHIVED_ID="+str(ident),flush=True)
        return report

def commit_json(path: str, raw: bytes):
    url=f"https://api.github.com/repos/{REPO}/contents/{path}"
    headers={"Authorization":"Bearer "+os.environ["GH_TOKEN"],
             "Accept":"application/vnd.github+json",
             "X-GitHub-Api-Version":"2022-11-28",
             "User-Agent":"beglin-finance-quota-backup/1"}
    for retry in range(6):
        doc={"branch":BRANCH,"message":"Ops: preserve readback-verified original Actions ZIPs on private Draft Release",
             "content":base64.b64encode(raw).decode()}
        try:
            req=request.Request(url+"?ref="+BRANCH,headers=headers)
            with request.urlopen(req,timeout=12) as resp:
                doc["sha"]=json.load(resp)["sha"]
        except error.HTTPError as e:
            if e.code!=404:raise
        req=request.Request(url,headers=headers,method="PUT",data=json.dumps(doc).encode())
        try:
            with request.urlopen(req,timeout=20) as resp:
                result=json.load(resp)
            return result["commit"]["sha"]
        except error.HTTPError as exc:
            if exc.code not in (409,422) or retry==5:raise
            time.sleep(retry+1)
    raise RuntimeError("UNREACHABLE")

def main():
    fail_unless(os.environ.get("GITHUB_REPOSITORY")==REPO and
                os.environ.get("GITHUB_REF")=="refs/heads/"+BRANCH,
                "NOT_DEDICATED_QUARANTINE_BRANCH")
    fail_unless(os.environ.get("GH_TOKEN"),"NO_EPHEMERAL_GITHUB_TOKEN")
    batch=int(os.environ["FINANCE_BACKUP_BATCH"])
    fail_unless(0<=batch<BATCHES,"INVALID_BATCH")
    inv=gh_json("api",f"repos/{REPO}/contents/ops/finance_quota_inventory_20261024.json?ref={BRANCH}")
    raw=base64.b64decode(inv["content"])
    report=json.loads(raw)
    all_rows=report["all_artifacts"]
    fail_unless(len(all_rows)==report["count"] and
                report["count"]<=2000 and
                all_rows==sorted(all_rows,key=lambda x:(-x["size"],x["id"])),
                "INVENTORY_IDENTITY_OR_ORDER")
    items=all_rows[batch*LIMIT_BATCH:(batch+1)*LIMIT_BATCH]
    fail_unless(len(items)==LIMIT_BATCH and all(x.get("digest") for x in items),
                "BATCH_INVENTORY_UNVERIFIED")
    state=gh_json("release","view",TAG,"--repo",REPO,"--json","isDraft")
    fail_unless(state["isDraft"] is True,"DRAFT_RELEASE_REQUIRED")
    results=[]
    # Dedicated GitHub-hosted runner; bound concurrency, bounded test copies.
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(artifact_backup,row):row["id"] for row in items}
        errors=[]
        for task in as_completed(tasks):
            try:results.append(task.result())
            except Exception as exc:errors.append({"id":tasks[task],"reason":str(exc)[:120]})
    document={"schema":"beglin.finance-quota-release-only-backup-batch/1",
              "recorded_at_utc":datetime.now(timezone.utc).isoformat(),
              "repo":REPO,"batch":batch,"source_inventory_count":report["count"],
              "source_inventory_bytes":report["measured_bytes"],
              "release_tag":TAG,"release_is_private_draft":True,
              "archived":[x for x in sorted(results,key=lambda x:x["artifact_id"])],
              "errors":errors,"original_actions_deleted":False,
              "d50_or_wd_black_backup_verified":False,"release_only_not_delete_authority":True}
    raw=(json.dumps(document,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
    commit=commit_json(OUTPUT.format(batch=batch),raw)
    print("BATCH_BACKUP_REPORT_COMMIT="+commit)
    print("BATCH_SUCCESS_COUNT="+str(len(results))+" BATCH_ERROR_COUNT="+str(len(errors)),flush=True)
    # Any error must preserve GitHub Actions originals and fail the job.
    if errors:sys.exit(2)

if __name__=="__main__":main()
