#!/usr/bin/env python3
"""Scratch trigger calibration for non-monotonic precision selection.

The runner first proves that a fixed replay is a real near-tie event attributed
to one target, then replays the exact prompt under a low-cost base precision and
one or more alternate qNg64 precisions. It writes trigger evidence only for the
scratch experiment; it never touches the production route, launchd service, or
promotion/control files.

One event is intentionally not enough for dynamic production selection.
precision_dynamic_selector.py requires multiple distinct event identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request

import gpu_runtime_control as grc
import precision_context as pc


REQUEST_RE = re.compile(
    r"\[moe gpu cb online\] req (\d+) .*? tokens:\s*([^\n\r]*)"
)
SUPPORTED_TRIGGER = "near_tie"


class CalibrationError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()


def minimal_env() -> dict[str,str]:
    out={}
    for key in ("PATH","HOME","USER","LOGNAME","LANG","LC_ALL","TMPDIR"):
        value=os.environ.get(key)
        if value:
            out[key]=value
    return out


def active_manifest_rows(path: str | Path) -> list[str]:
    rows=[
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise CalibrationError("source manifest contains no active rows")
    return rows


def materialize_manifest(source: str | Path, dest: Path, count: int) -> dict:
    rows=active_manifest_rows(source)
    first=rows[0]
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text((first+"\n")*int(count))
    return {
        "source_manifest":str(Path(source)),
        "source_manifest_sha256":sha256_file(source),
        "unique_source_rows":len(set(rows)),
        "selected_row_sha256":hashlib.sha256(first.encode()).hexdigest(),
        "path":str(dest),
        "sha256":sha256_file(dest),
        "requests":int(count),
    }


def run_process(binary: str, cwd: str, env: dict, log_path: Path, timeout: int) -> tuple[int,str]:
    proc=subprocess.Popen(
        [binary],cwd=cwd,env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
    )
    try:
        output,_=proc.communicate(timeout=int(timeout))
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            output,_=proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            output,_=proc.communicate(timeout=10)
        raise CalibrationError(f"worker timed out: pid={proc.pid}")
    log_path.write_text(output)
    return int(proc.returncode),output


def parse_jsonl(path: Path) -> list[dict]:
    rows=[]
    if not path.is_file():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        line=line.strip()
        if not line:
            continue
        try:
            obj=json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj,dict):
            rows.append(obj)
    return rows


def detect_trigger(
    *,
    binary: str,
    cwd: str,
    moe_base: str,
    safetensors: str,
    manifest: Path,
    root: Path,
    role: str,
    layer: int,
    req: int,
    pos: int,
    model: str,
    corpus: str,
    threshold: float,
    timeout: int,
) -> dict:
    root.mkdir(parents=True,exist_ok=True)
    event_log=root/"trigger.jsonl"
    combo=root/"hi-combos.txt"
    combo.write_text(f"{role} {int(layer)}\n")
    try:
        event_log.unlink()
    except FileNotFoundError:
        pass

    env=minimal_env()
    env.update({
        "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
        "QWEN_MOE_BASE":moe_base,
        "QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
        "QWEN_MOE_CB_SLOTS":"1",
        "QWEN_MOE_GPU_VALIDATION_REPORT":"1",
        "QWEN_MOE_NEARTIE_LOG":"1",
        "QWEN_MOE_NEARTIE_THRESHOLD":str(float(threshold)),
        "QWEN_MOE_NEARTIE_EVENTS_LOG":str(event_log),
        "QWEN_MOE_NEARTIE_MODEL":model,
        "QWEN_MOE_NEARTIE_CORPUS":corpus,
        "QWEN_MOE_NEARTIE_CORRECT":"1",
        "QWEN_MOE_NEARTIE_CORRECT_THRESHOLD":str(float(threshold)),
        "QWEN_MOE_NEARTIE_CORRECT_SAFETENSORS":safetensors,
        "QWEN_MOE_NEARTIE_HI_COMBOS":str(combo),
        "QWEN_MOE_ATTRIB":"1",
        "QWEN_MOE_ATTRIB_MODE":"add",
        "QWEN_MOE_ATTRIB_MAX_EVENTS":"1",
        "QWEN_MOE_ATTRIB_MAX_POS":str(int(pos)),
    })
    rc,output=run_process(
        binary,cwd,env,root/"trigger-worker.log",timeout
    )
    rows=parse_jsonl(event_log)
    events=[
        row for row in rows
        if row.get("kind")=="event"
        and int(row.get("req",-1))==int(req)
        and int(row.get("pos",-1))==int(pos)
    ]
    attrs=[
        row for row in rows
        if row.get("kind")=="attribution"
        and int(row.get("req",-1))==int(req)
        and int(row.get("pos",-1))==int(pos)
        and row.get("role")==role
        and int(row.get("layer",-1))==int(layer)
    ]
    event=events[-1] if events else None
    attr=attrs[-1] if attrs else None
    status=(
        "TRIGGER_CONFIRMED"
        if rc==0 and event is not None and attr is not None
        else (
            "TRIGGER_NOT_ATTRIBUTED"
            if rc==0 and event is not None
            else "NO_TRIGGER"
        )
    )
    return {
        "status":status,
        "returncode":rc,
        "event":event,
        "attribution":attr,
        "event_log":str(event_log),
        "event_log_sha256":sha256_file(event_log) if event_log.is_file() else None,
        "worker_log":str(root/"trigger-worker.log"),
        "worker_log_sha256":sha256_file(root/"trigger-worker.log"),
    }


def parse_requests(output: str) -> dict[int,list[int]]:
    out={}
    for match in REQUEST_RE.finditer(output):
        out[int(match.group(1))]=[int(v) for v in match.group(2).split()]
    return out


def policy_has_target(policy, role: str, layer: int, n: int) -> bool:
    return any(
        row.get("role")==role
        and int(row.get("layer",-1))==int(layer)
        and int(row.get("n",-1))==int(n)
        for row in (policy or [])
    )


def replay_precision(
    *,
    binary: str,
    cwd: str,
    moe_base: str,
    safetensors: str,
    manifest: Path,
    root: Path,
    role: str,
    layer: int,
    n: int,
    pos: int,
    prompt_len: int,
    reference_token: int,
    timeout: int,
) -> dict:
    run=root/f"n{int(n)}"
    run.mkdir(parents=True,exist_ok=True)
    promo=run/"promotion_nq.txt"
    ack_path=run/"applied_ack.json"
    txn_path=run/"txn.cmd"
    promo.write_text(f"{role} {int(layer)} {int(n)}\n")
    for path in (ack_path,txn_path):
        try:path.unlink()
        except FileNotFoundError:pass

    env=minimal_env()
    env.update({
        "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
        "QWEN_MOE_BASE":moe_base,
        "QWEN_MOE_NEARTIE_CORRECT":"0",
        "QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
        "QWEN_MOE_CB_SLOTS":"4",
        "QWEN_MOE_GPU_VALIDATION_REPORT":"1",
        "QWEN_MOE_GPU_APPLIED_ACK":str(ack_path),
        "QWEN_MOE_GPU_TXN_FILE":str(txn_path),
        "QWEN_MOE_PROMOTION_FILE_NQ":str(promo),
        "QWEN_MOE_PROMOTION_SAFETENSORS":safetensors,
    })
    rc,output=run_process(
        binary,cwd,env,run/"worker.log",timeout
    )
    requests=parse_requests(output)
    ack=None
    ack_error=None
    try:
        ack=grc.read_runtime_ack(ack_path)
    except Exception as exc:
        ack_error=str(exc)
    gen_idx=int(pos)-(int(prompt_len)-1)
    values=[
        toks[gen_idx]
        for toks in requests.values()
        if 0<=gen_idx<len(toks)
    ]
    hits=sum(v==int(reference_token) for v in values)
    applied=(
        isinstance(ack,dict)
        and policy_has_target(ack.get("active_policy"),role,layer,n)
    )
    passed=(
        rc==0
        and applied
        and len(values)==len(requests)
        and len(requests)>0
        and hits==len(requests)
    )
    return {
        "n":int(n),
        "status":"PASS" if passed else "FAIL",
        "returncode":rc,
        "requests":len(requests),
        "eligible_requests":len(values),
        "reference_hits":hits,
        "distinct_tokens":sorted(set(values)),
        "policy_applied":bool(applied),
        "ack_sha256":ack.get("ack_sha256") if isinstance(ack,dict) else None,
        "policy_hash":ack.get("active_policy_hash") if isinstance(ack,dict) else None,
        "weight_epoch":ack.get("weight_epoch") if isinstance(ack,dict) else None,
        "ack_error":ack_error,
        "worker_log":str(run/"worker.log"),
        "worker_log_sha256":sha256_file(run/"worker.log"),
    }


def credentials() -> tuple[str,str]:
    url=os.environ.get("QWEN_SUPABASE_URL","").rstrip("/")
    key=os.environ.get("QWEN_SUPABASE_KEY","")
    if not url or not key:
        raise CalibrationError("QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY are required")
    return url,key


def persist_evidence(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    url,key=credentials()
    req=urllib.request.Request(
        f"{url}/rest/v1/moe_precision_trigger_evidence_v1?on_conflict=evidence_id",
        data=json.dumps(rows).encode(),
        headers={
            "apikey":key,
            "Authorization":f"Bearer {key}",
            "Content-Type":"application/json",
            "Prefer":"resolution=merge-duplicates,return=representation",
        },
        method="POST",
    )
    with urllib.request.urlopen(req,timeout=30) as resp:
        landed=json.loads(resp.read())
    if not isinstance(landed,list) or len(landed)!=len(rows):
        raise CalibrationError("trigger evidence persistence was not verified")
    return landed


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",required=True)
    ap.add_argument("--cwd",required=True)
    ap.add_argument("--moe-base",required=True)
    ap.add_argument("--safetensors",required=True)
    ap.add_argument("--source-manifest",required=True)
    ap.add_argument("--root",required=True)
    ap.add_argument("--model",default="deepseek-v2-lite")
    ap.add_argument("--corpus",default="precision-trigger-calibration-v1")
    ap.add_argument("--role",default="shared_up_proj")
    ap.add_argument("--layer",type=int,default=3)
    ap.add_argument("--req",type=int,default=0)
    ap.add_argument("--pos",type=int,default=16)
    ap.add_argument("--prompt-len",type=int,default=9)
    ap.add_argument("--reference-token",type=int,default=1224)
    ap.add_argument("--base-n",type=int,default=5)
    ap.add_argument("--alternate-n",type=int,nargs="+",default=[6,9])
    ap.add_argument("--requests",type=int,default=12)
    ap.add_argument("--threshold",type=float,default=0.5)
    ap.add_argument("--context-hash")
    ap.add_argument("--timeout",type=int,default=180)
    ap.add_argument("--persist",action="store_true")
    ap.add_argument("--output")
    args=ap.parse_args()

    root=Path(args.root).expanduser().resolve(strict=False)
    allowed=Path("/Users/xox/vdsp_shadow_runs").resolve(strict=False)
    if allowed not in root.parents:
        raise CalibrationError("calibration root must stay under vdsp_shadow_runs")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    detect_manifest=materialize_manifest(
        args.source_manifest,root/"manifests/detect-one.txt",1
    )
    replay_manifest=materialize_manifest(
        args.source_manifest,root/"manifests/replay.txt",args.requests
    )
    detection=detect_trigger(
        binary=args.binary,cwd=args.cwd,moe_base=args.moe_base,
        safetensors=args.safetensors,manifest=Path(detect_manifest["path"]),
        root=root/"trigger",role=args.role,layer=args.layer,req=args.req,
        pos=args.pos,model=args.model,corpus=args.corpus,
        threshold=args.threshold,timeout=args.timeout,
    )
    # detect_trigger expects its root to exist.
    # Create lazily before retrying when needed.
    if detection["status"]!="TRIGGER_CONFIRMED":
        out={
            "schema":"beglin-precision-trigger-calibration-v1",
            "status":detection["status"],
            "production_write_allowed":False,
            "detection":detection,
            "evidence_rows":[],
        }
    else:
        ns=[args.base_n]+[
            n for n in args.alternate_n if int(n)!=int(args.base_n)
        ]
        replays={}
        for n in ns:
            replays[int(n)]=replay_precision(
                binary=args.binary,cwd=args.cwd,moe_base=args.moe_base,
                safetensors=args.safetensors,
                manifest=Path(replay_manifest["path"]),
                root=root/"replays",role=args.role,layer=args.layer,n=n,
                pos=args.pos,prompt_len=args.prompt_len,
                reference_token=args.reference_token,timeout=args.timeout,
            )
        event=detection["event"]
        attr=detection["attribution"]
        event_identity={
            "source_row_sha256":detect_manifest["selected_row_sha256"],
            "req":int(args.req),"pos":int(args.pos),
            "role":args.role,"layer":int(args.layer),
            "orig_argmax":attr.get("orig_argmax"),
            "corrected_argmax":attr.get("corrected_argmax"),
        }
        event_id=sha256_json(event_identity)
        evidence_rows=[]
        base=replays[int(args.base_n)]
        for n in args.alternate_n:
            alt=replays[int(n)]
            passed=(
                base["status"]=="PASS"
                and alt["status"]=="PASS"
                and detection["status"]=="TRIGGER_CONFIRMED"
            )
            metrics={
                "event_id":event_id,
                "distinct_events":1,
                "margin":event.get("margin"),
                "replay_margin_b1":event.get("replay_margin_b1"),
                "batch_size":event.get("batch_size"),
                "threshold":attr.get("threshold"),
                "orig_argmax":attr.get("orig_argmax"),
                "corrected_argmax":attr.get("corrected_argmax"),
                "base_reference_hits":base["reference_hits"],
                "alternate_reference_hits":alt["reference_hits"],
                "base_ack_sha256":base["ack_sha256"],
                "alternate_ack_sha256":alt["ack_sha256"],
                "source_manifest_sha256":detect_manifest["source_manifest_sha256"],
                "binary_sha256":sha256_file(args.binary),
            }
            preimage={
                "model_id":args.model,
                "role":args.role,"layer":int(args.layer),
                "from_n":int(args.base_n),"to_n":int(n),
                "trigger_type":SUPPORTED_TRIGGER,
                "event_id":event_id,
                "metrics":metrics,
            }
            evidence_id=sha256_json(preimage)
            evidence_rows.append({
                "evidence_id":evidence_id,
                "model_id":args.model,
                "context_hash":args.context_hash,
                "role":args.role,
                "layer":int(args.layer),
                "from_n":int(args.base_n),
                "to_n":int(n),
                "trigger_type":SUPPORTED_TRIGGER,
                "signal_bucket":{"trigger_only":True},
                "requests":min(base["requests"],alt["requests"]),
                "pass":bool(passed),
                "status":"PASS" if passed else "FAIL",
                "metrics":metrics,
                "source_run_id":event_id,
                "evidence_sha256":sha256_json({
                    "detection":detection,
                    "base":base,
                    "alternate":alt,
                }),
            })
        if args.persist:
            persist_evidence(evidence_rows)
        out={
            "schema":"beglin-precision-trigger-calibration-v1",
            "status":"PASS" if all(r["pass"] for r in evidence_rows) else "FAIL",
            "production_write_allowed":False,
            "auto_promotion_enabled":False,
            "distinct_events":1,
            "dynamic_selector_min_distinct_events":3,
            "coverage_sufficient_for_dynamic_selection":False,
            "detection":detection,
            "replays":replays,
            "event_id":event_id,
            "evidence_rows":evidence_rows,
        }

    text=json.dumps(out,indent=2,sort_keys=True)+"\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text,end="")
    return 0 if out["status"] in {"PASS","NO_TRIGGER","TRIGGER_NOT_ATTRIBUTED"} else 2


if __name__=="__main__":
    raise SystemExit(main())
