#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys, threading, time
from pathlib import Path
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps

DEFAULT_ROOT = Path("/Users/xox/vdsp_serving/precision-epoch-acceptance-v1")

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def source_head():
    try:
        return subprocess.check_output(["git","-C",str(Path(__file__).resolve().parents[1]),"rev-parse","HEAD"],text=True).strip()
    except Exception:
        return None

def checked_root(path: Path) -> Path:
    root=path.expanduser().resolve()
    serving=Path("/Users/xox/vdsp_serving").resolve()
    if root==serving or serving not in root.parents:
        raise SystemExit(f"root must be a child of {serving}: {root}")
    for item in (serving/"persistent-workers",serving/"persistent-stage"):
        if root==item or item in root.parents:
            raise SystemExit(f"refusing production path: {root}")
    return root

def policies():
    return (
        [{"role":"shared_up_proj","layer":3,"n":6},{"role":"shared_down_proj","layer":26,"n":5}],
        [{"role":"shared_up_proj","layer":3,"n":5},{"role":"shared_down_proj","layer":26,"n":6}],
    )

def sequential(binary: Path, root: Path) -> dict:
    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/"candidate")
    base_policy,target_policy=policies(); rows=[]
    try:
        worker.start(); pid=worker.health()["pid"]; prompt=base._read_first_certified_prompt()
        for name,policy in (("target-1",target_policy),("base",base_policy),("target-2",target_policy)):
            got=worker.submit_with_precision_policy([(prompt,10)],target_policy=policy,admission_id=name)
            h=worker.health()
            rows.append({"name":name,"finite_logits":got["finite_logits"],"tokens":got["responses"][0],
                         "precision_epoch":got["precision_epoch"],"runtime_policy_hash":h["runtime_policy_hash"],"pid":h["pid"]})
        log=worker.log_path.read_text(errors="ignore")
        cache_hits={
            "shared_up_L3_n5":log.count("CACHE_HIT role=shared_up_proj layer=3 n=5"),
            "shared_down_L26_n6":log.count("CACHE_HIT role=shared_down_proj layer=26 n=6"),
            "shared_up_L3_n6":log.count("CACHE_HIT role=shared_up_proj layer=3 n=6"),
            "shared_down_L26_n5":log.count("CACHE_HIT role=shared_down_proj layer=26 n=5"),
        }
        ok=(all(r["pid"]==pid and r["finite_logits"] for r in rows)
            and all(r["precision_epoch"]["transitioned"] for r in rows)
            and all(len(r["precision_epoch"]["changed_targets"])==2 for r in rows)
            and all(r["precision_epoch"]["after_epoch"]==r["precision_epoch"]["before_epoch"]+1 for r in rows)
            and rows[0]["runtime_policy_hash"]==pc.policy_hash(target_policy)
            and rows[1]["runtime_policy_hash"]==pc.policy_hash(base_policy)
            and rows[2]["runtime_policy_hash"]==pc.policy_hash(target_policy)
            and cache_hits["shared_up_L3_n5"]>=1 and cache_hits["shared_down_L26_n6"]>=1)
        return {"status":"PASS" if ok else "FAIL","pid":pid,"rows":rows,"cache_hits":cache_hits}
    finally:
        worker.stop(force=True)

def concurrency(binary: Path, root: Path) -> dict:
    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/"candidate")
    base_policy,target_policy=policies(); results=[]; errors=[]; result_lock=threading.Lock()
    try:
        worker.start(); pid=worker.health()["pid"]; prompt=base._read_first_certified_prompt()
        def invoke(name,policy,delay):
            time.sleep(delay); started=time.monotonic()
            try:
                got=worker.submit_with_precision_policy([(prompt,10)],target_policy=policy,admission_id=name)
                row={"name":name,"started":started,"ended":time.monotonic(),"pid":worker.health()["pid"],
                     "finite_logits":got["finite_logits"],"precision_epoch":got["precision_epoch"]}
                with result_lock: results.append(row)
            except Exception as exc:
                with result_lock: errors.append({"name":name,"error":repr(exc)})
        a=threading.Thread(target=invoke,args=("concurrent-target",target_policy,0.0))
        b=threading.Thread(target=invoke,args=("concurrent-base",base_policy,0.02))
        a.start(); b.start(); a.join(); b.join()
        results.sort(key=lambda r:r["precision_epoch"]["before_epoch"])
        ok=(not errors and len(results)==2 and results[0]["name"]=="concurrent-target" and results[1]["name"]=="concurrent-base"
            and all(r["pid"]==pid and r["finite_logits"] for r in results)
            and results[0]["precision_epoch"]["after_epoch"]==results[1]["precision_epoch"]["before_epoch"]
            and results[0]["precision_epoch"]["after_policy_hash"]==pc.policy_hash(target_policy)
            and results[1]["precision_epoch"]["after_policy_hash"]==pc.policy_hash(base_policy))
        return {"status":"PASS" if ok else "FAIL","pid":pid,"results":results,"errors":errors,"final_health":worker.health()}
    finally:
        worker.stop(force=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary",required=True); ap.add_argument("--root",default=str(DEFAULT_ROOT)); ap.add_argument("--output")
    args=ap.parse_args()
    binary=Path(args.binary).expanduser().resolve(strict=True); root=checked_root(Path(args.root))
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    seq=sequential(binary,root/"sequential"); con=concurrency(binary,root/"concurrency")
    result={"schema":"beglin-precision-epoch-xox-acceptance-v1","status":"PASS" if seq["status"]==con["status"]=="PASS" else "FAIL",
            "source_head":source_head(),"binary":str(binary),"binary_sha256":sha256_file(binary),
            "production_touched":False,"sequential":seq,"concurrency":con}
    raw=json.dumps(result,indent=2,sort_keys=True)+"\n"
    output=Path(args.output).resolve() if args.output else root/"result.json"
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(raw)
    print(raw,end=""); print("RESULT_SHA256",hashlib.sha256(raw.encode()).hexdigest())
    return 0 if result["status"]=="PASS" else 2

if __name__=="__main__":
    raise SystemExit(main())
