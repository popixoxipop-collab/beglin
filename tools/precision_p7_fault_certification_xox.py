#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil, time, urllib.request
from pathlib import Path
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
import precision_closed_loop_acceptance_xox as p2

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def live():
    with urllib.request.urlopen("http://127.0.0.1:18765/healthz",timeout=5) as f: x=json.load(f)
    return {"generation":x["route_generation"],"route":x["route_manifest_sha256"],
            "pids":{k:v["pid"] for k,v in x["workers"].items()},
            "binary":sha("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--binary",required=True); ap.add_argument("--root",required=True)
    a=ap.parse_args(); binary=Path(a.binary).resolve(strict=True); root=Path(a.root).resolve()
    if root.parent!=Path("/Users/xox/vdsp_serving") or not root.name.startswith("precision-p7-fault-cert-"):
        raise SystemExit("unsafe root")
    root.mkdir(exist_ok=False); ps.PERSISTENT_BINARY=binary; before=live()
    out={"schema":"beglin-p7-fault-certification-v1","production_before":before,"production_touched":False}
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/"candidate")
    try:
        worker.start(); prompt=base._read_first_certified_prompt(); req=[(prompt,10)]
        pid1=worker.health()["pid"]; startup_hash=pc.policy_hash(p2.STARTUP)
        baseline=worker.submit_with_precision_policy(req,target_policy=p2.STARTUP,admission_id="fault-baseline")
        # ACK corruption must fail before inference/transition.
        raw=json.loads(worker.ack_path.read_text()); raw["weight_epoch"]=int(raw["weight_epoch"])+999
        worker.ack_path.write_text(json.dumps(raw))
        ack_fault={}
        try:
            worker.submit_with_precision_policy(req,target_policy=[
                {"role":"shared_up_proj","layer":3,"n":5},
                {"role":"shared_down_proj","layer":26,"n":6},
            ],admission_id="fault-corrupt-ack")
            ack_fault={"status":"FAIL","reason":"unexpected admission"}
        except Exception as exc:
            ack_fault={"status":"PASS","type":type(exc).__name__,"message":str(exc)}
        # Kill the isolated worker, prove dead submissions fail, then clean restart.
        worker.stop(force=True)
        dead_fault={}
        try:
            worker.submit(req)
            dead_fault={"status":"FAIL","reason":"dead worker served request"}
        except Exception as exc:
            dead_fault={"status":"PASS","type":type(exc).__name__,"message":str(exc)}
        worker.start(); pid2=worker.health()["pid"]
        recovered=worker.submit_with_precision_policy(req,target_policy=p2.STARTUP,admission_id="fault-restart")
        health=worker.health(); after=live()
        assertions={
            "baseline_finite":baseline["finite_logits"],
            "ack_corruption_fail_closed":ack_fault["status"]=="PASS",
            "dead_worker_fail_closed":dead_fault["status"]=="PASS",
            "new_pid_after_restart":pid2!=pid1,
            "restart_finite":recovered["finite_logits"],
            "startup_policy_restored":health["runtime_policy_hash"]==startup_hash,
            "production_unchanged":before==after,
        }
        out.update(status="PASS" if all(assertions.values()) else "FAIL",assertions=assertions,
                   pid_before_kill=pid1,pid_after_restart=pid2,ack_fault=ack_fault,
                   dead_worker_fault=dead_fault,restart_health=health,production_after=after)
    except Exception as exc:
        out.update(status="FAIL",error_type=type(exc).__name__,error=str(exc))
    finally:
        worker.stop(force=True)
    raw=json.dumps(out,sort_keys=True,indent=2)+"\n"; (root/"result.json").write_text(raw)
    print(json.dumps({"status":out["status"],"assertions":out.get("assertions"),"ack_fault":out.get("ack_fault"),
                      "dead_worker_fault":out.get("dead_worker_fault"),"pid_before_kill":out.get("pid_before_kill"),
                      "pid_after_restart":out.get("pid_after_restart"),"result_sha256":hashlib.sha256(raw.encode()).hexdigest()},indent=2))
    return 0 if out["status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
