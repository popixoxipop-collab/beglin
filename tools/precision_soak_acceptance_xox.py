#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, threading, time
from pathlib import Path
import precision_context as pc
import precision_observability as po
import precision_soak as soak
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps

A=[{"role":"shared_up_proj","layer":3,"n":6},{"role":"shared_down_proj","layer":26,"n":5}]
B=[{"role":"shared_up_proj","layer":3,"n":5},{"role":"shared_down_proj","layer":26,"n":6}]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--binary",required=True)
    ap.add_argument("--root",required=True); ap.add_argument("--cycles",type=int,default=24)
    args=ap.parse_args(); binary=Path(args.binary).resolve(strict=True); root=Path(args.root).resolve()
    serving=Path("/Users/xox/vdsp_serving").resolve()
    if root==serving or serving not in root.parents: raise SystemExit("unsafe root")
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/"worker")
    lineage=root/"lineage.jsonl"; summary_path=root/"summary.json"
    rows=[]; rss=[]; failures={}
    try:
        worker.start(); pid=worker.health()["pid"]; prompt=base._read_first_certified_prompt()
        worker.configure_precision_observability(lineage_path=lineage,snapshot_path=summary_path,strict=True)
        for i in range(args.cycles):
            for suffix,policy in (("b",B),("a",A)):
                got=worker.submit_with_precision_policy([(prompt,2)],target_policy=policy,admission_id=f"soak-{i:04d}-{suffix}")
                h=worker.health(); ep=got["precision_epoch"]; tc=ep["transition_cost"]
                rows.append({"pid":h["pid"],"before_epoch":ep["before_epoch"],"after_epoch":ep["after_epoch"],
                             "policy_hash":h["runtime_policy_hash"],"cache_hits":tc["cache_hits"],
                             "cache_misses":tc["cache_misses"],"cache_bytes_added":tc["cache_bytes_added"],
                             "resident_cache_bytes":tc["resident_cache_bytes"],"rss_bytes":h["rss_bytes"]})
                rss.append(h["rss_bytes"])
        soak.verify_epoch_sequence(rows); soak.verify_cache_plateau(rows,warmup=4)
        # concurrent admissions: scheduler/worker RLock must serialize exact policies.
        conc=[]; errs=[]; lock=threading.Lock()
        def invoke(name,policy,delay):
            time.sleep(delay)
            try:
                got=worker.submit_with_precision_policy([(prompt,2)],target_policy=policy,admission_id=name)
                with lock: conc.append({"name":name,"epoch":got["precision_epoch"],"pid":worker.health()["pid"]})
            except Exception as exc:
                with lock: errs.append(repr(exc))
        t1=threading.Thread(target=invoke,args=("concurrent-b",B,0.0)); t2=threading.Thread(target=invoke,args=("concurrent-a",A,0.01))
        t1.start(); t2.start(); t1.join(); t2.join()
        conc.sort(key=lambda x:x["epoch"]["before_epoch"])
        if errs or len(conc)!=2 or conc[0]["epoch"]["after_epoch"]!=conc[1]["epoch"]["before_epoch"]:
            raise RuntimeError(f"concurrency serialization failed: {errs} {conc}")
        # ACK corruption must fail closed before inference; restore exact bytes afterward.
        ack=worker.ack_path; original=soak.corrupt_json(ack)
        try:
            worker.submit_with_precision_policy([(prompt,1)],target_policy=B,admission_id="fault-ack")
            failures["ack_corruption"]="FAIL"
        except Exception as exc:
            failures["ack_corruption"]="PASS:"+type(exc).__name__
        finally: soak.restore_bytes(ack,original)
        # lineage tamper must be detected; restore pristine bytes afterward.
        pristine=lineage.read_bytes(); soak.corrupt_lineage(lineage)
        failures["lineage_corruption"]="PASS" if soak.verify_lineage_failure(lineage) else "FAIL"
        soak.restore_bytes(lineage,pristine)
        worker.precision_observability._reload()
        # Worker crash: kill scratch child, prove it is dead, then start a fresh scratch worker.
        old_pid=worker.health()["pid"]; soak.terminate_worker(worker.proc)
        failures["worker_crash"]="PASS" if not worker.is_alive() else "FAIL"
        worker.stop(force=True)
        worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/"restart-worker")
        worker.start(); new_pid=worker.health()["pid"]
        restart=worker.submit_with_precision_policy([(prompt,2)],target_policy=B,admission_id="restart-b")
        restart_ok=(new_pid!=old_pid and restart["finite_logits"] and worker.health()["runtime_policy_hash"]==pc.policy_hash(B))
        failures["worker_restart"]="PASS" if restart_ok else "FAIL"
        # standalone retention exercise; production journal is never touched.
        rot=root/"rotation.jsonl"
        observer=po.PrecisionObservability(lineage_path=rot,max_bytes=1500,max_rotated_files=3)
        decision={"signal":{"active_triggers":[]},"selected_policy":[],"changes":[],"selection":{"targets":[]}}
        for i in range(20):
            result={"finite_logits":True,"responses":[[i]],"precision_epoch":{
                "before_policy_hash":"a"*64,"after_policy_hash":"a"*64,"before_epoch":0,"after_epoch":0,
                "transitioned":False,"changed_targets":[],"transition_cost":{},"inference_passes":1,
                "engine_wall_ms":1.0,"roundtrip_ms":2.0}}
            observer.record(admission_id=f"rot-{i}",worker_pid=999,request_count=1,decision=decision,result=result)
        rotated=soak.assert_retention(rot,3)
        rss_info=soak.rss_drift(rss)
        # generous guard: catch runaway growth, not allocator noise.
        rss_ok=rss_info["delta"] <= 256*1024*1024 and rss_info["peak"]-rss_info["start"] <= 384*1024*1024
        out={"schema":"beglin-precision-soak-xox-v1","status":"PASS" if rss_ok and all(str(v).startswith("PASS") for v in failures.values()) else "FAIL",
             "production_touched":False,"binary_sha256":sha(binary),"cycles":args.cycles,"transition_admissions":len(rows),
             "pid":pid,"epoch_start":rows[0]["before_epoch"],"epoch_end":rows[-1]["after_epoch"],
             "cache":{"misses":sum(x["cache_misses"] for x in rows),"hits":sum(x["cache_hits"] for x in rows),
                      "bytes_added":sum(x["cache_bytes_added"] for x in rows),"resident_final":rows[-1]["resident_cache_bytes"]},
             "rss":rss_info,"rss_guard_pass":rss_ok,"concurrency":conc,"failure_injection":failures,
             "rotation":{"rotated_files":len(rotated),"max_files":3},"lineage_records":len(po.PrecisionObservability(lineage_path=lineage).read_records())}
    finally:
        worker.stop(force=True)
    raw=json.dumps(out,indent=2,sort_keys=True)+"\n"; (root/"result.json").write_text(raw)
    print(raw,end=""); print("RESULT_SHA256",hashlib.sha256(raw.encode()).hexdigest())
    return 0 if out["status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
