#!/usr/bin/env python3
"""XOX isolated P4 acceptance: measured policy cost -> closed-loop scheduler."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

import precision_context as pc
import precision_closed_loop_acceptance_xox as p2
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps

A=p2.STARTUP
B=p2.TARGET
AH=pc.policy_hash(A); BH=pc.policy_hash(B)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def head():
    return subprocess.check_output(
        ["git","-C",str(Path(__file__).resolve().parents[1]),"rev-parse","HEAD"],
        text=True).strip()

def load_cost(path, expected_sha):
    path=Path(path).resolve(strict=True)
    if sha(path)!=expected_sha: raise RuntimeError("cost evidence SHA mismatch")
    out=json.loads(path.read_text())
    if not (out.get("status")=="PASS" and out.get("production_touched") is False):
        raise RuntimeError("cost evidence contract mismatch")
    return out

def accepted_rows(combo):
    b=dict(combo[0]); b["policy"]=B
    a={
        "policy":A,"policy_hash":AH,"status":"PASS","pass":True,
        "production_touched":False,
        "evidence_sha256":p2.COMBINED_SHA,
        "source_head":"ae1b06b3fbb86930e4c23e51f6ff82373272cde5",
    }
    return [a,b]

def meta(got):
    return {
        "finite_logits":got["finite_logits"],
        "token8":got["responses"][0][8] if len(got["responses"][0])>8 else None,
        "epoch":got["precision_epoch"],
        "closed_loop":got["precision_closed_loop"],
    }
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary",required=True)
    ap.add_argument("--root",default="/Users/xox/vdsp_serving/precision-e2e-cost-acceptance-v1")
    ap.add_argument("--cost-evidence",required=True)
    ap.add_argument("--cost-sha256",required=True)
    args=ap.parse_args()
    binary=Path(args.binary).resolve(strict=True); root=Path(args.root).resolve()
    serving=Path("/Users/xox/vdsp_serving").resolve()
    if root==serving or serving not in root.parents: raise SystemExit("unsafe root")
    if root.exists():shutil.rmtree(root)
    root.mkdir(parents=True)
    candidates,trigger,combo,src=p2.evidence_snapshot()
    cost=load_cost(args.cost_evidence,args.cost_sha256)
    if cost.get("binary_sha256")!=sha(binary):
        raise RuntimeError("cost evidence binary mismatch")
    accepted=accepted_rows(combo)
    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/"candidate")
    out={
        "schema":"beglin-precision-e2e-cost-xox-acceptance-v1",
        "source_head":head(),"binary_sha256":sha(binary),
        "cost_evidence_sha256":args.cost_sha256,"production_touched":False,
        "cost_evidence_source_head":cost.get("source_head"),
        "source_evidence":src,
    }
    try:
        worker.start(); pid=worker.health()["pid"]; prompt=base._read_first_certified_prompt()
        cfg=worker.configure_precision_closed_loop(
            candidates=candidates,trigger_evidence=trigger,
            combined_policy_evidence=accepted,cost_evidence=cost,
            policy_e2e_weight=1.0,
        )
        no_risk=worker.submit_with_closed_loop_precision(
            [(prompt,10)],signal={},admission_id="p4-no-risk-startup")
        h0=worker.health()
        risk_cold=worker.submit_with_closed_loop_precision(
            [(prompt,10)],signal={"low_margin":True,"margin":0.010715},
            admission_id="p4-risk-cold")
        h1=worker.health()
        no_risk_on_b=worker.submit_with_closed_loop_precision(
            [(prompt,10)],signal={},admission_id="p4-no-risk-hold-b")
        h2=worker.health()
        restore=worker.submit_with_precision_policy(
            [(prompt,10)],target_policy=A,admission_id="p4-restore-a")
        hr=worker.health()
        risk_warm=worker.submit_with_closed_loop_precision(
            [(prompt,10)],signal={"low_margin":True,"margin":0.010715},
            admission_id="p4-risk-warm")
        hw=worker.health()
        final_restore=worker.submit_with_precision_policy(
            [(prompt,10)],target_policy=A,admission_id="p4-final-restore")
        hf=worker.health()
        m0=meta(no_risk); mc=meta(risk_cold); mb=meta(no_risk_on_b); mw=meta(risk_warm)
        cold_cost=mc["epoch"]["transition_cost"]; warm_cost=mw["epoch"]["transition_cost"]
        no_risk_opt=m0["closed_loop"]["policy_cost_optimizer"]
        cold_opt=mc["closed_loop"]["policy_cost_optimizer"]
        hold_opt=mb["closed_loop"]["policy_cost_optimizer"]
        warm_opt=mw["closed_loop"]["policy_cost_optimizer"]
        out.update({
            "status":"PASS","pid":pid,"configuration":cfg,
            "no_risk_startup":{**m0,"health":h0},
            "risk_cold":{**mc,"health":h1},
            "no_risk_on_target":{**mb,"health":h2},
            "restore":{ "epoch":restore["precision_epoch"],"health":hr},
            "risk_warm":{**mw,"health":hw},
            "final_restore":{"epoch":final_restore["precision_epoch"],"health":hf},
        })
        ok=(
            h0["pid"]==pid and no_risk_opt["selected_policy_hash"]==AH
            and m0["epoch"]["transitioned"] is False
            and cold_opt["selected_policy_hash"]==BH
            and mc["token8"]==1224 and cold_cost["cache_misses"]==2
            and cold_cost["cache_bytes_added"]>0
            and hold_opt["selected_policy_hash"]==BH
            and mb["epoch"]["transitioned"] is False
            and hr["runtime_policy_hash"]==AH
            and warm_opt["selected_policy_hash"]==BH
            and mw["token8"]==1224 and warm_cost["cache_hits"]==2
            and warm_cost["cache_misses"]==0 and warm_cost["cache_bytes_added"]==0
            and warm_cost["transition_wall_ms"] < cold_cost["transition_wall_ms"]
            and warm_opt["selected_cost"]["expected_e2e_ms"]
                < cold_opt["selected_cost"]["expected_e2e_ms"]
            and hf["runtime_policy_hash"]==AH and hf["pid"]==pid
        )
        out["status"]="PASS" if ok else "FAIL"
    finally:
        worker.stop(force=True)
    raw=json.dumps(out,indent=2,sort_keys=True)+"\n"
    (root/"result.json").write_text(raw)
    print(raw,end="")
    print("RESULT_SHA256",hashlib.sha256(raw.encode()).hexdigest())
    return 0 if out["status"]=="PASS" else 2

if __name__=="__main__":
    raise SystemExit(main())
