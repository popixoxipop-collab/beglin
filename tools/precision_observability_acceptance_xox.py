#!/usr/bin/env python3
"""XOX isolated P6 acceptance for closed-loop lineage and aggregate metrics."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

import precision_closed_loop_acceptance_xox as p2
import precision_context as pc
import precision_e2e_cost_acceptance_xox as p4
import precision_observability as pob
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps

A=p2.STARTUP
B=p2.TARGET
AH=pc.policy_hash(A)
BH=pc.policy_hash(B)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def head():
    return subprocess.check_output(
        ["git","-C",str(Path(__file__).resolve().parents[1]),"rev-parse","HEAD"],
        text=True,
    ).strip()
def load_cost(path, expected_sha, binary_sha):
    path=Path(path).resolve(strict=True)
    if sha(path)!=expected_sha:
        raise RuntimeError("cost evidence SHA mismatch")
    out=json.loads(path.read_text())
    if not (
        out.get("status")=="PASS"
        and out.get("production_touched") is False
        and out.get("binary_sha256")==binary_sha
    ):
        raise RuntimeError("cost evidence contract mismatch")
    adaptive=out.get("adaptive_recovery_profile") or {}
    if not (
        adaptive.get("status")=="PASS"
        and adaptive.get("same_pid") is True
        and adaptive.get("observed_inference_passes")==[2]
    ):
        raise RuntimeError("cost evidence lacks measured adaptive pass profile")
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",required=True)
    ap.add_argument("--cost-evidence",required=True)
    ap.add_argument("--cost-sha256",required=True)
    ap.add_argument(
        "--root",
        default="/Users/xox/vdsp_serving/precision-observability-acceptance-v1",
    )
    args=ap.parse_args()
    binary=Path(args.binary).resolve(strict=True)
    binary_sha=sha(binary)
    root=Path(args.root).resolve()
    serving=Path("/Users/xox/vdsp_serving").resolve()
    if root==serving or serving not in root.parents:
        raise SystemExit("unsafe acceptance root")
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    candidates,trigger,combo,src=p2.evidence_snapshot()
    cost=load_cost(args.cost_evidence,args.cost_sha256,binary_sha)
    accepted=p4.accepted_rows(combo)

    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(
        route=base.candidate_route(),root=root/"candidate")
    lineage=root/"lineage.jsonl"
    snapshot=root/"observability-summary.json"
    out={
        "schema":"beglin-precision-observability-xox-acceptance-v1",
        "source_head":head(),
        "binary_sha256":binary_sha,
        "cost_evidence_sha256":args.cost_sha256,
        "production_touched":False,
        "source_evidence":src,
    }
    try:
        worker.start()
        pid=worker.health()["pid"]
        cfg=worker.configure_precision_closed_loop(
            candidates=candidates,
            trigger_evidence=trigger,
            combined_policy_evidence=accepted,
            cost_evidence=cost,
            policy_e2e_weight=1.0,
        )
        obs_cfg=worker.configure_precision_observability(
            lineage_path=lineage,
            snapshot_path=snapshot,
            strict=True,
        )
        prompt=base._read_first_certified_prompt()
        no_risk=worker.submit_with_closed_loop_precision(
            [(prompt,10)],signal={},admission_id="p6-no-risk-a")
        risk=worker.submit_with_closed_loop_precision(
            [(prompt,10)],
            signal={"low_margin":True,"margin":0.010715},
            admission_id="p6-low-margin-b",
        )
        hold=worker.submit_with_closed_loop_precision(
            [(prompt,10)],signal={},admission_id="p6-hold-b")
        restored=worker.submit_with_precision_policy(
            [(prompt,10)],target_policy=A,admission_id="p6-restore-a")
        final_health=worker.health()
        summary=worker.precision_observability.summary()
        records=worker.precision_observability.read_records()
        pob.verify_records(records)

        record_status=[
            no_risk.get("precision_observability",{}).get("status"),
            risk.get("precision_observability",{}).get("status"),
            hold.get("precision_observability",{}).get("status"),
            restored.get("precision_observability",{}).get("status"),
        ]
        expected_residency={
            "shared_down_proj/L26/n5":2,
            "shared_down_proj/L26/n6":2,
            "shared_up_proj/L3/n5":2,
            "shared_up_proj/L3/n6":2,
        }
        ok=(
            record_status==["RECORDED","RECORDED","RECORDED","RECORDED"]
            and summary["admissions"]==4
            and summary["requests"]==4
            and summary["trigger_counts"]=={"low_margin":1}
            and summary["triggered_admissions"]==1
            and summary["transitioned_admissions"]==2
            and summary["cache_misses"]==2
            and summary["cache_hits"]==2
            and summary["cache_hit_rate"]==0.5
            and summary["cache_bytes_added"]>0
            and summary["inference_pass_histogram"]=={"1":4}
            and summary["extra_pass_rate"]==0.0
            and summary["finite_logits_rate"]==1.0
            and summary["policy_residency_admissions"]=={AH:2,BH:2}
            and summary["precision_residency_admissions"]==expected_residency
            and summary["expected_e2e_ms_mean"] is not None
            and summary["actual_roundtrip_ms_mean"]>0
            and summary["lineage_head_sha256"]==records[-1]["record_sha256"]
            and records[0]["prev_record_sha256"] is None
            and records[1]["prev_record_sha256"]==records[0]["record_sha256"]
            and records[2]["prev_record_sha256"]==records[1]["record_sha256"]
            and risk["responses"][0][8]==1224
            and final_health["runtime_policy_hash"]==AH
            and final_health["pid"]==pid
        )
        out.update({
            "status":"PASS" if ok else "FAIL",
            "pid":pid,
            "closed_loop_configuration":cfg,
            "observability_configuration":obs_cfg,
            "record_status":record_status,
            "record_sha256":[r["record_sha256"] for r in records],
            "summary":summary,
            "lineage_path":str(lineage),
            "lineage_sha256":sha(lineage),
            "snapshot_path":str(snapshot),
            "snapshot_sha256":sha(snapshot),
            "risk_token8":risk["responses"][0][8],
            "restore_epoch":restored["precision_epoch"],
            "final_health":final_health,
        })
    finally:
        worker.stop(force=True)
    raw=json.dumps(out,indent=2,sort_keys=True)+"\n"
    (root/"result.json").write_text(raw)
    print(raw,end="")
    print("RESULT_SHA256",hashlib.sha256(raw.encode()).hexdigest())
    return 0 if out.get("status")=="PASS" else 2

if __name__=="__main__":
    raise SystemExit(main())
