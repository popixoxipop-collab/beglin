#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,uuid
from pathlib import Path

import gpu_runtime_control as grc
import precision_dynamic_selector as ds
import precision_risk_signals as prs
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps

PROMPT=[23757,49238,28947,3969,319,317,38470,440,6383]
STARTUP=[
    {"role":"shared_up_proj","layer":3,"n":6},
    {"role":"shared_down_proj","layer":26,"n":5},
]
L26_EVIDENCE=Path("/Users/xox/vdsp_serving/BEGLIN_DYNAMIC_A6F3E6D_LIVE_CUTOVER_2026-10-03.json")
L26_SHA="4056304c9b978883aa85e56dcbcd644982d5d4edc991497f33e5a1f7f61211d2"

def source_head():
    return subprocess.check_output(
        ["git","-C",str(Path(__file__).resolve().parents[1]),"rev-parse","HEAD"],
        text=True,
    ).strip()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def reb(worker,old,new):
    ack=grc.read_runtime_ack(worker.ack_path); tid="p5-"+uuid.uuid4().hex
    grc.prepare_rebind(
        ack_path=worker.ack_path,txn_path=worker.txn_path,txn_id=tid,
        expected_epoch=int(ack["weight_epoch"]),
        expected_policy_hash=str(ack["active_policy_hash"]),
        role="shared_up_proj",layer=3,expected_n=old,target_n=new)
    worker.submit_base([([1],1)])
    return grc.verify_terminal_ack(
        ack_path=worker.ack_path,txn_id=tid,
        allowed_statuses={"REBIND_APPLIED"})
def sample(worker,count):
    rows=[]
    for _ in range(count):
        got=worker.submit_base([(PROMPT,2)])
        rows.append({
            "tokens":got["responses"][0],
            "events":got.get("neartie_events",[]),
            "engine_wall_ms":got["engine_wall_ms"],
            "inference_passes":got.get("inference_passes",1),
        })
    return rows

def allocator_decision():
    return {"targets":[
        {
            "role":"shared_up_proj","layer":3,"status":"PROPOSED",
            "selected_n":6,"current_n":6,
            "dynamic_escalation":{"candidate_alternates":[
                {"n":5,"real_pass_events":6,"persistent_p50_ms":None}
            ]},
        },
        {
            "role":"shared_down_proj","layer":26,"status":"PROPOSED",
            "selected_n":5,"current_n":5,
            "dynamic_escalation":{"candidate_alternates":[
                {"n":6,"real_pass_events":2,"persistent_p50_ms":None}
            ]},
        },
    ]}

def selected_map(result):
    return {(r["role"],int(r["layer"])):int(r["selected_n"])
            for r in result["targets"]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary",required=True)
    ap.add_argument("--root",default="/Users/xox/vdsp_serving/precision-risk-signals-acceptance-v1")
    ap.add_argument("--samples",type=int,default=6)
    args=ap.parse_args()
    binary=Path(args.binary).resolve(strict=True); root=Path(args.root).resolve()
    serving=Path("/Users/xox/vdsp_serving").resolve()
    if root==serving or serving not in root.parents: raise SystemExit("unsafe root")
    if root.exists():shutil.rmtree(root)
    root.mkdir(parents=True)
    if sha(L26_EVIDENCE)!=L26_SHA: raise RuntimeError("L26 evidence SHA mismatch")
    l26=json.loads(L26_EVIDENCE.read_text())
    trigger_requests=l26.get("trigger_requests") or []
    if not (l26.get("status")=="PASS" and len(trigger_requests)>=2):
        raise RuntimeError("L26 low-margin evidence contract mismatch")

    ps.PERSISTENT_BINARY=binary
    ps.NEARTIE_TELEMETRY_THRESHOLD=0.05
    os.environ["QWEN_MOE_RISK_SIGNALS"]="1"
    worker=ps.AdaptivePersistentRouteWorker(
        route=base.candidate_route(),root=root/"candidate")
    out={
        "schema":"beglin-precision-risk-signals-xox-acceptance-v1",
        "source_head":source_head(),
        "binary_sha256":sha(binary),"production_touched":False,
        "l26_evidence_sha256":L26_SHA,
    }
    try:
        start=worker.start(); pid=start["pid"]
        base_rows=sample(worker,args.samples)
        ack5=reb(worker,6,5)
        target_rows=sample(worker,args.samples)
        ack6=reb(worker,5,6)
        return_rows=sample(worker,max(3,args.samples//2))
        base_fail=sum(r["tokens"][1]!=1 for r in base_rows)
        target_fail=sum(r["tokens"][1]!=1 for r in target_rows)
        return_fail=sum(r["tokens"][1]!=1 for r in return_rows)
        events=[e for r in base_rows for e in r["events"]
                if e.get("pos")==9 and e.get("predicted_token")==372
                and e.get("competing_token")==1]
        if len(events)!=args.samples:
            raise RuntimeError("missing P5 telemetry events")
        summary=prs.summarize_events(events)
        eps=1e-6
        buckets={
            "near_tie":{
                "margin_min":max(0.0,summary["min_margin"]-eps),
                "margin_max":summary["max_margin"]+eps,
            },
            "high_entropy":{
                "entropy_min":max(0.0,summary["min_entropy"]-eps),
                "entropy_max":min(1.0,summary["max_entropy"]+eps),
            },
            "routing_ambiguity":{
                "routing_ambiguity_score_min":max(
                    0.0,summary["min_routing_ambiguity_score"]-eps),
                "routing_ambiguity_score_max":min(
                    1.0,summary["max_routing_ambiguity_score"]+eps),
            },
        }
        signal=prs.derive_signal(
            events,
            near_tie_margin_max=buckets["near_tie"]["margin_max"],
            high_entropy_min=buckets["high_entropy"]["entropy_min"],
            routing_ambiguity_min=buckets["routing_ambiguity"][
                "routing_ambiguity_score_min"],
        )
        if not (signal["near_tie"] and signal["high_entropy"]
                and signal["routing_ambiguity"]):
            raise RuntimeError("P5 measured signal did not activate all triggers")
        p5_evidence=[]
        for trigger in ("near_tie","high_entropy","routing_ambiguity"):
            p5_evidence.append({
                "evidence_id":f"xox-l3-6-to-5-{trigger}",
                "role":"shared_up_proj","layer":3,"from_n":6,"to_n":5,
                "trigger_type":trigger,"status":"PASS","pass":True,
                "requests":args.samples,"signal_bucket":buckets[trigger],
                "metrics":{"base_failures":base_fail,
                           "target_failures":target_fail},
            })
        l26_row={
            "evidence_id":"xox-l26-5-to-6-low-margin",
            "role":"shared_down_proj","layer":26,"from_n":5,"to_n":6,
            "trigger_type":"low_margin","status":"PASS","pass":True,
            "requests":len(trigger_requests),
            "signal_bucket":{"margin_max":0.02},
            "metrics":{"base_failures":len(trigger_requests),
                       "target_failures":0},
        }
        decision=allocator_decision()
        all_evidence=p5_evidence+[l26_row]
        p5_selection=ds.select(decision,signal,all_evidence)
        low_selection=ds.select(
            decision,{"low_margin":True,"margin":0.010715},all_evidence)
        missing_selection=ds.select(
            decision,signal,[p5_evidence[0],p5_evidence[2],l26_row])
        p5_map=selected_map(p5_selection)
        low_map=selected_map(low_selection)
        missing_map=selected_map(missing_selection)

        p5_policy=[
            {"role":"shared_up_proj","layer":3,
             "n":p5_map[("shared_up_proj",3)]},
            {"role":"shared_down_proj","layer":26,
             "n":p5_map[("shared_down_proj",26)]},
        ]
        applied=worker.submit_with_precision_policy(
            [(PROMPT,2)],target_policy=p5_policy,
            admission_id="p5-selected-policy")
        applied_health=worker.health()
        restored=worker.submit_with_precision_policy(
            [(PROMPT,2)],target_policy=STARTUP,
            admission_id="p5-restore-startup")
        final_health=worker.health()
        ok=(
            base_fail==args.samples and target_fail==0
            and return_fail==len(return_rows)
            and worker.health()["pid"]==pid
            and p5_map=={
                ("shared_up_proj",3):5,
                ("shared_down_proj",26):5,
            }
            and low_map=={
                ("shared_up_proj",3):6,
                ("shared_down_proj",26):6,
            }
            and missing_map=={
                ("shared_up_proj",3):6,
                ("shared_down_proj",26):5,
            }
            and applied["finite_logits"]
            and applied["responses"][0][1]==1
            and final_health["runtime_policy_hash"]==ps._policy_hash(STARTUP)
            and final_health["pid"]==pid
        )
        out.update({
            "status":"PASS" if ok else "FAIL","pid":pid,
            "base_failures":base_fail,"target_failures":target_fail,
            "return_failures":return_fail,"signal_summary":summary,
            "signal":signal,"buckets":buckets,
            "p5_evidence":p5_evidence,
            "p5_selection":p5_selection,
            "low_margin_selection":low_selection,
            "missing_evidence_selection":missing_selection,
            "ack_to_n5":ack5,"ack_return_n6":ack6,
            "applied_p5_policy":{
                "policy":p5_policy,
                "tokens":applied["responses"][0],
                "precision_epoch":applied["precision_epoch"],
                "health":applied_health,
            },
            "restore":{
                "tokens":restored["responses"][0],
                "precision_epoch":restored["precision_epoch"],
                "health":final_health,
            },
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
