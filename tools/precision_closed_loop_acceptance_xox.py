#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
import precision_allocator as pa
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps

L3=Path("/Users/xox/vdsp_serving/l3n6-to-n5-trigger-evidence-20261003/result.json")
L3_SHA="b16764b20addac627aa523d3f5b0c1a5cb9f4d9a4bb03ca8010eeecd5418bee7"
L26=Path("/Users/xox/vdsp_serving/BEGLIN_DYNAMIC_A6F3E6D_LIVE_CUTOVER_2026-10-03.json")
L26_SHA="4056304c9b978883aa85e56dcbcd644982d5d4edc991497f33e5a1f7f61211d2"
COMBINED=Path("/Users/xox/vdsp_serving/precision-epoch-acceptance-ae1b06b/result.json")
COMBINED_SHA="5934f6d3f596c5d69baf97600af87c7fc41a844be50c3234fecd5148e0953b34"
STARTUP=[{"role":"shared_up_proj","layer":3,"n":6},{"role":"shared_down_proj","layer":26,"n":5}]
TARGET=[{"role":"shared_up_proj","layer":3,"n":5},{"role":"shared_down_proj","layer":26,"n":6}]
TARGET_HASH="e471dfb075ca86cd0c5341b972258df7991847bb3f1557b69447c8b6168042ae"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def load(path, expected):
    if sha(path)!=expected: raise RuntimeError(f"evidence SHA mismatch: {path}")
    out=json.loads(path.read_text())
    if not isinstance(out,dict): raise RuntimeError(f"invalid evidence: {path}")
    return out
def source_head():
    return subprocess.check_output(
        ["git","-C",str(Path(__file__).resolve().parents[1]),"rev-parse","HEAD"],
        text=True).strip()

def cand(role,layer,n,size,feasible,conditional=False,recoveries=None,prov=None):
    numel=int(size["numel"])
    return {
        "role":role,"layer":layer,"n":n,"feasible":feasible,
        "conditionally_feasible":conditional,
        "conditional_recoveries":list(recoveries or []),
        "blocked_reasons":[] if feasible else ["REAL_FAIL_EVENT"],
        "real_pass_events":4,"real_fail_events":1 if conditional else 0,
        "real_event_count":5,"real_pass_event_keys":[],"real_fail_event_keys":[],
        "g4_context_hash":f"certified-g4-{role}-{layer}-{n}",
        "g6_context_hash":f"certified-g6-{role}-{layer}-{n}",
        "tensor_count":int(size["tensor_count"]),"numel":numel,
        "effective_bpw":pa.effective_bpw(n),
        "estimated_bytes":numel*pa.effective_bpw(n)/8.0,
        "persistent_p50_ms":None,"persistent_p95_ms":None,
        "persistent_rss_bytes":None,"benchmark_pid":None,
        "acceptance_provenance":prov or {},
    }

def evidence_snapshot():
    l3=load(L3,L3_SHA)
    if not (l3.get("status")=="PASS" and l3.get("role")=="shared_up_proj"
            and int(l3.get("layer",-1))==3 and int(l3.get("from_n",-1))==6
            and int(l3.get("to_n",-1))==5 and int(l3.get("base_failures",0))>0
            and int(l3.get("target_failures",1))==0 and l3.get("same_pid") is True):
        raise RuntimeError("L3 evidence contract mismatch")
    l26=load(L26,L26_SHA)
    tr=l26.get("trigger_requests") or []
    if not (l26.get("status")=="PASS" and len(tr)>=2):
        raise RuntimeError("L26 evidence contract mismatch")
    margins=[]
    for req in tr:
        ap=req.get("adaptive_precision") or {}
        if ap.get("action")!="RECOVERY_N6": raise RuntimeError("L26 recovery missing")
        rows=[r for r in req.get("neartie_events",[]) if r.get("phase")=="base_n5"]
        if not rows: raise RuntimeError("L26 base event missing")
        margins += [float(r["margin"]) for r in rows]
    if any(m>0.02 for m in margins): raise RuntimeError("L26 margin outside bucket")
    combined=load(COMBINED,COMBINED_SHA)
    if not (combined.get("status")=="PASS"
            and combined.get("source_head")=="ae1b06b3fbb86930e4c23e51f6ff82373272cde5"
            and combined.get("production_touched") is False
            and combined.get("binary_sha256")=="4540599d86a59dd3cfd14fc863d21ed3f0cc5a8a1d872a821a04bc04ae23b490"):
        raise RuntimeError("combined evidence contract mismatch")
    hits=[r for r in combined.get("sequential",{}).get("rows",[])
          if r.get("runtime_policy_hash")==TARGET_HASH and r.get("finite_logits") is True
          and len(r.get("tokens") or [])>8 and r["tokens"][8]==1224]
    if len(hits)<2: raise RuntimeError("combined policy not repeatedly accepted")
    sizes={
        ("shared_up_proj",3):pa.target_numel(base.SAFETENSORS,"shared_up_proj",3),
        ("shared_down_proj",26):pa.target_numel(base.SAFETENSORS,"shared_down_proj",26),
    }
    candidates=[
        cand("shared_up_proj",3,5,sizes[("shared_up_proj",3)],True,prov={"sha256":L3_SHA}),
        cand("shared_up_proj",3,6,sizes[("shared_up_proj",3)],True,prov={"sha256":L3_SHA}),
        cand("shared_down_proj",26,5,sizes[("shared_down_proj",26)],False,True,
             [{"to_n":6,"trigger_type":"low_margin","signal_bucket":{"margin_max":0.02},
               "evidence_sha256":l26.get("adaptive_evidence_sha256")}],
             {"sha256":L26_SHA}),
        cand("shared_down_proj",26,6,sizes[("shared_down_proj",26)],True,prov={"sha256":L26_SHA}),
    ]
    trigger=[{
        "evidence_id":"xox-l26-production-low-margin","role":"shared_down_proj","layer":26,
        "from_n":5,"to_n":6,"trigger_type":"low_margin","status":"PASS","pass":True,
        "requests":len(tr),"signal_bucket":{"margin_max":0.02},
        "evidence_sha256":l26.get("adaptive_evidence_sha256"),
        "metrics":{"base_failures":len(tr),"target_failures":0},
    }]
    combo=[{
        "policy_hash":TARGET_HASH,"status":"PASS","pass":True,
        "production_touched":False,"evidence_sha256":COMBINED_SHA,
        "source_head":combined["source_head"],"binary_sha256":combined["binary_sha256"],
    }]
    return candidates,trigger,combo,{"l3":L3_SHA,"l26":L26_SHA,"combined":COMBINED_SHA,"margins":margins}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--binary",required=True)
    ap.add_argument("--root",default="/Users/xox/vdsp_serving/precision-closed-loop-acceptance-v1")
    args=ap.parse_args(); binary=Path(args.binary).resolve(strict=True); root=Path(args.root).resolve()
    serving=Path("/Users/xox/vdsp_serving").resolve()
    if root==serving or serving not in root.parents: raise SystemExit("unsafe acceptance root")
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    candidates,trigger,combo,src=evidence_snapshot()
    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/"candidate")
    out={"schema":"beglin-precision-closed-loop-xox-acceptance-v1",
         "source_head":source_head(),"binary_sha256":sha(binary),
         "production_touched":False,"source_evidence":src}
    try:
        worker.start(); pid=worker.health()["pid"]
        cfg=worker.configure_precision_closed_loop(
            candidates=candidates,trigger_evidence=trigger,combined_policy_evidence=combo)
        prompt=base._read_first_certified_prompt()
        signal={"low_margin":True,"margin":0.010715}
        first=worker.submit_with_closed_loop_precision(
            [(prompt,10)],signal=signal,admission_id="closed-loop-risk-1")
        h1=worker.health()
        try:
            worker.submit_with_closed_loop_precision(
                [(prompt,10)],signal={"low_margin":True,"margin":0.03},
                admission_id="closed-loop-unaccepted-base")
            blocked={"status":"FAIL","reason":"unexpectedly admitted"}
        except Exception as exc:
            blocked={"status":"PASS","type":type(exc).__name__,"message":str(exc),
                     "runtime_policy_hash":worker.health()["runtime_policy_hash"]}
        second=worker.submit_with_closed_loop_precision(
            [(prompt,10)],signal=signal,admission_id="closed-loop-risk-2")
        h2=worker.health()
        restored=worker.submit_with_precision_policy(
            [(prompt,10)],target_policy=STARTUP,admission_id="closed-loop-restore")
        hf=worker.health()
        out.update({"pid":pid,"configuration":cfg,
                    "first":{"token8":first["responses"][0][8],
                             "finite_logits":first["finite_logits"],
                             "epoch":first["precision_epoch"],
                             "closed_loop":first["precision_closed_loop"],"health":h1},
                    "blocked":blocked,
                    "second":{"token8":second["responses"][0][8],
                              "finite_logits":second["finite_logits"],
                              "epoch":second["precision_epoch"],
                              "closed_loop":second["precision_closed_loop"],"health":h2},
                    "restore":{"finite_logits":restored["finite_logits"],
                               "epoch":restored["precision_epoch"],"health":hf}})
        ok=(first["finite_logits"] and first["responses"][0][8]==1224
            and first["precision_closed_loop"]["selected_policy_hash"]==TARGET_HASH
            and len(first["precision_closed_loop"]["changes"])==2
            and h1["pid"]==pid and blocked["status"]=="PASS"
            and blocked["runtime_policy_hash"]==TARGET_HASH
            and second["finite_logits"] and second["responses"][0][8]==1224
            and second["precision_epoch"]["transitioned"] is False and h2["pid"]==pid
            and hf["runtime_policy_hash"]==pc.policy_hash(STARTUP) and hf["pid"]==pid)
        out["status"]="PASS" if ok else "FAIL"
    finally:
        worker.stop(force=True)
    raw=json.dumps(out,indent=2,sort_keys=True)+"\n"
    (root/"result.json").write_text(raw)
    print(raw,end=""); print("RESULT_SHA256",hashlib.sha256(raw.encode()).hexdigest())
    return 0 if out["status"]=="PASS" else 2

if __name__=="__main__":
    raise SystemExit(main())
