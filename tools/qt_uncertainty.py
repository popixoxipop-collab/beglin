#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

def canon(o): return json.dumps(o,sort_keys=True,separators=(",",":"),allow_nan=False)
def clamp(x): return max(0.0,min(1.0,x))

def classify(cell,budget):
    ns=sorted(int(n) for n in cell["supported_n"]); errs=[float(cell["error_by_n"][str(n)]) for n in ns]
    rec=int(cell["recommended_n"]); er=float(cell["error_by_n"][str(rec)])
    safe=er<=budget
    monotonic=all(errs[i+1]<=errs[i]+1e-15 for i in range(len(errs)-1))
    nearest=min(abs(e-budget) for e in errs); boundary=clamp(1.0-nearest/max(budget,1e-12))
    samples=int(cell.get("sample_count",0)); epistemic=1.0/math.sqrt(max(1,samples))
    spread=(max(errs)-min(errs))/max(budget,1e-12); aleatoric=max(0.0,spread)
    # Conservative evidence score, deliberately not a calibrated posterior yet.
    margin=(budget-er)/max(budget,1e-12)
    safe_p=clamp(0.5+0.35*math.tanh(2.0*margin)-0.15*epistemic)
    if not safe: cls="UNSATISFIED"
    elif not monotonic: cls="NON_MONOTONIC"
    elif boundary>=0.8: cls="BOUNDARY"
    else: cls="SAFE"
    priority=clamp(max(epistemic,boundary,0.9 if cls=="UNSATISFIED" else 0.8 if cls=="NON_MONOTONIC" else 0.0))
    return {"target_key":cell["target_key"],"classification":cls,"recommended_n":rec,"safe_probability":safe_p,"epistemic_uncertainty":epistemic,"aleatoric_proxy":aleatoric,"boundary_score":boundary,"active_observation_priority":priority}

def build(qmap):
    budget=float(qmap["error_budget"]); cells=[classify(c,budget) for c in qmap["cells"]]
    dist={k:sum(c["classification"]==k for c in cells) for k in ["SAFE","BOUNDARY","NON_MONOTONIC","UNSATISFIED"]}
    out={"schema":"beglin-q-uncertainty-v1","source_heatmap_sha256":qmap["heatmap_sha256"],"error_budget":budget,"cells":cells,"classification_distribution":dist}
    out["uncertainty_sha256"]=hashlib.sha256(canon(out).encode()).hexdigest(); return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--qheatmap",required=True);p.add_argument("--output",required=True);p.add_argument("--summary-output",required=True);a=p.parse_args()
    q=json.load(open(a.qheatmap));o=build(q);Path(a.output).write_text(json.dumps(o,sort_keys=True,indent=2)+"\n")
    ranked=sorted(o["cells"],key=lambda c:(-c["active_observation_priority"],c["target_key"]))
    summary={"schema":"beglin-q-uncertainty-summary-v1","status":"PASS","source_heatmap_sha256":o["source_heatmap_sha256"],"uncertainty_sha256":o["uncertainty_sha256"],"classification_distribution":o["classification_distribution"],"top_active_observation_targets":ranked[:64],"automatic_live_promotion":False}
    Path(a.summary_output).write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n");print(json.dumps(summary,sort_keys=True))
if __name__=="__main__":main()
