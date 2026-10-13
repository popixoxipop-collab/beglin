#!/usr/bin/env python3
"""Verified archived P7 risk signal coverage for P2. Read-only, no response payload export.

P7 used one reference prompt and operator-injected low_margin. This is a negative
control: never call it an independent multi-prompt training dataset.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,collections
import precision_context as pc
import precision_observability as po

SCHEMA="beglin-hcp-p2-lineage-survey-v1"
EXPECTED_RESULT_SHA="ebc6c8c52759e7164b555d19e7e330bf933835be2b32073cef634c13ef332c94"

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def checked_rows(root:pathlib.Path):
    root=root.resolve(strict=True)
    result=(root/"result.json").read_bytes()
    if sha(result)!=EXPECTED_RESULT_SHA:raise ValueError("P7 source receipt digest mismatch")
    rec=json.loads(result)
    if rec.get("status")!="PASS" or rec.get("production_touched") is not False or rec.get("summary",{}).get("admissions")!=600:
        raise ValueError("P7 result is not exact 600 admitted PASS")
    directory=root/"lineage.jsonl.segments"
    rows=[];prior=None
    for raw in (directory/"manifest.jsonl").read_bytes().splitlines():
        entry=json.loads(raw);claim=entry.get("manifest_sha256")
        copy={k:v for k,v in entry.items() if k!="manifest_sha256"}
        if claim!=pc.sha256_json(copy) or entry.get("prev_manifest_sha256")!=prior:
            raise ValueError("broken manifest SHA chain")
        prior=claim
        p=(directory/entry["segment_file"]).resolve(strict=True)
        if not p.is_relative_to(directory.resolve(strict=True)) or not p.is_file() or sha(p.read_bytes())!=entry["segment_sha256"]:
            raise ValueError("archived lineage segment hash mismatch")
        segment=[json.loads(s) for s in p.read_bytes().splitlines() if s]
        po.verify_records(segment)
        if not segment or len(segment)!=entry["records"] or segment[0]["record_sha256"]!=entry["first_record_sha256"] or segment[-1]["record_sha256"]!=entry["last_record_sha256"]:
            raise ValueError("archived lineage record mismatch")
        rows.extend(segment)
    active=[json.loads(s) for s in (root/"lineage.jsonl").read_bytes().splitlines() if s]
    po.verify_records(active);rows.extend(active)
    if len(rows)!=600 or len({r["admission_id"] for r in rows})!=600:
        raise ValueError("600 event identity coverage mismatch")
    if any(r.get("outcome",{}).get("finite_logits") is not True for r in rows):
        raise ValueError("nonfinite archived logits")
    return rows,prior

def summarize(rows:list[dict],manifest_sha:str)->dict:
    if len(rows)!=600:raise ValueError("expected 600 observations")
    risk=collections.Counter();nonnull=collections.Counter();contexts=collections.Counter()
    missing=collections.Counter();policies=collections.Counter();event_source=collections.Counter()
    safe_events=[]
    for idx,row in enumerate(rows):
        signals=row.get("signal") or {}
        triggers=row.get("active_triggers") or []
        for t in triggers:risk[t]+=1
        for k in ("margin","entropy","routing_ambiguity_score"):
            v=signals.get(k)
            if v is None:missing[k]+=1
            elif isinstance(v,(int,float)) and math.isfinite(float(v)):nonnull[k]+=1
            else:raise ValueError("invalid nonfinite signal")
        for k in ("evidence_snapshot_sha256","allocation_sha256","selection_sha256",
                  "combined_policy_evidence_sha256","cost_evidence_sha256"):
            if row.get(k):contexts[k]+=1
        policy=row.get("after_policy_hash")
        if not isinstance(policy,str) or len(policy)!=64:raise ValueError("invalid policy hash")
        policies[policy]+=1
        origin=str(row.get("admission_path",""))
        event_source[origin]+=1
        # The P7 schedule is operator-driven and non-generalizing.
        safe_events.append({"position":idx,"injected_low_margin":("low_margin" in triggers),
                            "margin":signals.get("margin"),
                            "entropy":signals.get("entropy"),
                            "routing_ambiguity_score":signals.get("routing_ambiguity_score"),
                            "policy_hash":policy,
                            "transitioned":bool(row.get("transitioned"))})
    out={"schema":SCHEMA,"status":"COVERAGE_MEASURED_TRAINING_NOT_APPROVED",
       "source_result_sha256":EXPECTED_RESULT_SHA,
       "source_manifest_head_sha256":manifest_sha,
       "event_count":len(rows),"unique_prompt_sources":1,
       "operator_injected_risk":True,
       "trigger_counts":dict(risk),
       "observed_numeric_signal_counts":dict(nonnull),
       "missing_numeric_signal_counts":dict(missing),
       "provenance_coverage":dict(contexts),
       "policy_hash_counts":dict(policies),
       "admission_path_counts":dict(event_source),
       "real_multidomain_latent_training_suitable":False,
       "real_condition_certificates_verified":False,
       "native_policy_execution_allowed":False,
       "production_touched":False,
       "sanitized_operator_sequence":safe_events}
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-root",type=pathlib.Path,required=True)
    parser.add_argument("--output",type=pathlib.Path,required=True)
    args=parser.parse_args()
    rows,head=checked_rows(args.source_root)
    x=summarize(rows,head)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(x,sort_keys=True,indent=2)+"\n")
    brief={k:v for k,v in x.items() if k!="sanitized_operator_sequence"}
    print("P2_ACTUAL_P7_SIGNAL_COVERAGE",json.dumps(brief,sort_keys=True))
if __name__=="__main__":main()
