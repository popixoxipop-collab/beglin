#!/usr/bin/env python3
"""Trigger-conditioned precision selector for Beglin allocator v1.

Non-monotonic qNg64 means "higher n" is never treated as inherently safer.
The allocator's low-cost selected_n is the default. A trigger may switch to an
alternate n only when trigger-conditioned evidence for that exact target,
from_n, to_n and trigger type has passed. Missing evidence always holds base.
This module is proposal/decision evidence only and never mutates production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request


TRIGGERS = ("near_tie", "low_margin", "high_entropy", "routing_ambiguity")


class DynamicSelectorError(RuntimeError):
    pass


def _credentials():
    url=os.environ.get("QWEN_SUPABASE_URL","").rstrip("/")
    key=os.environ.get("QWEN_SUPABASE_KEY","")
    if not url or not key:
        raise DynamicSelectorError("QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY are required")
    return url,key


def _headers(key, *, write=False):
    out={"apikey":key,"Authorization":f"Bearer {key}"}
    if write:
        out["Content-Type"]="application/json"
    return out


def fetch_trigger_evidence(model_id: str) -> list[dict]:
    url,key=_credentials()
    params={
        "model_id":f"eq.{model_id}",
        "select":"*",
        "order":"created_at.asc",
    }
    qs=urllib.parse.urlencode(params,safe=".,")
    req=urllib.request.Request(
        f"{url}/rest/v1/moe_precision_trigger_evidence_v1?{qs}",
        headers=_headers(key),
    )
    with urllib.request.urlopen(req,timeout=30) as resp:
        rows=json.loads(resp.read())
    if not isinstance(rows,list):
        raise DynamicSelectorError("trigger evidence query did not return an array")
    return rows


def normalize_signal(signal: dict) -> dict:
    active=[]
    for trigger in TRIGGERS:
        value=signal.get(trigger)
        if isinstance(value,bool):
            enabled=value
        elif value is None:
            enabled=False
        else:
            enabled=bool(value)
        if enabled:
            active.append(trigger)
    margin=signal.get("margin")
    if margin is not None:
        margin=float(margin)
    entropy=signal.get("entropy")
    if entropy is not None:
        entropy=float(entropy)
    return {
        "active_triggers":active,
        "margin":margin,
        "entropy":entropy,
        "routing_ambiguity_score":signal.get("routing_ambiguity_score"),
        "raw":signal,
    }


def _target_key(row: dict):
    return str(row["role"]),int(row["layer"])


def _signal_matches_bucket(signal: dict, bucket: dict) -> bool:
    if not isinstance(bucket, dict) or not bucket:
        return False
    if bucket.get("trigger_only") is True:
        return True

    checks = (
        ("margin", "margin_min", "margin_max"),
        ("entropy", "entropy_min", "entropy_max"),
        ("routing_ambiguity_score", "routing_ambiguity_score_min", "routing_ambiguity_score_max"),
    )
    constrained = False
    for field, lo_key, hi_key in checks:
        lo = bucket.get(lo_key)
        hi = bucket.get(hi_key)
        if lo is None and hi is None:
            continue
        constrained = True
        value = signal.get(field)
        if value is None:
            return False
        value = float(value)
        if lo is not None and value < float(lo):
            return False
        if hi is not None and value > float(hi):
            return False
    return constrained


def _evidence_proves_precision_benefit(row: dict) -> bool:
    metrics=row.get("metrics") or {}
    requests=int(row.get("requests",0))
    if requests <= 0:
        return False

    base_failures=metrics.get("base_failures")
    target_failures=metrics.get("target_failures")
    if base_failures is not None and target_failures is not None:
        try:
            base_failures=int(base_failures)
            target_failures=int(target_failures)
        except (TypeError,ValueError):
            return False
        return base_failures > 0 and target_failures == 0

    base_pass=metrics.get("base_pass")
    target_pass=metrics.get("target_pass")
    return base_pass is False and target_pass is True


def _passed_context_evidence(rows, *, role, layer, from_n, to_n, trigger, signal):
    matches=[]
    for row in rows:
        if (
            str(row.get("role"))==str(role)
            and int(row.get("layer",-1))==int(layer)
            and int(row.get("from_n",-1))==int(from_n)
            and int(row.get("to_n",-1))==int(to_n)
            and str(row.get("trigger_type"))==str(trigger)
            and row.get("status")=="PASS"
            and row.get("pass") is True
            and int(row.get("requests",0))>0
            and _signal_matches_bucket(signal, row.get("signal_bucket") or {})
            and _evidence_proves_precision_benefit(row)
        ):
            matches.append(row)
    return matches


def choose_for_target(target: dict, signal: dict, trigger_evidence: list[dict]) -> dict:
    if target.get("status")!="PROPOSED":
        return {
            "role":target.get("role"),
            "layer":target.get("layer"),
            "status":"NO_BASE_PROPOSAL",
            "selected_n":target.get("current_n"),
        }
    base_n=int(target["selected_n"])
    normalized=normalize_signal(signal)
    active=normalized["active_triggers"]
    if not active:
        return {
            "role":target["role"],"layer":int(target["layer"]),
            "status":"BASE_LOW_COST",
            "selected_n":base_n,
            "active_triggers":[],
            "production_write_allowed":False,
        }

    alternates=target.get("dynamic_escalation",{}).get("candidate_alternates",[])
    eligible=[]
    missing=[]
    for alt in alternates:
        n=int(alt["n"])
        if n==base_n:
            continue
        evidence_by_trigger={}
        ok=True
        for trigger in active:
            rows=_passed_context_evidence(
                trigger_evidence,
                role=target["role"],layer=target["layer"],
                from_n=base_n,to_n=n,trigger=trigger,signal=normalized,
            )
            if not rows:
                ok=False
                missing.append({"to_n":n,"trigger":trigger})
                break
            evidence_by_trigger[trigger]=rows
        if ok:
            eligible.append({
                "n":n,
                "real_pass_events":int(alt.get("real_pass_events",0)),
                "persistent_p50_ms":alt.get("persistent_p50_ms"),
                "evidence":evidence_by_trigger,
            })

    if not eligible:
        return {
            "role":target["role"],"layer":int(target["layer"]),
            "status":"HOLD_BASE_NO_TRIGGER_EVIDENCE",
            "selected_n":base_n,
            "active_triggers":active,
            "missing_trigger_evidence":missing,
            "required_calibrations":[
                {
                    "role":target["role"],
                    "layer":int(target["layer"]),
                    "from_n":base_n,
                    "to_n":int(alt["n"]),
                    "trigger_types":active,
                }
                for alt in alternates if int(alt["n"])!=base_n
            ],
            "production_write_allowed":False,
        }

    # Non-monotonic rule: never rank by larger n. Prefer the eligible alternate
    # with the smallest effective precision cost, then stronger event coverage.
    chosen=min(
        eligible,
        key=lambda x:(x["n"],-x["real_pass_events"]),
    )
    return {
        "role":target["role"],"layer":int(target["layer"]),
        "status":"TRIGGER_CONDITIONED_ALTERNATE",
        "selected_n":int(chosen["n"]),
        "base_n":base_n,
        "active_triggers":active,
        "evidence":chosen["evidence"],
        "production_write_allowed":False,
    }


def select(decision: dict, signal: dict, trigger_evidence: list[dict]) -> dict:
    targets=[
        choose_for_target(target,signal,trigger_evidence)
        for target in decision.get("targets",[])
        if target.get("status")=="PROPOSED"
    ]
    selected_policy=[]
    for target in targets:
        if target.get("selected_n") is not None:
            selected_policy.append({
                "role":target["role"],
                "layer":int(target["layer"]),
                "n":int(target["selected_n"]),
            })
    return {
        "schema":"beglin-dynamic-precision-selector-v1",
        "status":"DECISION_ONLY",
        "production_write_allowed":False,
        "nonmonotonic_precision":True,
        "signal":normalize_signal(signal),
        "selected_policy":selected_policy,
        "targets":targets,
    }


def persist(result: dict, model_id: str) -> dict:
    url,key=_credentials()
    canonical=json.dumps(result,sort_keys=True,separators=(",",":"))
    decision_id=hashlib.sha256(canonical.encode()).hexdigest()
    payload={
        "decision_id":decision_id,
        "model_id":model_id,
        "schema_version":result["schema"],
        "status":result["status"],
        "signal":result["signal"],
        "selected_policy":result["selected_policy"],
        "target_decisions":result["targets"],
        "production_write_allowed":False,
    }
    req=urllib.request.Request(
        f"{url}/rest/v1/moe_precision_dynamic_decisions_v1?on_conflict=decision_id",
        data=json.dumps(payload).encode(),
        headers={
            **_headers(key,write=True),
            "Prefer":"resolution=merge-duplicates,return=representation",
        },
        method="POST",
    )
    with urllib.request.urlopen(req,timeout=30) as resp:
        rows=json.loads(resp.read())
    if not isinstance(rows,list) or len(rows)!=1:
        raise DynamicSelectorError("dynamic decision persistence was not verified")
    return rows[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allocator-decision",required=True)
    signal_group=ap.add_mutually_exclusive_group(required=True)
    signal_group.add_argument("--signal-json")
    signal_group.add_argument("--signal-file")
    ap.add_argument("--model",default="deepseek-v2-lite")
    ap.add_argument("--persist",action="store_true")
    ap.add_argument("--output")
    args=ap.parse_args()
    decision=json.loads(Path(args.allocator_decision).read_text())
    if args.signal_file:
        signal=json.loads(Path(args.signal_file).read_text())
    else:
        signal=json.loads(args.signal_json)
    evidence=fetch_trigger_evidence(args.model)
    result=select(decision,signal,evidence)
    if args.persist:
        row=persist(result,args.model)
        result["persisted_decision_id"]=row["decision_id"]
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text,end="")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
