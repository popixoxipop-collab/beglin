#!/usr/bin/env python3
"""Choose the cheapest exact accepted precision policy under hard correctness constraints."""
from __future__ import annotations

import precision_context as pc


class PolicyCostError(RuntimeError):
    pass


def _policy_map(policy):
    return {(str(r["role"]),int(r["layer"])):int(r["n"]) for r in pc.normalize_policy(policy)}


def _combined_index(rows):
    out={}
    for row in rows or []:
        if not isinstance(row,dict) or row.get("status")!="PASS" or row.get("pass") is not True:
            continue
        policy=row.get("policy")
        if not isinstance(policy,list):
            continue
        normalized=pc.normalize_policy(policy)
        ph=pc.policy_hash(normalized)
        if str(row.get("policy_hash","")).lower()!=ph:
            raise PolicyCostError("combined evidence policy hash mismatch")
        if row.get("production_touched") is not False:
            raise PolicyCostError("combined policy evidence must prove production_touched=false")
        out[ph]={**row,"policy":normalized}
    return out


def _profile_index(cost_evidence):
    if not isinstance(cost_evidence,dict) or cost_evidence.get("schema")!="beglin-precision-e2e-cost-v1":
        raise PolicyCostError("unexpected policy cost evidence schema")
    rows=cost_evidence.get("accepted_policy_profiles")
    if not isinstance(rows,list) or not rows:
        raise PolicyCostError("accepted_policy_profiles must be non-empty")
    out={}
    for row in rows:
        policy=pc.normalize_policy(row["policy"])
        ph=pc.policy_hash(policy)
        if str(row.get("policy_hash","")).lower()!=ph:
            raise PolicyCostError("policy cost profile hash mismatch")
        out[ph]={**row,"policy":policy}
    return out


def _cache_has_policy(runtime_state, policy):
    cache={(str(r["role"]),int(r["layer"]),int(r["n"])) for r in runtime_state.get("qng64_cache",[])}
    return all((str(r["role"]),int(r["layer"]),int(r["n"])) in cache for r in policy)


def _transition(profile, current_hash, cache_state):
    for row in profile.get("transitions",[]):
        if str(row.get("from_policy_hash"))==current_hash and row.get("cache_state")==cache_state:
            return row
    return None
def _admitted_sets(allocation):
    out={}
    for target in allocation.get("targets",[]):
        if target.get("status")!="PROPOSED":
            continue
        key=(str(target["role"]),int(target["layer"]))
        allowed={int(target["selected_n"])}
        for alt in target.get("dynamic_escalation",{}).get("candidate_alternates",[]):
            allowed.add(int(alt["n"]))
        out[key]=allowed
    return out


def _selector_exact(selection):
    active=(selection.get("signal") or {}).get("active_triggers") or []
    if not active:
        return {}
    exact={}
    for row in selection.get("targets",[]):
        if row.get("selected_n") is not None:
            exact[(str(row["role"]),int(row["layer"]))]=int(row["selected_n"])
    return exact


def optimize_policy(
    *,
    current_policy,
    allocation,
    selection,
    combined_policy_evidence,
    cost_evidence,
    runtime_state,
    e2e_weight=1.0,
    cache_weight=0.0,
    transition_weight=0.0,
    active_memory_weight=0.0,
):
    if min(float(e2e_weight),float(cache_weight),float(transition_weight),float(active_memory_weight))<0:
        raise PolicyCostError("policy cost weights must be non-negative")
    current=pc.normalize_policy(current_policy)
    current_map=_policy_map(current)
    current_hash=pc.policy_hash(current)
    accepted=_combined_index(combined_policy_evidence)
    profiles=_profile_index(cost_evidence)
    admitted=_admitted_sets(allocation)
    exact=_selector_exact(selection)
    resident_before=int(runtime_state.get("resident_qng64_cache_bytes",0) or 0)

    candidates=[]
    for ph,evidence in accepted.items():
        policy=evidence["policy"]
        pmap=_policy_map(policy)
        if set(pmap)!=set(current_map):
            continue
        if any(key not in admitted or n not in admitted[key] for key,n in pmap.items()):
            continue
        if any(pmap.get(key)!=n for key,n in exact.items()):
            continue
        profile=profiles.get(ph)
        if profile is None:
            continue
        if ph==current_hash:
            transition_ms=0.0; bytes_added=0; cache_state="active"
        else:
            cache_state="warm" if _cache_has_policy(runtime_state,policy) else "cold"
            tr=_transition(profile,current_hash,cache_state)
            if tr is None:
                continue
            transition_ms=float(tr["p50_transition_ms"])
            bytes_added=int(tr.get("cache_bytes_added",0))
        passes=int(profile.get("expected_inference_passes",1))
        steady=float(profile["steady_roundtrip_p50_ms"])
        e2e=transition_ms+steady*passes
        resident_after=resident_before+bytes_added
        active_bytes=0.0
        for target in allocation.get("targets",[]):
            if target.get("status")!="PROPOSED": continue
            key=(str(target["role"]),int(target["layer"]))
            n=pmap.get(key)
            rows=[target.get("selected")]+list(target.get("pareto_frontier",[]))
            match=next((r for r in rows if isinstance(r,dict) and int(r.get("n",-1))==n),None)
            if match is not None:
                active_bytes+=float(match.get("estimated_bytes",0.0))
        candidates.append({
            "policy":policy,"policy_hash":ph,"cache_state":cache_state,
            "transition_p50_ms":transition_ms,"cache_bytes_added":bytes_added,
            "resident_cache_bytes_after":resident_after,
            "expected_inference_passes":passes,
            "steady_roundtrip_p50_ms":steady,
            "expected_e2e_ms":e2e,"active_weight_bytes":active_bytes,
            "combined_policy_evidence":evidence,
        })
    if not candidates:
        raise PolicyCostError("no exact accepted policy satisfies selector and allocator constraints")

    dims=[
        ("e2e","expected_e2e_ms",float(e2e_weight)),
        ("cache","resident_cache_bytes_after",float(cache_weight)),
        ("transition","transition_p50_ms",float(transition_weight)),
        ("active_memory","active_weight_bytes",float(active_memory_weight)),
    ]
    dims=[d for d in dims if d[2]!=0.0]
    if not dims:
        raise PolicyCostError("at least one policy cost weight must be non-zero")
    baselines={}
    for name,field,_ in dims:
        vals=[float(c[field]) for c in candidates]
        baselines[name]=(min(vals),max(vals))
    for c in candidates:
        score=0.0; used=[]
        for name,field,weight in dims:
            lo,hi=baselines[name]; value=float(c[field])
            norm=value/lo if lo>0 else value/max(hi,1.0)
            score+=weight*norm; used.append(name)
        c["objective_score"]=score; c["objective_dimensions"]=used
    chosen=min(candidates,key=lambda c:(c["objective_score"],c["expected_e2e_ms"],c["policy_hash"]))
    return {
        "schema":"beglin-policy-cost-optimizer-v1",
        "status":"SELECTED",
        "production_write_allowed":False,
        "current_policy_hash":current_hash,
        "selected_policy":chosen["policy"],
        "selected_policy_hash":chosen["policy_hash"],
        "selected_cost":chosen,
        "eligible_policies":candidates,
        "weights":{
            "e2e":float(e2e_weight),"cache":float(cache_weight),
            "transition":float(transition_weight),"active_memory":float(active_memory_weight),
        },
    }
