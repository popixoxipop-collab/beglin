#!/usr/bin/env python3
"""P8 evidence-driven precision policy refresh proposal and certification gate."""
from __future__ import annotations
import hashlib, json
import precision_allocator as pa
import precision_context as pc
import model_capability_bridge as mcb

SCHEMA="beglin-precision-policy-refresh-v1"
CERT_SCHEMA="beglin-precision-policy-certification-candidate-v1"

class PolicyRefreshError(RuntimeError): pass

def _sha(v):
    return hashlib.sha256(pc.canonical_json(v).encode()).hexdigest()

def propose(*, candidates, current_policy, lineage_summary, p7_certification,
            memory_weight=1.0, latency_weight=0.0, rss_weight=0.0,
            e2e_weight=0.0, min_admissions=100, model_capability_bundle=None):
    if int(lineage_summary.get("admissions",0)) < int(min_admissions):
        raise PolicyRefreshError("insufficient P6/P7 admission evidence")
    if float(lineage_summary.get("finite_logits_rate",0.0)) != 1.0:
        raise PolicyRefreshError("finite-logit rate is not 1.0")
    if p7_certification.get("status")!="PASS":
        raise PolicyRefreshError("P7 certification is not PASS")
    if p7_certification.get("production_touched") is not False:
        raise PolicyRefreshError("P7 evidence must prove production_touched=false")
    allocation=pa.optimize(
        candidates,current_policy=current_policy,memory_weight=memory_weight,
        latency_weight=latency_weight,rss_weight=rss_weight,e2e_weight=e2e_weight)
    current=pc.normalize_policy(current_policy)
    proposed=pc.normalize_policy(allocation["proposed_policy"])
    cur={(r["role"],int(r["layer"])):int(r["n"]) for r in current}
    nxt={(r["role"],int(r["layer"])):int(r["n"]) for r in proposed}
    if set(cur)!=set(nxt): raise PolicyRefreshError("policy shape change is not P8-v1 eligible")
    changes=[{"role":k[0],"layer":k[1],"old_n":cur[k],"new_n":nxt[k]}
             for k in sorted(cur) if cur[k]!=nxt[k]]
    ranked=[]
    targets={(r["role"],int(r["layer"])):r for r in allocation.get("targets",[])}
    for change in changes:
        row=targets[(change["role"],change["layer"])]
        if row.get("status")!="PROPOSED": raise PolicyRefreshError("changed target lacks allocator proposal")
        ranked.append({**change,"policy_mode":row.get("policy_mode"),
                       "objective_score":(row.get("selected") or {}).get("objective_score"),
                       "allocator_target_sha256":_sha(row)})
    ranked.sort(key=lambda r:(float("inf") if r["objective_score"] is None else r["objective_score"],
                              r["role"],r["layer"]))
    selected=ranked[0] if ranked else None
    capability_binding=None
    if model_capability_bundle is not None and selected is not None:
        try:
            capability_binding=mcb.validate_p8_target(
                bundle=model_capability_bundle,
                role=selected["role"],layer=int(selected["layer"]),
                target_n=int(selected["new_n"]),backend="mlx_metal")
        except mcb.CapabilityBridgeError as exc:
            raise PolicyRefreshError(f"model capability gate rejected target: {exc}") from exc
    proposal_identity={"current":current,"proposed":proposed,"selected":selected,
                       "lineage_head":lineage_summary.get("lineage_head_sha256"),
                       "p7":p7_certification.get("result_sha256")}
    if capability_binding is not None:
        proposal_identity["model_capability_bundle_sha256"]=capability_binding["model_capability_bundle_sha256"]
    proposal_id=_sha(proposal_identity)[:24]
    out={"schema":SCHEMA,"status":"PROPOSAL_ONLY","production_write_allowed":False,
         "automatic_live_promotion":False,"proposal_id":proposal_id,
         "current_policy":current,"current_policy_hash":pc.policy_hash(current),
         "proposed_policy":proposed,"proposed_policy_hash":pc.policy_hash(proposed),
         "changes":ranked,"selected_shadow_target":selected,
         "allocator_decision_sha256":_sha(allocation),
         "lineage_head_sha256":lineage_summary.get("lineage_head_sha256"),
         "lineage_admissions":int(lineage_summary["admissions"]),
         "p7_result_sha256":p7_certification.get("result_sha256"),
         "shadow_required":selected is not None,
         "manual_review_required":True}
    if capability_binding is not None:
        out.update({
            "model_capability_bundle_sha256":capability_binding["model_capability_bundle_sha256"],
            "capability_target_key":capability_binding["capability_target_key"],
            "capability_backend":capability_binding["backend"],
            "capability_binding_sha256":_sha(capability_binding),
        })
    return out

def shadow_candidate(proposal: dict) -> dict:
    if proposal.get("schema") != SCHEMA or proposal.get("status") != "PROPOSAL_ONLY":
        raise PolicyRefreshError("invalid P8 proposal")
    target = proposal.get("selected_shadow_target")
    if target is None:
        raise PolicyRefreshError("proposal has no changed shadow target")
    out={
        "candidate_id": "p8-" + proposal["proposal_id"],
        "status": "READY",
        "backend": "mlx_metal",
        "model": "deepseek-v2-lite",
        "role": target["role"],
        "layer": int(target["layer"]),
        "n": int(target["new_n"]),
        "event_count": int(proposal["lineage_admissions"]),
        "current_bits": int(target["old_n"]),
        "p8_proposal_id": proposal["proposal_id"],
        "p8_allocator_target_sha256": target["allocator_target_sha256"],
        "production_write_allowed": False,
        "requires_replay_provenance": True,
    }
    out.update(mcb.optional_lineage(proposal))
    return out


def certification_candidate(*, proposal, shadow_result):
    if proposal.get("schema")!=SCHEMA or proposal.get("status")!="PROPOSAL_ONLY":
        raise PolicyRefreshError("invalid P8 proposal")
    if proposal.get("production_write_allowed") is not False:
        raise PolicyRefreshError("proposal unexpectedly permits production writes")
    target=proposal.get("selected_shadow_target")
    if target is None: raise PolicyRefreshError("proposal has no changed shadow target")
    if shadow_result.get("shadow_status")!="SHADOW_ADMITTED":
        raise PolicyRefreshError("shadow result is not SHADOW_ADMITTED")
    if shadow_result.get("production_write_allowed") is not False:
        raise PolicyRefreshError("shadow result unexpectedly permits production writes")
    candidate=shadow_result.get("candidate") or {}
    for key in ("role","layer"):
        if candidate.get(key)!=target.get(key):
            raise PolicyRefreshError(f"shadow candidate {key} mismatch")
    if int(candidate.get("n",-1))!=int(target["new_n"]):
        raise PolicyRefreshError("shadow candidate n mismatch")
    shadow_sha=shadow_result.get("result_sha256") or _sha(shadow_result)
    out={"schema":CERT_SCHEMA,"status":"MANUAL_REVIEW_CANDIDATE",
         "production_write_allowed":False,"automatic_live_promotion":False,
         "proposal_id":proposal["proposal_id"],
         "proposal_sha256":_sha(proposal),"shadow_result_sha256":shadow_sha,
         "target":target,"proposed_policy":proposal["proposed_policy"],
         "proposed_policy_hash":proposal["proposed_policy_hash"],
         "required_next_action":"manual_review_and_separate_cutover_gate"}
    out.update(mcb.optional_lineage(proposal))
    return out
