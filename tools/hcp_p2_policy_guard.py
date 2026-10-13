#!/usr/bin/env python3
"""Pure P2 shadow policy gate; predictive latents can NEVER authorize qNg64.

Use original precision_closed_loop.materialize_selected_policy for independent
allocator/selector checks, followed by a stricter context-bound, exact-policy
acceptance check. No PrecisionEpochScheduler or production write interfaces.
"""
from __future__ import annotations
import hashlib
import re
import precision_context as pc
import precision_closed_loop as pcl

HEX64=re.compile(r"^[0-9a-f]{64}$")
SCHEMA="beglin-hcp-p2-shadow-evidence-guard-v1"

def hold(current,reason):
    c=pc.normalize_policy(current)
    return {"schema":SCHEMA,"status":"HOLD_UNCERTIFIED",
        "selected_policy":c,"selected_policy_hash":pc.policy_hash(c),
        "reason":reason,"learning_is_not_evidence":True,
        "production_write_allowed":False,"native_policy_change_allowed":False}

def propose(*,current_policy,context_hash,learned_prediction,
            allocation=None,selection=None,combined_evidence=None):
    current=pc.normalize_policy(current_policy)
    if not isinstance(learned_prediction,dict) or learned_prediction.get("source")!="REAL_QUANTIZED_AF_H2_TOP6_ROUTING":
        return hold(current,"MISSING_OR_UNTRUSTED_FORECAST")
    # The trained surrogate is not a calibrated margin/topK+1 score and may not
    # override native low-level selector or grant additional policy permissions.
    if not isinstance(context_hash,str) or not HEX64.fullmatch(context_hash):
        return hold(current,"MISSING_EXECUTION_CONTEXT_IDENTITY")
    if not (isinstance(allocation,dict) and isinstance(selection,dict) and isinstance(combined_evidence,list)):
        return hold(current,"MISSING_EXACT_ALLOCATOR_SELECTOR_CERTIFICATION")
    try:
        decision=pcl.materialize_selected_policy(
            current_policy=current,allocation=allocation,selection=selection,
            combined_policy_evidence=combined_evidence)
    except (ValueError,KeyError,TypeError,pcl.PrecisionClosedLoopError) as err:
        return hold(current,"EXISTING_BEGLIN_PRECISION_CONTRACT_REJECTED")
    active=set((selection.get("signal") or {}).get("active_triggers") or [])
    for row in selection.get("targets",[]):
        if row.get("status")=="TRIGGER_CONDITIONED_ALTERNATE":
            e=row.get("evidence") or {}
            if (e.get("status")!="PASS" or e.get("pass") is not True
                or e.get("context_hash")!=context_hash
                or e.get("trigger_type") not in active):
                return hold(current,"MISSING_EXACT_CONTEXT_TRIGGER_EVIDENCE")
    ph=decision["selected_policy_hash"]
    if ph==pc.policy_hash(current):
        return {"schema":SCHEMA,"status":"SHADOW_HOLD_EXACT_INCUMBENT",
            "selected_policy":current,"selected_policy_hash":ph,
            "source_selector_hash":pc.sha256_json(selection),
            "production_write_allowed":False,"native_policy_change_allowed":False}
    eligible=[]
    for row in combined_evidence:
        if not isinstance(row,dict):continue
        if row.get("status")!="PASS" or row.get("pass") is not True or row.get("production_touched") is not False:continue
        if row.get("policy_hash")!=ph or row.get("context_hash")!=context_hash:continue
        if not isinstance(row.get("evidence_sha256"),str) or not HEX64.fullmatch(row["evidence_sha256"]):continue
        policy=row.get("policy")
        if not isinstance(policy,list) or pc.policy_hash(policy)!=ph:continue
        eligible.append(row)
    if len(eligible)!=1:
        return hold(current,"MISSING_UNIQUE_EXACT_CONTEXT_CERTIFICATE")
    return {"schema":SCHEMA,"status":"SHADOW_CERTIFIED_PROPOSAL_ONLY",
        "selected_policy":decision["selected_policy"],
        "selected_policy_hash":ph,
        "exact_context_hash":context_hash,
        "certification_sha256":eligible[0]["evidence_sha256"],
        "source_selector_hash":pc.sha256_json(selection),
        "learning_is_not_evidence":True,
        "original_beglin_authoritative_decision":decision["schema"],
        "production_write_allowed":False,
        "native_policy_change_allowed":False,
        "needs_separate_native_scratch_certification":True}
