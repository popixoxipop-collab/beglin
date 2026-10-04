#!/usr/bin/env python3
"""P10 adapter from P9 manual-review bundle to existing dry-run canary contract."""
from __future__ import annotations
import hashlib,json
import manual_canary_contract as mc
import precision_context as pc
import model_capability_bridge as mcb

SCHEMA="beglin-precision-p10-canary-gate-v1"
class P10Error(RuntimeError): pass
def _sha(v): return hashlib.sha256(pc.canonical_json(v).encode()).hexdigest()

def materialize_canary_template(*,p9_bundle:dict,baseline_policy:list[dict],
 source_commit:str,binary_sha256:str,checkpoint_sha256:str,
 g4_run_id:str,g4_sha256:str,g6_run_id:str,g6_sha256:str)->dict:
 if p9_bundle.get("status")!="MANUAL_REVIEW_CANDIDATE": raise P10Error("P9 bundle is not manual-review candidate")
 if p9_bundle.get("automatic_live_promotion") is not False or p9_bundle.get("production_write_allowed") is not False:
  raise P10Error("P9 bundle violates manual promotion boundary")
 target=p9_bundle.get("target") or {}
 if not all(k in target for k in ("role","layer","old_n","new_n")): raise P10Error("P9 target incomplete")
 candidate=pc.normalize_policy(p9_bundle.get("proposed_policy") or [])
 baseline=pc.normalize_policy(baseline_policy)
 bm={(r["role"],int(r["layer"])):int(r["n"]) for r in baseline}
 cm={(r["role"],int(r["layer"])):int(r["n"]) for r in candidate}
 key=(str(target["role"]),int(target["layer"]))
 if set(bm)!=set(cm) or {k for k in bm if bm[k]!=cm[k]}!={key}: raise P10Error("P10 requires exactly one resident target delta")
 if bm[key]!=int(target["old_n"]) or cm[key]!=int(target["new_n"]): raise P10Error("P9 target does not match policies")
 template={"schema":"manual-canary-proposal-v1","mode":"dry_run","production_write_allowed":False,
  "proposal_id":"p10-"+str(p9_bundle["proposal_id"]),"proposer":"precision-p10-gate",
  "environment_id":"xox-isolated-canary","model_revision":"deepseek-v2-lite",
  "backend":"mlx_metal","architecture":"deepseek-v2-lite-moe",
  "source_commit":str(source_commit),"binary_sha256":str(binary_sha256),
  "checkpoint_sha256":str(checkpoint_sha256),
  "baseline_policy_hash":mc.sha256_json(baseline),"candidate_policy_hash":mc.sha256_json(candidate),
  "single_target":{"role":key[0],"layer":key[1],"before_n":bm[key],"after_n":cm[key]},
  "evidence_refs":[{"kind":"G4_A_B_R","run_id":g4_run_id,"sha256":g4_sha256},
                   {"kind":"G6_RESTART_CANARY","run_id":g6_run_id,"sha256":g6_sha256}],
  "budget":{"max_requests":16,"max_tokens":512,"max_duration_ms":30000,"max_memory_bytes":34359738368},
  "expected_epoch":0,"restart_instance_id":"P10_RUNTIME_PREIMAGE_REQUIRED",
  "kill_switch_scope":"single candidate worker only",
  "rollback_plan":"restore exact runtime preimage; fail closed on mismatch"}
 lineage=mcb.validated_optional_lineage(p9_bundle)
 out={"schema":SCHEMA,"status":"READY_FOR_RUNTIME_PREIMAGE_MATERIALIZATION",
  "production_write_allowed":False,"automatic_live_promotion":False,
  "p9_bundle_sha256":_sha(p9_bundle),"baseline_policy":baseline,"candidate_policy":candidate,
  "manual_canary_template":template,"trusted_production_approval_present":False,
  "production_cutover_allowed":False,"required_next_action":"materialize_runtime_preimage_then_human_approval"}
 out.update(lineage)
 return out

def final_gate(*,p9_bundle:dict,canary_pass:dict,rollback_drill:dict)->dict:
 if p9_bundle.get("status")!="MANUAL_REVIEW_CANDIDATE": raise P10Error("invalid P9 bundle")
 if canary_pass.get("state")!="CANARY_PASS_REVIEW_REQUIRED": raise P10Error("canary pass rehearsal missing")
 if rollback_drill.get("state")!="ROLLBACK_VERIFIED": raise P10Error("rollback rehearsal missing")
 lineage=mcb.validated_optional_lineage(p9_bundle)
 out={"schema":SCHEMA,"status":"AWAITING_TRUSTED_PRODUCTION_APPROVAL",
  "production_write_allowed":False,"automatic_live_promotion":False,
  "production_cutover_allowed":False,"p9_bundle_sha256":_sha(p9_bundle),
  "canary_pass_sha256":_sha(canary_pass),"rollback_drill_sha256":_sha(rollback_drill),
  "required_next_action":"trusted_human_approval_bound_to_fresh_runtime_preimage_and_cutover_plan"}
 out.update(lineage)
 return out
