#!/usr/bin/env python3
"""P9 queue -> repeated shadow -> certification bundle orchestration.

No production mutation path exists here. P9 consumes a P8 proposal and explicit
replay provenance, invokes the existing scratch-only gpu_shadow_runner, and can
emit only MANUAL_REVIEW_CANDIDATE.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import precision_context as pc
import precision_policy_refresh as p8
import gpu_shadow_runner as shadow

SCHEMA="beglin-precision-p9-shadow-cert-v1"
class P9Error(RuntimeError): pass
def _sha(v): return hashlib.sha256(pc.canonical_json(v).encode()).hexdigest()
def _file_sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
 return h.hexdigest()

def validate_provenance(provenance:dict)->dict:
 if not isinstance(provenance,dict) or provenance.get("schema")!="beglin-replay-provenance-v1":
  raise P9Error("missing replay provenance")
 if provenance.get("production_write_allowed") is not False: raise P9Error("provenance permits production write")
 raw=Path(str(provenance.get("raw_token_file",""))).resolve()
 manifest=Path(str(provenance.get("manifest",""))).resolve()
 if not raw.is_file() or not manifest.is_file(): raise P9Error("replay provenance files missing")
 if _file_sha(raw)!=provenance.get("raw_token_sha256"): raise P9Error("raw token SHA mismatch")
 line=manifest.read_text().strip().split()
 if len(line)!=2 or Path(line[0]).resolve()!=raw: raise P9Error("replay manifest does not bind raw token file")
 if int(line[1])!=int(provenance["max_new_tokens"]): raise P9Error("replay max_new_tokens mismatch")
 return dict(provenance)

def queue_item(*,proposal:dict,provenance:dict|None)->dict:
 candidate=p8.shadow_candidate(proposal)
 if provenance is None:
  return {"schema":SCHEMA,"status":"WAITING_FOR_REPLAY_PROVENANCE","production_write_allowed":False,
          "automatic_live_promotion":False,"proposal_id":proposal["proposal_id"],"candidate":candidate}
 prov=validate_provenance(provenance)
 return {"schema":SCHEMA,"status":"READY_FOR_SHADOW","production_write_allowed":False,
         "automatic_live_promotion":False,"proposal_id":proposal["proposal_id"],"candidate":candidate,
         "baseline_policy":pc.normalize_policy(proposal.get("current_policy") or []),
         "provenance":prov,"queue_sha256":_sha({"proposal":proposal,"provenance":prov})}

def candidate_spec(*,queue:dict,event:dict,reference:dict,cwd:str,binary:str,binary_sha256:str,
                   checkpoint_sha256:str,moe_base:str,safetensors:str)->dict:
 if queue.get("status")!="READY_FOR_SHADOW": raise P9Error("queue item is not READY_FOR_SHADOW")
 c=queue["candidate"]; p=queue["provenance"]
 return {"candidate_id":c["candidate_id"],"role":c["role"],"layer":int(c["layer"]),"n":int(c["n"]),
         "baseline_policy":pc.normalize_policy(queue.get("baseline_policy") or []),
         "event":event,"reference":reference,"prompt_len":int(p["prompt_len"]),
         "g4_manifest":p["manifest"],"g6_manifest":p["manifest"],"cwd":cwd,"binary":binary,
         "binary_sha256":binary_sha256,"checkpoint_sha256":checkpoint_sha256,"moe_base":moe_base,
         "safetensors":safetensors,"production_write_allowed":False,"p9_queue_sha256":queue["queue_sha256"]}

def run_repeated_shadow(*,queue:dict,spec:dict,shadow_root:str,autopilot:str,repeats:int=3,timeout:int=3600)->dict:
 if repeats<2 or repeats>10: raise P9Error("shadow repeats must be in [2,10]")
 runs=[]
 for i in range(repeats):
  got=shadow.run_shadow(spec,shadow_root=shadow_root,autopilot=autopilot,
                        run_id=f"p9-{queue['proposal_id']}-{i+1}",timeout=timeout)
  runs.append(got)
 if any(r.get("production_write_allowed") is not False for r in runs): raise P9Error("shadow unexpectedly permits production write")
 statuses=[r.get("shadow_status") for r in runs]
 status="SHADOW_ADMITTED" if all(x=="SHADOW_ADMITTED" for x in statuses) else "SHADOW_REJECTED"
 return {"schema":SCHEMA,"status":"SHADOW_REPEATED_COMPLETE","shadow_status":status,
         "production_write_allowed":False,"automatic_live_promotion":False,"repeats":repeats,
         "statuses":statuses,"runs":runs,"result_sha256":_sha(runs),"candidate":queue["candidate"]}

def certification_bundle(*,proposal:dict,repeated_shadow:dict)->dict:
 if repeated_shadow.get("shadow_status")!="SHADOW_ADMITTED": raise P9Error("repeated shadow is not unanimously admitted")
 cert=p8.certification_candidate(proposal=proposal,shadow_result={
  "shadow_status":"SHADOW_ADMITTED","production_write_allowed":False,
  "candidate":repeated_shadow["candidate"],"result_sha256":repeated_shadow["result_sha256"]})
 cert["schema"]="beglin-precision-p9-certification-bundle-v1"
 cert["p9_repeats"]=int(repeated_shadow["repeats"])
 cert["p9_shadow_statuses"]=list(repeated_shadow["statuses"])
 cert["p9_repeated_shadow_sha256"]=_sha(repeated_shadow)
 cert["automatic_live_promotion"]=False
 cert["production_write_allowed"]=False
 cert["required_next_action"]="manual_review_and_separate_cutover_gate"
 return cert
