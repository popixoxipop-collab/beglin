#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import precision_policy_refresh as p8
import precision_closed_loop_acceptance_xox as p2
P7=Path("/Users/xox/vdsp_serving/precision-p7-soak-long600-4baf28b/result.json")
P7_SHA="ebc6c8c52759e7164b555d19e7e330bf933835be2b32073cef634c13ef332c94"
CURRENT=[{"role":"shared_up_proj","layer":3,"n":6},{"role":"shared_down_proj","layer":26,"n":5}]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if sha(P7)!=P7_SHA: raise RuntimeError("P7 evidence SHA mismatch")
 p7=json.loads(P7.read_text())
 if p7.get("status")!="PASS" or p7.get("production_touched") is not False: raise RuntimeError("P7 evidence invalid")
 candidates,_,_,src=p2.evidence_snapshot()
 proposal=p8.propose(candidates=candidates,current_policy=CURRENT,lineage_summary=p7["summary"],
   p7_certification={"status":"PASS","production_touched":False,"result_sha256":P7_SHA})
 target=proposal["selected_shadow_target"]; shadow=p8.shadow_candidate(proposal)
 out={"schema":"beglin-p8-policy-refresh-acceptance-v1","status":"READY_FOR_SHADOW",
      "production_touched":False,"p7_result_sha256":P7_SHA,"source_evidence":src,
      "proposal":proposal,"shadow_candidate":shadow,
      "shadow_execution_blocker":"NEEDS_REPLAY_PROVENANCE","certification_candidate":None}
 raw=json.dumps(out,sort_keys=True,indent=2)+"\n"
 path=Path("/Users/xox/vdsp_serving/precision-p8-policy-refresh-ready.json"); path.write_text(raw)
 print(json.dumps({"status":out["status"],"proposal_id":proposal["proposal_id"],
   "selected_shadow_target":target,"shadow_execution_blocker":out["shadow_execution_blocker"],
   "result_sha256":hashlib.sha256(raw.encode()).hexdigest()},indent=2))
if __name__=="__main__": main()
