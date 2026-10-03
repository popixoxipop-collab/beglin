# P8 Evidence-Driven Precision Policy Refresh

P8 converts accumulated P5/P6/P7 evidence into proposal-only precision policy refresh candidates. It never writes production state and never performs automatic live promotion.

## Flow
P5 candidate evidence -> Precision Allocator -> P6 lineage maturity -> P7 certification gate -> one-target shadow proposal -> existing GPU shadow pipeline -> manual-review certification candidate.

## Gates
- minimum admission evidence (default 100);
- finite-logit rate must be 1.0;
- P7 certification must PASS and prove production_touched=false;
- resident policy shape cannot change;
- allocator remains proposal-only;
- at most one changed target is handed to shadow per cycle;
- shadow result must be SHADOW_ADMITTED and exactly match role/layer/n;
- certification output is MANUAL_REVIEW_CANDIDATE only;
- production_write_allowed=false and automatic_live_promotion=false throughout.

## Current XOX proposal
P7 canonical 600-run evidence:
- /Users/xox/vdsp_serving/precision-p7-soak-long600-4baf28b/result.json
- SHA-256 ebc6c8c52759e7164b555d19e7e330bf933835be2b32073cef634c13ef332c94

P8 acceptance:
- /Users/xox/vdsp_serving/precision-p8-policy-refresh-ready.json
- SHA-256 f91f74b3cddbd7f3cec1ff51222f0cea2e52228f6d8b3b9e4a38e16c5e07ff34
- status READY_FOR_SHADOW
- proposal id e25b8cced601cb4bd70de479
- selected first shadow target shared_up_proj / L3 / n6 -> n5
- production_touched=false

The current L3 recovery evidence proves the precision behavior on XOX but does not contain the replayable source manifest/provenance fields required by the existing gpu_shadow_pipeline. Therefore P8 intentionally stops at NEEDS_REPLAY_PROVENANCE and does not manufacture SHADOW_ADMITTED.

Once replay provenance is attached, the existing shadow pipeline can run the one-target candidate. Only a matching SHADOW_ADMITTED result can produce MANUAL_REVIEW_CANDIDATE. A separate human review/cutover gate remains required.
