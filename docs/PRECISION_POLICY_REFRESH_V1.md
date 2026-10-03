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

## Replay provenance capture for future refresh cycles
Persistent workers now have an opt-in replay provenance capture path for P8 shadow evidence. It is disabled by default and must be explicitly configured to a child directory of /Users/xox/vdsp_serving.

When enabled, only admissions that actually emit precision risk events are retained. Before the normal ephemeral request directory is removed, the worker stores a content-addressed copy of the raw int32 prompt, a one-entry replay manifest, and SHA-bound metadata. The record includes prompt length, max-new-tokens, event count/hash, raw-token SHA and production_write_allowed=false.

XOX scratch proof:
- /Users/xox/vdsp_serving/p8-replay-provenance-probe/provenance/989f093b187541dffb98df11/provenance.json
- raw token SHA-256 989f093b187541dffb98df11e792cadf5177dc7f3d5d3ca2525b0e788f83abba
- provenance SHA-256 86342477535e7755299a0f8f4b83c24c7d4451099df51d65c19a98981f5f8af5
- event_count=1
- production_write_allowed=false

This new capture does not retroactively manufacture provenance for the current L3 n6->n5 P8 proposal. That historical fixture did not preserve its original raw token source, so the existing proposal correctly remains NEEDS_REPLAY_PROVENANCE. Future evidence generated with capture enabled can enter the existing gpu_shadow_pipeline without weakening its provenance gate.
