# Precision Closed Loop v1

## Goal
P2 connects the already-certified components into one request-time path:

request signal -> Precision Allocator -> Dynamic Selector -> exact combined-policy evidence gate -> Precision Epoch Scheduler -> inference

Production live is not changed by this stage.

## Hot-path contract
Remote/Supabase fetches are not performed while admission is held. Candidate rows, trigger evidence and combined-policy acceptance records are loaded before serving and frozen into one evidence snapshot SHA.

For each admission the worker holds the same RLock used by ordinary submit(), then:
1. reads the exact runtime ACK/policy preimage;
2. runs the evidence-gated non-monotonic allocator;
3. runs the trigger-conditioned selector;
4. validates that selector n values are allocator-admitted;
5. materializes a full resident policy without adding/removing targets;
6. requires exact combined-policy evidence when the allocator says pairwise evidence is required;
7. passes the selected policy to Precision Epoch Scheduler;
8. verifies scheduler before/after policy hashes against the closed-loop decision.
## Fail-closed rules
- allocator must remain PROPOSAL_ONLY and production_write_allowed=false
- selector must remain DECISION_ONLY, nonmonotonic_precision=true and production_write_allowed=false
- selector cannot invent an n outside allocator base + admitted alternates
- trigger alternate requires contextual PASS evidence
- resident policy shape cannot change hot
- an unaccepted multi-target policy hash never reaches inference
- a stale scheduler preimage or mismatched post-policy fails the admission

## XOX source evidence used for integration acceptance
L3 shared_up 6->5 recovery:
- /Users/xox/vdsp_serving/l3n6-to-n5-trigger-evidence-20261003/result.json
- SHA-256 b16764b20addac627aa523d3f5b0c1a5cb9f4d9a4bb03ca8010eeecd5418bee7
- same PID, base failures 40, target failures 0

L26 shared_down 5->6 low-margin recovery:
- /Users/xox/vdsp_serving/BEGLIN_DYNAMIC_A6F3E6D_LIVE_CUTOVER_2026-10-03.json
- SHA-256 4056304c9b978883aa85e56dcbcd644982d5d4edc991497f33e5a1f7f61211d2
- observed base margin 0.010715, RECOVERY_N6, token 1224
Exact combined policy acceptance:
- policy: shared_up/L3 n5 + shared_down/L26 n6
- policy hash e471dfb075ca86cd0c5341b972258df7991847bb3f1557b69447c8b6168042ae
- /Users/xox/vdsp_serving/precision-epoch-acceptance-ae1b06b/result.json
- SHA-256 5934f6d3f596c5d69baf97600af87c7fc41a844be50c3234fecd5148e0953b34

## Pre-commit isolated result
With low_margin=true, margin=0.010715:
- allocator selected low-cost L3 n5 and conditional L26 n5
- selector held L3 n5 because no L3 escalation evidence applied to that signal
- selector escalated L26 n5->n6 from real trigger evidence
- exact target policy hash matched the combined XOX acceptance
- scheduler atomically changed L3 6->5 and L26 5->6
- finite logits true, token 1224, same PID

With margin=0.03 the selector produced an unaccepted combined low-cost policy. The evidence gate rejected it before inference and runtime remained at the last accepted policy.

A repeated 0.010715 request became a no-op epoch because the accepted target policy was already active.

## Production boundary
No new HTTP API is exposed and the generic closed-loop engine is not configured in the live supervisor. Existing adaptive L26 production remains unchanged. A future cutover requires exact-source acceptance and a separate deployment gate.
