# Precision E2E Cost v1

## Scope
P4 adds measured end-to-end cost optimization without weakening any correctness or evidence gate.

Correctness remains a hard constraint. Cost is used only among allocator-admitted widths and exact accepted multi-target policies.

The measured dimensions are:
- steady inference engine and roundtrip latency
- quiescent REBIND/REBIND_SET transition wall time
- qNg64 cache hit/miss counts
- cache bytes materialized by a transition
- total resident qNg64 cache bytes
- expected inference pass count
- active quantized weight bytes
- expected end-to-end latency
## Native telemetry
The durable GPU precision ACK now carries:
- transition_wall_ms
- transition_cache_hits
- transition_cache_misses
- transition_cache_bytes_added
- resident_qng64_cache_bytes
- qng64_cache entries keyed by role/layer/n with exact allocated bytes

Transition timing starts when a validated CAS command is accepted and includes the quiescent boundary/drain before durable ACK.

The qNg64 cache registry is structural rather than inferred from logs, so a request can distinguish cold materialization from a warm cache hit.

Backward compatibility is preserved: old ACKs may omit P4 telemetry and existing non-P4 code paths continue to work.
## Cost model
tools/precision_e2e_cost.py enriches target candidates using the current runtime ACK and an offline measured cost snapshot.

tools/precision_policy_cost_optimizer.py evaluates only exact PASS combined policies.

If triggers are active, the selector precision decision is pinned and cost optimization is forbidden from reducing it.

With no active trigger, the optimizer may avoid an unnecessary safe-policy transition when the current exact accepted policy has lower measured E2E cost.

If a requested cost dimension lacks complete measurement, the allocator refuses to estimate it.

No Supabase/network request occurs while the serving admission lock is held.
## XOX pre-commit measurements
Scratch-only benchmark:
/Users/xox/vdsp_shadow_runs/precision_e2e_cost/p4-policy-20261003/result.json

Benchmark SHA-256:
3faf5269632e030811b0e5e51ce7e7720066cb14eb82a8c0c505025b42bcae5b

Native benchmark binary SHA-256:
b7dbe0fe768f53ed22dd2ec88d44d61a52d647a93509af9f57d63f98fa2d4c86

Exact policy A:
shared_up/L3 n6 + shared_down/L26 n5
steady roundtrip p50 about 181.10 ms.

Exact policy B:
shared_up/L3 n5 + shared_down/L26 n6
steady roundtrip p50 about 186.73 ms.
A -> B cold transition:
- p50 about 621.00 ms
- 2 cache misses
- 8,650,752 bytes newly resident

A -> B warm transition:
- p50 about 307.50 ms
- 2 cache hits
- 0 newly resident bytes

Individual cold materialization:
- shared_up/L3 n5: about 294.60 ms, +3,964,928 bytes
- shared_down/L26 n6: about 326.74 ms, +4,685,824 bytes

The two individual cache allocations sum to the measured combined +8,650,752 bytes.
## P4 scratch acceptance
Pre-commit acceptance:
/Users/xox/vdsp_serving/precision-e2e-cost-acceptance-v1/result.json
SHA-256:
098a913c372b6e60c4e1eda8d79ed1935b2d77344353ed348540812953a052fc

Observed behavior:
- no-risk on policy A: cost optimizer kept A, transition cost 0
- low-margin risk: correctness forced policy B despite its cold cost; token 1224
- cold B transition: actual 625.5 ms, 2 misses, +8,650,752 bytes
- no-risk while already on B: optimizer kept B instead of thrashing back to A
- later low-margin transition A -> B was warm: actual 308.161 ms, 2 hits, 0 added bytes
- final scratch state restored to policy A
- production_touched=false

## Exact-source P4 evidence
Implementation commit:
5c9e111bfd3ff12e1274864cb7fe47fa287d7e11

Exact-source cost benchmark:
/Users/xox/vdsp_shadow_runs/precision_e2e_cost/p4-5c9e111/result.json
SHA-256:
e4e5ed9cc05a5d85b565b0accbae4df90143b9703d54aac4c3a52404b3c463b4

The benchmark reports source_head 5c9e111bfd3ff12e1274864cb7fe47fa287d7e11 and native binary SHA-256 b7dbe0fe768f53ed22dd2ec88d44d61a52d647a93509af9f57d63f98fa2d4c86.

Exact-source P4 closed-loop acceptance:
/Users/xox/vdsp_serving/precision-e2e-cost-acceptance-5c9e111/result.json
SHA-256:
86f6c2a5d3cd747236920207c79bbc919509de08d5706d210aaaf57202c978cf

Observed exact-source acceptance:
- no-risk startup kept policy A, transition 0 ms
- correctness-triggered cold A -> B stayed mandatory, actual transition 616.699 ms, 2 misses, +8,650,752 bytes, token 1224
- no-risk while B was active kept B instead of paying a transition back to A
- later A -> B warm transition measured 314.29 ms, 2 hits, 0 new bytes, token 1224
- final scratch state restored to policy A on the same PID
- production_touched=false

Certified successor regression:
- run 37118827288
- head 5c9e111bfd3ff12e1274864cb7fe47fa287d7e11
- conclusion SUCCESS
## Production boundary
P4 is not enabled in production by this implementation stage.

Current production remains generation 6 on the existing adaptive L26 supervisor, with auto-promotion OFF and no external exposure.

A later cutover must separately pin the new native binary, the measured cost snapshot, and exact-source P4 acceptance evidence.

## Inference-pass measurement hardening
The original P4 acceptance measured REBIND/cache/resident-memory costs directly, while accepted closed-loop policies were one-pass and therefore carried expected_inference_passes=1.

This hardening makes the worker result itself authoritative for pass count:
- every ordinary PersistentRouteWorker admission reports inference_passes=1;
- AdaptivePersistentRouteWorker reports inference_passes=2 when a low-cost base pass is followed by n6 recovery;
- the scratch P4 benchmark records an adaptive_recovery_profile from the real XOX worker and requires observed_inference_passes=[2];
- the P4 acceptance refuses cost evidence that does not contain this real two-pass measurement.

Exact-source evidence is generated after the implementation commit is fixed.

## Exact-source inference-pass evidence
Implementation commit:
78740c0209d4f0483de9b85b47fdb99767c2c7f4

Exact-source benchmark:
/Users/xox/vdsp_shadow_runs/precision_e2e_cost/p4-pass-metrics-78740c0/result.json
SHA-256:
9647217bf707741fdbf11c7ff0f32bbf719fd090b61060c016bf9e8f4b923871

Adaptive recovery profile:
- same PID 82470
- 3 measured samples
- observed_inference_passes=[2]
- expected_inference_passes=2
- p50 engine about 668.033 ms
- p50 roundtrip about 701.572 ms

Exact-source P4 acceptance:
/Users/xox/vdsp_serving/precision-e2e-cost-pass-metrics-78740c0/result.json
SHA-256:
0251105c3a37703681a8b9ad209b980dcb444c7849f02d341c84e4af834fe14b

The acceptance is source-bound to 78740c0209d4f0483de9b85b47fdb99767c2c7f4 and explicitly carries the measured adaptive 2-pass profile while closed-loop exact-policy admissions report inference_passes=1.

Certified successor regression:
- run 37122034952
- head 78740c0209d4f0483de9b85b47fdb99767c2c7f4
- conclusion SUCCESS
