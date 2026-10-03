# P6 observability integration and review

## Source boundary
The P6 work in `precision-observability-v1-20261003` was already present when
this continuation started. This branch preserves commits through bd56172 and a
snapshot of the uncommitted work (patch SHA-256
11d5bfff30a48361119923377e8df6911e1907290a6b68e168407d5719d6ce5f).
The original working directory was not edited by this review.

## Request lineage
All opt-in paths share the worker admission RLock through recording:
closed-loop selection, explicit policy, ordinary native submission and the
legacy adaptive two-pass path. Nested calls produce one terminal record.
Decision rejections and execution failures are separate from successful returns;
a rejected decision has zero inference passes, not an invented measurement.
Existing production enablement and precision policies are unchanged.

Each line records admission/worker identity, source commit, supervisor and binary
SHA-256, checkpoint identity, sanitized signal fields, evidence references,
selected policy, before/after weight epoch, expected/actual cost, outcome, and a
response digest. Prompt text and raw generated token arrays are not journaled.
The hash chain can be checked against its separately retained terminal head; it
is not a substitute for externally trusted storage or a signing service.

## Metric definitions
Trigger rates count triggered **admissions**, not affected requests in a batch.
Request count is reported separately. Cache hit rate uses measured hit+miss
counts. Extra-pass rate uses admissions with a measured positive pass count;
pre-inference rejections are excluded from that denominator.
Precision residency is **successful admission end-state count**, not elapsed
wall-clock residency or time spent in an intermediate adaptive policy.
Finite logits mean finite numbers only, not reference accuracy or response quality.
Unknown telemetry remains null and is counted as missing, not zero latency.

## Journal behavior
The normal record path checks file identity/size/timestamps and updates aggregate
counters without rescanning the full history. Full replay happens on startup,
external file changes, or explicit verification. A process advisory lock protects
cooperating writers. Inputs and policy/epoch continuity are validated before
append so a rejected record cannot poison the existing journal.
Unique admission IDs are retained for duplicate detection; their memory footprint
grows with history. Log retention/rotation and remote ingestion are not in v1.

Snapshot failure is reported as degraded after a successful durable journal
append. Default non-strict observation failures do not change inference output;
strict mode remains for acceptance. Malformed/incomplete existing logs are not
silently repaired or discarded.

## Read-only consumption
The successor includes `GET /observability`, returning per-worker summary or
NOT_CONFIGURED. The current production binary and process were not replaced.
Offline audit: `PYTHONPATH=tools python3 tools/precision_observability.py --lineage PATH`.

## Validation
Added tests cover concurrent writers, invalid/tampered/truncated data, pre-append
continuity checks, snapshot failure, unknown costs, nested admissions, rejection,
execution failure, nonfatal sink behavior and read-only routing.
The XOX acceptance checks all four paths plus a rejected policy, negative-control
inference, actual adaptive recovery, exact journal replay, source/binary identity,
and unchanged production PID/binary/route snapshots. Exact-source results are
recorded after the implementation commit is fixed.

## Exact-source XOX evidence

Implementation: `f1f8307048e92bbfa3467e2bb6dfe65a73eb2ed6`. Native binary: `a9b35e4e62480899962c6d3dc2890f5042c9a893abbae313b8366becb44cb028`.

Cost evidence: `/Users/xox/vdsp_shadow_runs/precision_e2e_cost/p6-review-f1f8307/result.json`
SHA-256: `eda23b8d594cd80a935975994d9b61f9882f4cc76a8ec7479efabe971d441336`.

Acceptance: `/Users/xox/vdsp_serving/precision-observability-review-f1f8307/result.json`
SHA-256: `72ffe37957fe2af291cb613523d91aada2b70c4bc292a40e985a075fb2ee1814`.

Nine records: 8 SUCCESS, 1 REJECTED; closed_loop 4, explicit_policy 2, direct 1, adaptive 2.
Native transaction/ACK did not change on rejection.
Pass histogram: 0-pass 1, 1-pass 6, 2-pass 2; extra-pass rate 2/8 = 25%.
Measured transition cache hits 6, misses 2 (75% hit rate). Five admissions changed
precision, containing eight target changes in total. Same isolated worker PID `13854`.
Fresh in-memory, persisted and restarted/offline summaries were exactly equal.
Three recovery checks returned token 1224. The deliberate direct negative-control
request returned token 16822; it is not mislabeled as reference accuracy PASS.

Recording cost in this small run: median 0.995 ms,
maximum 1.162 ms over 8 successful admissions.
This is a tiny scratch sample, not a production SLO or a throughput claim.

Validation: GPU unit/guard suite 214/214, focused allocator/selector/supervisor
58/58, native transition 1/1. Certified successor regression run 37126150619
SUCCESS on exact implementation f1f8307048e92bbfa3467e2bb6dfe65a73eb2ed6.

Production pre/post snapshots were equal: generation 6, PIDs 35481/35493,
binary 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9.
P6 remains opt-in and is not deployed to production.
