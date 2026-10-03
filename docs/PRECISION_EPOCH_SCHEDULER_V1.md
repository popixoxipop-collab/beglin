# Precision Epoch Scheduler v1

## Scope
This stage generalizes the proven single-target GPU REBIND path into an admission-scoped multi-target precision epoch. Production live remains on the existing adaptive L26 path until this successor is separately promoted.

## Runtime contract
REBIND_SET changes up to 8 resident qNg64 targets in one transaction.

- one immutable preimage: expected_epoch + active-policy hash + every target expected n
- admission pauses at the existing quiescent boundary
- MLX synchronizes once before mutation
- every current binding is snapshotted before the first new binding is installed
- all target bindings are materialized and verified before commit
- runtime graph/KV epoch reset happens once
- weight_epoch increments exactly once regardless of target count
- any partial failure restores the full preimage and emits REBIND_SET_FAILED
- legacy DEMOTE and one-target REBIND remain supported

Command:
REBIND_SET txn_id expected_epoch count policy_sha256 (role layer expected_n target_n repeated count times)

## Scheduler contract
tools/precision_epoch_scheduler.py serializes:
read ACK -> CAS plan -> REBIND_SET -> inference -> terminal ACK verification

A second admission cannot read or mutate an intermediate policy epoch. The scheduler shares the worker same reentrant admission lock with ordinary submit(), so policy-aware and direct worker admissions cannot interleave either.

v1 intentionally refuses hot policy-shape changes. All role/layer targets must already exist in the worker startup policy. Adding or removing a target requires worker restart.

PersistentRouteWorker.submit_with_precision_policy() is the opt-in bridge. Existing production submit() and current L26 adaptive path are unchanged.

## XOX isolated evidence
Candidate native binary SHA-256:
4540599d86a59dd3cfd14fc863d21ed3f0cc5a8a1d872a821a04bc04ae23b490

Multi-target sequence:
- startup: shared_up/L3 n6 + shared_down/L26 n5, epoch 2
- epoch 3: L3 6->5 AND L26 5->6
- epoch 4: L3 5->6 AND L26 6->5
- epoch 5: L3 6->5 AND L26 5->6

All three admissions used PID 42643, finite logits stayed true, and repeated widths hit qNg64 cache for both targets.

Evidence path:
/Users/xox/vdsp_serving/precision-epoch-multitarget-acceptance-20261003/result.json
SHA-256:
de7031f3127f44916938db660b62c480eb8529054c445b1464df50fa28aef68d

Concurrent admission evidence:
- request A requested the two-target alternate policy
- request B arrived 20 ms later requesting base
- same PID 42780
- A completed epoch 2->3
- only then B executed epoch 3->4
- no policy or epoch interleaving occurred

Evidence path:
/Users/xox/vdsp_serving/precision-epoch-concurrency-acceptance-20261003/result.json
SHA-256:
ed582876737c948f5c2e77b0d39720f384876c6617dcc894d6081f236c1b2a69

## Current production boundary
This stage does not enable generic precision-policy requests on live HTTP and does not replace the live binary.

Production remains:
- route generation 6
- adaptive L26 enabled
- automatic promotion disabled
- external network exposure disabled

Next gate: allocator and selector policy materialization into the scheduler, then separate production acceptance.

## Post-commit acceptance
The implementation was fixed at source commit:
ae1b06b3fbb86930e4c23e51f6ff82373272cde5

Exact-source XOX isolated acceptance:
/Users/xox/vdsp_serving/precision-epoch-acceptance-ae1b06b/result.json
SHA-256:
5934f6d3f596c5d69baf97600af87c7fc41a844be50c3234fecd5148e0953b34

The evidence reports source_head ae1b06b3fbb86930e4c23e51f6ff82373272cde5,
sequential PASS, concurrency PASS, production_touched=false, and native binary
SHA-256 4540599d86a59dd3cfd14fc863d21ed3f0cc5a8a1d872a821a04bc04ae23b490.

Certified successor regression:
- run 37114905775
- head ae1b06b3fbb86930e4c23e51f6ff82373272cde5
- conclusion SUCCESS
