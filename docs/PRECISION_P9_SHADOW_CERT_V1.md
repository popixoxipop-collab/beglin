# Precision P9 Shadow Certification v1

P9 closes the orchestration gap between a P8 policy-refresh proposal and the existing scratch-only GPU shadow pipeline. It has no production mutation path.

Flow:
P8 proposal -> replay-provenance validation -> READY_FOR_SHADOW queue item -> repeated gpu_shadow_runner executions -> unanimous SHADOW_ADMITTED -> MANUAL_REVIEW_CANDIDATE.

Safety contracts:
- a proposal without replay provenance remains WAITING_FOR_REPLAY_PROVENANCE and cannot execute shadow;
- raw token SHA, replay manifest path and max-new-token count are revalidated before queue admission;
- shadow repeats are 2..10 and all runs must be SHADOW_ADMITTED;
- one rejection blocks certification;
- every shadow child keeps production_write_allowed=false;
- certification can emit only MANUAL_REVIEW_CANDIDATE;
- automatic_live_promotion=false and a separate manual review/cutover gate remains mandatory.

The historical L3 n6->n5 P8 proposal remains blocked because its original replay input was not preserved. P8 replay provenance capture added in PR #48 applies to future evidence and does not retroactively manufacture this missing input.

Unit/regression acceptance: P9/P8 focused 8/8 PASS; GPU suite 227/227 PASS.
