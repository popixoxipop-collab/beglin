# Agent F — GPU coverage harness

This file belongs to the parallel Agent-F track. It prepares and validates
coverage evidence; it does not perform production mutation.

## Scope

Historical rows retained for reference only:

| target | historical positive | historical G5 |
|---|---|---|
| shared_down_proj/L4 n=6 | G4/G6 PASS | n=5 rollback PASS |
| kv_a_proj_with_mqa/L11 n=7 | custom-Metal G4/G6 PASS | unverified |
| shared_up_proj/L3 n=6 | G4/G6 PASS | n=7 rollback PASS |

The remaining real-hardware priorities are:

1. kv_a_proj_with_mqa/L11 bounded bad-candidate search (n=5/6/7 only when the certified runtime reports those widths supported);
2. actual G5 rollback only if a genuine failing candidate is observed;
3. G6 positive/regression/budget/inconclusive matrix for kv_a/L11 and shared_up/L3;
4. custom-Metal single-slot vs multi-slot regression check.

## Evidence discipline

gpu_coverage_f.py refuses to label decision-only rows as real GPU PASS/FAIL.
Budget and inconclusive decisions derived by changing controller thresholds on
one hardware observation remain decision_only with reused_hardware_run=true.

A regression row needs an explicit failing observation. The helper does not
flip target_replay_pass on a passing hardware observation and then call that a
real regression.

## Current execution gate

Real GPU execution requires both:

- an A0 XOX GPU resource lease; and
- an Agent-B checkpoint handoff with status=VERIFIED and a 64-hex actual
  checkpoint identity.

Without those, the generated plan sets real_gpu_ready=false. Unit/fixture work
can continue.

Example command:

    PYTHONPATH=tools python3 tools/gpu_coverage_f.py plan \
      --base-sha <certified-base-sha> \
      --checkpoint-handoff /scratch/B/checkpoint_handoff.json \
      --output /scratch/F/coverage_plan.json

The real runner on the certified XOX worktree is intentionally a separate
integration step. It must use a unique scratch control root and one owner-run
harness for ACK/txn sequencing; do not drive mid-run demotion by manual remote
round-trips.
