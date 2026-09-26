# Agent D Handoff — GPU Shadow Observability

Status: **VERIFIED**  
Date: 2026-09-26 KST  
Draft review PR: **#10**  
Production mutation: **OFF**

## Final integration

```text
pre-D certified XOX HEAD
067d4d8989b597785c756b1673bbf5a08564e61d

conflict-resolved D integration unit
bb76ecf07db3f4d3859424c397d056c2abdf4c57

integrated XOX HEAD
db88d77ffe75547fba7a1f4e667ec708f74daa6e
```

The original staging squash conflicted only in
`tools/gpu_shadow_pipeline.py` because certified XOX already had the real
production `discovery._json_safe(...)` fix. The conflict was aborted without
touching pre-existing dirty experiment files.

The final merge preserves both:

```text
discovery._json_safe(result)
+
SHADOW_CYCLE_FAILED / SHADOW_CYCLE_UNCLASSIFIED => nonzero exit
```

That combination was fixture-tested before the conflict-resolved integration
unit was applied.

## Tailnet

Final live release used for D integration:

```text
g6-0.4.0-alpha.33
preflight 10 OK / 0 WARN / 0 FAIL
```

Only narrow D fetch/cherry-pick/reporter argv were added. No generic production
write path was opened.

## D behavior

- `launch_id -> cycle_id -> shadow_run_id`
- CURRENT / PREVIOUS / LEGACY_UNLINKED result relation
- terminal-aware context dedupe
- bounded retry for SHADOW_ERROR/unclassified results
- retry exhaustion -> MANUAL_REVIEW_REQUIRED
- explicit SHADOW_CYCLE_COMPLETE / FAILED / UNCLASSIFIED
- read-only `gpu_shadow_report.py`
- `real_gpu` only with current linkage and explicit G4/G6 stage evidence

## Integrated regression

On `db88d77...`:

```text
test_gpu_*.py                151/151 PASS
test_manual_canary.py          35/35 PASS
git diff --check                    PASS
```

Request IDs:

```text
GPU suite    8e1627f6-e77f-4af8-af0e-77671f0d0361
canary suite 20e48ad7-e532-49cc-ba48-585833ae22d9
diff check   0de83747-51b7-4f5e-af08-089d8e74c829
```

## Live read-only reporter

Executed twice against:

```text
/Users/xox/vdsp_shadow_runs
```

Observed:

```text
cycle.status             NO_NEW_READY_CANDIDATE
pipeline_state           not_run
gpu_validation.state     not_run
evidence_level           not_run
source_freshness         legacy_unlinked
ready_count              1
candidate_count          100
dedupe_reason            reusable_terminal_same_candidate_and_runtime
production_write_allowed false
```

The state predates Agent D IDs, so these warnings are correct:

```text
LEGACY_CONTEXT_UNVERIFIED
LEGACY_CYCLE_WITHOUT_LAUNCH_ID
```

It is **not** relabeled as a new GPU PASS.

The four observed source artifact hashes were identical across both reporter
runs, proving the reporter did not mutate the observed state:

```text
launcher_status.json
9478ba25576cb97579807409b174ac2a541dc61a843fce24287c5f12d80e9c5b

last_cycle.json
0a4cd2b456667a737c5c346a8f90b837461026cf7c8f6ea92adfe7be6c56d215

candidate_history.json
81942763c65759602bbf0d8c3f3f5733638109c65e2e4ec4e4995201dd963d4c

discovery.json
de353587b5bdbc9d02c97f9d12ad602b9eaee83d210bcf0132342677f65c3c67
```

## Scope of VERIFIED

Verified:
- D code integrated on certified XOX
- complete GPU regression compatibility
- live reporter classification
- reporter read-only input invariance
- production write remained false

Not claimed:
- a new GPU candidate run
- new G4/G6 model-quality evidence
- production canary
- production promotion

## Next owner

A0/C can now ingest the Agent D result into the global certification manifest.

Do not force a shadow rerun merely to create launch/cycle IDs. A future
naturally-triggered cycle with new evidence will exercise those fields live.
