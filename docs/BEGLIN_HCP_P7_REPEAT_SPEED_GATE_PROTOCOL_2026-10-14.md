# Beglin HCP P7 — preregistered repeated precision-policy performance gate (2026-10-14)

## Authority, starting point and reason
- Based on immutable P5 seal \`b5cf761089153a6326855f267f1381fadb3f414e\` and the isolated P6 AF-to-qNg64 logit study.
- Original P5 12-domain 106–112-token paired B-minus-A engine time mean **+26.2745 ms**, 95% bootstrap about **[-155.28, +226.75] ms**, p95 **A 860.96 / B 993.60 ms**. These data do **not** prove B faster. The median B advantage is insufficient.
- Never modify main, \`/Users/xox/vdsp_serving/persistent-stage\`, live route, production precision policy, release plane, or stored checkpoint. No automatic live promotion.
- Keep REBIND_SET and terminal ACK verification enabled (not monkeypatched), require stage and route SHA before/after, run under unique RUNNER_TEMP scratch, one GPU request slot, RSS <14 GiB, physical RAM >=40 GiB, free scratch >=3 GiB, source/checkpoint digests pinned.
- DeepSeek-V2-Lite MLA only; 104–112 prompt tokens plus at most four generation tokens. No general KV128, GQA, BF16-teacher or quality claim.

## Two distinct hypotheses, tested separately
1. **Warm steady inference hypothesis:** B actually has lower end-to-end request cost than A when the worker stays at the selected policy, even without precision switching.
2. **Adaptive switching hypothesis:** Policy rebind provides enough net savings over a predetermined workload to amortize every REBIND_SET, queue/planner, GPU warmup and roundtrip cost. A lower median engine-only time alone does not satisfy this.

## Frozen workload
- Independently preregistered 12 source domains, same 12 P5 prompt token hashes and original 36-prompt P4 lineage SHA \`c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53\`.
- Keep original Python policy A (L3 shared_up n6, L26 shared_down n5) and B (L3 shared_up n5, L26 shared_down n6). No tuning on holdout or in-place algorithm edits.
- Same prompt tokens and 4-token budget each treatment, no extra corrective forward pass, finite logits, unchanged resident PID, exact per-policy ACK and epoch/txn ID.
- Pilot first: 2 prompts × 3 complete temporal blocks. Do not execute full study unless pilot validates complete tracing, no OOM, and no interference or worker restart.
- Formal repeated target: **at least 20 complete temporal blocks**, 12 matched prompts per block, ABBA versus BAAB block order balanced and alternated according to precommitted seed. Pair by (block, domain) to address drift. Record sequence and any early stopping explicitly. No results from partial/failed blocks in success denominator.
- Distinguish (a) first-request cold/warmup, (b) same-policy repeated steady admissions (no epoch change), (c) A↔B transitions. A claim about steady-state cannot use switched requests, and a claim about dynamic policy cannot omit transitions.
- Observe wall-clock thermal state/power availability if exposed without elevated sensor action; never artificially control or mask real thermal limitations.

## Per-admission measurements
- \`request_roundtrip_ms\` from application submit to verified completion; \`engine_wall_ms\`; \`transition_wall_ms\`; scheduler/planner time if externally measured; \`gpu_rss_bytes\`; \`TTFT\`; output count and token/s when supported.
- Policy SHA, exact admission ID, epoch before/after, txn ID, ACK status, no corrective inference flag, source/checkpoint/binary hashes.
- Never export raw user IDs, prompt tokens, generated token IDs, full logits, weights or live worker logs. Export only bounded aggregate records, block-level paired deltas and SHA receipts.
- Report per-policy p50/p95 and B-A mean/median paired differences. Do not conflate throughput, engine time and end-to-end latency.

## Pre-registered statistical reporting
- Estimate B-minus-A paired roundtrip and engine differences using **temporal-block cluster bootstrap**, 10,000 draws, fixed seed, 95% two-sided intervals. Also report signed per-block medians; no post hoc exclusions.
- Report p95 request tail for A/B and worst block; show both absolute milliseconds and percent of A.
- A practical speed-claim gate requires all: (1) 95% upper interval for B-A **roundtrip** is <0, (2) best-estimate end-to-end roundtrip benefit >=3%, (3) B p95 <=1.05×A p95, (4) no ACK mismatch/precision rollback, error, OOM, or production mutation, and (5) P6 fidelity comparison does not show an unresolved material regression. Even this establishes only scoped experimental speedup, not universal model advantage or safety.
- Explicit \`NO_PROVEN_SPEEDUP\` whenever interval includes 0, measurement incomplete, mixed workloads unreported or p95 regresses.
- Independently report **switch amortization break-even** = typical transition cost divided by observed per-request warm time saving, if the saving is credibly positive. If <=0, break-even is undefined/no amortization.
- If P6 is inconclusive on fidelity, do not trade unknown quality for speed.

## Isolation / release
- Research branch and run-owned scratch only. No environment variable in the deployed supervisor, no write to active-route, no mutation of release APIs.
- Before-and-after SHA of immutable stage and active route; fail closed upon mismatch. Preserve successful metrics, receipts, source SHA, action ID; remove scratch checkpoint copies, raw token IDs, logits and workers on *every* exit.
- Promotion remains \`BLOCKED\` until independent production certification, concurrency/memory/cold-restart tests and formal authority. Any eventual Release Plane change requires fresh current/live/runtime verification, rebase on latest release, and CAS with \`expected_current\` after stage/verify PASS.

## Current disposition
- This file is a preregistered protocol, **not** executed P7 evidence.
- The P6 same-input AF versus qNg64 logit study must be interpreted before drawing a P7 cost/quality conclusion.
