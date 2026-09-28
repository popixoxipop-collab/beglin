# BEGLIN P8 — Agent C Independent Baseline Audit

**Agent:** C — independent evidence auditor  
**Status:** `BASELINE_AUDIT_PASS_AWAITING_AGENT_A_B_RESULTS`  
**Rule:** Agent A/B runtime source was not modified. No merge or release action was performed.

## 1. Baseline identity

- Physical run: `p7-20260927T155716Z-2576`
- Tailnet job: `job_60ceca6c048312368a891bdae5ce337c`
- Runner: `xox-gpu-sweep-v2-instrumented`
- Source commit: `bc5f3d058d9e819bd088f42452b6ded0e12f3060`
- Binary SHA-256: `9b4c54d48587c18007076e16ff702d02fac0fac8d5d12000fa5936df489f4f79`
- Checkpoint SHA-256: `1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898`
- Real XOX `mlx_moe.cpp` SHA-256 observed directly: `7a2f69fe96e7df12aae532a9c9642d825a1f6a96e2ba9b5d7838f82ea99a9712`
- Public summary SHA-256 observed directly: `63b7131080c2c3095eb032f07f328fb8d4ceb4796512d19f98ff2ff87e6c129a`

## 2. Independent 8-cell recalculation

| Cell | rc | Terminal | Class | Requests | Activation | Observed n | Policy hash match |
|---|---:|---|---|---:|---|---:|---|
| kv_a/L11 n4 | 1 | FAILED_RUNTIME | UNSUPPORTED_OR_RUNTIME_ERROR | 0 | no | — | — |
| kv_a/L11 n5 | 0 | FAILED_EVIDENCE | POLICY_NOT_APPLIED | 12 | yes | 5 | yes |
| kv_a/L11 n6 | 0 | FAILED_EVIDENCE | POLICY_NOT_APPLIED | 12 | yes | 6 | yes |
| kv_a/L11 n7 | 1 | FAILED_RUNTIME | UNSUPPORTED_OR_RUNTIME_ERROR | 0 | yes | 7 | yes |
| shared_up/L3 n4 | 1 | FAILED_RUNTIME | UNSUPPORTED_OR_RUNTIME_ERROR | 0 | no | — | — |
| shared_up/L3 n5 | 0 | FAILED_EVIDENCE | POLICY_NOT_APPLIED | 12 | yes | 5 | yes |
| shared_up/L3 n6 | 0 | FAILED_EVIDENCE | POLICY_NOT_APPLIED | 12 | yes | 6 | yes |
| shared_up/L3 n7 | 0 | FAILED_EVIDENCE | POLICY_NOT_APPLIED | 12 | yes | 7 | yes |

Recalculated totals:
- `FAILED_RUNTIME = 3`
- `FAILED_EVIDENCE = 5`
- activation-bearing cells = 6
- performance-bearing cells = 5
- `requested_policy_applied=true` = 0
- activation n mismatch = 0
- activation policy-hash mismatch = 0
- activation role/layer mismatch = 0
- production write allowed = false
- production control unchanged = true

## 3. Blocking findings for P8

### C-F1 — Runtime failures are not one category

The two n=4 cells have no activation and no completed requests. In contrast, `kv_a/L11 n7` already produced a valid **n=7 activation binding** before the process failed.

**A0 implication:** Agent A must not collapse all three failures into one generic "policy not applied" bucket. n4 and kv n7 have different evidence boundaries.

### C-F2 — Strong attribution contradiction

All five cells that exit with rc=0 and complete 12 requests are labelled `POLICY_NOT_APPLIED`. However their activation record:

- matches requested role,
- matches requested layer,
- matches requested n,
- matches the cell `policy_hash`.

This is a strong **policy-attribution false-negative candidate**.

**A0 implication:** Agent B must repair or replace the policy-applied classifier using actual bound-state evidence. Merely changing the displayed classification is insufficient.

### C-F3 — kv_a/L11 n7 bound first, failed later

`kv_a/L11 n7` has:
- activation n = 7,
- matching role/layer/policy hash,
- then rc=1 / FAILED_RUNTIME.

This is evidence of successful binding followed by a later runtime failure.

### C-F4 — No activation-record attribution mismatch found

For all six activation-bearing cells, Agent C found no n/role/layer/policy-hash mismatch inside the activation evidence itself.

This **does not** authorize reclassifying P7 terminal states. It is a baseline invariant to compare against Agent B's fix.

## 4. Before/after fingerprints frozen for A/B

- `qwen_infer.c`: `4599edae10d7a093a08b0bb2e0681c368a227cb66ce29300210499b92ecdf414`
- `CMakeLists.txt`: `9c0c53cc4cc0acc854de541b6486363dfb43324fd283cebb23ebeda570f6e1d8`
- `mlx_moe.cpp`: `7a2f69fe96e7df12aae532a9c9642d825a1f6a96e2ba9b5d7838f82ea99a9712`
- `mlx_moe.h`: `2cf88a47b4383d12610a99e3e9d1661d981ab1f31bfb6abb4e85dd58d0299725`

Agent C will use these only as baseline fingerprints. Agent C will not modify those files.

## 5. Evidence-source triangulation

Primary:
1. original Tailnet job status/output,
2. P7 raw result,
3. P7 public summary,
4. direct XOX `mlx_moe.cpp` SHA read.

Cross-check results:
- raw/public run ID: match
- result SHA: match
- source commit: match
- binary SHA: match
- checkpoint SHA: match
- MLX source SHA: match
- terminal counts: match
- per-cell terminal/classification: match
- expected normalization only: public experiment IDs are lowercased

Secondary corroboration:
- project certifier reports six verified `layer.activation_summary` events and no instrumentation problems.

## 6. Audit limitation

Direct read-only `sqlite3` access to the dashboard DB is denied by the alpha.53 exec policy. Therefore journal rows were not queried with sqlite3 by Agent C. Journal event IDs are secondary corroboration only; the raw Tailnet job and physical evidence remain primary.

The binary/checkpoint underlying files are outside the currently exposed read-file workspace, so this pass cross-checks their hashes between original job output and immutable raw/public evidence rather than re-hashing those files directly.

## 7. P8 re-audit gate for Agent A/B

Agent C will reject an A/B integration result if any of the following happens:

- a negative P7 row disappears without replacement evidence;
- `POLICY_NOT_APPLIED` becomes PASS only because classifier text changed;
- requested n is claimed without actual bound-state proof;
- source/binary/checkpoint identity changes without a new control group;
- silent fallback appears;
- runtime-valid cells lack finite-logits evidence;
- Agent C reports or A0 release state are modified by A/B.

Required for Agent C final PASS:
- all 8 cells explicitly attributed,
- runtime crash eliminated or explicit unsupported state,
- actual bound n known for every supported cell,
- source/binary/checkpoint/policy identity consistent,
- finite logits for runtime-valid cells,
- zero silent fallback,
- no production mutation.

## 8. Next

Wait for Agent A and Agent B evidence. Then Agent C performs a **read-only before/after audit** and emits:

`P8_C_INDEPENDENT_INTEGRATION_AUDIT_2026-09-28.{json,md}`

No merge/release action will be performed by Agent C.
