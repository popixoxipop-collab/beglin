# P8 Agent C — B Interim Static Audit

**Branch:** `prod/p8-policy-binding`  
**Audited HEAD:** `7d931750ff4e36a782e805b210ed57ba128675dd`  
**Base:** `a0b8ab2c90af70fa8095556033e605477ec186ae`  
**Verdict:** `IMPLEMENTATION_PLAUSIBLE_NOT_GATE_READY`

## What C independently verified

GitHub compare is currently **ahead 20 / behind 0** and changes only:

- `mlx_moe.cpp`
- `mlx_moe.h`
- `tests/policy_binding/**`

Ownership is clean.

The branch introduces runtime truth APIs that read the same representation maps used by MLX dispatch:

1. `g_qng64_tensors`
2. `g_tensors`
3. `g_dtensors`

Missing state and requested-n mismatches fail closed.

Role/layer lookup is centralized inside the backend and supports the P8 target roles including `kv_a_proj_with_mqa` and `shared_up_proj`.

Instrumentation now derives n/representation from the canonical binding truth and emits `BEGLIN_POLICY_BINDING_V1`.

## Strong point

This design directly addresses C baseline finding **C-F2**: P7 showed exact activation role/layer/n/policy-hash matches while the old classifier still emitted `POLICY_NOT_APPLIED`. B's new API uses actual representation maps instead of relying on optional promotion/debug text.

## Why C does not pass B yet

### C-B1 — Synthetic 8-cell probe is not the canonical workload

`binding_runtime_probe.cpp` binds tiny synthetic buffers with `mlx_gpu_bind_af()` and proves the query/assert API for:

- kv_a/L11 n=4/5/6/7
- shared_up/L3 n=4/5/6/7

That is useful contract/runtime-unit evidence, but it does **not** prove the actual model loader and inference path under the real checkpoint.

Final B evidence must show actual requested n vs actual bound n from the canonical P8 physical/integration run.

### C-B2 — Immutable gate evidence is still absent

The branch can print:

`P8_POLICY_ATTRIBUTION_8_OF_8`

but C found no immutable final P8-B evidence document that binds this to source/binary/checkpoint/policy identities and raw command output.

### C-B3 — Finite-logits/runtime correctness belongs to integrated A+B run

B cannot independently establish P8 completion. A0 must integrate **B then A**, run the canonical 8-cell matrix, and pass the result back to C.

### C-B4 — Classifier change alone is not evidence

A future result is rejected if it only changes `POLICY_NOT_APPLIED` to PASS.

C requires, per supported canonical cell:

- requested n,
- observed bound n,
- representation,
- role/layer,
- policy hash / epoch,
- immutable source/binary/checkpoint identity.

## Current C position

B implementation is structurally promising and ownership-clean, but remains:

`IMPLEMENTATION_PLAUSIBLE_NOT_GATE_READY`

C will not issue `P8_EVIDENCE_INDEPENDENT_PASS` until B final evidence and the post-integration canonical 8-cell run exist.
