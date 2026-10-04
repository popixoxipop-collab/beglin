# P12 Model Capability Contract v1 — Phase 1

This change begins P12 without touching the live P11 production runtime.

## Landed in this slice

- 13 versioned model/capability JSON contracts under `contracts/model/`.
- Deterministic source, architecture, tensor-role, operator, skeleton and capability identities.
- Fail-closed unknown-architecture behavior.
- Evidence-required `VERIFIED` backend capability cells.
- Backend-symmetric transition planning:
  - CPU reports `RESTART_REQUIRED` for precision changes.
  - MLX can report `HOT_REBIND_SINGLE` / `HOT_REBIND_MULTI` only for explicitly verified resident targets.
  - policy-shape changes remain restart-only.
- Read-only source inspection for:
  - GGUF identity,
  - single safetensors,
  - sharded safetensors + index integrity,
  - config-based architecture name discovery.
- Read-only `tools/inspect_model.py` Phase-1 front door.
- Unit tests for determinism, fail-closed behavior, evidence requirements, source integrity and CPU/GPU schema symmetry.

## Deliberate boundary

This slice does **not**:
- enable a new production mutation endpoint,
- auto-promote any model,
- claim arbitrary GGUF architecture discovery,
- claim tokenizer/loader verification beyond the evidence already present,
- infer semantic tensor roles from unknown architecture names.

Next slices compile real architecture adapters, tensor/operator graphs, tokenizer/loader registries and bind the resulting `ModelCapabilityBundle` into P8-P11.
