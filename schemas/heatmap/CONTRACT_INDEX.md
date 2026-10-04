# Q/T Heatmap Contract Index

Canonical contracts for adaptive local precision and selective training.

- `observation-identity-v2.schema.json` — canonical model/layer/role/group identity plus checkpoint, capability, policy and weight epochs
- `numeric-observation-v1.schema.json` — cheap inference-side numeric telemetry
- `quant-perturbation-v1.schema.json` — per-candidate-n perturbation and backend parity evidence
- `training-sensitivity-v1.schema.json` — gradient/Fisher-proxy/selective-training observations
- `qheatmap-v2.schema.json` — precision-allocation heatmap
- `theatmap-v2.schema.json` — selective-training heatmap
- `local-precision-policy-v1.schema.json` — BEVAL-approved group-local precision policy
- `selective-training-policy-v1.schema.json` — BEVAL-approved train/freeze/LR policy
- `qt-beval-evidence-v1.schema.json` — Q/T evaluation evidence

Identity invariants:
1. checkpoint, skeleton, capability bundle, group size, and weight epoch are part of the accumulation boundary.
2. observations from different identity generations MUST NOT be merged.
3. tensor source names are provenance; canonical `target_key` is the decision identity.
4. v1 local allocation unit is qNg64 group64. Element-level allocation is not enabled by this contract set.

Authority boundary:
- BSKEL defines schema and legal states.
- BECODER may estimate Q/T heatmaps and candidate policies.
- BEVAL is required before a policy is marked approved.
- Runtime and trainer MUST reject unapproved policies.
- Unknown, stale, low-confidence, unsupported, or provenance-mismatched cells fail closed.
