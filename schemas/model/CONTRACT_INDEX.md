# P12 Model Contract Index

Canonical contracts for the model-capability front door.

- `architecture-descriptor-v1.schema.json`
- `backend-capability-v1.schema.json`
- `backend-transition-result-v1.schema.json`
- `loader-contract-v1.schema.json`
- `loader-runtime-plan-v1.schema.json` — explicit source-file-bound loader execution plan
- `model-capability-bundle-v1.schema.json`
- `model-skeleton-v1.schema.json`
- `model-source-v1.schema.json`
- `operator-graph-v1.schema.json`
- `pipeline-eligibility-v1.schema.json`
- `quant-capability-v1.schema.json`
- `runtime-mutation-v1.schema.json`
- `tensor-role-graph-v1.schema.json`
- `tokenizer-contract-v1.schema.json`
- `validation-plan-v1.schema.json`
- `verification-evidence-v1.schema.json`

Identity rule: timestamps and absolute source locations are excluded from stable identity hashes; content/checkpoint hashes are not. Unknown architecture/operator/tokenizer/quantization capability must fail closed. `IMPLEMENTED_UNVERIFIED` is never equivalent to `VERIFIED`.

Verification rule: any capability reported as `VERIFIED`, `IN_ENGINE_VERIFIED`, or `EXTERNAL_VERIFIED` must carry a `beglin-verification-evidence-v1` reference whose checkpoint/architecture/backend identity matches the inspected model. A boolean flag or file presence alone is never verification.
- `tokenizer-runtime-plan-v1.schema.json` — explicit in-engine/external tokenizer execution plan with executable hash binding

## QT Heatmap v2 foundation

- `qt-observation-identity-v1.schema.json` — checkpoint/skeleton/capability-bound group identity
- `qt-numeric-observation-v1.schema.json` — cheap runtime numeric statistics for a canonical cell
- `qt-quant-perturbation-v1.schema.json` — per-bit quantization perturbation and backend-parity evidence
- `qt-training-sensitivity-v1.schema.json` — gradient/Fisher-proxy evidence for selective training
- `qheatmap-v2.schema.json` — precision recommendation cells; BECODER candidate only
- `theatmap-v2.schema.json` — selective-training sensitivity cells; BECODER candidate only
- `local-precision-policy-v1.schema.json` — BEVAL-gated local group precision policy
- `qpolicy-candidate-v1.schema.json` — deterministic BECODER output; hard-unapproved until BEVAL

QT invariant: BSKEL defines identity/schema only; BECODER proposes Q/T cells; runtime must not apply a local precision policy unless `approved_by_beval=true`. Initial local unit is qNg64 group64; element-level policy is out of QT-0 scope.

- `q-uncertainty-v1.schema.json` — uncertainty-aware QT-3 BEVAL evidence: safe/boundary/non-monotonic/unsatisfied classification and active-observation priority. `safe_probability` is an evidence score until calibration evidence upgrades it to a posterior.
