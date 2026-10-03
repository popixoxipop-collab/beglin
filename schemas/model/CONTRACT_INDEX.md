# P12 Model Contract Index

Canonical contracts for the model-capability front door.

- `architecture-descriptor-v1.schema.json`
- `backend-capability-v1.schema.json`
- `loader-contract-v1.schema.json`
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

Identity rule: timestamps and absolute source locations are excluded from stable identity hashes; content/checkpoint hashes are not. Unknown architecture/operator/tokenizer/quantization capability must fail closed. `IMPLEMENTED_UNVERIFIED` is never equivalent to `VERIFIED`.
