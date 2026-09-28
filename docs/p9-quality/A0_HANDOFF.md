# P9 A0 execution handoff

Status: **P9_QUALITY_FAIL / P9_EVIDENCE_INDEPENDENT_PASS**

Branch: `prod/p9-integration`
Integration base: `727d950ef420320455b5071969efd49e3751e8c8`

## Completed

The canonical P8 handoff is accepted with:

- `P8_RUNTIME_VALID_8_OF_8`
- `P8_POLICY_ATTRIBUTION_8_OF_8`
- `P8_EVIDENCE_INDEPENDENT_PASS`
- overall `P8_INTEGRATION_PASS`

P9 now has an explicit integrated adapter instead of a null placeholder:

- adapter: `p9-xox-mlx-metal-v1`
- hardware/backend: XOX / MLX Metal
- corpus: eight pinned prompts, two repetitions each
- cells: safe reference, n4, n5, n6, n7
- external score: deterministic token edit agreement against the reference
- physical runner: `beglin_p9_quality_fixture.py`
- exact Tailnet argv: `["python3","beglin_p9_quality_fixture.py"]`

The generated plan is `P9_EXECUTION_PLAN_2026-09-29.json`. Exact fixture registration is now frozen by `P9_TAILNET_REGISTRATION_EVIDENCE_2026-09-29.json`, and adapter preflight has no blockers.

The first physical run completed as durable XOX job `job_f640ce1213e392ed07d7d627482aa975` with run ID `p9-20260928T182317Z-56784`. The job state is `SUCCEEDED`, the runner result is `P9_PHYSICAL_RAW_PASS`, and all five logs contain 16 complete requests with finite logits.

The versioned root-cause follow-up completed as XOX job `job_a760be0f9dc67ef5b971a326a1dd1449` with run ID `p9a-20260928T192110Z-66994`. Tailnet alpha.62 admitted only `["python3","beglin_p9_single_target_ablation_fixture.py"]` on `xox/sandbox`, pinned to runner SHA `749abcc325bc4d10f41ade3be242b8de3b7119fd3795ea074a3f35283f0e22d3`. A live request with an added `--self-test` argument was rejected. All seven cells passed raw validation, and the independent audit passed 225/225 checks.

## Long-context implementation

The pinned retrieval prompt is 15,033 tokens and requires 15,045 positions with generation. The P9-only build sets a 16,384-position limit.

The original expanded MLA float32 cache would require roughly 9 GiB and the existing ragged `take` path could replicate the same B=1 cache for all 64 prefill rows. The isolated P9 path therefore:

- stores expanded K/V as symmetric int8 with one float16 scale per layer/head/position,
- preserves full causal attention rather than truncating or sliding the context,
- broadcasts the single slot cache across query rows with manual attention matmuls,
- allocates only the configured model layer count,
- leaves the default/P8 build path unchanged.

The registered runner recomputes the compact-cache projection from the pinned model config and fails closed if the projection is unexpectedly at or above 3 GB.

## Required physical evidence

The runner must produce one raw log for each of reference, n4, n5, n6, and n7. Every log must have 16 parsed request records with:

- finite logits across prompt and generation rows,
- complete teacher-forced NLL counts,
- deterministic output token IDs,
- promotion count,
- live activation summaries for `kv_a_proj_with_mqa/L11` and `shared_up_proj/L3`,
- expert-router selection-boundary near-tie counts,
- long-context output evidence.

Raw physical success is not a P9 quality PASS. EOE normalized all five artifacts, evaluated the complete metric set against the pre-candidate frozen policy, and completed the independent final audit.

## Current prohibition

`configs/quality/quality_policy_v1.json` remains `FROZEN`, hash-bound to the physical reference artifact and timestamped before any candidate log was inspected. The complete result is n4 PASS, n5 FAIL, n6 FAIL, and n7 FAIL. Independent evidence audit passes, but the overall gate is `P9_QUALITY_FAIL`; `production_write_allowed=false` and no production promotion or go-live is authorized.

The single-target boundary is now characterized. KV/L11 independently changes `constraint_words` at n5/6/7 and `ppl_short_1` at n6/7. Shared/L3 independently changes `short_reasoning` at n5 and `constraint_words` at n6/7. The combined path is nonlinear: at n6/n7 `constraint_words` follows the shared-only output, while the KV-only `ppl_short_1` change at n7 is cancelled when both targets are applied. This explains the failure but does not alter it.

## Final evidence

- physical result SHA-256: `430c507e061f561880eac2267a0ad2df07680fafcbf6d05f735ad4209819e4d3`
- evaluation hash: `ceb40a9c18a92231f484d9b528aaeca124218ff48d40ab76d9a52f466f954938`
- independent audit SHA-256: `481746339b5e87c28c9084301797284b7887c70dc397f9f6e701bab5e7d91074`
- audit file SHA-256: `1c2de16bf274f728d6711d943b30a5a357d7f210af464d02279b37abd752ffbc`
- single-target raw result hash: `65a79a88fe9a89e7f3d6cdd773323561407d8145f011dc12394f10e210b15fb4`
- single-target analysis hash: `e76d79a44fff9c24c21b385f85e8a6dac28ce96802669b2551e42d3f1f0b2ff9`
- single-target independent audit hash: `30e7dd91f58fb50a6af508e21f166bad09f22fca21c732bdda00c6dc7212c9c5`
- single-target audit file SHA-256: `0439142bbab1546c8b0340f04e2463ea863ad68a0e1eced33dac56cd2d6ce17d`

## Next exact action

Keep the frozen policy, failed matrix, and versioned ablation immutable. Do not promote n5, n6, or n7. A later engineering iteration must change the precision assignment or add a targeted mitigation, then submit a new complete versioned matrix; it must not overwrite this evidence or reinterpret n4's isolated PASS as approval of the full matrix.
