# P9 single-target boundary report

Status: **P9_SINGLE_TARGET_EVIDENCE_INDEPENDENT_PASS**

This report characterizes the output drift previously observed when `kv_a_proj_with_mqa/L11` and `shared_up_proj/L3` were promoted together. It does not change the frozen P9 matrix or authorize production.

## Physical experiment

- Tailnet release: `g6-0.4.0-alpha.62`
- XOX job: `job_a760be0f9dc67ef5b971a326a1dd1449`
- Run: `p9a-20260928T192110Z-66994`
- Exact argv: `["python3","beglin_p9_single_target_ablation_fixture.py"]`
- Cells: reference plus each target alone at n=5, n=6, and n=7
- Diagnostic prompts: `constraint_words`, `ppl_short_1`, `short_reasoning`
- Repetitions: 2 per prompt, 6 requests per cell
- Result: all 7 cells PASS, all logits finite, all repeated token sequences deterministic

The reference output on all diagnostic requests exactly matches the corresponding reference requests from the frozen full P9 run. The binary and checkpoint hashes also match. This makes the old combined cells and the new isolated cells comparable at the observed greedy-token boundary.

## Isolated target boundaries

| Target | n=5 | n=6 | n=7 |
|---|---|---|---|
| `kv_a_proj_with_mqa/L11` | `constraint_words` | `constraint_words`, `ppl_short_1` | `constraint_words`, `ppl_short_1` |
| `shared_up_proj/L3` | `short_reasoning` | `constraint_words` | `constraint_words` |

Both targets therefore cross at least one observed token-selection boundary at every tested n. This is not a monotonic precision threshold: each n is an independent categorical observation.

This diagnostic isolates the failed n5/n6/n7 cells. It does not claim an individual-target n4 threshold because isolated n4 cells were not run; the frozen full matrix separately establishes only that the combined n4 cell passes its complete policy.

## Combined-result attribution

| n | Prompt | Isolated causal result | Combined behavior |
|---:|---|---|---|
| 5 | `constraint_words` | KV alone crosses at token 2; shared alone matches reference | Combined crosses at token 2, then follows a distinct continuation |
| 5 | `short_reasoning` | shared alone crosses at token 0; KV alone matches reference | Combined crosses at token 0, then follows a distinct continuation |
| 6 | `constraint_words` | both targets independently cross at token 2 | Combined output equals the shared-only output |
| 6 | `ppl_short_1` | KV alone crosses at token 1; shared alone matches reference | Combined output equals the KV-only output |
| 7 | `constraint_words` | both targets independently cross at token 2 | Combined output equals the shared-only output |
| 7 | `ppl_short_1` | KV alone crosses at token 1; shared alone matches reference | The combined pair returns to the reference output; the KV-only change is cancelled |

The direct cause is therefore precision-induced representation drift in **both** promoted targets, amplified or cancelled by their nonlinear composition near greedy token-selection boundaries. It is not random execution instability: the experiment is finite and bitwise deterministic within each repeated request.

## Evidence

- runner SHA-256: `749abcc325bc4d10f41ade3be242b8de3b7119fd3795ea074a3f35283f0e22d3`
- plan SHA-256: `edb06370dbdf0ae6f1f61cb4ae7f82d320a8507a3504b2f118eb576455e8427c`
- raw result claimed hash: `65a79a88fe9a89e7f3d6cdd773323561407d8145f011dc12394f10e210b15fb4`
- raw result file SHA-256: `0bdcca94681f737155d2becfe6006184a96f8e01e51858653c1181efe025a348`
- boundary analysis claimed hash: `e76d79a44fff9c24c21b385f85e8a6dac28ce96802669b2551e42d3f1f0b2ff9`
- boundary analysis file SHA-256: `293e76eb7329e41f12944546ac5dc9466dd2782037d119cabd542fbf1a40beb4`
- independent audit claimed hash: `30e7dd91f58fb50a6af508e21f166bad09f22fca21c732bdda00c6dc7212c9c5`
- independent audit file SHA-256: `0439142bbab1546c8b0340f04e2463ea863ad68a0e1eced33dac56cd2d6ce17d`
- audit checks: 225 PASS, 0 FAIL

## Decision boundary

The original P9 verdict remains `P9_QUALITY_FAIL`: n4 is the only eligible isolated candidate, while n5, n6, and n7 remain rejected by the frozen complete-matrix policy. This ablation explains the failure; it does not turn the failed matrix into a production approval. `production_write_allowed=false` and `production_release_authorized=false` remain fixed.
