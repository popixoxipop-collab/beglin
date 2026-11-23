# JE01 — true original27 routed FFN activation, first real sensitivity pass (2026-11-22 KST)

## What is actually measured

On original DeepSeek-V2-Lite BEGLIN 27-layer native Apple Metal with the **unchanged AF4 checkpoint** and SHA-matched real WikiText TRAIN16 input, previous isolated native scratch tracing recorded **45 true Top-6 routed `h2` FFN inputs**, each 2048 float32 elements plus native router probabilities. Source trace binary SHA-256 `238642649f68d111df3b60f60083a47d3a6074ada881ccb36f813dd606f9a5b4`, original full-gold output `b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c`, NLL **2.291799702354307**, same as original baseline.

On EOE D50, full original AF4 9.8GB and four source Chat BF16 shard SHA checks passed, then real native group64 BEGLIN Q4/Q5 encoder/decoder were applied to the original 8 Experts' **gate_proj** and **up_proj** roles (16 roles in total). Every resulting matrix was multiplied by the **actual recorded original-model h2 input** (no synthetic activation); output squared difference from true source BF16 was weighted by that Expert's actual Top-6 router probability and normalized by source output energy.

| 16 real gate/up roles, 45 actual routed training observations | Mean-of-role activation-output relative RMSE |
| --- | ---: |
| Original affine AF4 | **0.06549721022313063** |
| Symmetric native Q4g64 | **0.10883960251207181** |
| Symmetric native Q5g64 | **0.05096151463578087** |

All 16/16 gate/up roles have Q5's relative output-MSE below AF4, and all have Q4's above AF4. This is **measured true native activation-output fidelity**; a distinct observable from unweighted matrix coefficient-space RMSE. Real CPU original BF16/AF4/Q4/Q5 GEMV, original model trained h2 values. EOE job `job_0bce0a10f72d2129c0ebfc04bf534614` SUCCEEDED, signed complete result receipt **`d55f4b506076a3a30f73c582ede270c38d41ac12b06190e959ac5a7383157a89`**.

Source evidence:
- [Signed 45 routed h2 and original27 16-role activation-weighted GEMV results](https://github.com/popixoxipop-collab/beglin/blob/research/je01-moe-bayesian-bit-allocation-20261117/docs/research/JE01_ACTUAL_ORIGINAL27_TRUE_H2_16GATEUP_ACTIVATION_WEIGHTED_GEMV_20261122.json).
- [Independent canonical receipt, gate/up shape/source checks and explicit non-Fisher assertions CI 44963732279 — PASS](https://github.com/popixoxipop-collab/beglin/actions/runs/44963732279).

### Scope and limitations

This test uses only **16 TRAIN prompts / 45 selected routed observations**, with repeated pilot exposure. It does **not** observe `down_proj`'s true SwiGLU input, all 256 training examples, token NLL perturbation or validation. It does not compute loss gradients, Fisher, Hessian, nor any automatic-differentiation backward pass. In particular, `E[p ||(W_b-W_{BF16}) h_2||²]` is a **true-input-conditioned FFN output perturbation proxy**, not `E[g_{gold}g_{gold}^T]` for the model's next-gold-token log-likelihood. These cannot be interchanged.

## Evidence-backed next 256-train measurement

Original 512 frozen DeepSeek tokenizer IDs, 256 calibration prompt files, and complete original content SHA receipt `2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34` were retrieved from immutable GitHub Actions [source run 44065613386](https://github.com/popixoxipop-collab/beglin/actions/runs/44065613386). The official relocation utility reverified all 512 prompt bytes (NOT labels) and issued new EOE D50 manifests, signed receipt `a97d8e6acf3111c8c60e9210908311c9c4255f5429d5cb506445aab75f82b713`. This **does not use** validation128/OOD128 gold labels.

A new disposable original27 Apple Metal build has compiled with actual 2048d h2 plus 1408d original SwiGLU captures for all selected Top-6 Experts at the last token of each of the **256 training prompts**, R16 × 16 chunks; original C/C++ and operating checkpoint unchanged. Compiled research binary SHA `77fd53111cf8316c9d9186f21ace65bccaf3c6fb1aba54c43c698076d5dc914d`, [native build CI 44963171023 PASS](https://github.com/popixoxipop-collab/beglin/actions/runs/44963171023). Each chunk is checked against original file and prompt IDs, each run has memory-pressure hard stops and fresh binary receipt. First chunk's actual h2 is required to numerically match the preceding gold16 h2 trace before the remaining 240 previously unselected calibration prompts run; overall frozen source-256 selected Expert counts must match the previously attested original 64-way router totals of **682**.

Executable driver: `tools/je01_eoe_original27_real256_h2_swiglu_runner.py`; planned actual `gate/up` inputs are 2048d h2, `down` input is 1408d original AF4 SwiGLU. The downstream true pretrained AF4, native Q4 and Q5 GEMV analysis `tools/je01_eoe_actual_cal240_h2_sw_native_activation_dp.py` strictly excludes the first **16** used in earlier pilots from policy selection and uses only 240 remaining training prompts, solving exact **43,253,760B** 24-role DP under both separately reported raw output SSE and relative role MSE objectives.

**Execution status at this report:** durable Tailnet stage request `je01-eoe-original27-cal256-real-hidden-h2-sw-ffn-source-stage-20261122-v1`, and full256 actual native GPU capture request `je01-eoe-actual-original27-cal256-h2-downsw-all-682-routes-first16control-20261122-v1`, both **WAITING / NO_CAPACITY** because EOE reserved CPU slots were unavailable; their result outputs do NOT yet exist. No unsupported claim of complete256 activation, activation-conditional DP, Fisher or Hessian.

All operations are research-only D50, PR #94 Draft, no production/native release modification or promotion; user-owned Tailnet current/live release remains `g6-0.9.0-operator.900500` until independently checked again.
