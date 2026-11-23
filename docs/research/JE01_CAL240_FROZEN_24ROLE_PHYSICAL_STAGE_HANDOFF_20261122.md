# JE01 — CAL240 source-frozen 24-role physical-stage handoff (2026-11-22 KST)

## Measurement versus inference boundary

Real original27 Apple Metal recorded 256/256 frozen TRAIN prompt last-prompt activations (682 actually routed selected Expert events). The already-inspected first16/45 are **parity controls only**. For policy fitting, the remaining **240 TRAIN prompts / 637 routed records** were measured at 2048-d `h2` (gate/up) and 1408-d original SwiGLU output (down), for all eight frozen Experts × three FFN roles.

Signed, independently replayed [full 24-role physical-source reconstruction ledger](JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json): canonical SHA `35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb`, file SHA `479c8548f74de0a8efab51547559be7db5fcd0e34e3e2722192c67c10c8f1dbd`.

- Absolute router-weighted individual FFN-role output SSE: original AF4 **7192.091944662442** → frozen mix **5584.880989209061**, relative reduction **22.34691891899631%** on *training reconstruction proxy*.
- Sum of role-normalized MSE: all original AF4 **0.134837209449249**; the SSE-chosen mix **0.1664768825277238** (**23.46509% worse**). Thus the objective determines the answer.
- The SSE decrease is concentrated in down: gate **+132.16441 worse**, up **+133.46406 worse**, down **−1872.83943 better**. L24/E26/down alone contributes SSE decrease **1115.1181**. It is not uniform Expert improvement.
- This is **not Fisher/Hessian, not actual original27 model NLL, not an independent heldout effect, and not a reason for deployment**.

## Frozen 43,253,760-byte role policy

| Expert | gate_proj | up_proj | down_proj |
|---|---|---|---|
| L12/E13 | Q4 symmetric | Q4 symmetric | original affine AF4 |
| L13/E22 | Q4 symmetric | Q4 symmetric | original affine AF4 |
| L13/E28 | Q4 symmetric | Q4 symmetric | original affine AF4 |
| L14/E63 | original affine AF4 | original affine AF4 | original affine AF4 |
| L20/E17 | original affine AF4 | original affine AF4 | Q5 symmetric |
| L21/E61 | original affine AF4 | original affine AF4 | Q5 symmetric |
| L24/E26 | Q5 symmetric | Q5 symmetric | Q5 symmetric |
| L24/E60 | original affine AF4 | original affine AF4 | Q5 symmetric |

The 24 roles each have 2,883,584 original coefficients; 12 AF4 ×1,802,240B + 6 Q4 ×1,622,016B + 6 Q5 ×1,982,464B = **43,253,760B**, exactly the eight-Expert original physical byte baseline. Original AF4 has real, potentially signed scale and a separate real bias. Native symmetric Q4 has nibble and scale and requires implicit −8×scale; symmetric Q5 has code bitplanes and scale. **Do not treat original AF4 and native symmetric Q4 as the same encoding.**

## Research-only implementation now in PR #94

- [Physical original-source bundle builder](../../tools/je01_eoe_cal240_frozen_sse_exactbyte_role_bundle_20261122.py). It pins the completed source ledger and original Chat BF16/AF4 hashes, re-quantizes only the 12 changed roles, copies actual original AF4 packed+scale+bias for 12 unchanged roles, checks 60 contiguous physical spans, writes binary/TSV/signed JSON exclusively under a new D50 research folder. It does **not** bind the model or run NLL.
- [Physical-bundle policy and provenance tests](../../tests/test_je01_cal240_frozen_sse_physical_role_bundle.py), four synthetic-only tests PASS in [Actions 44993280495](https://github.com/popixoxipop-collab/beglin/actions/runs/44993280495).
- [Source-only seven-Expert mixed-role MLX overlay stage](../../tools/je01_eoe_original384_cal240_sse_role_specific_overlay_stage.py). Existing scratch loader required each Expert's three roles to use one codec; the frozen SSE optimizer requires **role-level mixing**. New isolated source patch binds role-specific E1 original AF4/Q4/Q5 for seven changed Experts, preserves all-AF4 L14/E63 native E64, checks 43MB, 24 roles, 60 regions, exactly seven registered changed Expert overlays.
- [Six offline fail-closed scratch-stage tests](../../tests/test_je01_cal240_sse_role_specific_overlay_contract.py) PASS in [Actions 44993663749](https://github.com/popixoxipop-collab/beglin/actions/runs/44993663749). This does **not** mean actual native C++ compile, Metal E1/E64 parity, or NLL have passed.

## Actual remote jobs and blockers

1. Real CAL240 GEMV: `job_1b8f08a4d48513f62b2d9b504d613bcb` **SUCCEEDED / Exit0**; separate signed-ledger DP and moment SHA verification: `job_5d295246feb1cffe4926e11bfc093d70` **SUCCEEDED / Exit0**.
2. Frozen same-layer Expert interaction and real gate→SiLU→up→down FFN replay: resource key `je01-eoe-cal240-frozen-sse-joint-swiglu-fullffn-replay-20261122-v1` was last checked **WAITING / NO_CAPACITY**, no backend job. **Do not duplicate**.
3. Physical source bundle: initial `...physical43m-20261122-v1` job **FAILED before computation** at the Git-blob source preflight because the command escaped NUL incorrectly; no data/model modified. Corrected `je01-eoe-real-cal240-frozen-sse-rolebundle-12af4-6q4-6q5-physical43m-20261122-v2` is a single durable admitted request, last checked **WAITING / NO_CAPACITY**. Verify terminal status/physical SHA before claiming creation.

## Strict next acceptance gates

1. Receive **actual** frozen joint FFN and co-routed Expert output replay; respect original SwiGLU parity guard and 240-prompt-only sample scope. If regression, record as FAIL; never rescue by retuning on validation/OOD.
2. Receive actual 43MB physical bundle with complete source, role-span, full-file SHA receipts. Existing v1 preflight failure is not evidence the v2 bundle exists.
3. Stage the mixed-role overlay strictly in isolated scratch. **Compile and independently verify** signed Q4/Q5 decoding and original AF4 E1 groupwise bias semantics; demand exact OFF original27 output SHA parity and specific original-AF4 E1 versus E64 control before any ON claim.
4. With frozen physical policy, run paired 27-layer teacher-forced NLL against original AF4 and a matched same-tier-count permuted-policy placebo using TRAIN data. Do not use the initial 16 seen prompts as confirmatory evidence.
5. Freeze candidate, thresholds, SHA and paired prompt-cluster confidence procedure before accessing untouched validation128/OOD128. Require genuine full-model quality, memory and latency non-inferiority and no unacceptable route/Expert regressions. Empirical Fisher/Hessian remains separate and must be measured from actual loss derivatives, not renamed output SSE.

**Decision: original affine AF4 stays active; PR Draft/unmerged; no source weights, production runners or Tailnet live release changed. Latest previously verified Tailnet current/live: `g6-0.9.0-operator.900500`.**
