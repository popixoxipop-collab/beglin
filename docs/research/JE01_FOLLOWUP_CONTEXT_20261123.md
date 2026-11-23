# JE01 follow-up: native role sensitivity verification

**Handoff date:** 2026-11-23 KST. **Decision:** research only; AF4 retained; no allocation adoption.

The original Gate JVP and the new **Gate/Up/Down Q4 full-prefix native diagnostic both PASS**, with an independent raw-file audit confirming all three new directions. The 14-cell native job and the independent audit completed successfully. This establishes derivative agreement with converged fixed-route finite differences on previously seen TRAIN16; improved model likelihood and allocation quality remain unmeasured by this diagnostic.

- Research PR: [beglin #94](https://github.com/popixoxipop-collab/beglin/pull/94), open Draft, unmerged; follow-up code checkpoint `fa36e03ea365010fef8d7e95f9aecc3e13118587`. The final evidence is identified by its separate receipt and file hashes below.
- Viewer: [JE01 native sensitivity](https://je01-native-sensitivity-20261123-abdb.sweet-eft-6723.chatgpt.site).
- Runtime was checked at `g6-0.9.0-operator.900520`, with current/live convergence. This work does not activate a release or modify production weights.

## Recovered context and decision

JE01 is the BEGLIN/DeepSeek native language-model expert-precision experiment. The current work follows the failed same-byte allocation candidates and tests actual token-NLL sensitivities. It is separate from the H3 image-model torus experiments.

| Evidence stage | Observed result | Permitted interpretation |
|---|---|---|
| Original full TRAIN256 | 11,024 gold tokens; NLL `2.46035653719724` | Baseline on the larger calibration set; not comparable directly with TRAIN16 NLL |
| TRAIN240 FFN objective | Linear SSE −22.35%; nonlinear FFN −22.71%; joint expert FFN −25.09% | Local reconstruction improved; whole-model NLL still required |
| Same-byte candidate A | 43,253,760 bytes; TRAIN16 NLL `2.294761262072885` versus AF4 `2.291799702354307` | ΔNLL `+0.002961559718578117`; no adoption |
| Follow-up candidates B/C | NLL B `2.2974258514952206`; C `2.2959651985430702` | Both worse than the same AF4 baseline; no adoption |
| Ordinary-route Gate/Up/Down FD | Coarse/fine derivative disagreement exceeded limits by approximately 10.38/10.21/10.53× | Numerical sensitivity estimates rejected |
| Fixed hard-route Gate FD | RMS disagreement `4.8163325945271946e-05`; limit `0.0008819161658322316` | One direction converged on previously seen TRAIN16 |
| Original Gate native JVP | RMS versus fine FD `4.374933887532656e-05`; cosine `0.9999960901377418` | One full-prefix fixed-route computational derivative validated |

Earlier evidence is retained in PR #94 under `docs/research/JE01_ORIGINAL_AF_REAL256_FULL_TARGET_INDEPENDENT_AUDIT_20261118.json`, `JE01_CAL240_JOINT_NONLINEAR_FFN_NATIVE_REPLAY_20261122.json`, and `JE01_REAL27_POSTNLL_A_B_C_656GOLD_INDEPENDENT_PAIRED_AUDIT_20261123.json`. Ordinary/fixed sensitivity audits are linked by the viewer. The current independent Gate audit below reopens the original native raw outputs, rather than trusting those status labels.

## Independently verified Gate result

[JE01_NATIVE_ONE_COLUMN_INDEPENDENT_AUDIT_20261123.json](JE01_NATIVE_ONE_COLUMN_INDEPENDENT_AUDIT_20261123.json) reports `PASS_INDEPENDENT_RAW_NATIVE_JVP_FIXED_FD` at 2026-11-23 08:45:56 KST. Its canonical receipt is `b7f20261c382f065ef5f6dc70ec04ed41450fa7ea495507a7014c385778e3cac`; file SHA-256 is `71c304b6b69b8a07e4911c336352d142c12acf86c6522cd054cd2c48744d009e`.

The audit rechecked the complete 9,814,343,680-byte original AF4 file, native source/build/adapter identity, input targets, all six fixed-FD cells, the new OFF control, and every JVP record. OFF and JVP gold/routes were byte-identical to the original baseline. It recomputed both FD convergence and JVP agreement across 656 gold tokens. This was a raw-output audit, not another model inference.

Full-prefix counters were 4,334 processed tokens, 32 case resets, 117,018 layer steps, and 656 scored gold tokens. The 32 resets are native execution counters, not 32 independent evaluation sentences: the diagnostic has 16 sentences.

## What this follow-up changes

The new patch generalizes the previous alpha-zero Gate injection to the original frozen bank's 24 roles and Q4/Q5 alternatives. The actual diagnostic runner deliberately admits only role indices 0/1/2: L12/E13 Gate/Up/Down Q4.

Gate and Up retain one 2,048-channel input and six 1,408-channel expert outputs. Down retains six distinct `[6,1,1408]` SwiGLU inputs and six 2,048-channel outputs. Its alternative inputs are not collapsed to one row. Each selected role uses the alpha-zero tangent `d_native + d_alpha * (alternative − native)`; all other roles retain their native primal and input tangent.

The original OFF implementation, guarded native adapters, hard-route replay, dynamic router scores/selected weights, complete hidden/K/V tangent propagation, and exact source hashes remain enforced. Unsupported reverse-mode and direct-vmap transforms remain rejected. No installed MLX package is modified.

**Pre-run plan:** the runner seals `PLAN_RECEIPT.json` before any native cell. It fixes the 14-cell order, three roles, Q4 tier, alpha grid, full-prefix counters, and comparison thresholds. Every new run and final result is bound to this plan. Reports are append-only; resume reopens and verifies raw files. A completed diagnostic cannot be retried to obtain a different numerical outcome. A new attempt requires an earlier execution/integrity failure tied to the same plan.

The completed attempt's sealed plan canonical SHA is `12ae8c5eb7a3d6409eea2aa79337de9da9c56453bac33ec084c4e4a64aec8a14`.

The executed sequence was OFF once; Gate JVP once using its existing fixed-FD reference; then Up and Down each with alpha-zero, ±0.05, ±0.025 fixed-FD cells and one alpha-zero JVP. All zero/JVP primals matched original gold bytes, and every fixed route matched the baseline route bytes. For each direction, both FD convergence and JVP-versus-fine-FD agreement passed the predeclared rule `RMS(error) <= 0.0001 + 0.05 * RMS(fine FD)` across all 656 tokens.

## Source/build pins

[JE01_THREE_ROLE_NATIVE_JVP_BUILD_20261123.json](JE01_THREE_ROLE_NATIVE_JVP_BUILD_20261123.json) records the successful native build. Its `new_role_runtime_validation: NOT_RUN` is the immutable state at build time. The subsequent diagnostic and independent audit completed with PASS in separate receipts; the historical build receipt was not rewritten.

| Object | SHA-256 / canonical receipt |
|---|---|
| New build receipt | `2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008` |
| New native binary, 1,278,360 bytes | `110b1978d97bcd7c6bdfd6d42142569e56e9dacffa4ff462dc0786b3128e49f4` |
| New role patch | `68c8cd6541657312bca6604dcbe74dcdc8156ff4da80de9c8fd672a1ad08d755` |
| New diagnostic runner | `9be8372fa3858577011eb5ab1d4ad654f746e151a95012aaf23e515fd8a7eb90` |
| Patched C++ | `6055ca3b261075fe5fcf76e5829f0fb20b8ecdd52599248f4b9b1032cfbeb49f` |
| Patched C | `5e2a6ea76c22c1ec0ff9f2fb6d093b7d3a471679adbe439ed1c4b898a56a52a6` |
| Guarded adapter | `9ae9d98b921334bae321e583908e71bdc9c08178cbf3ca5d7a1d4d4e552a09ec` |
| Guarded smoke receipt | `b0825ab7a048f7d95b155592760e10d73e3df2b112230adbf7b348f388ca1595` |
| Original AF4 weights | `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4` |

Local verification passed 8/8 targeted tests: three patch tests, including compilation of the exact C++ selector guard for 48 allowed and 23 rejected configurations; and five runner tests covering scope expansion, malformed JVP records, finite-difference alignment, append-only receipts, and resume source/plan/role identity. Independent code review found no blocking defect. The native numerical results and independent raw-output audit below provide the separate runtime evidence.

## Reproduction and resume

The EOE research root is `/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_followup_20261123_v1`. Preserve its parent source/data/adapter evidence roots. The following command revalidates attempt 1 from its hash-verified completed cells. With the complete evidence intact, resume reopens the existing outputs without rerunning inference. Obtain fresh resource admission before any execution that could launch native cells; never run concurrent copies against the same attempt.

```bash
je01_root=/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_followup_20261123_v1
/opt/homebrew/opt/python@3.11/bin/python3.11 "$je01_root/tools/je01_native_role_jvp_runner.py" diagnostic \
  --root "$je01_root" \
  --smoke-receipt /Volumes/D50/TailnetCommander/work/je01_native_jvp_smoke_20261123_v4/JE01_NATIVE_JVP_SMOKE_RECEIPT.json \
  --expected-smoke-receipt b0825ab7a048f7d95b155592760e10d73e3df2b112230adbf7b348f388ca1595 \
  --expected-adapter-sha 9ae9d98b921334bae321e583908e71bdc9c08178cbf3ca5d7a1d4d4e552a09ec \
  --patch "$je01_root/tools/je01_native_role_jvp_patch.py" \
  --expected-patch-sha 68c8cd6541657312bca6604dcbe74dcdc8156ff4da80de9c8fd672a1ad08d755 \
  --expected-patched-cpp-sha 6055ca3b261075fe5fcf76e5829f0fb20b8ecdd52599248f4b9b1032cfbeb49f \
  --expected-patched-c-sha 5e2a6ea76c22c1ec0ff9f2fb6d093b7d3a471679adbe439ed1c4b898a56a52a6 \
  --expected-build-receipt 2cbeb4dc1a7e1ab5642a4abb22395498c10dde43bee051eec3c303c7e53fa008 \
  --fixed-audit /Volumes/D50/TailnetCommander/work/je01_role_directional_fixed_20261123_v1/JE01_FIXED_ROUTE_GATE_INDEPENDENT_AUDIT.json \
  --attempt 1 --timeout-seconds 900 --resume
```

Use the same MLX 0.32.1 Python environment that built this receipt. For a genuinely new build, use a fresh `je01_native_role_jvp_*` root and positional `build`; omit runtime-only `--expected-build-receipt`, `--fixed-audit`, and `--resume`. A fresh binary must obtain its own build receipt and fresh OFF/native validation; the current receipt must never be copied as evidence for another build.

## Source closure required in the research commit

The new runner imports these five immutable legacy Python files; include all five with their exact build-receipt hashes: `tools/je01_native_one_column_jvp_runner.py`, `tools/je01_role_directional_native_runner.py`, `tools/je01_role_directional_fixed_native.py`, `tools/je01_role_directional_fixed_routes_patch.py`, and `tools/je01_role_directional_bank.py`.

Also retain the new `je01_native_role_jvp_runner.py` and `je01_native_role_jvp_patch.py`, the pinned `je01_mlx_native_jvp_adapters.h`, and the guarded smoke source trio `je01_mlx_native_jvp_smoke.py`, `je01_mlx_native_jvp_smoke.cpp`, and that same adapter header. The smoke validator is dynamically imported from the immutable smoke evidence root. Preserve the original one-column patch for provenance, both new test files, independent auditor source(s), and their evidence JSON. The pre-existing CAL240 ledger `docs/research/JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json` is required by bank verification. Its builder `tools/je01_eoe_cal240_frozen_sse_exactbyte_role_bundle_20261122.py` is required to reconstruct the bank, but is not invoked by this diagnostic.

This source closure does not make the experiment standalone: the SHA-bound native parent source/binaries, MLX runtime, original AF4 checkpoint, bank, frozen tokens, and prior raw receipts remain explicit EOE prerequisites. Do not substitute another checkpoint or rewrite guards to recreate missing artifacts.

## Completed three-direction result and independent audit

[JE01_THREE_ROLE_JVP_DIAGNOSTIC_20261123.json](JE01_THREE_ROLE_JVP_DIAGNOSTIC_20261123.json) records `THREE_ROLE_NATIVE_JVP_FIXED_FD_PASS`. The native job `job_3b09545af2076ea404721f268b248754` finished `SUCCEEDED`, exit 0, in 1,266.242 seconds. This is the preserved content of EOE `JE01_THREE_ROLE_JVP_DIAGNOSTIC_attempt1.json`.

All rows below use L12/E13, tier Q4, 16 previously seen sentences, and the same 656 gold-token targets. FD convergence compares central differences at `h = 0.05` and `h = 0.025`; native JVP agreement uses the finer difference. Values are rounded here; the linked JSON retains full precision.

| Direction | FD convergence RMS error | JVP–fine FD RMS error | Shared RMS acceptance limit | JVP–fine FD cosine | Verdict |
|---|---:|---:|---:|---:|---|
| Gate, role 0 | `4.81633259e-05` | `4.37493389e-05` | `8.81916166e-04` | `0.9999960901` | Both PASS |
| Up, role 1 | `4.85076422e-05` | `4.38810871e-05` | `8.32416498e-04` | `0.9999955134` | Both PASS |
| Down, role 2 | `4.70092776e-05` | `4.44129557e-05` | `9.63228112e-04` | `0.9999966957` | Both PASS |

[JE01_THREE_ROLE_INDEPENDENT_AUDIT_20261123.json](JE01_THREE_ROLE_INDEPENDENT_AUDIT_20261123.json) independently reports `INDEPENDENT_AUDIT_CONFIRMED_THREE_ROLE_PASS`, with both evidence integrity and the numerical verdict PASS. It verified **132 files**, reopened all **14 new native-cell outputs**, checked the external plan pin and prior Gate evidence, and recomputed the three direction comparisons and Gram from raw records. The audit performed no additional native inference. Its completion was observed at 2026-11-23 09:17:54 KST; job `job_9ef614c72a23fe297059e97c9a7e40af` finished `SUCCEEDED`, exit 0, in 4.600 seconds.

All three JVP executions satisfied the sealed full-prefix counters: 4,334 processed tokens, 32 resets, 117,018 layer steps, and 656 scored gold tokens. The audit checked zero/JVP primal byte identity, route byte identity, finite values, token/target alignment, native configuration logs, and source/build identity. The complete original AF4 file was rehashed after the run and remained `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4`.

The final evidence pins are:

- Diagnostic canonical receipt: `1448de4fdfb34082c7bd6976676d8a5404f078d649d730adbaf331fb233857df`; file SHA-256: `22f929b8e92eff027358fa7c2104044cda7967ab46503199793c42618b5ae2b5`.
- Independent audit canonical receipt: `5ce09188e65d5cfcc0f956447b2918fe9c80bf81f14e8486ba97864f7137b87b`; file SHA-256: `9d82c2d727d4d20eb799e9a5b7625474311c954aaf7694063793bfaf09f18fcc` (226,736 bytes).

## Exploratory observed-label Gram and remaining boundary

After all three directions passed, the runner formed `G[i,j] = mean_t(dNLL_t/dalpha_i * dNLL_t/dalpha_j)` over the same 656 observed labels. The independent auditor reproduced it. Coordinate order is Gate, Up, Down at L12/E13 Q4.

| G | Gate | Up | Down |
|---|---:|---:|---:|
| Gate | `2.4459528428e-04` | `-4.5962191321e-05` | `1.5396998603e-05` |
| Up | `-4.5962191321e-05` | `2.1456025280e-04` | `2.4116178817e-05` |
| Down | `1.5396998603e-05` | `2.4116178817e-05` | `2.9812013568e-04` |

The signed mean directional gradients are respectively `+0.0002523155735237539`, `−0.0004978168552929503`, and `+0.00017885948196584748`. They and the Gram were not used to select an allocation policy.

This 3×3 matrix is exploratory and conditional on the fixed hard route and previously seen TRAIN16. The measured scope covers three Q4 computational directions; general selector support for 24 roles × two tiers does not establish a 48-direction numerical result. Full-parameter Fisher and Hessian claims, natural finite-step routing stability, and allocation quality remain unsupported by this experiment. The report explicitly records `full48_empirical_fisher: null`, `train240_allocation: NOT_RUN`, `allocation_selected: false`, `independent_validation_ood_accessed: false`, and `production_modified: false`. TRAIN240 fitting and untouched validation128/OOD128 remain subsequent gated work. PR #94 remains Draft; AF4 is retained.
