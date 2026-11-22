# JE01 follow-up: native role sensitivity verification

**Handoff date:** 2026-11-23 KST. **Decision:** research only; AF4 retained; no allocation adoption.

The original Gate JVP now has an independent raw-file audit **PASS**. The new Gate/Up/Down Q4 native build **PASSed**, and its 14-cell diagnostic is **RUNNING / numerical verdict pending** under EOE job `job_3b09545af2076ea404721f268b248754`. Build success does not establish derivative agreement or improved model likelihood.

- Research PR: [beglin #94](https://github.com/popixoxipop-collab/beglin/pull/94), open Draft, unmerged when rechecked; observed HEAD `c1541582bd54e1fc7ee9e60951026142e5e81fd3`.
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

The live attempt's sealed plan canonical SHA is `12ae8c5eb7a3d6409eea2aa79337de9da9c56453bac33ec084c4e4a64aec8a14`.

The planned sequence is OFF once; Gate JVP once using its existing fixed-FD reference; then Up and Down each with alpha-zero, ±0.05, ±0.025 fixed-FD cells and one alpha-zero JVP. All zero/JVP primals must match original gold bytes, and every fixed route must match the baseline route bytes. For each direction, both FD convergence and JVP-versus-fine-FD agreement must pass `RMS(error) <= 0.0001 + 0.05 * RMS(fine FD)` across all 656 tokens.

## Source/build pins

[JE01_THREE_ROLE_NATIVE_JVP_BUILD_20261123.json](JE01_THREE_ROLE_NATIVE_JVP_BUILD_20261123.json) records the successful native build. Its `new_role_runtime_validation: NOT_RUN` is the immutable state at build time; the subsequent diagnostic job is running.

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

Local verification passed 8/8 targeted tests: three patch tests, including compilation of the exact C++ selector guard for 48 allowed and 23 rejected configurations; and five runner tests covering scope expansion, malformed JVP records, finite-difference alignment, append-only receipts, and resume source/plan/role identity. Independent code review found no blocking defect. These checks do not substitute for the running native numerical diagnostic.

## Reproduction and resume

The EOE research root is `/Volumes/D50/TailnetCommander/work/je01_native_role_jvp_followup_20261123_v1`. Preserve its parent source/data/adapter evidence roots. The following command reproduces validation of the current attempt and resumes only hash-verified completed cells; do not launch it while the recorded job is running. Obtain fresh resource admission before execution.

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

## Result insertion point — pending

Append the final job state/exit, sealed plan SHA, `JE01_THREE_ROLE_JVP_DIAGNOSTIC_attempt1.json` canonical/file hashes, all three FD/JVP verdicts, original-AF4 post-run SHA, and an independent raw-output audit here after completion. At this handoff, none of the new three-role numerical verdicts is final.

Even if all three directions pass, the optional 3×3 observed-label directional Gram is exploratory and conditional on the fixed hard route and previously seen TRAIN16. It is not the full parameter Fisher, a Hessian, a 48-direction result, evidence of natural finite-step routing stability, or an allocation policy. TRAIN240 fitting and untouched validation128/OOD128 remain subsequent gated work. PR #94 remains Draft; no quality improvement or production adoption is authorized by these derivative diagnostics alone.
