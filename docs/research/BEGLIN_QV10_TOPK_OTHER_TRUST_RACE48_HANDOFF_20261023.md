# BEGLIN Qv10 — Top2 + OTHER grouped output trust allocation, RACE48 native handoff

**Research date:** 2026-10-23 KST
**Branch:** `research/qv10-topk-other-trust-20261023` (isolated; no `main` merge)
**Model:** pinned original `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
**Final scientific decision:** SOURCE_BF16_PASS / SOURCE_LOCAL_1SWAP_PASS / NATIVE_RACE_3ARM_PASS / INDEPENDENT_AUDIT_PASS / GENERALIZATION_NOT_PROVEN / AUTO_PROMOTION_OFF.

## Motivation and novel experimental contract

Qv7 Llama, Qv8 joint signed loss, and Qv9 four-choice conditional Gauss–Newton did not establish general advantage over the complete Q4 baseline. Qv10 tests whether a less truncated source-output proxy and a precommitted trust-region penalty can better predict native-model quality.

At every one of the 24 pinned source cases, the **original unmodified BF16 Qwen** next-token vocabulary is partitioned into seven output bins: four exact A/B/C/D letter IDs, the two highest-probability other vocabulary tokens, and the exact probability mass of all remaining vocabulary ("OTHER"). The BF16 `logsumexp` of OTHER and its original BF16 gradient are computed without truncating the source output vocabulary. These are combined with the existing Qv9 A/B/C−D gradients, giving six signed relative-logit projections per each of the **196** model projection tensors. Tensor residuals are from the real BEGLIN original C Q4G64 and qNg64-n8 encoders.

**Important mathematical limitation:** the original full-vocabulary *probability mass* is preserved exactly among the seven bins, but within-OTHER Fisher curvature is lost. The resulting grouped KL is exact only for **linearly predicted seven-bin logits**; it is not actual quantized-model KL or the full vocabulary Hessian.

The precommitted source-only objective was:
- Max of even/odd 12-source-case mean grouped linearized KL plus relative-logit-shift penalty, threshold 2 and coefficient 0.05.
- Plus 0.10 times mean isolated per-tensor grouped Gauss–Newton sum, divided by 196.
- Strictly exactly 28 n8 + 168 Q4 projections, original Apple NEON CPU memory/serial/stage proxy budgets. Six deterministic 1-exchange starts; no global binary optimum claim.

## Source calibration, independent diagnostic, and source-bound local optimality

The original Qv9 ARC-Challenge train24 and diagnostic24 were reused with exact prior SHA; these **are not pristine across the entire research history**, though train and diagnostic are disjoint within this experiment. Source calibration and RACE evaluation were not mixed.

Source 24 original BF16 grouped probability mass (mean):
- Four letter bins: **99.8704%**.
- Two leading non-letter tokens: **0.1092%**.
- OTHER all remaining vocabulary: **0.0204%**.

Independent diagnostic 24:
- Four letter bins: **99.9102%**.
- Two leading non-letter tokens: approximately **0.080%**.
- OTHER: approximately **0.010%**.

Source training/diagnostic scores below are **model-gradient proxies, NOT native-model next-token NLL**.

| Mask | Source train grouped KL + trust + ridge ↓ | Independent source diagnostic objective ↓ | Diagnostic grouped KL alone ↓ |
|---|---:|---:|---:|
| All Q4 | 343.0381 | 650.0376 | 16.3656 |
| Qv6 Fisher64 | 146.2318 | 42.9015 | **5.8591** |
| **Qv10 Top2+OTHER** | **0.3723** | **32.2145** | 15.4890 |

The Qv10 diagnostic composite is lower than both baselines, but its diagnostic **KL alone is far higher than Fisher64**. This is central negative evidence about proxy-to-real-loss generalization. On all 24 independent source diagnostic questions Qv10 also has at least one relative-logit shift magnitude greater than 2; the linearized approximation can be unreliable.

Selected source policy mask SHA-256: `5801452577e1957fc635d1e295ae7cde2d8a50c294b04d9a65e8e48dc7b32196`.
Original Apple CPU NEON budget proxies:
- Selected projection bytes: **757,039,104 B** (Q4 baseline **736,985,088 B**).
- Serial single-token estimate: **83.1024 ms**.
- Isolated async-stage cost proxy: **49.8765 ms**.
- These are not sustained throughput, real p95 serving latency, or live GPU claims.

Independent source-only search audit checked **4,704 possible one-tensor exchanges**: 1,551 meet cost constraints, zero have an improving objective. This certifies a feasible **one-exchange local minimum only** (best feasible objective delta +0.023118), not global optimality.

## Prelocked evaluation and real Apple CPU BEGLIN results

External development evaluation: `ehovy/race@2fec9fd81f1dc971569a9b729c43f2f0e6436637`, RACE-middle validation, **48 distinct original articles**, each full passage/question/all options untruncated, with exact single-token A/B/C/D answer labels. Selection and all original token SHA were locked before mask or native scoring.

The three BEGLIN arms reused the **same original Qv5 audited real C Qwen 28-layer native binaries**, original BF16-sourced weights and 48 frozen RACE input windows. Q4 and two mixed n8 masks were never trained or adjusted on RACE outcomes. Native four-choice accuracy uses argmax among A/B/C/D; the native correct-label NLL uses the full model vocabulary normalization.

| Arm | Correct / 48 | Correct-label full-vocab mean NLL ↓ | Q4 NLL difference |
|---|---:|---:|---:|
| Original unmodified HF BF16 / MPS (different arithmetic path) | **42** | **0.821231** | separate model arithmetic |
| Complete BEGLIN Q4 | 32 | 1.183725 | reference |
| **Previous Qv6 Fisher64** | **37** | **0.938313** | −0.245412 |
| New Qv10 grouped KL+trust | 35 | 1.036851 | −0.146875 |

Qv10 improved observed RACE correctness and NLL relative to Q4 but did **not** outperform the pre-existing Fisher64 policy.

Same 48-article paired exploratory comparison, seed 20261023 and 5,000 bootstrap resamples:

| Qv10 relative to | Mean NLL difference | Paired bootstrap 95% percentile interval | Correct-count delta | McNemar two-sided exact p |
|---|---:|---:|---:|---:|
| All Q4 | −0.146875 | [−0.451386, +0.144367] | +3 | 0.453125 |
| Fisher64 | +0.098538 | [−0.119535, +0.348491] | −2 | 0.625 |

Both intervals include zero. These 48 public development questions cannot prove broad model/data generalization. BF16/MPS and native C use different numerical paths and cannot be treated as exact arithmetic counterparts.

## Independence and security evidence

- Qv10 source-policy public SHA: `ae35b59d98d2b08ded60f4af10b2485ecf4c1c176eb7f113a810b33f51259ece`.
- Source/diagnostic separate original BF16 independent audit SHA: `8b4cce9494bd2a8d4129cdfade431ad00954296465801a72a45a54e9d29c4679`.
- Source feasible one-exchange independent audit SHA: `e51a07c2ca02f9e1ed74207ba8b249a2ea54764867df647e1a3446820f533d93`.
- Actual native **144** original C process/question score independent audit SHA: `6ca7c75d71cbce09d61f251bba33146255196657a4d5e007738ae55666463207`.
- Original BF16 external MPS reference public SHA: `580592f319059f18e58f51c9d8088e63d302f19e1cad6f494744d7d787c714c2`.
- Native auditor independently verified full passage/token SHA, exact A/B/C/D labels, 48 windows each arm, original C binary SHA, 28 n8 role count, totals/means and paired statistics.
- EOE public-contract/negative trace tests: **28/28 PASS**.
- Source model, raw gradients, individual question tokens and native per-question logs stay private under `lab_local/`. Only source, masks, aggregate public evidence and handoff are suitable for GitHub.

## Gates and next step

Do **not** merge to `main`, change the immutable Tailnet operator release, auto-promote Qv10, enable per-request dynamic precision, or assert Apple Metal/GPU advantages. The observed source calibration gain is much larger than the native quality effect and is not a proof of generalized precision allocation.

Next useful experiments: (a) more independent source calibrations and untouched evaluation sets across **both Qwen and Llama**, (b) model-output Jacobian curvature that does not collapse all OTHER tokens, (c) trust-region diagnostics on actual quantized fullmodel output, and (d) low-load sustained Apple CPU p50/p95/tokens-per-second and memory. Preserve all earlier negative Qv5–Qv9 findings.

**Final verdict: IMPLEMENTED_AND_AUDITED / RACE_Q4_BETTER_THAN_NEW_Qv10_NO / RACE_FISHER64_BEATS_Qv10 / GENERALIZATION_NOT_PROVEN / AUTOMATIC_PRECISION_PROMOTION_OFF.**
