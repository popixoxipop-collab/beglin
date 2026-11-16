# BEGLIN Qv21 actual native failure → B11 no-promotion hold

**Read-only audit:** 2026-11-17 KST; EOE D50, original-C 96-case calibration receipts, 96 frozen input rows, Qv21 trained public/private policy SHA verified. No new native scoring was performed.

## Exact real-native source pins

| Artifact | SHA256 |
| --- | --- |
| Qv21 preexec frozen public | eaf4633703f7ebc4139cbe6c970e36076a51693afa24141054d39d9fcaa93e40 |
| Original native Qwen28 C | f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221 |
| Calibration96 inputs | 982455c845d4c46ff66811e3ddd38fa70e9ade1cbcbba5880919943c8656fe2f |
| Qv8 source public aggregate | 9a292c6a4df81f5f3bceae391ea55d6dbb1c5ffbaea42593599d0e1c327adba5 |
| random73 public aggregate | 583ec0384095ce331dda40c51093569df3fea9991f8fc46e973f490e0a95e5fc |
| Qv8 private scores | b7e44c602576299a7524e5bb6c13307a1692a74ce3e64a5d18b4795d759ad41a |
| random73 private scores | c4ff22f3719d01f9db807c23a3ec9c03b745aad806ee55ed1e2b3574ce659d40 |
| Qv21 frozen private policy | 003310c75b754172007787bcc75bc517c0b0847adc2aaea31a61c6ef829419d8 |
| Qv21 frozen public policy | b287a98c4377d2748877034e95b5f7c9a005a369ab45e2f78c8fc554713d9ef5 |

## Frozen preregistered Qv21 calibration result

| Metric | Value |
| --- | ---: |
| Qv8 gold-token NLL | 1.6790517345178098 |
| random73 gold-token NLL | 1.8080851244203597 |
| 5-fold crossfit dynamic NLL | 1.6910129988806037 |
| Crossfit gain vs best static | **-0.011961264362793905**, FAIL vs required >=+0.04 |
| Paired bootstrap lower95 for crossfit gain | **-0.15095505407587895**, FAIL vs >0 |
| Crossfit random73 choices | 18/96, PASS vs minimum 12 |
| Severe selected-event ratio | 3/18 = 16.67%, FAIL vs <=10% |
| Native source correct | 56/96 |
| Native random correct | 59/96 |
| Policy state | CALIBRATION_BEST_STATIC, static_fallback_index=0 |
| Dynamic admitted? | **False** |

## Exploratory post-hoc paired diagnostics on same cal96

These numbers are **training-data diagnostics**, not independent B11 test or OOD evidence. The original C per-question NLL vectors and input SHA were verified before recomputation.

| Indicator | Value |
| --- | ---: |
| Per-case lower NLL for random73 | 67/96 |
| Large random73 harm, delta >=0.75 | 18/96 |
| Short/long severe harm | 9/48 and 9/48 |
| Average random73 minus Qv8 NLL | +0.12903338990255003 |
| 20,000-replicate paired-bootstrap CI, deterministic LCG seed 20261117 (exploratory only) | [-0.24193348959381825, +0.519924853088886] |
| Three largest loss increases | +6.66964, +5.99730, +5.80971 |
| Per-case hindsight oracle minimum mean NLL | 1.2239026762 (not runtime-feasible) |
| Same-data fitted policy chooses random73 | 16 cases |
| Same-data fitted-policy NLL | 1.5678647748 (NOT heldout; cannot validate promotion) |
| Frozen input-only full-model severe-risk AUC | 0.75926 (in-sample only) |

Random73 reduces NLL on many items but its heavy tail makes average NLL worse. Observed static-pair bootstrap CI includes zero. Fitting to and testing on the same cal96 gives an optimistic apparent gain, while the preregistered out-of-fold policy fails and includes 3 severe losses. Do not alter the preregistered thresholds or claim cross-family generality.

## Independent frozen baseline-only evaluations

- ARC-Challenge train48 Qv8: correct 35/48; mean NLL 1.0617700096003102; public SHA b82e635d12f186b99748ed2e52801af7b5a1cbacd94a262fc993373900d58aaf.
- OpenBookQA train48 Qv8: correct 30/48; mean NLL 1.6054994686709565; public SHA 0e81646f8797bc597d44f81acd128d2d5c7364bbad244ab541b48dfe18588c6d.
- random73 static results were not present at the checked evaluation paths. Baseline-only results do NOT demonstrate dynamic selection quality.

## Binding decision

B11 requires an accepted real original-Qv21 calibration policy. The only observed frozen Qv21 policy has dynamic_admitted=False; thus B11's new 400-sample scoring and promotion are **NOT AUTHORIZED** under the existing protocol. Its Bayesian Student-t / Conformal code and 48 synthetic CI contracts remain a reusable draft method, not a live optimizer. No Qv20 sealed validation was opened.

The scientifically justified next study is independent, using runtime-visible Qv8 signals (for example output probability margin, entropy and quantization sensitivity) to predict rare high-loss precision-mask failures. It must freeze new data, source/feature SHA and objective before any new gold-token NLL evaluation; leave Qv21 frozen result and thresholds unchanged. Reuse neither B9/B10 image-model cutoff values nor calibration data as evidence of BEGLIN performance.
