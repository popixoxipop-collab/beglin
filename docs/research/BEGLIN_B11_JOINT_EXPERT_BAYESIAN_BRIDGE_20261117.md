# B11 × BEGLIN — Bayesian Student-t / Conformal veto-only bridge

**Status: B11 CODE/CI PASS BUT NO_PROMOTION; ORIGINAL Qv21 NATIVE CAL96×2 PASS, Qv21 DYNAMIC GATE FAIL; NO B11 NEW ORIGINAL-C SCORES / NO DEPLOYMENT.**
**Date:** 2026-11-17 KST.
**Source branch:** research/qv21-tailrisk-prefix-selector-preregister-20261025, commit 7e65e5f0926060bb5d5b9f4473108c9a0c75911b.

## What was actually learned from Joint-Expert Bayesian research

The source model is pretrained DiT-MoE-S/2-8E2A, an image diffusion model, NOT BEGLIN's native-C Qwen inference engine.

| Stage | Evidence | Interpretation |
| --- | --- | --- |
| B3 | Gated Joint MAP tied Independent MAP in all 100 heldout cases, MSE gain 0% | No demonstrated cross-expert improvement |
| B4 | Supported cross-expert covariance links 0/6 | Keep block-diagonal independent MAP |
| B5 | Independent posterior maps passed 9/9 unit tests and synthetic replay | Not a full-weight epistemic posterior |
| B9 | Heldout joint conformal coverage 152/160 (95%), Student-t NLL -0.40% | Source-model-only evidence |
| B10 | Oxford-IIIT Pet, two breeds, N=120: frozen raw cutoff 120/120, frozen normalized 119/120, Oxford recalibration 113/120 (FAIL), Student-t NLL 3.879794 -> 3.853776 (-0.67%) | Recalibrating cutoffs can tighten too much |

B9 execution ID: job_c340032a2d0ef50fd0806152b02e9867; frozen method SHA256: 24d6bfb13f0eabcee134b15d4397193138c5d441f5a8b24704bb112040b35f7d.
B10 execution ID: job_57784d4194909a6f9da88baba8a3a846; result SHA256: d6ef706681508b46de39b6a31e7f6b84d29395914ed300367ca9a49cec913701.
Neither modifies a model or Router. Do not generalize the B10 two-breed OOD result to 10 breeds or to language models.


## 2026-11-17 newly recovered native evidence — crucial negative result

After original publication of this protocol, the EOE D50 original-Qv21 policy was found and verified read-only (private policy SHA256 003310c75b754172007787bcc75bc517c0b0847adc2aaea31a61c6ef829419d8; public policy SHA256 b287a98c4377d2748877034e95b5f7c9a005a369ab45e2f78c8fc554713d9ef5).

- Actual Qv21 original C cal96: Qv8 NLL 1.6790517345 (56 correct), random73 NLL 1.8080851244 (59 correct)
- Official 5-fold crossfit dynamic NLL 1.6910129989; gain against best static **-0.0119613**, bootstrap 95% lower **-0.150955**
- Risk: 18 random73 selections with 3 large losses = **16.67%**; official limit <=10%
- Official frozen policy: **CALIBRATION_BEST_STATIC**, static source index 0, policy_dynamic_admitted=False
- Therefore B11's strict admitted-Qv21 prerequisite FAILS. **No B11 fit, calibration, 400-case original C execution, evaluation or deployment.**
- Full original-96 exploratory analysis, receipt hashes, independent static-only 48+48, and correct in-sample-versus-out-of-fold distinction: [BEGLIN_QV21_REAL_NATIVE_NEGATIVE_B11_HOLD_20261117.md](BEGLIN_QV21_REAL_NATIVE_NEGATIVE_B11_HOLD_20261117.md).

The pure B11 code now additionally verifies the hash-consistent Qv21 private/public frozen policy bytes, original native C and mask pins, all four official gate booleans, numeric thresholds and the frozen fallback index before allowing fit. Offline CI includes false-admission and tamper tests. This PR must remain Draft/NO_PROMOTION and should not be merged as an actual optimizer.

## BEGLIN-specific research question

Can a frozen, split-calibrated Bayesian upper bound on the **paired gold-token NLL difference** veto dangerous switches to random73 without losing the original Qv21 policy benefit?

Qv21's existing policy is FROZEN: 13 prefix-ID-only features, ridge mean-delta and logistic severe-harm probability, random73 selected only when predicted NLL delta < -0.20 AND predicted severe-harm probability < 0.15. Its specification SHA256 is 408c661e66259784480b18eefc15f75d3bc7d034124194baaf3a218637e2f2d0. This new experiment does **not** change Qv21's original pre-registration, model, masks, Router, weights, or deployment rules.

Qv8 and random73 are *candidate precision masks*, not DiT-MoE routed Experts. B11 transfers a technique for representing and testing uncertainty, NOT the trained B9/B10 posterior, cutoff or score. B11 does NOT estimate cross-expert covariance.

### Fixed method

1. New original-C paired observations for both frozen masks yield delta = random73 gold-token NLL minus Qv8 gold-token NLL. The existing frozen Qv21 model supplies predicted delta and predicted severe-harm probability from prefix token IDs ONLY.
2. Fit **independent per-stratum 2x2 Bayesian residual regressions** on a completely separate B11 fitting split (short prefix 18..84, long 85..208). Basis x=(1, log(prefix_len/85)), prior coefficients|variance ~ N(0,variance*I2), variance ~ IG(alpha0=2,beta0=1). The resulting input-conditioned predictive is Student-t, with mean x^T beta and scale sqrt(beta_post/alpha_post * (1+x^T covariance_post x)). This posterior covers two regression coefficients per stratum, **not** pretrained network weights.
3. On a distinct B11 calibration split, form one-sided normalized residuals (actual delta - predicted delta - input_specific_posterior_mean)/input_specific_predictive_scale. Select empirical sorted score at index ceil((N+1)*0.95), separately per length stratum. This finite-sample split-conformal upper bound has group-wise marginal coverage only under within-group exchangeability assumptions.
4. At inference, bound = predicted_delta + input_specific_posterior_mean + frozen_conformal_q * input_specific_predictive_scale. B11 can select random73 only if the original Qv21 rule already selected it AND bound < -0.20. Thus B11 may veto an unsafe switch but can never create a switch absent in the baseline.
5. Fail closed on SHA drift, nonfinite values, overlapping fit/calibration/sample IDs, uncalibrated groups or prediction records containing outcome labels. A separate sealed-holdout scorer may open the observed deltas; the runtime prediction API cannot.
6. Report BOTH pooled raw and per-stratum raw split-conformal upper bounds alongside the input-conditioned Student-t normalized bound. Both raw controls are implemented as veto-only arms in the offline module; native and OOD evaluations remain NOT_RUN.


**Degeneracy check:** With a *constant* posterior mean/scale inside each stratum, normalized conformal scores rescale back into the same raw residual quantile and Bayesian parameters cancel. B11 explicitly uses the two-parameter length-dependent Student-t predictive to make the posterior affect the conditional upper bound. Whether that conditionalization improves real calibration is NOT established; raw pooled and stratified conformal ablations remain mandatory.

Implementation: tools/b11_bayesian_veto.py.
Synthetic safety tests: tests/test_b11_bayesian_veto.py.
No network, filesystem I/O, model inference, or external ML libraries are required in the B11 module.

### B11 proposed new dataset, not yet frozen or scored

- Fit: 120 new ARC-Challenge train items, short60/long60.
- Calibration: 80 different ARC-Challenge train items, short40/long40.
- In-domain evaluation: 80 further ARC-Challenge train items, short40/long40.
- Cross-source OOD evaluation: 120 new OpenBookQA train items, short60/long60.
- All 400 IDs, source sample hashes, tokenizer, original Qwen28 C executable hash, Qv8 and random73 mask hashes, and both original-C arms must be fixed and independently audited **before** outcomes are opened.
- Exclude every Qv20 and Qv21 calibration and sealed evaluation item, including original Qv20 48-question heldout. No Qv21 test label may be used to fit or choose B11.
- If 400 properly stratified fresh items are unavailable, STOP rather than choosing replacement samples after observing labels.

### Promotion criteria proposed for the new study

ALL necessary: original Qv21 real native calibration 96x2 and its original admission gate PASS; B11 never chooses random73 when Qv21 does not; at least 95% empirical upper-bound coverage in each length stratum on both new in-domain and OOD tests; observed severe NLL-harm rate among selected cases <=10%; random73 used >=12 times; no paired-mean NLL regression versus original frozen Qv21; positive improvement versus best static with paired 20,000-bootstrap lower 95% bound >0; all IDs/SHA/receipts audited. If any fails: B11 NO_PROMOTION, retain Qv21 best-static fallback. Report selective coverage and abstention counts; a policy vetoing everything is NOT an optimization success.

NLL comparisons within BEGLIN must use identical original C, equivalent 765,493,248 projection packed bytes per mask, identical thread settings and the same questions. DiT-MoE Student-t NLL and Qwen gold-token NLL use different prediction spaces, so numeric gains are never directly compared.

## Live operation boundary

On 2026-11-17, Tailnet Commander current and live release were both g6-0.9.0-operator.900484 with converged=true. The EOE host refused the requested read-only command at the fleet admission gate: insufficient_free_cpu, with 0 allocatable CPUs after existing reservations and swap pressure around 95%. This is not evidence of any real B11 native inference. Do not override admission, cancel another project's work, edit the production serving process, or activate a Tailnet release for this experiment.

**Current status: Implementation prepared only; all real B11 test metrics NOT_RUN.**
