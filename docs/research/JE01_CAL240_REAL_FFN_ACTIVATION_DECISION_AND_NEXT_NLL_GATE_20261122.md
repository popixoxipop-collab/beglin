# JE01 — 240 unseen TRAIN inputs: true FFN activation sensitivity, exact-byte decision, and next native NLL gate

Date: 2026-11-22 KST. Scope: BEGLIN DeepSeek-V2-Lite-Chat original 27-layer native Apple Metal, unchanged affine AF4 source. Research-only, **not** a deployment or a Fisher/Hessian result.

## Trace and original evidence

- The previous true original27 CAL256 run `job_d3d0c7a07095e03207cfe7fbc90df0e6` was SUCCEEDED/exit0; **256/256** fixed original WikiText TRAIN prompts and **682** actual routed records, each with native 2048d `h2`, native 1408d SwiGLU `down_proj` input and native original router probability.
- 16 original prior pilot prompts / 45 routed events are strict parity controls (native h2 and routing probability max abs difference **0**). **Exclude** all 16 from new precision policy fitting. Other **240 TRAIN prompts / 637 routed records** are the training optimization cohort, not heldout validation.
- Signed CAL256 activation receipt SHA: `b911aef00007b1f9b44214996353f93c7218fe51c82daf91966e450ebba460b6`. Original AF4 source SHA: `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4`.
- The separate 240-input real-pretrained code `tools/je01_eoe_actual_cal240_h2_sw_native_activation_dp.py` was executed by EOE D50 registered `job_1b8f08a4d48513f62b2d9b504d613bcb`, SUCCEEDED/exit0, with **all 24** gate/up/down roles. The real output objective uses original BF16 as a reference and native actual AF4, symmetric Q4, symmetric Q5; it weights each actual `W_b x - W_BF16 x` output squared error by its own routed probability.
- Completed experimental record claims signed canonical result receipt `35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb`. This identifies the EOE result. The full role-level ledger had **not yet** been independently reopened for this report; quoted optimizer outcomes below are verified from the existing job's stdout, **not** from an independent role-level replay.

## Actual output-error DP: two objectives yield opposite policy choices

Baseline selected 8 Experts × gate/up/down = 24 original AF4 role payloads: **43,253,760 bytes** (24 × 1,802,240). Native symmetric Q4 is 1,622,016B (−180,224B/role), native Q5 is 1,982,464B (+180,224B/role). A physically equal-budget alternative requires **number of Q4 roles = number of Q5 roles**.

Measured output metrics for each (Expert, role) use actual original 240 TRAIN prompt FFN activations:

- Raw router-weighted output `SSE`: output-level `sum_t p_{r,t} ||(W_b-W_BF16)x_{r,t}||_2^2`, summed across all 24 roles.
- Relative role `MSE`: each role's weighted `SSE` divided by that role's own original BF16 weighted output energy; then sum across all 24 roles.

| Strict exact-byte DP objective | Winner | Native Q4 count | Native Q5 count | Original AF4 count | Difference vs all-original-AF4 |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw output SSE total | Nonbaseline mixed allocation | 6 | 6 | 12 | **−1,607.2109554533818** raw SSE units |
| Sum of relative per-role MSE | Original all-AF4 | 0 | 0 | 24 | **0** |
| Sum of relative per-role MSE, best *nonbaseline* policy | Worse than original | 1 | 1 | 22 | **+0.0024733855809832266** summed relative-MSE units |

SSE decrease is **not** a relative percent improvement (no baseline denominator in this receipt excerpt) and **not** NLL improvement. Both objectives use the same actual observation source but different normalizations. A large-output-energy or heavily routed role dominates raw SSE; relative-MSE weights the 24 roles comparably. Neither is a genuine supervised loss gradient/Hessian/Fisher, and separately measured `gate/up/down` perturbations ignore the compositional error and cross-role interaction of the SwiGLU FFN.

In particular, no policy should be called the Bayesian posterior optimum or deployed solely because the SSE DP is negative.

## Independent audit and queue ownership

- Added independent stdlib-only signed-ledger and DP verifier: `tools/je01_cal240_real_dp_independent_audit_20261122.py`, which checks receipt SHA, source trace SHA, 8 Expert × 3 role dimensions/counts, `SSE / energy` relative MSE identity, moment NPZ digest, role assignments, all 43,253,760 byte totals and independently replays the exact DP.
- Added six synthetic adversarial unit tests: `tests/test_je01_cal240_real_dp_independent_audit_20261122.py`.
- GitHub Actions JE01 offline contract suite **161/161 PASS** at run `44992413401` (synthetic only; not remote raw output rehash).
- After discovering the successful previous real 24-role EOE job, redundant untouched request `rq_46993a5acba54e2abe50f4c59edea0fa` was **CANCELLED while WAITING**, backend job null. No other task cancelled.
- A *new, distinct audit-only* registered request `je01-eoe-cal240-real-signed-24role-independent-dp-audit-20261122-4bc7eae-v1`, queue `rq_d2b6c0a9eae6493b8b0fc250f2347e2d`, was submitted to independently reopen and replay the existing finished result. At submission it was **WAITING**, not complete. Do **not** mark its outcome PASS without a terminal receipt.

## Next scientific decision (frozen before any independent labels)

1. **First get the signed 24-role SSE-policy exact assignment** from the remote result ledger, execute the independent DP audit, then freeze that assignment and complete a physical 43,253,760B artifact with signed BF16/AF4/Q4/Q5 per-role provenance. No use of WikiText validation128 or HellaSwag OOD128 labels during choice.
2. Use existing original native 27-layer OFF mode to require exact original gold output SHA parity. Run the **single frozen SSE policy ON**, and a same-tier-count shuffled/permuted role assignment as a placebo, under equal 43,253,760B. The earlier 16 pilot cases remain development-only because they have already been studied. Use remaining 240 TRAIN full-gold data for development diagnostics if needed, not an independent significance claim.
3. Before unsealing validation/OOD labels: preregister paired token-weighted NLL delta, prompt-cluster bootstrap confidence interval, latency/memory noninferiority and original-router parity/flip counts. Evaluate untouched WikiText validation128 and HellaSwag OOD128 only after candidate and acceptance threshold have been sealed and source SHA audited.
4. Whether SSE policy ultimately beats baseline or not, the *true* Bayesian experiment needs gold-token `grad log p(y|x)` or Hessian-vector products. With a nonlinear gated FFN, include cross-role effects rather than equating FFN output identity metric with the language-model loss curvature.

**Decision now:** original affine AF4 baseline remains active. Equal-byte raw SSE mixed policy is a *test candidate only*; relative-role-MSE policy is all-original AF4; new candidate 27-layer paired NLL, genuine Fisher/Hessian and untouched validation/OOD scores **not measured**. PR #94 remains Draft/unmerged; release current/live `g6-0.9.0-operator.900500` unchanged.
