# OML–ACS C1 source-only OOF weighting protocol (2026-10-21)

**Classification:** development-only candidate protocol, written before this C1 weighted experiment; *not* a frozen final preregistration. The target dev split was already used in previous B experiments. No claim of blind evaluation.

We have three revision-pinned BF16 model families (Qwen1.5B, Mistral7B, Phi4-mini), shared SHA-pinned 300 anchor prompts and shared 600 real SST-2/CARER **train/validation** prompts and labels. Data, activation tensors, and model-specific anchor projectors stay private on Alienware WSL. This C1 test must not change historical B, V6 outcomes or production.

## Source-only training algorithm
1. For each axis, take **100 positive +100 negative** training prompts, align rows by private SHA across the three models. Independently make the same **five stratified folds** for each axis, with 20 of each class per fold, fixed seed 42 (axis-specific deterministic offset).
2. For each source model M and fold k, build a signed native positive-minus-negative behavior axis using only the other four folds. Project this via M's fixed **300-anchor projector**; normalize only for scalar classification. Form point projections `Ahat_M @ normalize(h_M-mu_M)` for training and OOF samples.
3. Obtain the classification cutoff as the midpoint of **only the training subset's** two mean projected scores. Predict labels on the left-out fold. Record 200 **out-of-fold binary errors** on the same labeled questions for each source model. Never fit model weights using target-model train/dev, and never use any dev labels for fold construction, sign selection, cutoff selection or source calibration.
4. Across the two allowed sources for a target, let E be the aligned 200×2 binary OOF error matrix. Estimate simplex nonnegative weights by minimizing quadratic `w^T mean(E^T E) w + ridge * ||w||²` using existing `weights_from_source_errors`, with **ridge 1e-6** and 2000 iteration cap. Compare `dependence=0` (diagonal reliability only) and `dependence=1` (correlated/co-failure errors). Baselines: equal 1/2 average, each source alone, source-OOF-accuracy best single. Keep target-dev oracle only as explicitly *post-hoc upper comparison*.
5. Combine full-training source canonical axes as a weighted **raw** Eq.4 average, then normalize the result once only for target detection scoring. No target-dev-dependent polarity flip.

## Evaluation and threshold
- For each of 3 held-out model families ×2 behavior axes, use **50 positive +50 negative target validation** examples, in exactly the same private row/SHA order; calculate AUROC and paired stratified 999-bootstrap CI for correlated−uniform and correlated−reliability-only.
- **Exploratory primary criterion** `AUROC_correlated - AUROC_uniform >=0.02` per target/axis, interpreted descriptively and with full unrounded values and uncertainty. Do not cherry-pick successful target/axes or change method after seeing scores. No final hypothesis acceptance based on this dev reused split.
- Save OOF fold-assignment digest, paired 200-row source OOF error summaries and correlations, source weights, per-condition target AUROC, comparisons and confidence intervals; store raw row identities privately and only publish aggregate receipts.

**Excluded:** output mixture A, C2 combined output/activation, real BEGLIN 8/4-bit Q, generation steering of C1, final IID/OOD, already consumed 24 V6 records. All remain separately unproven.
