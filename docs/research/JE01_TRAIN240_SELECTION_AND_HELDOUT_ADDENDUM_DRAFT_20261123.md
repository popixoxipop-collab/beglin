# JE01 TRAIN240 selection and heldout evaluation — addendum draft

**Date:** 2026-11-23. **Status: DRAFT FOR REVIEW; NOT SEALED OR EXECUTABLE AUTHORIZATION.**

This document proposes explicit statistical and selection rules for the 48-direction TRAIN240 continuation. It does not change any previously sealed plan, source, threshold, runner, audit or result. It records no TRAIN240 derivative, selected allocation, validation result or OOD result. The full48 TRAIN16 execution can continue independently of this review.

The controlling source review used immutable PR94 commit `a0bac6efaa11182177e88c3f6561fabbebc2f758`: the [original JE01 preregistration][prereg], the [TRAIN240 activation decision and next native NLL gate][caldecision], and the [48-direction gradient next gate][nextgate]. The local [TRAIN240 collection plan][collection] supplies the recovered source chain and execution prerequisites. The original preregistration's final NLL, routing, latency, placebo and independent-review gates remain controlling; proposed operational details below must be reviewed and prospectively sealed before use.

## 1. Review conclusion and decision points

The collection plan's definitions `g = sum(d)/N`, `F = sum(outer(d,d))/N`, per-case sufficient statistics and exact 12/6/6 byte constraint are mathematically consistent with the newer gradient gate. The most consequential unresolved details are the meaning of training bootstrap/fold diagnostics, the full set of preregistered controls, and the exact heldout comparison rule. These must not be chosen after seeing their results.

| Before this boundary | Required written decision |
|---|---|
| First TRAIN240 derivative | Seal aggregation, eligibility, optimizer, folds, bootstrap and all implementation/source pins. |
| First finite-policy diagnostic | Freeze exactly one candidate and all control maps, physical bytes, source hashes and native evaluation protocol. |
| First validation/OOD content or score access | Complete the development gate; seal cohort handling, paired statistics, confidence convention, placebo rule and single-candidate multiplicity scope. |
| Any adoption proposal | Independent evidence review must verify every original final gate; no deployment is implied by this addendum. |

The open protocol mappings in section 9 are part of the draft review, not silently resolved amendments.

## 2. TRAIN240 matrix aggregation

Use exactly the original TRAIN rows 16..255: 240 source rows and **10,368 gold targets**. Rows 0..15 remain previously seen development diagnostics. Let `d_it` be the 48-vector of full-prefix derivatives of the gold target NLL at AF4, for target `t` in source case `i`.

For each case retain:

```
N_i = number of scored gold targets in case i
a_i = sum_t d_it
B_i = sum_t outer(d_it, d_it)
N   = sum_i N_i = 10368
g   = sum_i a_i / N
F   = sum_i B_i / N
```

All 48 columns use the **same denominator N**. Do not normalize by a role's routing count, the number of nonzero derivatives, chunk count, prefix-token count, two-pass counter, or the count of examples containing that role. Retain zero derivative entries and every aligned target. Do not mask a derivative because the role was not routed at the current gold position: earlier prefix computations can affect that target through the model state.

`B_i` is the sum of per-token outer products, not `outer(a_i,a_i)`. Combining chunks requires sums of sufficient statistics followed by division by total gold targets; averaging the 15 chunk means is incorrect when their target counts differ. Accumulate in float64 with stable sums and independently reproduce the final matrix from all aligned raw columns. Preserve original row, gold index, target ID, absolute position, direction, source and binary identities.

Check all declared coverage, finite values, symmetry, diagonal second moments and the predeclared PSD tolerance `lambda_min(F) >= -1e-8 * max(1,trace(F))`. Do not clip negative eigenvalues or tune the tolerance after a result. PSD is a numerical/algebraic check; it does not establish that all derivatives are correct.

For this implementation, describe F precisely as the **uncentered observed-label score second moment in the 48 fixed-route directions**. Its use as a quadratic risk surrogate is specified by the next gate. It is not an evaluated true NLL Hessian, exact generalized Gauss–Newton matrix, Bayesian posterior, or prediction guarantee at the finite quantized policy.

Count routed eligibility using distinct original TRAIN source cases, with one count per selected Expert slot per case over its complete prefix. Two passes and repeated token events do not create independent cases. Preserve the earlier minimum-32-observation rule; document any source-document dependence. If an eligibility exclusion makes exact 12/6/6 impossible, return HOLD without reducing the threshold or changing the frozen slots. This route-count check does not itself establish the earlier method's separate finite paired-NLL eligibility claim.

## 3. Deterministic selection and training diagnostics

Represent a candidate as binary `x[r,Q4]` and `x[r,Q5]`, with `x[r,Q4]+x[r,Q5] <= 1` for each of the 24 roles. Require six Q4 coordinates, six Q5 coordinates and twelve AF4 roles. The resulting selected role payload must be exactly **43,253,760 bytes**. The baseline remains all-AF4 and is a reference, not a feasible mixed candidate under the exact-count constraint.

Minimize the newer gate's fixed objective:

```
Q(x) = g^T x + 0.5 x^T F x
q_i(x) = a_i^T x + 0.5 x^T B_i x
Q(x) = sum_i q_i(x) / sum_i N_i
```

Retain all cross-role terms. Do not replace this objective with an additive role DP, tune a curvature multiplier, add an outcome-selected regularizer, or change the normalization. The original 2,498,640,144 assignment count applies when every role/tier is eligible; eligibility restrictions can reduce it.

Proposed deterministic heuristic details, to be fixed in source before TRAIN240 derivatives:

- Encode tiers as AF4=0, Q4=1, Q5=2 in the immutable role order. Use 64 distinct feasible starts, including the lexicographically first feasible map. Generate the other starts with pinned NumPy/PCG64 seed 20261123 and a fixed documented permutation/rejection procedure independent of derivative values.
- At each iteration consider every different-tier role swap and both cycles of every three-role set containing one role of each tier. Choose the largest decrease exceeding absolute tolerance `1e-12`; resolve ties by the lexicographic resulting 24-role tier tuple. Preserve all eligibility and count constraints.
- Stop when there is no accepted improvement or after 512 accepted moves per start. Record cap hits and local convergence. Pick the lowest objective among all completed starts using the same tie rule.
- Independently recompute the winner's objective both from `(g,F)` and from the per-case `q_i`. For an incremental implementation, the exact score change is `delta^T(g+Fx) + 0.5 delta^T F delta`; retain the quadratic term.

Report **deterministic heuristic; global optimum unproven**. Search starts, fold fits and bootstrap diagnostics are training analyses and cannot become alternative heldout candidates.

Proposed five-fold interpretation: freeze cluster IDs, hash rule, fold assignment and source pins before derivatives. For each fold, fit the identical fixed optimizer using only the other four folds and evaluate its proxy on the excluded fold. Pool numerator/denominator sums across the excluded folds. Do not score a full-data fitted map on those same rows and call it cross-validation. Do not select the best fold's map or tune the optimizer from fold results. The one final candidate is selected by the already fixed rule on all TRAIN240 data.

Proposed training bootstrap interpretation: **20,000 shared cluster draws with seed 20261123 for the one full-data-selected, fixed x**, using `sum_i w_i q_i(x) / sum_i w_i N_i`. This is a conditional, descriptive training diagnostic. It does not account for choosing x on the same data and cannot certify prospective gain. It does not rerun the optimizer 20,000 times. Selection stability, if reported, uses the five fixed fold fits above. A future optimizer-refit bootstrap would require its own prospective definition and compute budget.

Cluster by source document when TRAIN-only provenance supports that mapping. Otherwise use frozen source-row clusters, report the missing document grouping and call the intervals sentence-cluster conditional. No token-level resampling and no claim of proven document independence is permitted.

## 4. Freeze and development gate

Before any finite-policy run, preserve one immutable bundle containing:

- The candidate's exact role/tier map, source bank/dequantization identities, original model SHA, physical packed file SHA and independently audited 12/6/6 counts and 43,253,760 bytes.
- The baseline and both controls in section 5, including their maps, seeds, hashes and physical packing evidence.
- Optimizer, aggregation and comparison source; immutable TRAIN/validation/OOD cohort and tokenizer identities; native binaries, backend/library identities, scoring construction, route definition and latency schedule.
- This protocol's final confidence convention, placebo rule and declared number of eligible heldout candidate plans.

Run the one frozen candidate and AF4 on the already seen TRAIN16 using complete teacher-forced gold NLL and **natural, unforced routing**. Preserve paired token and case losses plus full-prefix hard routes. A fixed-route finite-step experiment may be reported separately but cannot satisfy the natural-route development gate.

The existing collection-plan development screen remains: HOLD on an integrity/kernel failure, nonfinite value, any candidate hard top-K mismatch, candidate p95 token-latency regression greater than 5%, or positive candidate-minus-AF4 token-weighted NLL. Do not replace a failing candidate with another optimizer start, fold result, placebo, heuristic or historical A/B/C policy in this sealed attempt. Passing TRAIN16 is a development check, not independent significance or adoption evidence.

## 5. Heldout arms and negative controls

The original preregistration P2 names baseline, selected plan, equal-byte random-permutation control and a constant-tier frequency heuristic. The newer gradient gate changes the selection objective; it does not expressly remove either control. Run all four frozen arms on both original heldout cohorts, keeping their results separate. [Original P2 comparison and gates][prereg]

| Arm | Prospective definition proposed here |
|---|---|
| AF4 baseline | Unchanged original native model and all 24 selected role payloads in original AF4. |
| One gradient candidate | The single full-TRAIN240 map from the frozen heuristic. No choice among heldout results. |
| Equal-byte permutation placebo | Exactly one PCG64-seeded permutation of the 24-role multiset containing 12 AF4, 6 Q4, 6 Q5, generated without derivative or heldout outcomes. Fix seed, library version, resulting map and bytes before finite-policy scoring. Do not redraw a weak placebo. |
| Constant-tier frequency heuristic | A proposed implementation assigns Q5 to all three roles of the two most frequently routed frozen Expert slots, Q4 to all three roles of the two least frequent slots, and AF4 to the remaining four slots. Rank by TRAIN240 distinct-case routing count, with frozen slot order breaking ties. This yields the same 12/6/6 counts and bytes without using NLL derivatives. |

The specific frequency rule and placebo generator above are **new operational proposals**, not claims that the old preregistration already specified them. Their exact implementations must be reviewed and frozen before evaluation. Controls are not eligible replacement candidates even if their heldout scores are better. Report their route changes and failures rather than dropping them or choosing another control.

## 6. Paired heldout collection

Evaluate the original **128 WikiText same-source validation** and **128 HellaSwag OOD** inputs only after the freeze and development gate. Do not use heldout prompts, endings, labels, baseline NLL or control outcomes for fitting, tuning, eligibility, policy choice or control construction. Existing split IDs/hashes may identify the sealed cohorts; this draft did not open their content.

For all arms, use identical original input/target identities, complete target scoring, tokenizer, backend, batch and memory settings. No truncation, shortened target, selected subset, dropped failure or replacement example is allowed. Preserve full-vocabulary gold NLL sums and target counts per original source case, with target and position alignment verified independently.

Compare the candidate's naturally selected hard top-K Expert IDs with baseline over the full matching prefix at every routed layer. Use the predeclared native hard-route identity rule; continuous routing-weight changes are not silently substituted for hard-ID flips. Do not force baseline routes during this gate. Preserve tie handling, route mismatches, nonfinite values and unsupported kernels.

Measure p95 token latency under matched hardware, load and native settings. Fix warm-up, arm order/blocking, repetition counts and which token events enter the latency distribution before results. Record concurrent load and memory evidence. Repeated timing runs must not create additional independent NLL observations, and the fastest run must not be selected after measurement. Freeze the evaluator and exact schedule as part of the source closure before heldout access; no new numerical latency threshold is introduced here.

## 7. Paired statistics and confidence conventions

For cohort C, let `L_i^A` be the sum of gold-token NLL for arm A and `N_i` the common target count. Define positive gain as improvement:

```
G_C(candidate) = sum_i (L_i^AF4 - L_i^candidate) / sum_i N_i
A_C(placebo)  = sum_i (L_i^placebo - L_i^candidate) / sum_i N_i
```

Averaging per-case mean NLL is a secondary descriptive statistic only; it cannot replace this primary token-weighted ratio after seeing an outcome.

Use the original **20,000 paired bootstrap repeats**. Resample source clusters with replacement, retaining all tokens in each sampled case and using the same draw weights for every arm/contrast within a cohort. Compute each replicate as a ratio of weighted loss sums to weighted target counts. Validation and OOD retain separate estimates and intervals.

Proposed confidence convention: a two-sided percentile 95% interval, with lower bound at the 2.5th percentile and upper bound at the 97.5th percentile, using an explicitly pinned quantile interpolation method. The original phrase “lower 95% bound” does not itself settle one-sided versus two-sided convention; this choice must be written into the prospective evaluation receipt rather than inferred after scores. A published earlier JE01 evaluator with an applicable immutable convention should be checked before adopting this proposal.

Freeze the clustering algorithm before heldout content access and assign groups before examining scores. Use verified document/source groups where available. Otherwise label row-cluster intervals conditional on that grouping and explicitly report the unverified cross-row independence; independent review must not certify stronger nominal population coverage from unavailable document membership. Report the number of source rows and resampling clusters separately.

## 8. Final decision and multiplicity

The original final thresholds are preserved:

| Gate | Required evidence |
|---|---|
| Identity/packing | Independently verified source, checkpoint, selected slots, all compared maps and exact packed bytes. |
| Validation quality | Candidate token-weighted mean NLL gain **at least 0.02**, and paired bootstrap lower 95% bound **strictly greater than 0**. |
| OOD quality | Candidate paired bootstrap lower 95% bound for gain **at least 0**, with the original nonregression requirement. |
| Routing/kernels | **Zero candidate hard top-K mismatches** against baseline; no nonfinite values or unsupported kernel path. |
| Latency | Candidate p95 token-latency regression **no greater than 5%** under hardware/load parity. |
| Placebo | The gain must not be explained by the equal-byte permutation control. Proposed operationalization: validation `A_C(placebo)` must have a paired lower 95% bound **greater than 0**; report the OOD contrast separately. |
| Frequency control | Complete and disclose the prespecified frequency arm on both cohorts; it is not an outcome-selected replacement. The original preregistration does not explicitly impose a candidate-versus-frequency significance threshold. |
| Review | Independent reviewer verifies all final gates before any adoption proposal or operating change. |

The explicit candidate-versus-placebo confidence rule is a prospective **proposal to operationalize** the original qualitative placebo requirement, not an already recorded result or a retroactive interpretation. If it is not accepted in review, a different concrete rule must be sealed before any heldout access.

This draft proposes **one eligible candidate plan**. The placebo and frequency arm are controls and cannot be selected for adoption from their heldout scores. Training starts, five fold maps and bootstrap variants are never scored as competing heldout candidates. Therefore no selection among multiple heldout plans is performed in this draft; the original familywise-correction requirement remains applicable if that scope changes.

Testing more than one eligible candidate would require a prospectively declared candidate family, selection rule and familywise correction before heldout access. Do not increment the family after seeing scores or treat a newly tuned replacement as an independent test on the same 128+128 inputs. Any subsequent research must preserve the failed result and follow a separately justified evaluation design. All required gates are conjunctive; marginal confidence intervals must not be described as simultaneous familywise intervals.

Any failed or unverified required gate retains AF4. Report numerical FAIL separately from missing-source/integrity HOLD, preserve all raw outputs, and do not claim prospective improvement from the TRAIN240 proxy or TRAIN16 diagnostic. A passing research comparison does not itself authorize production promotion.

## 9. Items to resolve before prospective sealing

1. **Earlier versus newer selection method.** The original preregistration used independent NIG effect estimates, an additive DP and a predicted-reduction pilot screen of at least 0.15 NLL. The newer next gate specifies the 48-direction quadratic surrogate and exact 12/6/6 constraint. This gradient protocol does not implement or claim to pass the older posterior or its finite paired-NLL eligibility screen. The authoritative protocol mapping, including the old 0.15 pilot screen, must be stated explicitly; this draft does not silently waive it or change the final 0.02 heldout threshold.
2. **Control implementation.** Review and fix the exact frequency ranking, placebo seed/generator, physical maps and bytes before scoring. Do not omit the frequency comparator merely because the abbreviated collection plan mentioned only placebo.
3. **Confidence and placebo decision.** Confirm the existing applicable evaluator convention or adopt the proposed percentile convention and placebo contrast prospectively. Preserve the 20,000-repeat requirement and original final thresholds.
4. **Clustering and latency implementation.** Seal deterministic grouping and the complete matched-latency schedule, report their limitations, and verify that no heldout result influenced these choices.

No old source or evidence was edited during this review. No remote job, derivative collection, policy pack, heldout access or deployment was performed to prepare this draft.

[prereg]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/docs/research/JE01_MOE_BAYESIAN_EXPERT_BIT_ALLOCATION_PREREG_20261117.md
[caldecision]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/docs/research/JE01_CAL240_REAL_FFN_ACTIVATION_DECISION_AND_NEXT_NLL_GATE_20261122.md
[nextgate]: https://github.com/popixoxipop-collab/beglin/blob/a0bac6efaa11182177e88c3f6561fabbebc2f758/docs/research/JE01_ACTUAL27_ABC_NLL_BAYESIAN_FISHER_NEXT_GATE_20261123.json
[collection]: JE01_48_DIRECTION_TRAIN240_PLAN_20261123.md
