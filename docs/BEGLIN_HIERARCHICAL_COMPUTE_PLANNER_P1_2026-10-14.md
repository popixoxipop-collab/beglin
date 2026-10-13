# Beglin H-JEPA-inspired Hierarchical Compute Planner — P1 research handoff (2026-10-14)

## Verdict

**PROTOTYPE IMPLEMENTED, UNIT TESTS PASS, XOX ARCHIVED EVIDENCE VERIFIED; NO GENERAL SPEEDUP AND NO NATIVE MODEL QUALITY CLAIM.**

Our micro-expert research previously reached ~11% offline local FFN output-relative-L2 even at 95% channel retention, in exchange for only ~4.55% idealized Micro-FFN operations saved. This P1 research therefore changes the problem: retain the ordinary macro-Expert router and all model weights, but optimize the *duration and frequency of precision policy decisions*, rather than deleting neuron groups.

H-JEPA inspired this separation of expensive upper planning and lower execution, **NOT** the numerical result or a pretrained hierarchical latent JEPA. No separately learned upper/lower latent spaces were built or trained in this version.

## Architecture and original Beglin contracts

- **Upper-level budget planner**: propose one exact policy for several *requests* (horizon 4, 16, 64, 256 or 512); consider measured steady-state costs, cold/warm transition costs and past-only risk observations. Never inspect future risk flags to decide the current action.
- **Lower-level guard**: if a request has a context-certified precision requirement, defer to it immediately. For the studied DeepSeek-V2-Lite `low_margin` example, the approved recovery is the alternate combined policy B, **not** a blind return to startup policy A. Earlier cost-only exploratory replay had this direction backwards and is superseded by the corrected run below.
- All proposals are research-only, no `PrecisionEpochScheduler` invocation, no resident worker rebinding and no policy write. The true dynamic policy acceptance remains in Beglin's `precision_allocator`, `precision_dynamic_selector`, `precision_policy_cost_optimizer` and `precision_closed_loop` modules.
- qNg64 bits are nonmonotonic in quality; **do not infer higher bits are safer**. Candidate safety cannot be inferred from cost profiles alone. Exact combined-policy and condition-bucket certification are separate prerequisites for any actual use.
- `tools/hcp_replay.py` is a deterministic proof-of-mechanism simulator. It models **two** known policy cost profiles only; it is not native inference or an accepted new production controller.

## Real XOX measured-cost input, safety qualification

- [XOX P0 inventory read-only SUCCESS #38439941610](https://github.com/popixoxipop-collab/beglin/actions/runs/38439941610).
- SHA-bound sanitized `tools/evidence_inventory.json`: 10 archived `beglin-precision-e2e-cost-v1` PASS profiles. Every policy-hash is recalculated with canonical `precision_context.policy_hash`; incorrect/missing cost or transition provenance fails closed.
- Example `p7-soak-precommit/result.json`: startup A 183.1025 ms/request, combined B 181.596458 ms/request; first A→B cost 615.009 ms, B→A warm restore 306.777 ms. In identical conditions, transition+restore needs ~612 favorable requests to amortize at the recorded steady-state difference. This is illustrative only: requests include a 10-token inference benchmark, not per-token latency.
- Across the 10 archived cost snapshots, the rank order A vs B varies. Benchmarks include different source commits/binaries/cache contexts and are **not independent samples** of a single stationary cost distribution.
- P0 inventory contains **no condition-specific combined-policy acceptance records and no original risk event stream**. Treat every subsequent risk-eligibility simulation as HYPOTHETICAL, not as real authorization.

## Experiment A — XOX archived cost cross-profile A/B

- [Corrected Ubuntu + XOX SUCCESS #38442269527](https://github.com/popixoxipop-collab/beglin/actions/runs/38442269527), 8/8 Python unit tests on each execution host.
- Static reference = alternate policy B assumed *hypothetically* risk-compatible across the simulated trace; flat controller = recompute one-step cost per request; hierarchical controller = plan once per H requests, replan/guard on observed risk. The lower risk guard requires B in this example.
- 10 archived planning cost profiles × 10 archived evaluation cost profiles × 3 **synthetic** risk patterns = 300 combinations. Of the 270 off-diagonal combinations, H=512 had **47 lower modeled costs, 61 higher and 162 unchanged** relative to the static assumed-risk-compatible B reference. Median difference 0%; observed modeled range **−7.936825% to +21.697014%** (negative = lower modeled cost).
- Short horizons 4–256 mostly held the incumbent rather than paying expensive transitions. Upper planning is cheaper in call frequency but that is not a demonstrated native LLM acceleration.
- The output metric is a *sum of historical per-policy p50 cost estimates plus transition estimates*, not elapsed time of real counterfactual policy sequences. Each risk sequence is synthetic and context permission assumed in the cost-only experiment.

### XOX evidence
- [A/B run artifact 11834654603](https://github.com/popixoxipop-collab/beglin/actions/runs/38442269527/artifacts/11834654603).
- ZIP SHA-256: `6a8be0bb690d5fd8d91cb277b18e3613a9baadea6c7605e5b8daad807d33aa88`.
- Modelled result JSON SHA-256: `f8a612511f79f0818b864c12b540252f3efeb5011bf5b00f53d257e965b5f213`.

## Experiment B — independently verified 600 actual P7 admissions

- [P7 full archived lineage hash-chain read-only PASS #38442567975](https://github.com/popixoxipop-collab/beglin/actions/runs/38442567975).
- Source-result SHA-256 `ebc6c8c52759e7164b555d19e7e330bf933835be2b32073cef634c13ef332c94`. All 59 sealed 10-record segments (590 records) and the final 10 active records passed content hash, manifest-chain and record-value checks; exactly **600 unique admissions** and finite original logits.
- P7 *test procedure* injected `low_margin` at **200** admissions of the repeated **same reference prompt**, producing **400 actual precision transitions**, 3 observed policy hashes. Median actual original-engine wall time was **419.8785 ms** and original roundtrip **492.0448755 ms**. This does not measure the new planner, and is not a natural varied-user-input distribution.
- [Cost-only risk-order replay SUCCESS #38442749012](https://github.com/popixoxipop-collab/beglin/actions/runs/38442749012) takes that operator-injected 600-step risk-flag sequence, runs 10×10 archived cost-pair comparisons, and models a simplified *two-policy* controller. It cannot replicate the **third actual P7 policy** and does not replay any native tokens under new policy.
- H=512 out-of-archive *modeled* cost delta vs hypothetical static B ranged **−0.012923% to +0.036295%**, median **0%**. The improvement is neither robust nor practically meaningful.

### Temporal evidence
- [P7 trace + counterfactual artifact 11837955402](https://github.com/popixoxipop-collab/beglin/actions/runs/38442749012/artifacts/11837955402).
- ZIP SHA-256: `c829f1e8fac7a82d57962acd31b146c534fb6ea09e5c6aa5f539b3cf6c7c7960`.
- Verified temporal summary JSON SHA-256: `efda34dab9c7af35269dc9165ad670598248d6bdc128cae9b15c1a4706f1728c`.
- Temporal cost-only counterfactual JSON SHA-256: `ee5b6956d2014757bbce58fda0f845d66b748d7bb493c54d9a31183f10432ab5`.

## Failed claims / remaining gates

1. **No measured native inference speedup** of the two-level planner; historical cost simulation does not establish throughput, p95 latency or energy.
2. **No real per-request context certification consumed by the planner**. Beglin's original trigger evidence contract must be verified and delegated to existing selection/epoch guard before allowing any backend rebinding.
3. **No learnt H-JEPA-style upper JEPA and lower latent execution model**. This is a deterministic planning baseline inspired by two timescales; do not call it an H-JEPA-trained architecture.
4. **Only two policy profiles modeled**, while verified original P7 used three.
5. The original P7's 600 admissions recycled one fixed prompt and injected signals; the historical archives include different binaries. These are not generalization-grade prospective quality experiments.
6. No actual token-logit comparison, model task quality, native qNg64 regression, adaptive GPU/CPU crossover, or total runtime memory measurement of this new controller.
7. The optimizer must *never* assume `low_margin ⇒ return to startup A` when the conditionally certified selector requires B. This semantic error in the superseded initial cost replay was fixed and separately unit-tested.

## Next P2 engineering gate

- First extract sanitized, actually observed request-context signals with full per-request evidence IDs and accepted-policy sets, including the third policy, from a diverse set of isolated prompts. Track source binary/checkpoint, preserve chronological ordering and train/validation/holdout splits by request source, not the repeated fixed prompt.
- Compare static/frozen policy, existing flat evidence-gated selector, the deterministic 2-level planner, and **a separately learned slow/fast latent predictor** with the same constrained policy action space. Price *planner overhead and all transitions*.
- Promote only after exact accepted-policy/trigger evidence matches on the target backend and a **separate scratch-only native full-model A/B** demonstrates actual logit/accuracy consistency, token throughput/p95, RSS/Metal memory and net cost benefit. Production and `main` remain untouched; automatic live promotion is OFF.
