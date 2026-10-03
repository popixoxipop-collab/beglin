# Precision Risk Signals v1

## Scope
P5 expands request-time precision evidence beyond low_margin with three measured signals:

- near_tie: final-output top1 minus top2 raw-logit gap
- high_entropy: normalized full-vocabulary softmax entropy, H/log(V)
- routing_ambiguity: MoE expert-boundary ratio next_unselected / kth_selected

Correctness evidence remains mandatory. Signals do not themselves authorize a precision change.

## Runtime telemetry
The existing near-tie JSONL event now optionally carries:

- entropy
- routing_ambiguity_score
- routing_ambiguity_layer
- routing_boundary_selected
- routing_boundary_next

GPU routing ambiguity is observation-only and opt-in through QWEN_MOE_RISK_SIGNALS=1.
When disabled, the MLX router keeps the existing lazy path and performs no telemetry eval/readback.
When enabled for calibration, the MLX ragged scheduler reads the router softmax row and records the maximum ambiguity across MoE layers for each physical slot.

The score is bounded to [0,1]:
- 1.0 means the first unselected expert is effectively tied with the kth selected expert.
- smaller values mean a clearer top-k routing boundary.

The production worker environment defaults QWEN_MOE_RISK_SIGNALS to 0.

## Selector contract
precision_risk_signals.py converts measured request events into Dynamic Selector-compatible signals.

No metric means no trigger. Missing entropy or routing telemetry never becomes a synthetic risk signal.

Dynamic Selector continues to require exact contextual PASS evidence for every active trigger.
If near_tie, high_entropy and routing_ambiguity are simultaneously active, all three matching evidence rows are required before an alternate n is eligible.

This preserves the non-monotonic qNg64 rule: higher n is never assumed safer without evidence.
## XOX pre-commit calibration
Known risk fixture:
[23757,49238,28947,3969,319,317,38470,440,6383]

Under shared_up_proj/L3 n6:
- output [55222,372], incorrect second token
- margin 0.043699
- normalized entropy 0.232097685
- routing ambiguity 0.999847949
- most ambiguous router layer 24
- kth selected probability 0.027097726
- next unselected probability 0.027093606

Under shared_up_proj/L3 n5:
- output [55222,1]
- 6/6 target runs succeeded

Returning to L3 n6 reproduced the failure 3/3 times on the same worker PID.

Measured exact calibration buckets use only a +/- 1e-6 envelope around the observed base values; no arbitrary threshold was substituted.
Pre-commit isolated acceptance:
- /Users/xox/vdsp_serving/precision-risk-signals-acceptance-v1b/result.json
- SHA-256 a5e38605e6d353f588f0e7a08140e6a819714baf764c9d2f212de11c85e4508c

Selector behavior in that acceptance:
- P5 triad near_tie + high_entropy + routing_ambiguity -> L3 n5, L26 n5
- existing low_margin 0.010715 -> L3 n6, L26 n6
- P5 triad with high_entropy evidence removed -> L3 n6, L26 n5

The P5-selected policy produced [55222,1] and then the scratch worker was restored to L3 n6 + L26 n5.

## Production boundary
P5 telemetry and evidence are successor-only in this stage.
No production binary or route is changed.
The existing generation-6 adaptive L26 service remains the live path.
Exact-source acceptance and certified regression are sealed after the implementation commit is fixed.
