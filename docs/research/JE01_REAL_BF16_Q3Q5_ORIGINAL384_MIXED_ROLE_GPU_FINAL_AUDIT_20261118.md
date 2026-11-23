# JE01 — native original BF16 Q3/Q5 Expert/role GPU correction audit (2026-11-18 KST)

## Status and immutable references

- Branch: research/je01-moe-bayesian-bit-allocation-20261117; PR #94 Draft, no merge, no promotion.
- Original immutable 27-layer BEGLIN compressed AF baseline: 256 WikiText calibration cases, 11,024 gold tokens, NLL 2.46035653719724; 8 selected original Expert compressed FFN payloads = 43,253,760 physical bytes.
- Real BF16 source: exact 4/4 original DeepSeek-V2-Lite-Chat full-shard SHA; real q3/q5 physical tiers and three exact-budget files sealed previously.
- Existing NIG role policy selected 6 demoted q3 FFN roles; 3 other frozen control policies. Exact physical source evidence: [GitHub Actions run 44181761880](https://github.com/popixoxipop-collab/beglin/actions/runs/44181761880), SHA receipt 8d3be1509b0484b9f97687c98d0eaceeea3a6e2bde836c94e0a30ac242898d58, 4 x 43,253,760 B. The NIG selected L13 E28 gate/up/down and L24 E60 gate/up/down as its six low-bit roles. That means the observed NIG policy collapsed to two complete Experts and does not establish additional role-level allocation value.
- Source code staging for original 384 cbatch: commit bc86ee3a7a11c56df4d1927dacae706fc7e3ecbf, static fail-closed 5 tests and real source anchor CI [44180739115](https://github.com/popixoxipop-collab/beglin/actions/runs/44180739115) PASS. No canonical mlx_moe.cpp/h changed: all patching happens inside an isolated source copy.

## Real native GPU evidence (all real pretrained original BF16-derived weights)

| Test | CI run | Result |
| --- | --- | --- |
| E1 Q3/Q5 8 Expert × 3 roles Metal matvec against independent CPU native decode | [44179126248](https://github.com/popixoxipop-collab/beglin/actions/runs/44179126248) | 24/24 PASS; q3 6 roles, q5 18 roles; worst relative RMSE 1.184402499e-6; worst absolute 4.053115845e-6; MLX 47,592,964B peak |
| Original384 B1 scratch cbatch output correction, actual BF16-source L13 E22 q3 gate/up/down; isolated router mock/fallback | [44181161001](https://github.com/popixoxipop-collab/beglin/actions/runs/44181161001) | PASS; relative FFN correction error vs independently decoded CPU 1.447673722e-6; nonselected fallback 0; registry-clear fallback 0; peak independent context |
| Same scratch original384 correction with actual mixed-role L24 E26 q3 gate, q3 up, q5 down from weight-MSE control candidate; mock top6 | [44182229286](https://github.com/popixoxipop-collab/beglin/actions/runs/44182229286) | PASS; relative FFN correction error 1.580747387e-6; maximum absolute error 1.583710872e-6; unselected fallback 0; clear fallback 0 |

### What the correction actually computes

In an isolated GPU cbatch graph, the original 64-Expert q4 baseline routing and all six original per-Expert FFN outputs are evaluated first. For a frozen selected (layer,Expert) E1 n3/n5 overlay, correct the sum with

    routed_sum_new = routed_sum_original
                     + original_topk_weight[e]
                       * (overlaid_expert_FFN_output - old_selected_expert_FFN_output).

The original Expert route IDs and topk weights are not themselves edited in this layer. Unselected Expert output is unchanged by construction. The graph correction is model-free tested on actual E1 pretrained FFN weights with controlled input activations and controlled baseline-output tensors, including exact nonselected/registry-clear negative controls. This IS NOT yet an original 27-layer end-to-end model inference/parity claim. Downstream layer routing may change after modified previous-layer activations and still requires measured topk comparisons on real sequences.

### Statistical limit of NIG allocation

The real-weight synthetic-activation role NIG posterior pilot was trained with independent synthetic 32 cases/Expert and tested with fresh synthetic 32 cases/Expert. Although NIG proxy had lowest mean FFN output MSE 0.0783658 versus fixed-role permutation 0.0809883, weight-MSE 0.0820944, and prelabel-frequency 0.0982682, 20,000 paired bootstrap repetitions resampling **8 Expert clusters** gave 95% confidence intervals crossing 0 for every contrast and Holm-adjusted p values non-significant. See [44178972375](https://github.com/popixoxipop-collab/beglin/actions/runs/44178972375). This does NOT justify promoting NIG nor calling it a true gold-token NLL posterior.

### Unresolved scientific and runtime gates

1. Prove same-source original AF q4 checkpoint bytes/quantization parity to original Chat BF16; original AF baseline model 4bit and BF16-derived n3/n5 are not automatically identical provenance.
2. Integrate a verified 8 Expert overlay binder/manifest loader into a scratch-only real original 384 full-cbatch model runner and demonstrate NO-overlay full-logit parity before candidate test. Merely compiling the patch is not enough.
3. Check real 256 calibration prompts with 11,024 gold tokens using genuine BEGLIN original AF full-model logits, 32 or more independently routed paired gold-NLL observations per (Expert,tier); NIG risk posterior must use **real paired NLL**, not synthetic FFN MSE. Lock any candidate before independent heldout.
4. Compare baseline, freeze-chosen NIG, matched-budget controls on completely untouched WikiText128 and HellaSwag128 with 20k **paired** bootstrap. Required holdout gain >=0.02 NLL, lower 95% bound >0, OOD non-regression, 0 hard topk flips, p95 token latency <=5% regression, multiplicity correction and independent audit.
5. XOX read-only host command to inspect original AF source was blocked by Tailnet fleet admission: insufficient free CPU/RAM and no eligible scratch storage pool (~24.6GiB free below ~36GiB reserve). This safety limit was **not bypassed**; no real 27-layer alt model forward was run. User operational weights and Tailnet current/live g6-0.9.0-operator.900485 unchanged.

**Final conclusion: Native E1 n3/n5 kernel and scratch cbatch mixed-role correction mechanics PASS. Bayesian NLL improvement HOLD. PR Draft, no production promotion.**
