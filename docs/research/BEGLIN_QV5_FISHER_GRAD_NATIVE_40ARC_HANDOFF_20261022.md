# BEGLIN Qv5 — original BF16 directional empirical Fisher + actual 40 ARC native engine handoff

- **Date:** 2026-10-22 KST
- **Research branch:** \`research/qv5-fisher-grad-allocator-20261022\`
- **Baseline:** Qv4 research worktree, not \`main\`
- **Model:** \`Qwen/Qwen2.5-1.5B-Instruct\` original BF16 source revision \`989aa7980e4cf806f80c7fef2b1adb7bc71aa306\`
- **Dataset:** \`allenai/ai2_arc\` revision \`210d026faf9955653af8916fad021475a3f00453\`, ARC-Challenge train 24 / validation 40, SHA-sorted IDs, source/development non-overlap
- **Scientific verdict:** Source Fisher and native prototype execution PASS; **quality generalization NOT ESTABLISHED**, full production dynamic precision/Metal auto promotion **OFF**.

## A. Why Qv5 exists

Qv1 weight reconstruction, Qv2 diagonal BF16 activation and Qv3 original-input off-diagonal covariance all produced exact cost-constrained 28-of-196 precision allocations, but their previous whole-prompt development PPL did **not** prove superior model quality versus Q4/seed73. Qv4 added answer-suffix NLL on only eight simple authored questions; those data and the old V6 24 unknown provenance identifiers are not used here.

This research evaluates the *direction* of the native quantization residual against the **unmodified BF16 source model's task-gradient**:

\`\`\`text
For original BF16 model weights W_i and correct one-letter source answer loss L_s:

g_(i,s) = ∂L_s / ∂W_i
delta_(i,b) = NativeBEGLIN_Q_b(W_i) - W_i

F_(i,b) = mean_over_source_items [(g_(i,s) · delta_(i,b))^2]
gain_(i,8) = F_(i,4) - F_(i,8)

max sum(gain_(i,8) * x_i), x_i in {0,1},
sum x_i = 28, native packed bytes <= 830570496,
predicted serial Qwen CPU token latency <= 100ms,
source-measured independent async-stage additive estimate <= 50ms.
\`\`\`

Important: directional empirical Fisher quadratic form is a **first-order loss-perturbation variance proxy**, **not** the exact Hessian, actual model PPL, or a per-tensor full-model causal intervention. The Fisher idea is established prior art; BEGLIN's research integration is with actual native C bitplanes/NEON costs, source-frozen tensor policies, and native full-model verification, not a claim of having invented Fisher quantization.

## B. Original BF16 source calibration

All **196 projection tensors** in 28 Qwen decoder layers contributed gradients on **24 prelocked train questions** and exact BEGLIN C Q4G64/n8 packed residuals; no quantized model forward or downstream single-tensor intervention. BF16 MPS backward gradients were promoted to FP32 and directional dot products accumulated in FP64; quantizer residuals use an FP16 intermediate. This is not full exact F64 Fisher.

- Source score private receipt SHA: \`6d87dfef842fcce932cdb85c9a42bc3ede380493917ae608663e2072f8143417\`
- Original BF16 train answer-NLL: **1.227782**
- Source split-half (odd/even 12 vs 12) Fisher rank Spearman: **0.479680**. Limited stability, not a proven generalizable ranking.
- 8-bit directional proxy gain positive in **196/196** source projection tensors.
- Source held-out validation inputs were frozen before Fisher fitting and were not scored by the optimizer.

## C. Source-only exact allocation

- MILP status **0**, gap **0.0**, native CPU resource inputs unchanged from Qv2.
- Policy mask SHA: \`88fdc38e6f55a16b0bdd4f7385e70f6de053db577a26d334a3ef3cb831113533\`
- 8-bit role counts: down_proj=1, gate_proj=0, k_proj=5, o_proj=7, q_proj=1, up_proj=0, v_proj=14
- Packed projection bytes **757,039,104** (baseline full Q4 = 736,985,088); predicted serial decode **83.000ms**, isolated stage DAG sum **49.931ms**. **These are cost predictions**, not actual Qv5 sustained tokens/s or real full-model async speedups.
- Research policy is source-frozen. Dynamic/request-wise runtime precision mutation **not supported by Qv5**.

## D. Real 28-layer Apple BEGLIN CPU native results, 40 ARC-Challenge validation questions

All six arms use the **same locked 40 prompts, token IDs, A/B/C/D choice IDs, original source model and native 28-layer Qwen code**. Every question has a single exact-letter answer-token NLL and argmax among four options, from one native forward of the prefix per arm. Q4 and F32 use the same base binary; all four mixed policies use the same C mixed/repacked-n8 research binary. Different precision-family binaries are separately hash-pinned.

| Policy | Correct /40 | Accuracy | Answer-only correct token NLL ↓ |
|---|---:|---:|---:|
| f32 | 31/40 | 77.5% | 1.348649 |
| q4 | 31/40 | 77.5% | 0.980812 |
| role_random | 31/40 | 77.5% | 1.076543 |
| v2_diag | 30/40 | 75.0% | 0.995317 |
| v3_cov | 31/40 | 77.5% | 1.013715 |
| v5_fisher | 31/40 | 77.5% | 1.009328 |

Relative to native Q4, **Qv5-minus-Q4 mean answer NLL +0.028516**; verdict: **development mean NLL NOT lower than Q4**.

### Paired case-level bootstrap (exploratory, no multiplicity correction)

Difference is *baseline NLL minus Qv5 NLL*: positive favors Qv5. Deterministic 5,000 resamples over 40 paired source-frozen questions. Not a multi-model or official holdout guarantee.

| Baseline | Mean paired NLL difference | 95% bootstrap percentile interval | Questions with lower Qv5 NLL |
|---|---:|---:|---:|
| f32 | +0.339321 | [-0.076269, +0.821736] | 11/40 |
| q4 | -0.028516 | [-0.277730, +0.205226] | 22/40 |
| role_random | +0.067216 | [-0.287973, +0.390792] | 15/40 |
| v2_diag | -0.014011 | [-0.224536, +0.198526] | 26/40 |
| v3_cov | +0.004388 | [-0.216775, +0.217592] | 16/40 |

## E. Independent original Transformers BF16 arithmetic control

- Original unmodified Qwen Transformers 2.13 MPS BF16, same frozen 40 questions: **32/40 correct, answer NLL 1.337771**.
- Native BF16-source *widened-F32* vs independent *BF16* reference are not the same arithmetic path. Diagnostic comparison: 39/40 agree on whether an answer was correct; mean absolute per-question correct-label NLL difference 0.215676. Do not claim byte-exact model parity.
- Separate independent native transcript auditor re-parsed all original 40×6 correct-label NLL lines, four-choice decision logs, SHA-frozen evaluation inputs, actual binary SHA and 196 gradient source arrays. Private audit SHA: \`644a324a82698529ec818810230a1055d93a949568f3d4dac229a1ba044da387\`.

## F. Gate / limitations / next action

- **NO live activation.** \`main\`, Qwen original checkpoint, Tailnet Commander immutable operator release and live Apple Metal inference remain untouched.
- Six-arm native CPU completion and 40-question small development sample **cannot establish quality superiority or generalization**. No OOD dataset, multiple model-family replication, actual cost p50/p95/energy, CPU/GPU runtime dynamic admission or long-term serving evidence.
- 24 source questions produce at most modest source rank stability; gradients can saturate for easy items, and signed quant residual alignment can be unstable. More data and independent source/dataset splits are required.
- Do not tune on the 40 validation labels and then present the same sample as independent evidence. Next: pre-register a *new* dataset and held-out model family, rank-stability ablations (24→64→128), uncertainty intervals and a fully native sustained ABBA CPU benchmark on a quiet host; only after this consider additional bitwidths/Metal.
- Independent audit is **log/process/contract and arithmetic reproduction**, not a separate end-to-end kernel implementation; no new dynamic allocation release.

## G. Files and reproduction boundaries

Source/data lock: \`configs/research/BEGLIN_QV5_FISHER_GRAD_PRELOCK_20261022.json\`, \`q_v5_freeze_arc_inputs.py\`.
Real original BF16 source-gradient + original native C quantized residual: \`q_v5_original_bf16_fisher_source.py\`.
Measured Apple kernels + MILP: \`q_v5_fisher_milp.py\`.
Actual native 40-ARC C CPU programs and frozen masked accuracy/NLL: \`q_v5_native_arc_40eval.py\`.
Independent Torch BF16 reference: \`q_v5_independent_hf_bf16_oracle.py\`.
Independent native source/score audit: \`q_v5_independent_native_audit.py\`.
Regression: \`tests/research/test_q_v5_fisher_native.py\`.

Private raw model gradients, quant residuals, 40 validation token arrays and per-question native transcripts remain in \`lab_local/qv5_fisher_20261022/\` with owner-only permissions, excluded from Git. Commit only public aggregate JSONs, code, masks, tests and this handoff.

Academic anchors: HAWQ-V2 (Dong et al., NeurIPS 2020), FAMPWQ (Lee et al., arXiv:2608.24945), InfoQ (Akbulut et al., AAAI 2026). These are prior work comparisons; no new novelty or quality win is declared by Qv5 alone.
