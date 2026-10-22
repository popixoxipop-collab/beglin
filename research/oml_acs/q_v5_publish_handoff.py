#!/usr/bin/env python3
"""Seal a research-only Markdown handoff from independently audited native receipts."""
from __future__ import annotations
import hashlib,json,os,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
FISHER=ROOT/"docs/research/BEGLIN_QV5_FISHER_SOURCE_AGGREGATE_20261022.json"
MILP=ROOT/"docs/research/BEGLIN_QV5_FISHER_HIGHS_SOURCE_MILP_20261022.json"
ARC=ROOT/"docs/research/BEGLIN_QV5_FISHER_40ARC_NATIVE_PUBLIC_RESULT_20261022.json"
HF=ROOT/"docs/research/BEGLIN_QV5_ORIGINAL_BF16_HF_40_ARC_AGGREGATE_20261022.json"
OUT=ROOT/"docs/research/BEGLIN_QV5_FISHER_GRAD_NATIVE_40ARC_HANDOFF_20261022.md"
ORDER=("f32","q4","role_random","v2_diag","v3_cov","v5_fisher")
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
    if OUT.exists():raise FileExistsError("do not overwrite handoff based on mutable native evidence")
    for p in (FISHER,MILP,ARC,HF):
        if not p.is_file():raise RuntimeError("missing source/quality/hardware/independent proof")
    f=json.loads(FISHER.read_text());m=json.loads(MILP.read_text())
    a=json.loads(ARC.read_text());h=json.loads(HF.read_text())
    if a["status"]!="PASS_ORIGINAL_BF16_GRADIENT_AND_NATIVE_C_REPLAY_AUDIT_NOT_FINAL_RESEARCH_SUCCESS" or \
       a["policy_SHA"]!=m["mask_sha256"] or not m["source_only_objective"] or \
       a["Qv5_fisher_research_promotion_allowed"] or len(a["native_summary"])!=6:
        raise RuntimeError("Qv5 test failed or unverified; no success handoff")
    rows=[]
    for name in ORDER:
        d=a["native_summary"][name]
        rows.append(f"| {name} | {d['correct']}/40 | {d['accuracy']*100:.1f}% | {d['answer_only_mean_NLL']:.6f} |")
    q4=a["native_summary"]["q4"];v5=a["native_summary"]["v5_fisher"]
    change=v5["answer_only_mean_NLL"]-q4["answer_only_mean_NLL"]
    verdict=("development mean NLL lower than Q4, not a final generalization win"
             if change<0 else "development mean NLL NOT lower than Q4")
    comparisons=[]
    for name,v in a["paired_bootstrap"].items():
        comparisons.append(f"| {name} | {v['baseline_minus_Fisher_NLL_mean']:+.6f} | [{v['bootstrap95_interval'][0]:+.6f}, {v['bootstrap95_interval'][1]:+.6f}] | {v['nll_fisher_better_on_cases']}/40 |")
    src=f"""# BEGLIN Qv5 — original BF16 directional empirical Fisher + actual 40 ARC native engine handoff

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

max sum(gain_(i,8) * x_i), x_i in {{0,1}},
sum x_i = 28, native packed bytes <= 830570496,
predicted serial Qwen CPU token latency <= 100ms,
source-measured independent async-stage additive estimate <= 50ms.
\`\`\`

Important: directional empirical Fisher quadratic form is a **first-order loss-perturbation variance proxy**, **not** the exact Hessian, actual model PPL, or a per-tensor full-model causal intervention. The Fisher idea is established prior art; BEGLIN's research integration is with actual native C bitplanes/NEON costs, source-frozen tensor policies, and native full-model verification, not a claim of having invented Fisher quantization.

## B. Original BF16 source calibration

All **196 projection tensors** in 28 Qwen decoder layers contributed gradients on **24 prelocked train questions** and exact BEGLIN C Q4G64/n8 packed residuals; no quantized model forward or downstream single-tensor intervention. BF16 MPS backward gradients were promoted to FP32 and directional dot products accumulated in FP64; quantizer residuals use an FP16 intermediate. This is not full exact F64 Fisher.

- Source score private receipt SHA: \`{f['private_full_source_receipt_SHA256']}\`
- Original BF16 train answer-NLL: **{f['source_full_correct_answer_avg_nll']:.6f}**
- Source split-half (odd/even 12 vs 12) Fisher rank Spearman: **{f['split_half_odd_even_spearman']:.6f}**. Limited stability, not a proven generalizable ranking.
- 8-bit directional proxy gain positive in **{196-f['negative_gain_tensors']}/196** source projection tensors.
- Source held-out validation inputs were frozen before Fisher fitting and were not scored by the optimizer.

## C. Source-only exact allocation

- MILP status **{m['optimizer']['source_only_solver_status']}**, gap **{m['optimizer']['mip_gap']}**, native CPU resource inputs unchanged from Qv2.
- Policy mask SHA: \`{m['mask_sha256']}\`
- 8-bit role counts: {", ".join(f"{role}={count}" for role,count in m["optimizer"]["selected_role_counts"].items())}
- Packed projection bytes **{m['optimizer']['estimated_projection_weight_bytes']:,}** (baseline full Q4 = 736,985,088); predicted serial decode **{m['optimizer']['original_fullmodel_serial_estimated_ms']:.3f}ms**, isolated stage DAG sum **{m['optimizer']['isolated_async_CPU_stage_total_ms']:.3f}ms**. **These are cost predictions**, not actual Qv5 sustained tokens/s or real full-model async speedups.
- Research policy is source-frozen. Dynamic/request-wise runtime precision mutation **not supported by Qv5**.

## D. Real 28-layer Apple BEGLIN CPU native results, 40 ARC-Challenge validation questions

All six arms use the **same locked 40 prompts, token IDs, A/B/C/D choice IDs, original source model and native 28-layer Qwen code**. Every question has a single exact-letter answer-token NLL and argmax among four options, from one native forward of the prefix per arm. Q4 and F32 use the same base binary; all four mixed policies use the same C mixed/repacked-n8 research binary. Different precision-family binaries are separately hash-pinned.

| Policy | Correct /40 | Accuracy | Answer-only correct token NLL ↓ |
|---|---:|---:|---:|
{chr(10).join(rows)}

Relative to native Q4, **Qv5-minus-Q4 mean answer NLL {change:+.6f}**; verdict: **{verdict}**.

### Paired case-level bootstrap (exploratory, no multiplicity correction)

Difference is *baseline NLL minus Qv5 NLL*: positive favors Qv5. Deterministic 5,000 resamples over 40 paired source-frozen questions. Not a multi-model or official holdout guarantee.

| Baseline | Mean paired NLL difference | 95% bootstrap percentile interval | Questions with lower Qv5 NLL |
|---|---:|---:|---:|
{chr(10).join(comparisons)}

## E. Independent original Transformers BF16 arithmetic control

- Original unmodified Qwen Transformers 2.13 MPS BF16, same frozen 40 questions: **{h['correct']}/40 correct, answer NLL {h['correct_label_NLL']:.6f}**.
- Native BF16-source *widened-F32* vs independent *BF16* reference are not the same arithmetic path. Diagnostic comparison: {a['BF16_external_oracle_crosscheck']['native_vs_torch_BF16_per_case_correctness_agreement']}/40 agree on whether an answer was correct; mean absolute per-question correct-label NLL difference {a['BF16_external_oracle_crosscheck']['native_vs_torch_BF16_per_case_absolute_NLL_mean']:.6f}. Do not claim byte-exact model parity.
- Separate independent native transcript auditor re-parsed all original 40×6 correct-label NLL lines, four-choice decision logs, SHA-frozen evaluation inputs, actual binary SHA and 196 gradient source arrays. Private audit SHA: \`{a['private_independent_full_audit_SHA256']}\`.

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
"""
    data=src.encode()
    with OUT.open("xb") as s:s.write(data);s.flush();os.fsync(s.fileno())
    print("QV5_HANDOFF_SEALED",sha(data),len(data),"quality_verdict",verdict,flush=True)
if __name__=="__main__":run()
