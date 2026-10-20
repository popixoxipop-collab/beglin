# G0 source/paper audit — 2026-10-21

**Status: G0 started; numerical baseline verified; real LLM detection/transfer NOT verified.** All live production changes prohibited. Draft branch: `research/oml-acs-prereg-20261021`.

## Paper source
Kim & Han, *Cross-Family Universality of Behavioral Axes via Anchor-Projected Representations*, arXiv:2605.09875v1 (May 11, 2026), https://arxiv.org/html/2605.09875v1. Baseline mathematical contracts:

1. Extract final-token post-block residual `a_m(x;L_m)` at the designated model layer.
2. Fixed 300 anchors: 15 HELM scenarios ×20, seed 42, text length 10–300 chars after cleanup, exact dedup; keep anchor IDs/order identical across all models.
3. `mu_m=mean_i A_m[i]`; `Ahat_m[i]=normalize(A_m[i]-mu_m)`.
4. **Point** `pi_pt(h)=Ahat_m @ normalize(h-mu_m)`. **Direction** `pi_dir(v)=Ahat_m @ normalize(v)`; never subtract `mu` from differences.
5. Source native direction `v=normalize(mean(pos)-mean(neg))`; source canonical `c=normalize(mean_m pi_dir(v_m))`. Target reconstruction `normalize(Ahat_target.T@c)` is an adjoint, not an inverse.
6. Source axes 100 positive-negative pairs, independent test 100 pairs, paper ten axes. First engineering pilot uses sentiment SST-2 and emotion CARER (anger vs joy). Our extra 50 development pairs are a **planned deviation** and never counted as exact source reproduction.
7. Pilot small variants were paper Appendix A scale sweep (Qwen2.5-1.5B layer 14/28, Llama3.2-3B layer 14/28, Phi4-mini layer 16/32). Paper primary study used bigger models and selects models/layers by grid. Our **explicit zero-based indexing** may differ from author's undisclosed source convention; verify before any exact-reproduction claim.
8. Steering applies a constant residual addition with `alpha in [-5,-3,0,3,5]`; must compare no intervention, norm-matched random, shuffled anchors, prompt instruction, and label-dependent target-native upper bound separately.
9. Paper reports within-cluster (Llama/Qwen/Mistral/Phi) stronger alignment than Gemma, so no unconditional universality claim. Refusal OOD example is **+0.46 absolute refusal-rate change**, not '0.46 percentage points' (arXiv abstract formatting is inconsistent).

## Beglin source checked

- `README.md`: CPU Accelerate/vDSP+NEON and optional MLX/Metal, qNg64 tensor-local and expert-local precision, current serving.
- `tools/model_capability.py`: architecture/checkpoint/tensor role graph source inspection (not an activation extractor).
- `tools/model_capability_bridge.py`: model capability bundle/hash and target-gated backend precision roles.
- `tools/qt_beval.py`: provenance/evidence contract for quant policy review (not behavior metrics).
- `tools/precision_e2e_cost.py`: measured serving cost accounting (not LLM correctness).
- Current XOX `vdsp_engine` working tree was dirty and behind GitHub main; **not touched**.

## Implementation and honest verdicts

- `research/oml_acs/acs_core.py`: NumPy mathematical baseline; no model execution or target-label access.
- `research/oml_acs/manifest.py`: exact text/group collision detector and V6 consumed-holdout guard. **No independent semantic near-duplicate detector** yet.
- `research/oml_acs/hf_adapter.py`: optional pinned BF16 HF model hooks for future actual execution, refuses unpinned revisions, unknown model, implicit network downloads, unsupported decoder blocks, and remote-code execution. This adapter has passed **contract tests only**, not actual model integration tests.
- EOE Python 3.9 environment: NumPy present; Torch/Transformers absent. Alienware WSL: Torch 2.12.1+cu130, Transformers 5.12.1, GPU RTX 5070 Ti and locally cached pinned Qwen 1.5B.
- Source-first, no new paid API, no new hardware, no automatic production promotion.

## Blockers before G1 scientific evidence

- Exact HELM 300 anchor **IDs/text list** not retrieved. This is a method reimplementation until source correspondence is verified.
- Full V6 **24 used holdout content SHA-256** still unavailable: preregistration freeze is BLOCKED even if new data splits appear disjoint.
- Full real BoolQ/ARC-Challenge/CommonsenseQA source rows and behavior label splits have not been loaded or frozen.
- Pinned Llama3.2 model is gated; Phi trust_remote_code requirement and code audit unresolved.
- No three-family leave-one-out detection/steering results yet. No AUROC/NLL/efficiency scientific claim.
- Need an actual end-to-end HF adapter hook test and explicit numerical sign/prompt/template checks.
