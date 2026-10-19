# BEGLIN — full 27-layer DeepSeek-V2-Lite BF16 vs local qNg64 logits/NLL (2026-10-18)

**Status: PASS engine numerical experiment; NOT model-quality certified; NOT deployed.**

## What was actually run

Using the original DeepSeek-V2-Lite modeling classes, original tokenizer and read-only **D50 BF16 research copy** of the model, a source-defined **27/27-decoder-layer** CPU reference was executed one layer at a time. For each MoE layer, the actual `MoEGate` Top-6 router was recomputed from the current hidden state and only the selected experts loaded. The complete BF16 `lm_head` was evaluated over **102,400 vocabulary logits** in bounded BF16 chunks.

The new BEGLIN C qNg64 bitplane module was then applied to the **first decoder's seven Attention/FFN projection matrices** under two policies, leaving all later 26 layers in BF16:

- Uniform qNg64 n7: `[7,7,7,7,7,7,7]`, code+F32 scale **75,939,840 bytes**.
- Previously train-selected mixed qNg64: `[9,9,9,7,6,7,7]` for `[Q,KV-A,KV-B,O,Gate,Up,Down]`, **75,530,240 bytes**.

A fourth arm used the same first-layer mixed bits plus **qNg64 n7 on ALL actually selected layer-1 routed experts and always-on shared experts' Gate/Up/Down**, while preserving layer-1 source BF16 router, Attention and norms. It recomputed the remaining 25 layers from the changed hidden state and final full-vocabulary logits.

These are **first-layer-only** and **first-two-layer-partial** quantizations, **not all 27 layers compressed**. No genuine full-model decode tokens/s, dataset accuracy, official 480-eval, or GPU end-to-end inference was measured in these CPU C experiments.

## Strict replay and evidence

BF16 prompt #0 full logits were first produced in a 27/27 source reference process, yielding logits shape `[5, 102400]`, mean next-token self-NLL **5.297430038**, and an immutable logit SHA. A separate BF16 rerun reproduced **the exact same logit-byte SHA**, not merely a similar L2 norm. Both test prompts' BF16, uniform7 and mixed-first-layer runs were executed twice and **all matching arms' final logit SHA, final hidden SHA, top-token and NLL were identical** across independent runs.

The actual BEGLIN C `qng64_matmul_f32` bitplane kernel (not a quantization-error simulator) computed quantized modules. Source code, model index and research model are SHA-pinned in the research runner. No weights/logits are committed to GitHub; the authoritative experiment JSONs remain on the authorized research host.

## Final 27-layer logits vs original BF16

**Two self-authored short prompts** (5 and 7 tokenizer positions). The evaluation measures all positions' full-vocabulary logits, not merely a projection-layer proxy.

| Policy | Prompt #0 full-logits rel L2 | Prompt #1 full-logits rel L2 | Prompt #0 NLL nats | Prompt #1 NLL nats |
|---|---:|---:|---:|---:|
| Original BF16 | 0 | 0 | 5.297430 | 7.592266 |
| First layer uniform n7 | 0.011579 | 0.016338 | 5.239494 | 7.613499 |
| **First layer mixed [9,9,9,7,6,7,7]** | **0.004484** | **0.014005** | 5.311430 | **7.545403** |
| First layer mixed + layer-1 routed/shared MoE n7 | 0.008060 | 0.018682 | 5.251354 | 7.611679 |

The first-layer mixed policy gives **lower full-logit relative L2 than uniform n7 in both prompts at 409,600 bytes less first-layer packed weight payload**. However, **NLL and KL comparisons do NOT follow the same ranking**. Prompt #0 mean `KL(BF16||policy)` is 0.002008 (uniform), 0.000951 (first-layer mixed), 0.001466 (mixed+MoE). Prompt #1 KL is 0.001693, 0.002243, 0.001769 respectively. No universal model-quality advantage can be inferred from two tiny sentences.

All arms preserved the **last-position BF16 Top-1 token** in both prompts. Across *all* logit positions, however, the first-layer-mixed+MoE arm preserved **4 of 5** on prompt #0 (vs **5/5** for the other quant arms); all policy arms preserved **7/7** on prompt #1. Thus preserving a final Top-1 does not imply preservation of every output position.

Quantization changed MoE routing inside later layers (e.g., 15 or 24 of the 26 source-routed layers changed their Top-6 index fingerprint). This is why whole-model logits are required as the quality target instead of an isolated tensor reconstruction proxy.

## Layer-1 selected MoE weight scope

The BF16 D50 source was read-only. Expert IDs were recomputed using the **real source router for each arm's changed hidden values**; no precomputed/static expert lists were injected.

| Prompt | Actual layer-1 routed experts | n7 qNg64 projection modules including shared (3 each) | BF16 source-weight equivalent | Native packed code+scale |
|---|---:|---:|---:|---:|
| #0 | 22 | 69 | 415,236,096 B | 194,641,920 B |
| #1 | 27 | 84 | 501,743,616 B | 235,192,320 B |

The 7-bit qNg64 encoded weight payload for these **active** MoE groups costs **46.875% of BF16 weight bytes** (53.125% saved), but they were not shown to improve downstream logits; in these two probes they **increased** full-logit relative L2 compared with first-layer-mixed-only. This is not a whole-model resident/peak memory measurement; active expert sets can change with input.

## What the results support

1. **PASS:** real DeepSeek first two local quantization configurations propagate successfully to **all 27 decoder layers**, finish the full 102,400-token vocabulary output, and produce finite next-token NLL.
2. **PASS:** the preselected mixed first-layer bits reduce BF16-relative global **logit L2** compared with uniform first-layer bits on both test prompts while spending slightly fewer first-layer packed weight bytes.
3. **NEGATIVE:** adding 7-bit compression to all currently routed MoE experts in layer 1 does **not** monotonically improve model-output fidelity. Conditional expert precision is necessary, not one-size-fits-all n7.
4. **NOT CERTIFIED:** the samples have only **2 authored prompts, 12 token positions, 10 next-token targets**, no genuine held-out quality benchmark, and at most 2 partially quantized layers. Neither production quality nor final decode throughput, Metal-native full-model execution, original-XOX source attestation, and full-model exact memory budget are established.

The separately run, read-only audit rechecked all four experiment JSON SHA-256 pins, asserted rerun logit-byte equality and layer/route/size invariants, and returned `PASS_FULL27_ENGINE_NUMERIC_EVIDENCE_AUDIT_NOT_QUALITY_CERTIFIED`.

Read-only audit receipt SHA-256: `6561ff9449f998d5503dee470ff6c82fa74f51845e5f74001af4c73a9c758bdb`.

Experiment JSON receipt SHA-256:
- First-layer-only #0: `c1615b23f04eff3bfe84f0a92cbb4045d7ec9184f3c5cdabfd45eb2a6888be86`
- First-layer-only #1: `917d49f5c6b477440c75339493787b0c2d003e469c5b4c96e4e0ef940ca40ac9`
- First-layer+MoE #0: `15e263f6649b605423007639cf0bf2b2929ecb5780afdc3853482e84c6a6f6b9`
- First-layer+MoE #1: `28f396fd19635420806eb3bd5d256f3c4ddeeba30429b91a4310eb29ab2a3941`

## Engine next steps

- Build a **model-level qNg64 precision optimizer** for individual Attention/FFN/shared/routed expert matrices, using held-out next-token logits/KL/NLL, routing-change rates and *entire-model* weight-byte budgets instead of weight-reconstruction SSE alone.
- Integrate/source-verify the first-layer plus MoE native packed paths in a **non-serving BEGLIN C++/MLX build**. Existing Python+original C kernel evidence does not prove live XOX binary performance.
- Quantize and run **all 27 layers** with matched BF16, uniform and mixed weight-byte budgets; measure real prefill/decode tokens/s, true unified-memory peak, KV/weight epoch, and useful held-out quality evaluations.
- Keep `automatic_live_promotion=false` until independent quality/performance and source/model-provenance gates are satisfied.

**Branch:** `research/local-quant-budget-20261017`; **PR #63 remains DRAFT and unmerged**. Only engine research outputs were changed.
