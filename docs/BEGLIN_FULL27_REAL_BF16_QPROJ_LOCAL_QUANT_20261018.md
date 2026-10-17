# BEGLIN P18 — Real 27-layer DeepSeek BF16 versus qNg64 CPU Q-projection inference

Date: **2026-10-18 (KST)** · Status: **research-only, Draft PR** · No model/serving/infra mutations.

## What was actually executed

A source-defined **DeepSeek-V2-Lite 27-decoder BF16 reference** was streamed from a read-only D50 research copy, loading only the experts selected by each layer's **real source MoEGate** (64 experts total, top-6 per token). The source-defined final RMSNorm and entire 102,400-vocabulary `lm_head` were executed in bounded BF16 chunks, producing true final logits and teacher-forced next-token self-NLL for short self-authored text.

A parallel CPU inference experiment replaced **only each layer's attention Q projection** (27 tensors, all 3072×2048) with real BEGLIN C qNg64 encoded **bitplanes and float32 scales**, executed through the isolated NEON/Apple Accelerate transient-64-row-tiled GEMM. Outputs were cast to BF16 at the same source module boundary. Every other weight — K/V-A, K/V-B, O, FFN, MoE router, shared and routed experts, embedding, norm, and lm_head — remained **original BF16**.

All four trajectories used exactly the **same token IDs and source model checkpoint**. Expert routing was recomputed from each trajectory's own hidden activation, and each layer loaded the union of actually used BF16 experts. The BF16 trajectory's entire logits matrix SHA-256 was verified byte-for-byte against the previously completed independent BF16 baseline for **each holdout prompt**.

This is a **27-layer model with q-projection-local quantization**, *not* full-model all-weights quantization, production Metal inference, or a 480-item quality benchmark.

## A. BF16 source reference and 27-layer calibration

- Original BF16 calibration prompt 0, 5 tokens: all **27/27 layers PASS**, logits **[5,102400]**, average teacher-forced self-NLL **5.297430**, top-1 token ID **429**. Receipt SHA-256 `3de1973704e0364ebf206ce51e54ed48d2a5906bea735df05b0fb21b661051fc`.
- On this **calibration prompt only**, each of the 27 Q projection tensors was encoded/executed at **5, 6, 7, 9 bits** (108 real C projection probes). Each local Q projection BF16-relative squared error was used in an exact additive-budget Pareto search. No holdout logits were used to select bitwidths.
- Under **exactly the uniform 7-bit Q projection payload** of **159,252,480 bytes** (codes + one F32 scale/64 weights), the best measured local-Q-error policy was **uniform 7-bit across all 27**: there is **no same-byte mixed-precision win on this calibration metric**.
- At **+2% Q-projection-only budget** (162,398,208 bytes), the train-selected policy upgraded layers **2 and 7** to 9 bits. At **+5%** (167,116,800 bytes), layers **2, 3, 5, 7 and 18** were upgraded to 9 bits. All remaining 27 Q-projections remained 7-bit. Strictly the +2/+5 policies have **larger** Q-projection storage budgets, although their other BF16 model weights are unchanged.
- Uncompressed BF16 weight payload for these 27 Q projection tensors alone = **339,738,624 bytes**. Uniform packed qNg64 n7 = 53.125% fewer Q-projection weight bytes. This excludes the rest of the model and runtime activation/KV memory.
- Calibration full logits SHA checked against original independent reference; no drift.

Calibration receipt SHA: `8319e8f4ed455441d8cb28ec50ce4d4584f2fec84c739e7ed61a84f981fe9466`.

**Frozen before independent inference holdouts** policy receipt SHA: `4358df662376b615100bb46bf29aa1884de2cd9ac100dc5c4f7c96aeb19b3431`.

## B. Three independent *short prompt* holdouts: final logits and NLL

All three holdout prompts were distinct from calibration prompt 0 and were evaluated after the bit choices were frozen. Each arm executed all 27 layers and the complete `lm_head`. For each holdout, the independent original BF16 logits SHA was reproduced **exactly**.

| Original BF16 self-NLL versus quantized Q projections | Prompt 1 (7 tokens) | Prompt 2 (8 tokens) | Prompt 3 (10 tokens) | Mean of 3 prompts |
|---|---:|---:|---:|---:|
| Original BF16 | 7.592266 | 5.381042 | 4.392442 | **5.788583** |
| Uniform Q n7 (strict equal-Q-byte reference) | 7.569014 | 5.390790 | 4.431130 | 5.796978 |
| Local Q +2% budget | 7.533412 | 5.363383 | 4.398029 | **5.764941** |
| Local Q +5% budget | 7.530890 | 5.383953 | 4.404769 | 5.773204 |

The three prompts contain **25 token positions** in total; the autoregressive self-NLL uses **22 predicted target positions**. The token-weighted NLL averages are BF16 `5.579676`, Q n7 `5.592263`, local +2% `5.560291`, local +5% `5.568906`.

- Top-1 agreement at all 25 token positions: uniform Q n7 **24/25**, local +2% **25/25**, local +5% **25/25**. All three variants preserved the *final-position* top-1 ID in all three prompts.
- Average whole-prompt logits relative L2 to BF16: Q n7 **0.014507**, +2% **0.013151**, +5% **0.013138**.
- NLL improved vs BF16 on **1/3**, **2/3**, and **1/3** prompts respectively for Q n7, local +2%, and local +5%. **The per-prompt improvement is not universal.**
- The +2% policy has the lowest three-prompt mean NLL and a lower average logits L2 than uniform Q n7 **on these three tiny authored prompts only**. This is **NOT statistical proof of model task-quality improvement** or a production-ready quantization policy.
- Quantization changed some source MoE TopK expert assignments; the engine did not pretend routes remained fixed. Actual source BF16 expert weights were lazily loaded for the routes selected by each trajectory.

Paired 27-layer inference receipts:
- Holdout prompt 1: `f044513470379327f6fb512a408bf173faeee75fae90efb38047638e290479c4`
- Holdout prompt 2: `44953c95d27c30d348fb43ffb5bd75a12e07e899f470475050d2d22a4d9f4fb7`
- Holdout prompt 3: `b3f4c0bef8d775edfc49b5e185f86eb8c81fea3b73e7c974b4087d7d376a8f4d`
- Combined frozen-result summary: `38fc7018fb27657e79fed59933c078d4fd24c5ff5649743dd454caf05adf4320`

## C. Native engine implementation

`research/local_quant_cpu_metal_20261017/qng64_tiled_cpu_linear_research.py` implements a reusable BF16 CPU PyTorch `nn.Module`:

- Quantizes a source BF16 rank-2 weight matrix ONCE into native C qNg64 group-64 signed n-bit planes and F32 scales.
- Calls `beglin_qng64_tiled_cblas_matmul` on BF16 hidden inputs via a bounded 64-row temporary F32 tile and Apple Accelerate SGEMM; returns BF16 outputs at the source projection boundary. No persistent full FP32 dense-weight cache in the module.
- Validates code/scales byte sizes, n=2..15 except distinct q4g64 n4, input dimensions/dtype, dynamic library pin, finite scales/output, and fails closed.
- Companion `test_qng64_tiled_cpu_linear_research.py`: **5/5 local EOE Apple Silicon CPU tests PASS** with independent bitplane-decoded FP32 numerical oracle, invalid arguments/dimensions, packed bit mutation/recovery. This is **isolated research code** and is *not* wired into the production BEGLIN C++ serving engine.
- Native encoder source SHA-256 `cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956`.
- Research tiled-C source SHA-256 `1d409f9ead007471960246c20eb742edd98301967cac689f3ace50b4db07f68d`.

## D. Explicit limitations / next engine gates

1. Only **Q projections** across all 27 layers were replaced; the full MoE routed FFN weights/other attention and final head remained BF16 in this 27-layer trial. The separate 2-layer full-active-MoE Metal tests are a **different limited experiment**, not additive proof of full-model all-weight compression.
2. Only **one calibration prompt and three tiny holdout prompts** (22 self-prediction target positions) were tested, not an established QA task suite or frozen 480 benchmark. No statistical significance or general task-quality win.
3. CPU tiled C inference was verified but **warm actual end-to-end prefill/decode tokens/s, peak device memory and power were not compared**. This validation run computed multiple model trajectories inside one process, and elapsed wall time is not a standalone per-policy throughput benchmark.
4. The D50 model copy was used READ ONLY. The original XOX source issuer attestation remains unavailable; the running XOX BEGLIN backend binary has not been changed or freshly shown identical to the research code.
5. Before merging, add source-identified native backend compatibility tests and compare BF16 / uniform / same-total-model-byte mixed precision across **all actual model weight roles** (Q/KV/O, shared/routed MoE, embeddings/head where supported), plus NLL/task evaluations on a meaningful held-out suite.

**Release:** Draft PR only; `automatic_live_promotion=false`, no live engine or infrastructure change.
