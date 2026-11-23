# JE01 original BEGLIN native 384-position GPU forward — final evidence, 2026-11-17

## What is now proven

**Real original BEGLIN compressed DeepSeek-V2-Lite AF weights were executed end to end on XOX Metal with a scratch-only 384-position, one-slot native C/MLX binary.** This is the first real native original BEGLIN 384 forward, not the earlier unrelated MLX Chat-4bit reference. It is distinct from executing the original unquantized 31GB BF16 source.

| Actual isolated run | Prompt positions | Native first output token | Timed GPU wall | Exit |
|---|---:|---:|---:|---|
| 384 model, frozen shortest OOD prompt | 10 | 279 | 142.05 ms | PASS |
| 384 model, frozen longest calibration prompt | 217 | 853 | 3,062.19 ms | PASS |
| 384 model, calibration prefix + synthetic repeated suffix | 330 | 3572 | 4,692.66 ms | PASS |
| **384 model, 383 input positions + next token at position 384** | **383** | **3572** | **5,507.27 ms** | **PASS** |
| Existing operating-stage 32-position binary, same real OOD prompt | 10 | 279 | 125.10 ms | PASS |

Each actual GPU run loaded the complete 27-layer AF model, successfully bound **269/269 native tensors**, executed one-slot B=1 with per-step prefill budget 1, and exited 0. The 330/383 extended cases were **synthetic stress tests** that repeated the final native token of an already frozen calibration prompt, NOT official heldout-quality scores.

## More convincing native compatibility: 16/16 frozen calibration inputs

The exact original 512 native inputs were independently reverified **512/512** on XOX, using the original sealed 512-input receipt SHA **2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34**. The XOX-local destination input manifest index receipt is **0aef144a8d549aa9e33c3a9fd9e899e0301e826bd563ceb92ad541cff1bbf768**.

Sixteen different short **real calibration** prompts were selected by native length and input ID only, **no gold labels**, with input lengths 15–25 positions, 355 total positions per model. They were run through the existing live-worker-stage binary in a SEPARATE process and through the new 384 research binary on identical native input files.

**All 16/16 first output token IDs matched:**

[11, 279, 2238, 31460, 330, 72672, 185, 1501, 245, 11, 457, 285, 327, 20171, 889, 68307]

Both isolated processes completed successfully with 27 native layers and 269 AF tensor bindings; the operating route and actual worker PIDs were unchanged. The 384 arm processed 355 positions in one timed 4,957.54 ms pass (~71.608 tokens/s), the existing 32 arm in 4,444.62 ms (~79.872 tokens/s). These are **single-run timing observations, not controlled evidence of a stable performance difference**, because build/runtime ABI and JIT environments differ.

[All 16 per-input actual outputs, both run receipts and provenance](JE01_ACTUAL_XOX_NATIVE384_VS32_16_REAL_CAL_PARITY_20261117.json), canonical SHA **d10f987e57d802dbd86e29f8c2576e215f38cda10abbff5f21fe6119277674fd**.

## Binary provenance and MLX ABI repair

The previously built hosted ARM64 MLX-0.32.1 research binary, SHA **bd7d79661d25370d50529b92cddaa3871cd60be6211263dbc29f35aeacd35b6f**, was downloaded into scratch on XOX and fully SHA-checked. It initially failed BEFORE model load because its C++ MLX ABI differed from the XOX resident libmlx.dylib. No installed library or operating binary was replaced.

A new local research-only source tree was exported from the existing clean XOX BEGLIN git commit **e827b522959a5bb01852686c76c4e65ac16b63ba**. Strict SHA/exact anchor transforms set CPU and MLA GPU position limits to 384, restrict all four run-gate entry points to B=1, prevent multi-column layer calls (A=1), and give the binary a unique name. Build against XOX's **existing** MLX C++ wheel and rpath succeeded.

Actual executed local binary SHA **60a0b559496180d29b65d5a9cb86bba4d16a12238a140cdfdd157602fcc90437**, file name qwen_infer_gpu_je01_context384_xoxabi. Source-stage receipt SHA **aa2ccc6090644373b09f0e96d06dff17ecad3bbf219b7fea8f770a52f1231656**. The exact [host-local source stage recipe](../../tools/je01_xox_original_localmlx_native384_stage.py) and [memory/production-health guarding native runner](../../tools/je01_xox_original_localmlx384_safe_shadow_runner.py) are committed to this Draft PR for reproducibility; running them on another machine requires correct host-specific paths and a new safety assessment.

## Complete original compressed AF file SHA256 provenance

Five actual XOX original AF files were streamed read-only and fully hashed with SHA256; no 9.8GB EOE copy was needed.

| Native source name | Bytes | SHA256 |
|---|---:|---|
| deepseek_moe_af.bin | 9,814,343,680 | 9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4 |
| deepseek_moe_f32.bin | 14,137,344 | 1cc5386c8b8773ee802e059be22f0b7b317dacfc3c9bbf0b724b4e411b596033 |
| layout_af.txt | 25,625 | ba3898f3bf366b1aec8471d9896885f101bf64b78c816828ea82bd7d58851c3e |
| layout_f32.txt | 6,695 | 2ecc5fcb2b4d26302fdd26e9f7d9167811b431a5649d467fa5e5912dbb5a5e4c |
| arch_config_moe.txt | 372 | 43f205631553478b204f761d5894aaeec67100ed276aca762a24c62e63a91c50 |
| **Total** | **9,828,513,716** | Combined source receipt **8399be688722e6732c67ed9c0518ff808403e3ca5d82597aad7407906b915870** |

The compressed AF SHA values must **not** be equated with the unquantized 31GB original BF16 checkpoint canonical identity **36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc** without a separately verified mapping.

## Operating safety and remaining research gates

Research runs took place exclusively under XOX's isolated research scratch directory: /Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original384_20261117 . Each subprocess checked healthz=ok, route generation 6, active route beglin-candidate-shared-up-l3-n6 and unchanged worker PIDs **35481, 35493** before/during/after. The runner sampled memory pressure and worker identity every second, terminated its own child on free memory below 35%, worker changes or timeout. Lowest free memory across the 384 tests **47%**, and for both 16-prompt arms **48%**. No original source, weights, worker process, operating route or release changed; production worker PIDs were conserved.

The canonical machine-readable [real source SHA, model run, input and short-prefix receipt](JE01_ACTUAL_XOX_ORIGINAL_AF_NATIVE384_5_RUNS_20261117.json) has SHA **3bef35d1fc892e651df74dc03f5bf9bda31824753411e2bfccf2af118d403ec0**.

**What has NOT been demonstrated:** exact numerical parity of all 102,400 original model logits between old32 and new384, real BF16-source end-to-end hidden state/logit parity, full **512/512** original native model inference (18 official prompts were run, plus 2 synthetic stress prompts), JE01 preregistered teacher-forced gold-token NLL, full original routed Expert posterior, actual equal packed-byte Bayesian vs uniform6 comparison, and any model-quality gain. Only the first output token is compared for parity. Keep PR #94 Draft, no merge/promotion.

**Decision: ORIGINAL_BEGLIN_NATIVE384_FORWARD_PASS / ACTUAL_384_BOUNDARY_PASS / 16_OF_16_REAL_CALIBRATION_GREEDY_TOKEN_PARITY_PASS / SOURCE_9.8GB_FULL_SHA_PASS / ORIGINAL_TEACHER_FORCED_NLL_HOLD / PRODUCTION_UNCHANGED.**
