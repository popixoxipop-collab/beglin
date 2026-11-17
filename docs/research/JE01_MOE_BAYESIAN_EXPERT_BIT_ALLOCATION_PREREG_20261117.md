# JE01 — Actual routed-MoE Expert Bayesian precision allocation (BEGLIN)

**Preregistration / new independent research, 2026-11-17 KST.**
**Research code only. No native MoE perturbation observations, no checkpoint, Router or production edits.**
**Default decision: HOLD / preserve baseline.**

## Why this is a distinct research branch

The Joint-Expert Bayesian B3/B4 study did NOT demonstrate useful Expert cross-covariance (0/6 supported pairs); B5 independent Expert posterior maps passed synthetic/numerical checks, but were not pretrained-weight posterior distributions. B9/B10 Student-t and conformal calibration tested image-model feature uncertainty, not BEGLIN.

The BEGLIN Qv21 experiment, although actually run in original Qwen28 C, involved **two precision masks** on a dense Qwen2.5-1.5B model, not routed MoE Experts. Its 96-case native 5-fold gate officially FAILED: source NLL 1.679052, random73 NLL 1.808085, dynamic 1.691013, selected severe loss 3/18. See [Qv21 original native negative audit in B11 Draft #93](https://github.com/popixoxipop-collab/beglin/pull/93). JE01 must not reuse its calibration/evaluation as JE01 validation.

BEGLIN's README and main engine document per-(layer, expert_id) precision selection, routed FFN role quantization and real DeepSeek-V2-Lite E=64 top-k gather. **JE01 targets those actual routed Experts**, not Qv21 mask variants.

## Hypothesis and parameterization

For routed Expert e and measured byte tier b, estimate the distribution of per-case signed gold-token NLL perturbation relative to the immutable baseline assignment. Use **one independent Normal-Inverse-Gamma scalar posterior per (e,b)**, with prior mean 0, kappa 1, alpha 2, beta 1. The predictive distribution is Student-t. This is only a posterior over the *local paired NLL effect estimate*, NOT an epistemic posterior over the pretrained neural weights.

Each alternative is scored conservatively with Bayesian location + 2 × Student-t predictive scale; this 2-scale term is a *fixed risk surrogate* (NOT a calibrated 95% confidence limit without independent validation). If a candidate has fewer than 32 independent routed calibration observations, missing paired NLL, nonfinite values, or any observed router flip, mark the tier ineligible.

Optimization is exact-byte multiple-choice DP over at most **8 frozen selected Expert slots**:

- Each Expert chooses one of its actually measured packed-bit tiers;
- sum of measured selected packed bytes must **exactly equal** immutable baseline bytes for those same slots;
- all other model weights, Expert slots and layer routing remain unchanged;
- minimize the sum of independent Expert conservative loss scores; require predicted reduction at least 0.15 NLL (pilot screen only);
- no cross-Expert covariance or additive model quality claim is made until full MoE-forward validation.

The pure planner is in tools/je01_bayesian_expert_allocator.py and is intentionally capped at eight slots. It does not run inference, edit precision maps, or activate code. Decision output is always SHADOW_ONLY_NO_WEIGHT_EDIT.

## P0 — native grounding, not yet executed

Pin exact full-model SHA, tokenizers, BEGLIN engine commit, compiled reference binary, backend (CPU/Metal), rotary/routing configuration, output-logit parity, baseline per-slot packed bytes, and data source hashes. Proposed real model family is BEGLIN-supported DeepSeek-V2-Lite MoE, but **the exact checkpoint artifact has not been pinned in JE01**.

A separate, isolated instrumentation build must collect actual top-K routed Expert IDs, pre-change router score gaps, Expert activation norms and per-question gold-token NLL with no modification to the immutable baseline path. No arbitrary high-level "Expert trust" score is accepted as a native observation. Token/logit parity with original backend must PASS before training candidates.

Freeze the **8 pilot Expert slots by pre-label routing frequency on a new unlabeled calibration split**, not by whichever candidate later improves loss. If any missing pinned source, model, byte measurement, parity receipt or routing trace: STOP.

## P1 — new evidence and split

Proposed, to be frozen before any native scoring:
- Source model inputs: independent clean text prompts from WikiText-103, not Qv20/Qv21 ARC/OpenBookQA cases.
- Calibration for Expert-effect posterior: 256 unique inputs, with each tested (e,b) requiring at least 32 real routed Expert activations. Unmeasured tiers remain ineligible.
- Heldout same-source: 128 different inputs. Cross-source OOD: 128 from HellaSwag validation (scored on compatible gold-token NLL construction).
- Source file bytes and content ID/overlap hashes, tokenizer construction, per-alt exact packed byte receipts, seeds and resulting cohort IDs must be immutable before gold results.
- Independent auditor must verify each paired baseline/candidate native score and that the candidate Expert was actually routed on that prompt. No oracle selection using heldout labels.

These named source datasets and counts are **proposed and NOT yet SHA-frozen**. If the exact source partitions lack enough routed observations, record NOT_FEASIBLE rather than silently adjusting thresholds or using the old Qv21 heldout.

## P2 — full pretrained model validation required

Compare (a) frozen baseline bit map, (b) independent Bayesian DP shadow plan, (c) equal-byte random-permutation control, (d) constant-tier frequency heuristic. Run **actual full pretrained MoE forward**, not an additive surrogate alone, on all heldout and OOD samples. Apply identical question IDs, prompt tokens, backend, generation/scoring mode, CPU/thread count and permitted memory ceiling; measure both full-vocabulary paired gold-token NLL and router top-K flip counts. GPU latency must be measured independently under equivalent concurrent load.

Adoption proposal, ALL required:
1. Every source and checkpoint SHA, sampled slot ID and exact packed byte identity independently audited.
2. End-to-end heldout NLL mean improvement against baseline ≥0.02, paired bootstrap 20,000 repeats lower 95% bound >0.
3. OOD 128 mean NLL non-regressing, paired bootstrap lower 95% bound for gain >=0.
4. Zero observed hard router top-K mismatches relative to baseline, no NaNs/unsupported kernel paths.
5. No worse than 5% p95 token latency under hardware-parity conditions.
6. Predicted gain must not be explained by equal-byte random-permutation placebo; familywise multiple-testing correction for any selection among multiple plans.
7. Independent reviewer certifies measurements BEFORE any production or operating-release change.

If any gate fails, keep unchanged baseline. Never use this pilot planner to perform a production release activation. B12/JE02 cross-Expert covariance is out of scope until co-routing evidence and negative controls support nonzero interaction. This is intentionally a **new preregistration**, separate from the negative Qv21 and the B9/B10 image-model results.

## Current engineering artifact

Research-only implementation: tools/je01_bayesian_expert_allocator.py.
Synthetic contracts: tests/test_je01_bayesian_expert_allocator.py.
This code can produce an exact-byte proposed bit layout with independent Student-t risk scores on synthetic observations. Synthetic CI is **not** native-model evidence.


## 2026-11-17 P0 audit follow-up (observations, not a change to preregistered test gates)

DeepSeek-V2-Lite's official [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json) specifies 64 routed Experts with Top-6, two always-on shared Experts, a dense first layer, and no normalization of selected Top-K probabilities. BEGLIN's native GPU backend is Apple MLX/Metal; an Alienware CUDA evaluation is a separate reference and **not native BEGLIN parity**.

The BEGLIN eager native interface mlx_gpu_layer_step() can return the selected Expert IDs and weights, and mlx_gpu_risk_signal_telemetry_get() returns a risk summary. But these existing outputs alone cannot reconstruct all 64 pre-Top-K scores. A separate **isolated, non-production instrumented reference build** is still required to emit all 64 softmax values, selected IDs, and SHA-bound per-question records, then show baseline-logit parity before any candidate calibration.

New shadow-only source:
- tools/je01_checkpoint_attest.py: hashes indexed safetensors shards, config, tokenizer, binary, and engine git commit with path traversal and symlink-escape checks. Without externally pinned identities it reports discovery only, never native parity PASS.
- tools/je01_router_trace_audit.py: rejects outcome labels, invalid 64-probability normalizers, top-6 mismatches, duplicate/unfrozen prompts, forged alternative traces; freezes eight pilot Expert slots by independent prompt routing counts, no gold outcomes.
- tools/je01_bayesian_expert_allocator.py: limited to direct expert precision tiers 2/3/4/5/6/7/8/16/32, rejecting layer 0 (dense). The BEGLIN shipped routed Expert precision default is int8; baseline=4 in synthetic fixtures is not a real model baseline assertion.

CI: 43/43 offline synthetic tests PASS at [GitHub Actions run 43983912466](https://github.com/popixoxipop-collab/beglin/actions/runs/43983912466). CI is NOT real DeepSeek model inference.

Fresh Tailnet Commander current/live remained g6-0.9.0-operator.900484. BEGLIN-native Metal GPU resource_plan recommended no host: EOE lacked reservable RAM and CPU; XOX lacked RAM, CPU and eligible storage pool. No GPU job was run.

XOX read-only *admitted* metadata job job_900786c808f71b9c2fe86774941785f6 finished SUCCEEDED, exit 0. Its inspected standard HuggingFace hub contained the matching model name models--mlx-community--Qwen2.5-1.5B-Instruct-4bit, but no DeepSeek name in that directory; this does not prove no DeepSeek exists elsewhere. EOE Hugging Face cache file reads were denied due to a protected symlink component; directory search and CPU work were refused by existing fleet admission. Therefore a local DeepSeek checkpoint, entire safetensors SHA and tokenizer are NOT YET PINNED.

Final state: P0_OFFLINE_GUARDS_PASS / P0_ACTUAL_MODEL_SHA_NOT_VERIFIED / P1_NATIVE_ROUTER_TRACES_NOT_RUN / P2_FULL_FORWARD_OOD_NOT_RUN / NO_PROMOTION.

The original JE01 method, controls, target sample counts 256/128/128, thresholds and safety rules remain unchanged. New scoring is blocked until the scheduler authorizes actual model-cache inventory, the real full-checkpoint identity is sealed, and an isolated native route trace is proven parity-equivalent.


## P1 input-only row-slice freeze — 512 real inputs, 2026-11-17 KST

**Confirmed advance:** The XOX managed read-only dataset job job_d5d3ec2b9c494eb826577ef79bcf3868 completed SUCCEEDED/exit 0 in 3722 ms (0.3155 process-tree CPU core-seconds; peak RSS ~28.47 MB). A SHA-verified copy of tools/je01_data_freeze.py (Git blob 00c2c299eb7bc93aa0f9be87efc5fde2a1f75437) queried the public Hugging Face dataset-viewer API. No real DeepSeek weights or loss labels were used for selection, and no model was run.

Committed provenance: [JE01_FROZEN_WIKITEXT_HELLASWAG_512_20261117.json](JE01_FROZEN_WIKITEXT_HELLASWAG_512_20261117.json). Manifest canonical SHA256 **a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147**.

| Frozen source | Version x-revision | Input count | Row cohort SHA256 |
|---|---|---:|---|
| WikiText-103-raw training | b08601e04326c79dfdd32d625aee71d232d685c3 | **256 calibration** | ed59b82a8c8cd5effc5a7087e41003fe76470ded1aef24cb85398e9b677f1a07 |
| WikiText-103-raw validation | b08601e04326c79dfdd32d625aee71d232d685c3 | **128 holdout** | fa98276743518ec452fbe70463ce6e997ba026d84e95b7f576aae86f437adf3c |
| HellaSwag validation | 218ec52e09a7e7462a5400043bb9a69a41d06b76 | **128 OOD holdout** | e8dc68aee55a5ca90c654418c7e958012dedd74234d3ade35db75e3855d7fe1a |

All 34 queried API pages returned exact frozen x-revision, partial=false and expected row indexes. Each page-response SHA, selected prompt SHA and deterministic row-index ID is stored. **All 512 selected source IDs and prompt IDs are unique.** HellaSwag gold label was never accessed for choosing rows.

Recorded raw-page limitation: 10 predetermined WikiText validation 100-row windows contained a 22-row overlap. This predeclared sampler was not changed after the result. The selected row/prompt ID guard ensures the actual selected cohorts are disjoint; candidate counts are per-page records and can include overlap.

**Material incomplete gate:** The full underlying Parquet file-byte SHA was not independently measured, even though selected page-response and prompt SHA values are sealed. The sampler must be described as a *row-slice freeze*, NOT as complete full-source-byte attestation. The real DeepSeek indexed checkpoint SHA, MLX backend parity, Expert routing trace, per-Expert paired NLL, full-model improvement and latency remain NOT_VERIFIED. Do not promote or edit baseline Expert weights.

Supporting source/tests: tools/je01_data_freeze.py; tests/test_je01_data_freeze.py. Native run ID job_d5d3ec2b9c494eb826577ef79bcf3868. Current decision: **P1_ROW_IDS_512_FROZEN / P1_FULL_PARQUET_SHA_NOT_VERIFIED / P0_NATIVE_MODEL_SHA_NOT_VERIFIED / P2_FULL_FORWARD_NOT_RUN / NO_PROMOTION**.


## P1 completion supplement — four complete Parquet SHA256, same day

**Resolves the preceding row-slice-only limitation for the four source files actually used.** XOX managed job job_bda3b78a6c5c3c60db4f2a3317d72395 (SUCCEEDED, exit 0, ~29 seconds) streamed **321,049,738 bytes** in RAM from the exact immutable Hugging Face dataset revisions without writing source datasets to disk. Each file passed exact byte length, both start/end Parquet magic bytes, and a streaming full-file SHA256. The original GitHub research-only source was git-blob-verified prior to execution. External source data was treated as untrusted; no labels were used in selection and no model was executed.

New independent attestation: [JE01_FULL_SOURCE_PARQUET_SHA_20261117.json](JE01_FULL_SOURCE_PARQUET_SHA_20261117.json); SHA256 of canonical receipt **0924b13ac6022bf18e895c76dbd1498432079cb73c69e43e00fe01b9f63226d5**, bound to the existing frozen 512-row manifest hash a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147.

| Source file | Full bytes | Independently calculated SHA256 |
|---|---:|---|
| WikiText-103-raw train shard 0/2 | 156,987,808 | 74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212 |
| WikiText-103-raw train shard 1/2 | 157,088,770 | ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720 |
| WikiText-103-raw validation | 657,209 | 204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c |
| HellaSwag validation | 6,315,951 | 899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd |

The last two hashes additionally match Hugging Face's published per-file pointer SHA. Full-file hashing here covers exactly these 4 pinned files used for 256/128/128, not every split/config in the public dataset repository.

**Current corrected gate:** P1_SOURCE_FILES_SHA_PASS + P1_512_FROZEN_ROWS_PASS; still P0_ORIGINAL_DEEPSEEK_CHECKPOINT_UNVERIFIED, P0_ORIGINAL_MLX_PARITY_NOT_RUN, P2_PAIRED_EXPERT_NLL_NOT_RUN, and NO_PROMOTION. The earlier row-slice-only observation remains a valid chronological record, but is now superseded for the four underlying source files by this SHA attestation. No BEGLIN runtime or operating release was modified.

Implementation: tools/je01_parquet_source_sha_audit.py; offline contract tests: tests/test_je01_parquet_source_sha_audit.py.


## P0 actually recovered native model — 2026-11-17 KST, corrected checkpoint status

**A previous source-missing assumption was disproved by managed read-only host inspection.** EOE D50 has a full original DeepSeek-V2-Lite BF16 indexed safetensors checkout at /Volumes/D50/deepseek_v2lite_bf16_safetensors; an alternative full source exists at /Volumes/D50/vdsp/deepseek_v2_lite_hf. XOX has both the same four-indexed-shard model files at /Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors and a BEGLIN proprietary compressed model directory at /Users/xox/vdsp_local_data/moe_base_deepseek/weights_moe. This is **not** a reason to download another 31 GB copy.

- EOE managed read-only directory verification job: job_31e5ad7cd492cfdc4a82824c809d8330, SUCCEEDED exit 0.
- EOE full 31,413,626,576-byte BF16 SHA audit job: job_dc3bfed1b7bb31276834abc56e39dd70, SUCCEEDED exit 0, ~19.7 seconds. The running code was committed before execution at tools/je01_native_bf16_sha.py, Git blob SHA ed4a072d738df353f5c21b62ac0220fe5868d6de.
- **Original BF16 checkpoint content identity SHA256:** 36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc; reproducible evidence: [JE01_REAL_BF16_CHECKPOINT_SHA_20261117.json](JE01_REAL_BF16_CHECKPOINT_SHA_20261117.json). Receipt SHA: 689c78abe0280c1cc961222f8c8601e54ad89c97641aeee62ef3c4e05951fe7b.
- All four BF16 shard lengths and complete SHA256 hashes are present, plus pinned config, index, tokenizer SHA. Tokenizer SHA 41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669 **matches previous real QV26B DeepSeek experiment**. Both independent D50 model directories' config and index hashes match.
- This original BF16 checkpoint identity is different from the already-deployed *quantized* XOX serving checkpoint identity 1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898; **no equivalence between compression output or per-Expert NLL can be inferred from these two SHA values.** Actual quantized checkpoint source/byte identity and original BEGLIN Metal logit parity remain unverified in JE01.

Actual relevant prior native-model control from EOE Downloads report BEGLIN_QV26B_DEEPSEEK_SHORT_HOLDOUT_2026-11-11.md, SHA 23b86f4d2e327b2be8faaa380f2e398729994b04373481f6900158b46fe4ac16: XOX isolated BEGLIN DeepSeek-V2-Lite workers previously ran 64 paired short Y/N prompts under an immutable **MOE_CBATCH_MAXPOS=32** model position budget. Both fixed shared_up_proj layer3 n6 and n5 yielded 31/64 correct and did **NOT** justify a precision change. This is a real native DeepSeek test, but it is a **single shared projection**, NOT a test of independently allocated routed Experts or JE01 NLL effects. Its prompts were limited to 12..28 tokens plus up to 4 new tokens; the originally planned QV26 longer ARC/OB prompt evaluation failed at the 32-position limit. No extrapolation to the current 512 JE01 frozen WikiText/HellaSwag inputs.

**New native feasibility blocker identified:** Before using the 512 frozen corpus on the original running BEGLIN binary, independently attest the exact tokenizer-derived input lengths and whether each frozen prompt can be scored with 32 positions, or certify a separate isolated longer-context backend with uninstrumented-baseline logit parity. DO NOT silently truncate or relabel the pre-registered inputs. A new alternate short-context conditioning policy must be preregistered before use. New deepseek route metadata should report actual 64-way router scores, not infer Expert identity from a proxy.

**Corrected current gate:** P0_BF16_SOURCE_CHECKPOINT_SHA_PASS / P0_XOX_EXISTING_QUANTIZED_MODEL_FOUND / P1_512_INPUTS_AND_FULL_SOURCE_FILES_SHA_PASS / P0_QUANTIZED_BINARY_PARITY_NOT_VERIFIED / P2_NATIVE_JE01_ROUTER_AND_PAIRED_NLL_NOT_RUN / NO_PROMOTION. Existing serving route and immutable Tailnet operating release unchanged.


## P2 original DeepSeek token-window feasibility — 2026-11-17 KST

The original BEGLIN production-compatible DeepSeek persistent GPU worker in the 2026-11-11 QV26B source had a **hard 32-position ragged/cbatch limit**. Its own frozen QV26 long prompt trial failed at 80 tokens. A new real-model tokenizer-only prereg replay of the *existing frozen 512 inputs* was run on EOE in the managed read-only job job_809788fb70b00589c74b5f0538d2469a, SUCCEEDED/exit 0, using the full actual BF16 DeepSeek tokenizer JSON verified at SHA 41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669. This exact run re-fetched all 34 pinned source pages, regenerated all 512 previously selected SHA-identical source rows and prompts, and **used no answer/gold labels**.

[Actual token-window receipt](JE01_ACTUAL_512_NATIVE_TOKEN_FEASIBILITY_20261117.json) has canonical receipt SHA **b0da7e361afbad5faae69f86d08651a21e5c6f95ffa15e7eab3bec51a7401064**.

| Frozen cohort | Total | Raw input plus assumed BOS and one gold target fits 32 | Does not fit 32 | Median raw tokens |
|---|---:|---:|---:|---:|
| WikiText calibration | 256 | 24 | 232 | 101 |
| WikiText heldout | 128 | 6 | 122 | 94.5 |
| HellaSwag OOD | 128 | 24 | 104 | 57 |
| **Total** | **512** | **54** | **458** | — |

This is a *feasibility-only* precheck, NOT 54 real forward passes or real Bayesian NLL scores. It is a conservative lower-bound fit based on the HF tokenizer without special tokens plus one assumed BOS and one next-token gold; exact BEGLIN native token ID parity and BOS semantics remain unverified. For HellaSwag, a full ending answer would require additional positions beyond this prompt-only count, so even 24/128 is an upper bound for an untruncated multi-token target.

**Important source-level correction:** BEGLIN native code distinguishes MOE_MAXPOS=160 (single-sequence CPU path) from MOE_CBATCH_MAXPOS=32 (ragged/cbatch), and mlx_moe.cpp has separate MLA_L0_MAXPOS=32 for Metal's eager attention path in the inspected QV26 source. Increasing MOE_MAXPOS alone does not fix persistent GPU cbatch; raising MOE_CBATCH_MAXPOS blindly can substantially enlarge batch K/V storage and risk production memory. There is no evidence that simply changing one integer creates a safe JE01 216-token Metal model.

**Gate decision:** keep the originally preregistered 512 IDs and prompts exactly as frozen. Do not silently truncate, select only the 54 short prompts, or rewrite scoring labels. Existing 32-position persistent worker is officially **NOT_FEASIBLE for the full P2 protocol**. The next allowed engineering task is an **isolated, uninstalled longer-context native BEGLIN reference path** which (a) can score at least 216 raw input tokens plus target, (b) independently passes identical short-prefix logit parity and exact tokenizer IDs versus original current native worker, (c) records exact precision-map and Expert route telemetry without production writes, and (d) completes a clean resource-plan admission and memory-cap check before any heldout scores. If it cannot be made resource-safe, keep JE01 HOLD, do not weaken the predeclared sample counts or NLL thresholds.


## P2a isolated M1 MLX 384 real allocation — 2026-11-17 KST

After sealing the full original 512-input gold/ending context lengths (max 331 BOS-inclusive, so **384 positions required**), a distinct, research-only 384-position / 1-slot native BEGLIN target was built and tested on a **real Apple Silicon M1 GitHub-hosted macOS-15 runner**, without altering original model weights, production binaries, live routes or operating releases.

- Original model checkpoint SHA retained: 36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc.
- Frozen input manifest retained: a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147.
- Native BEGLIN source is copied from a *clean, exact Git-tracked commit* into an entirely separate scratch checkout (tools/je01_isolated_context384_stage.py). Changes are confined to the scratch copy: MOE_MAXPOS and ragged/cbatch MOE_CBATCH_MAXPOS=384, deepseek MLA_L0_MAXPOS=384, all four C cbatch launch gates B=1, MLX batch and ragged A=1/slot0 only. GQA architecture constants and original source files are unchanged. An isolated-only C ABI function calls the **actual original** lazy fused MLA K/V initializer. CMake produces a distinct qwen_infer_gpu_je01_context384 arm64 binary; no system-install or deploy target is invoked.
- Real source-code contract tests at the time of first native arm64 build: 66/66 PASS, with new native-device and receipt tests subsequently added.
- Native M1 MLX 0.32.1 **actual GPU memory materialization**, [Actions run 44017766019](https://github.com/popixoxipop-collab/beglin/actions/runs/44017766019), [sealed receipt](JE01_ACTUAL_METAL384_DEVICE_KV_20261117.json). The **isolated native binary SHA256 is bd7d79661d25370d50529b92cddaa3871cd60be6211263dbc29f35aeacd35b6f**.
- The DeepSeek dimensions use 32 allocated layer slots (exact original code), 16 heads, FP32 K192+V128, B1 and 384 positions. The expected fused K/V GPU bytes are 251,658,240. MLX **device-active bytes before=0, after=251,658,240, delta=251,658,240**; peak=251,658,496. Independent host process RSS increment was 251,691,008 bytes, measured separately from the **actual GPU allocator**. B2 and B64 are rejected, so the 15 GiB B64 plan cannot be activated by this isolated target. This is real MLX cache allocation, not a numerical forecast.
- Native M1 binary compilation and both device probes completed successfully; the proven functionality is **native build, Metal usable, host MLA config, single-slot guard, actual GPU K/V initialization**.

**Hard evidence boundary:** The native probe deliberately loaded **no pretrained DeepSeek weights**, performed **no transformer token forward**, generated **no 64-way routed-expert probabilities or logits**, and did **not** measure JE01 Bayesian Expert NLL, placebo controls or heldout/OOD improvement. Therefore native context384 full-model parity, token-ID equivalence and quality improvement remain NOT_RUN/UNKNOWN. Exact BF16 source hash is not automatically equivalent to XOX compressed-serving weight hash. The 512-sample preregistration remains unchanged.

An attempted EOE D50 isolated source clone job job_be7e97d11507baa8924441be0831d271 failed after the Git network clone timed out; this did not modify operating source/weights. Subsequent isolated native build and GPU tests succeeded on GitHub macOS-arm64 runner using exact Git source instead. EOE inventory did find /opt/homebrew/bin/python3.11 with local MLX and mlx_lm available; D50 GGUF and source BF16 are present, but the BEGLIN compressed routed-Expert AF package was only **confirmed** on XOX, where managed job admission reports no eligible SSD storage pool at the reserve threshold. Do not bypass this policy to run heavy experiments.

**Current gate:** P0_ORIGINAL_BF16_SHA_PASS / P1_FULL_SOURCE_SHA_AND_512_FROZEN_PASS / P2A_NATIVE_ARM64_MLX_384_GPU_KV_PASS / P2B_REAL_MODEL_FORWARD_LOGIT_PARITY_NOT_RUN / P2C_EXPERT_PAIRED_NLL_NOT_RUN / NO_PRODUCTION_PROMOTION.


## 2026-11-17 — Real BF16 Expert FFN precision proxy and new-input falsification

This is a **separate exploratory proxy**, NOT an amendment to JE01's preregistered 256+128+128 **full-model gold-token NLL** gates. The original model checkpoint and full model-image SHA pins, full public evaluation-source hashes, and the independent JE01 holdouts remain unchanged and unused by this proxy.

### Real weights / synthetic input evidence

Managed EOE read-only job \`job_e48e673f3b756f065cb29f2e5bc187de\`: SUCCEEDED, exit 0. It accessed the genuine full-attested BF16 DeepSeek-V2-Lite source at \`/Volumes/D50/deepseek_v2lite_bf16_safetensors\`, source SHA \`36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc\`. It read gate/up/down BF16 matrices for **layer 11 Experts 0,7,15,23,31,39,47,55** from the indexed shard without ever loading the entire 31GB checkpoint; all 24 consumed matrices have individual BF16-byte SHA hashes in the receipt.

Each Expert FFN was evaluated on **16 deterministically generated synthetic RMS-normalized hidden vectors (seed 20261117)**, comparing original BF16 FFN outputs against independently quantized affine min/max per-64-coefficient **4, 6, and 8bit proxy** matrices.

| Proxy precision | Mean across 8 Experts: relative output RMSE vs original BF16 |
|---|---:|
| 4-bit | **0.1445032** (14.45%) |
| 6-bit | **0.0344652** (3.45%) |
| 8-bit | **0.00851887** (0.85%) |

Full evidence: [JE01_ACTUAL_BF16_EXPERT_8_FFN_PRECISION_PROXY_20261117.json](JE01_ACTUAL_BF16_EXPERT_8_FFN_PRECISION_PROXY_20261117.json), receipt SHA \`555e11d05ca3228a4cb8a828d6a2a3eef153ff5e4f21b4aa547f5820fdecaf2d\`.

### Equal-formula-byte mixed allocation and independent unseen synthetic inputs

A later, separately measured **16-vector synthetic heldout seed 20261118** was not used in bit selection. The first seed's **Expert-specific observed output MSE** chose **4-bit** for Experts **0,15,39,55** (lowest incremental loss) and **8-bit** for **7,23,31,47**. The proxy group packing formula uses one n-bit plane per value plus **two FP32 scale/bias parameters per 64-group**, giving 60,555,264 *theoretical proxy bytes* for both 4×4bit+4×8bit and all 8 Experts at 6bit. **No BEGLIN native compressed packed-byte identity has been measured.**

Managed EOE read-only heldout job \`job_cc6c6bdf5f1898cf1b7e143c6c1dda56\`: SUCCEEDED, exit 0. On actual BF16 Expert matrices and completely new synthetic inputs:

| Budget-comparable proxy condition | Mean heldout relative FFN output **MSE** |
|---|---:|
| Mean of all 70 4×4bit/4×8bit possible assignments | **0.010658290** |
| Training-selected 4×4bit/4×8bit allocation | **0.009793141** |
| Uniform 6-bit (same theoretical proxy bytes) | **0.001198055** |

The selected mixed layout ranked **1 of all 70** equal-proxy-byte 4/8 assignments on the new synthetic inputs, with an **8.12% relative MSE reduction vs the 70-plan mean**. However, the **uniform middle 6bit** layout still has **8.17× less MSE** than the optimized 4/8 mix. The train-selected 4/8 layout is **not** the best layout once 6bit alternatives are permitted. The synthetic heldout was introduced after reviewing the first seed, so this is **exploratory independent-input testing**, not a fully prospective third-party trial.

Heldout evidence: [JE01_REAL_BF16_EXPERT_8_EQUALBUDGET_HELDOUT_20261117.json](JE01_REAL_BF16_EXPERT_8_EQUALBUDGET_HELDOUT_20261117.json), canonical SHA \`441b08e520a512f10377fb801aab0ed1ba3d24852f8280abf524be42cca5c64a\`. Native weight data and source snippets are not copied into GitHub; receipts contain only measurement scores and per-tensor digests. Source: \`tools/je01_real_pretrained_expert_ffn_proxy.py\` and \`tools/je01_real_expert_proxy_heldout.py\`, with fail-closed receipt tests \`tests/test_je01_real_bf16_expert_proxy_evidence.py\` (latest offline JE01 suite 70/70 PASS).

### Why neither result authorizes JE01 promotion

- **Pretrained Expert weights were real** but Expert **hidden activations were artificial**, not sampled from the source model's router. The selected eight Experts are a fixed exploratory set, not the unlabeled router-frequency selected top eight pre-registered for JE01.
- The simple per-64 group affine reconstructor is **not** BEGLIN's native 2–15bit qNg64 implementation or bit-packed memory measurement. The proxy byte equality is a formula, not an actual compressed artifact or equal-runtime-memory result.
- The observed scalar is **FFN output MSE/RMSE**, **not** logit NLL, routed full-model loss, calibration, OOD or runtime latency. Cross-Expert interactions, co-routing, and the Bayesian posterior over real paired gold-token NLL remain **unmeasured**.
- The separate EOE cached MLX \`DeepSeek-V2-Lite-Chat-4bit-mlx\` has now been fully streamed and SHA pinned, but its tokenizer and source checkpoint identities differ from the BEGLIN source, so it must never be substituted for the JE01 model parity gate. Reference receipt: [JE01_EXTERNAL_MLX_CHAT_4BIT_FULL_SHA_20261117.json](JE01_EXTERNAL_MLX_CHAT_4BIT_FULL_SHA_20261117.json), SHA \`8e58b547b733e6bf10aa727381bd788bbb43376d59ec96d385c15acfb9f19063\`.
- Two bounded EOE MLX-LM Python model import preflights timed out **before model load** (\`job_8297f499f08fb3941f6e9d1cee1004c3\` 90s, \`job_0b962868fb6182e367e2ddd6f4022fb0\` 70s). \`mlx.core\` imported in ~0.02s in the second probe, but the MLX-LM model import never completed. This does not establish that pretrained full-forward is impossible; it establishes that no successful EOE reference full-forward was recorded in these runs. No admission override.

**Decision:** ORIGINAL_JE01_P2_FULL_FORWARD=NOT_RUN, NATIVE_EXPERT_NLL=NOT_MEASURED, PROXY_EXPERT_BIT_SENSITIVITY=PASS, PROXY_MIX_4_8_VS_RANDOM=BETTER, PROXY_UNIFORM_6BIT_VS_MIX_4_8=BETTER, PRODUCTION=HOLD. The next real gate remains the isolated pretrained BEGLIN full-model 384-position original-tokenizer parity and routed paired NLL evaluation, followed by the prespecified same-source/OOD quality and latency checks.


### Posthoc exhaustive three-tier negative control: 1,107 formula-equal bit plans

Following the independent synthetic-input heldout, an exhaustive diagnostic enumerated **all 3^8 = 6,561 candidate 4/6/8-bit maps** for the same eight pretrained BF16 Expert weight tensors, restricting to **sum(bits)=48**, i.e. 1,107 equal-formula-byte assignments. The optimization objective is the **unweighted mean per-Expert relative FFN output MSE**, deliberately independent of actual router frequencies (unknown). The full grid is posthoc analysis of the existing synthetic-activation measurements, NOT a new full-pretrained-model, prospective JE01 NLL trial.

[JE01_REAL_BF16_EXPERT8_FULLGRID_6BIT_DECISION_20261117.json](JE01_REAL_BF16_EXPERT8_FULLGRID_6BIT_DECISION_20261117.json) records both training-selected and heldout diagnostic optima. **Uniform 6bit for all eight Experts won in both**:

| Exact proxy-formula bit sum 48 condition | Train MSE | New synthetic-input heldout MSE |
|---|---:|---:|
| Uniform 6bit, best of 1,107 | **0.001192629** | **0.001198055** |
| Best non-uniform map on training: promote E7 6→8 and demote E55 6→4 | 0.002698156 | 0.002556908 |

Even the *best* nonuniform choice evaluated on the heldout costs at least **2.119×** uniform 6bit heldout MSE. This sharpens the preceding restricted 4/8-mix finding: choosing the right Experts matters when the tiers are forced to extremes, but once native-available intermediate 6bit is allowed, the **simple uniform midpoint is stronger than all unequal tier maps under this proxy**. This is compatible with the large asymmetry between the 6→4 demotion error and the 6→8 promotion benefit. It is not proof that all6 is globally optimal on actual BEGLIN NLL; real routed activation frequency, decoder nonlinear interactions, and native qNg64 quantization may alter the ranking.

The P2 decision stays **HOLD**. Next measurement must use the unchanged 512 input SHA manifest, native BEGLIN 384-context full model and tokenizer parity, real routed 64-way scores, same-bytes actual packed artifacts and paired NLL; the proxy cannot replace these gates.


## P2 real reference forward + original BF16 Router verification, 2026-11-17 KST

**New real hardware evidence, without changing JE01 preregistration.**

**MLX-LM loader root cause and safe isolation.** EOE Python3.11 faulthandler localized earlier 70-90s import hangs to huggingface_hub → filelock/_strict.py → _probe_link_follow_symlinks → os.link, before model weights were ever loaded. Moving the temporary directory did not fix the os.link stall. An isolated research process registered a package namespace and imported the unchanged installed mlx_lm.models.deepseek_v2 class WITHOUT executing the heavy mlx_lm/__init__.py, WITHOUT patching filelock or the OS. Model class import succeeded in **0.04s** (job_740491b26371aee46b586d9e9a69641d, exit0).

**First real 27-layer pretrained reference forward.** The SHA-attested (separate, not original BEGLIN source) 8.84GB mlx-community DeepSeek-V2-Lite-Chat-4bit-mlx snapshot, revision a33a8ad64477b9ca7a65b6dc1d8c06c997f87216, was loaded with **915 tensors** and passed an actual 27-layer Metal forward on a fixed public six-token prompt. Final logit tensor shape (1,6,102400), last argmax 8913, actual MLX peak 8,847,736,168 bytes below hard limit 10,200,000,000. Managed EOE job_c415161e5e2b0a13a09c52e209b944ce SUCCEEDED exit0. Evidence: [external pretrained reference forward](JE01_EXTERNAL_CHAT4BIT_REAL_PRETRAINED_FORWARD_20261117.json). This is NOT original BF16-source or native BEGLIN logit parity. No JE01 calibration/holdout data were used.

**First real source-independent L11 full64 route trace.** EOE job_872e6243e827001459f371088469a33e SUCCEEDED. At reference layer11, six real token events yielded each actual 64way softmax probabilities, native Top6 IDs and selected weights; min positive boundary margin 0.00022125244140625. Adding the monitoring hook changed **zero** full reference logits (max abs 0.0). Actual FP16 per-event probability sum deviated from exactly one by up to 0.00015544891357421875; initial 1e-4 guard was rejected and changed ONLY for this reference FP16 trace to 1e-3, with no change in model math (float32 remains 1e-4). Evidence: [L11 real full-64 router trace](JE01_EXTERNAL_CHAT4BIT_REAL_ROUTER_64WAY_6TOKENS_20261117.json).

**All 26 original BF16 router matrix numerical correspondence.** EOE job_9134bfbfaed70bf226b9a9ccddba45d0 SUCCEEDED exit0. Independently compared all 26 real source BF16 router gate weight matrices (each [64,2048]) against the same reference Chat4bit F16 gates. **26/26 PASS, 3,407,872 coefficients compared, maximum absolute coefficient deviation 2.9802322387695312e-8, maximum relative RMS 5.399142856518017e-8.** Includes per-layer raw tensor SHA receipts: [original vs reference router matrices](JE01_REAL_BF16_VS_EXTERNAL_CHAT_26_ROUTER_WEIGHT_PARITY_20261117.json), canonical SHA 5865d0b0e1e9b0f6e55a2c05257db82a1c66c8c72eb16f1f2e19ad55c3c2d7d6.

**Original BF16 Router evaluated on ACTUAL reference hidden activations, with full-logit parity.** EOE job_01207a825f8680362c46148e4d9b3744 SUCCEEDED exit0. At each routed layer1..26 and each of six real reference token hidden activations, evaluated BOTH the original BF16 router (converted to reference FP16 arithmetic) AND the original reference F16 router. Their **156 pairs** of full64 scores were **byte-identical** in IEEE754 FP16. All source and reference probabilities and actual Top6 selected IDs/probs for all 156 events are stored as compact, reversible little-endian FP16 base64. Source-vs-reference max64 probability diff **0.0**, hard Top-K mismatches/ties **0/0**, max FP16 sum deviation 0.0002465248107910156, complete uninstrumented-vs-instrumented reference model final logits max abs **0.0**, peak MLX bytes 8,857,712,600 under 10.2GB limit. [156-event original BF16 gate on reference real hidden state receipt](JE01_REAL_26_LAYER_ORIGINAL_BF16_ROUTER_ON_REFERENCE_HIDDEN_20261117.json), SHA 1d94acbbd801c52faa2ec546c8595b1feae00b48437fb8d922dda16aa0bb9623. Source: tools/je01_full26_real_router_original_gate_on_reference_state.py. Two full-proof receipt tests independently re-decode all 156 full64 distributions and verify the complete per-gate source SHAs; JE01 CI **75/75 PASS** (GitHub Actions run 44023100088).

**Correct P2 boundary.** These experiments prove all original BF16 router gate coefficient matrices can reproduce identical full64 routing functions **when given the same real reference hidden states**. They do NOT prove that the original BF16 model produces identical hidden states or logits end-to-end; non-router quantization and distinct tokenizer identity remain. No original native BEGLIN 384-context inference, JE01 256 calibration/128 same-source/128 OOD paired NLL, covariance, exact packed bytes, or latency quality was evaluated here. Status: REFERENCE_PRETRAINED_FULL_FORWARD_PASS / SOURCE_GATE_SAME_REFERENCE_HIDDEN_ROUTER_156_EVENTS_PASS / ORIGINAL_BF16_NATIVE_FULL_FORWARD_NOT_RUN / ORIGINAL_JE01_NLL_NOT_MEASURED / NO_PROMOTION. Original serving routes and release were not modified; last observed operating release g6-0.9.0-operator.900485 was converged.
