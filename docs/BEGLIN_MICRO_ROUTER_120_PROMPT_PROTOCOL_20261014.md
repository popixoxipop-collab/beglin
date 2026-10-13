# Beglin hierarchical Micro-Expert — 120-prompt independent holdout protocol

## Status / non-production boundary
Research-only successor to the 24-prompt pilot. The model, active weights, precision policy, GitHub main and production routing remain unchanged. No promotion or live intervention is permitted. All raw hidden states and token ID lists remain in the XOX per-run scratch and are cleaned after receipt upload.

## Dataset and immutable split
- Corpus: four version-controlled JSON prompt-bank files, 12 subject categories * ten unique natural-language questions, 120 prompts total.
- Within each category: first six train, next two validation, last two final holdout; global 72/24/24.
- Tokenizer: XOX local DeepSeek-V2-Lite tokenizer, offline, no downloaded dataset or service API.
- Every prompt uses a fresh verification process (independent KV/cache state). Inputs are bounded by 24 tokens; source hashes and per-prompt capture hashes are retained.
- The XOX preflight verifies all 120 tokenizations have distinct ID sequences and no malformed or short prompts. Note: distinct prompts are *not* necessarily statistically independent observations from the language distribution.

## Primary comparison
Primary configuration selected before measurement: layer 1, width 16, channel keep fraction 0.50. Measure:
1. static calibration-only channel choice;
2. learned channel choice without correction;
3. learned choice with mean correction;
4. learned choice with dense kernel-Ridge residual;
5. learned choice with rank-8 low-rank residual;
6. static with mean correction.

Methods are selected using validation prompts only. All final holdout methods are reported, with prompt-block paired bootstrap confidence interval for fractional error reduction against static. Secondary configurations: widths 16/64 and fractions 25%, 50%, 75%. The complete routed macro FFN output from the same BF16 expert weights is the offline reference.

## Cost and model quality
- Compare estimated resident FP32 coefficients for standard Ridge, rank-8 and constant-mean residual correction. This estimate is not CPU/GPU kernel timing.
- Report exact fallback-to-full pair counts and idealized selected-channel fraction. Router, correction and memory movement overhead are *not* included in that theoretical fraction.
- A separate shadow measurement may use Beglin's real qng64_dense.c on actual DeepSeek projection weights. Such a local CPU projection proves neither micro-router integration nor token-logit accuracy.
- Do not claim production suitability without native qNg64 representation parity, measured full-model token/task metrics, real wall-time+RSS benchmarks, and isolated combined-policy acceptance.
- Provisional engineering gate for investigating micro runtime: holdout output relative L2 <= 0.05, paired improvement lower confidence bound > 0, nontrivial full-expert fallback rate, and measured total cost benefit. This is a research gate, not a substitute for task quality.

## Known limitations
- Inputs are actual quantized-AF Beglin pre-FFN hidden vectors, whereas offline selected expert FFNs use the BF16 Safetensors checkpoint (representation mismatch).
- Final holdout has only 24 distinct prompts, 12 domains; conclusions may not generalize beyond the selected prompt set.
- Only layer 1 and the original Top-K macro-router are studied, not whole-model behavior after perturbation.
- Generalized Tupper's formula losslessly encodes a routing mask, not model parameters and does not itself compress the LLM.

## Evidence and provenance
- XOX 120-prompt preregistration preflight: https://github.com/popixoxipop-collab/beglin/actions/runs/38430034596
- Rank-8 unit suite: https://github.com/popixoxipop-collab/beglin/actions/runs/38430472890
- 120-prompt XOX execution: https://github.com/popixoxipop-collab/beglin/actions/runs/38430695333
- Source branch: research/trainable-micro-router-120prompt-20261014
