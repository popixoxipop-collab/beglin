# JE01 — original AF4 versus true DeepSeek-V2-Lite-Chat BF16 source and native NIG Q3/Q5: real-weight attribution

**Study status:** original-checkpoint mismatch hypothesis **not supported** for the frozen selected eight Experts; mixed NIG's damaging 3-bit tier is the dominant measured *weight-reconstruction* failure. This is not yet a causal layer-wise model-NLL ablation, and the exact historical AF4 exporter has not been re-executed.

## Evidence, not surrogate assumptions

- Actual original serving-equivalent BEGLIN AF4 file: **9,814,343,680 bytes**, full SHA-256 `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4`. Original layout SHA `ba3898f3bf366b1aec8471d9896885f101bf64b78c816828ea82bd7d58851c3e`.
- Actual pinned original DeepSeek-V2-Lite-Chat BF16 four-shard checkpoint, all full SHA-256 rechecked, immutable index SHA `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`, config SHA `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d`.
- Frozen original calibration-selected eight (layer,Expert) slots: (21,61), (24,26), (13,28), (20,17), (12,13), (24,60), (14,63), (13,22), each with gate/up/down.
- **Actual 24 role tensors = 69,206,016 individual source BF16 coefficients**. No selected role was replaced with synthetic weights. Original AF4 was independently reconstructed using the C runtime's exact *affine* formula `nibble × signed_scale + bias`, including valid negative scales. In L21/E61/gate, 22,465 of 45,056 original group scales were negative; treating them as invalid caused a preliminary fail-closed and was corrected before measuring.
- Native NIG original-source 43,253,760B physical payload, SHA `dab4f50b6233688d789facf37a7fc94e22b0f4aa33298c36e5f69d89063db8cd`, has real symmetric grouped native qNg64 **3bit for all FFN roles of L13/E28 and L24/E60** (6 roles), and **5bit for all three roles of the other 6 Experts** (18 roles). This allocation was previously frozen before gold-NLL scoring.
- One-role positive/negative source pairing: original AF4 L21/E61/gate versus **correct BF16 E61** gives coefficient correlation **0.9958537118**, and nibble-code agreement using original AF4 scale+bias **99.78908192%**. The *same-layer incorrect* BF16 E60 gives correlation **0.0017487654** and 4bit-code agreement **8.98357738%**. This strongly differentiates correct same-source alignment from arbitrary same-shape weights.
- Full 24-role original-AF4 versus actual Chat BF16 comparison: minimum correlation **0.99571354**, coefficient-weighted code agreement **99.788686%**, coefficient-weighted mean of per-role relative RMSE **0.091368**. Independently ratio-aggregated full original AF4 relative weight RMSE **0.09138458**.

## Direct three-way quantization comparison

All metrics below are **coefficient-space relative RMSE against the same verified original BF16 source**, NOT model logit MSE nor gold-token NLL.

| Actual 24-role subset | Original affine AF4 RMSE | NIG symmetric Q3/Q5 RMSE | Roles and coefficients |
| --- | ---: | ---: | --- |
| The six original frozen low-bit Expert FFN roles | 0.09155 | **0.35718** (Q3) | 6 roles, 17,301,504 coefficients |
| The other eighteen FFN roles | 0.09132 | **0.07116** (Q5) | 18 roles, 51,904,512 coefficients |
| All frozen 24 roles | **0.09138** | **0.19397** (Q3/Q5 mixture) | 24 roles, 69,206,016 coefficients |

Actual NIG-to-original-AF4 relative RMSE: **0.21417**. The additional source-normalized squared-error ratio of the full NIG mixture versus original AF4 is **+0.02927323**. Q3 contributes the strong positive extra error; Q5 alone improves weight reconstruction relative to original AF4, but cannot compensate.

**Real model quality check already completed**, run on EOE Mac Studio, original pretrained 27-layer native AF baseline and identical native experiment binary with NIG OFF/ON, train-only 16 WikiText cases / 656 gold tokens:

- Original/overlay OFF NLL: **2.291799702354307**; source and exact model-output SHA matched previous independent XOX reference.
- Original+Q3/Q5 NIG ON NLL: **2.3067548780457776**, delta **+0.014955175691470757 worse**; **13/16** cases regressed.
- This is only a small train development pilot, **not** 256-case calibration or independent WikiText128/HellaSwag128 evidence. Stronger proof that the two Q3 Experts caused the NLL change requires paired Expert-level and role-level ablations.

## Decision and next experiment

**Hypothesis A: different original models/changepoint caused the observed NIG loss — weakly supported / no matching evidence in selected 24 roles.** The original AF4 is quantized and therefore cannot be directly bitwise-equal to BF16; the exact historical native AF4 export algorithm has not been regenerated and other layers' source weights were not checked.

**Hypothesis B: severe Q3 Expert-wide quantization distortion drives the NIG regression — strong measured weight-level evidence, causal model-level proof pending.** The original BF16-weight NIG 3bit roles lost 35.72% relative coefficient fidelity versus 9.15% of AF4 in those same roles, despite better 5bit fidelity in the other roles.

Do NOT blindly promote the current Bayesian 3/5bit policy. The next *pre-registered exact-byte challenger* should eliminate 3bit:

- Same 24 frozen Expert FFN roles each hold 2,883,584 coefficients and 45,056 grouped f32 scales.
- Symmetric qNg64 4bit = **1,622,016 bytes per role**, 5bit = **1,982,464 bytes per role**.
- **12 roles × 4bit + 12 roles × 5bit = 43,253,760 bytes exactly**, the measured original AF4 eight-Expert payload budget, **with no 3bit roles**. This assumes the same native packed+scale budget scope and no extra sidecar cost; a physically saved candidate must be size/SHA audited to certify it.
- Freeze role assignment on calibration-only routing plus separate training-reconstruction risk; measure real source AF4/Q4/Q5 per-role and FFN differences, then ORIGINAL full 27-layer paired train NLL. Holdout labels remain sealed until the policy and thresholds are frozen.
- If Q4/Q5 under this exact-budget contract still loses, prioritize *affine* quantizer/scale-bias preservation or activation-aware quantization over further Bayesian-bit swapping; NIG prior/uncertainty alone is not validated.

**Provenance:** EOE research jobs `job_3bde0c6a84736f7b6e31a8baa591e07e` (corrected 24-source-pair comparison) and `job_75bfd6f10cde32d0f9395587135f60ab` (original AF4/Chat BF16/NIG 24-role direct attribution with wrong-Expert control); results are SHA-sealed in `docs/research/JE01_ORIGINAL_AF4_CHAT_BF16_NIG_Q35_REAL_24ROLE_WEIGHT_ATTRIBUTION_20261122.json` with canonical receipt SHA-256 `82f047f6e323be2ace5936b5acc177a2fd2f36b9144b64035a75d29a6e8a69a4`; original 24-role source receipt `be513c64e6af0a91613abdc34f1051a666023870561c143e4ad231e93a6e9233`. Original 16-case native NIG ON receipt `880f0cae56288adaa8e29645779c03e94cd002e178a06f9e31e284ee7b405ec5`. Both AF/BF16 audit scripts committed to the research branch.

**Research gate:** NIG Q3/Q5 quality **FAIL on true 16/656 pilot**; source numeric alignment **PASS for selected Expert 24 roles**; fully regenerated historical AF4 encoder **NOT YET VERIFIED**; new 12×Q4 +12×Q5 **design only, NOT executed**; full 256 and independent 128+128 **HOLD**. PR #94 stays Draft, original AF weights/operating services and Tailnet current/live release remain unchanged.
