# JE01 — Original AF4 vs true-source 12Q4/12Q5 43.25 MB, actual 27-layer 656-gold train pilot

**Experiment completed, EOE managed job `job_e1258ff079f18b7a6fd1d26ffb366c23` SUCCEEDED (exit 0)**. Original 27-layer DeepSeek-V2-Lite-Chat-based BEGLIN native GPU, same 16 WikiText train cases and **656 teacher-forced gold tokens**. Independently SHA-attested original AF4 9.814 GB and true-source 43,253,760B 12 Q4 symmetric packed-nibble +12 Q5 symmetric bitplane overlay. Source and executable artifacts were previously checked using actual Apple Metal GEMV across 24/24 roles.

## Direct, paired full-model measured NLL (not weight-space surrogate)

| Full real native 27-layer mode | Average gold-token NLL | Versus original |
| --- | ---: | ---: |
| Original AF4, correction OFF (control) | **2.291799702354307** | 0 |
| Previous true BF16-derived NIG 6×Q3 +18×Q5 ON | **2.3067548780457776** | +0.014955175691470757 (worse) |
| **Current true BF16-derived 12×Q4 +12×Q5 ON** | **2.2975364339707807** | **+0.005736731616473811 (worse, +0.2503155756% relative NLL)** |

Current 4/5bit candidate recovers approximately **61.64% of the NIG 3/5bit pilot's NLL deterioration** but **does not beat original AF4**. Exactly equal eight-Expert physical input bit budget **43,253,760 bytes**; this does not assert equal runtime GPU allocation.

## Reproducibility and actual data

- Q45 physical full-file SHA: `b4bd15db43f60f5aeb3ae8d6b9af148fa38545e5289ac049cd26a57a82932e6b` (12 Q4 native symmetric group64, 12 Q5 group64; actual pretrained BF16, NOT synthetic matrices).
- Original AF4 file SHA: `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4`.
- Native 27-layer executable SHA: `1eb8dd5311cc23efba11a54a2891e9ec77fc9bb5cb88cfb60bdd6183310059ab`.
- OFF actual output SHA: `b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c`; **bit-identical** to prior EOE original native reference. Off result receipt `9b63d843b2a8bde9bd0633ddd46e635ea7e3eaba4a35a10d31807c5a1638f72b`.
- Q45 ON output SHA: `27f5864fe11606ad1f494893e32144aea90cb9ee406e954aee118b91b78b2090`; ON receipt `48e7f1f808cfaea23caaf0f265a0379391036283adfb7dddb143aee1666fdfc4`.
- Original OFF per-case max absolute difference = **0.0**; control mean exactly **2.291799702354307**.
- Pilot **10/16 sentences worsened**, **6/16 improved**; largest per-case NLL sum worsening row4 **+0.939400948083545** across 39 gold tokens, row3 **+0.8911844497059036** across 31, row2 **+0.7737386624873892** across 35. Total candidate-origin NLL sum +3.763295940406632 over 656 tokens.
- Memory pressure minimum free **71%** in each mode; OFF 61.122 seconds, ON 63.685 seconds (wall timing, not a controlled throughput benchmark). Host GPU utilization peaked ~86%; actual managed job reservation 2 CPU cores, 0.95 GPU fraction, 24 GiB RAM, admitted through Tailnet Commander resource scheduler using calibrated prior CPU peak 1.025 cores.
- Full per-token and per-case gold data are retained in `docs/research/JE01_REAL_ORIGINAL27_Q45_GOLD16_OFF_20261122.json` and `docs/research/JE01_REAL_ORIGINAL27_Q45_GOLD16_ON_20261122.json` with original hash-bound receipts.

## Interpretation and decision

**Pilot efficacy gate: NO-GO to promotion.** Physical size equal, Metal path verified, but actual gold-target NLL higher than original AF4, despite improvement over the earlier 3/5bit NIG design.

This is **exploratory**, not confirmatory: 4/5bit role selection was made after inspecting failed Q3/Q5 train16 evidence. For the new paired 16-case dataset, a seeded 50,000-iteration *post-hoc sentence-cluster paired bootstrap* gave approximate 95% CI of per-token delta **[-0.000732, +0.011942]**, crossing 0; sign 10 regressions / 6 improvements. Not a pre-registered generalized statistical test; no claim of population degradation at controlled alpha. It is nevertheless sufficient to **withhold replacement** until larger, independent evidence shows a gain.

**Next controlled investigation:** compare real source and activation-conditioned quantization error from *affine* original AF4 against new symmetric Q4, and perform role-level/Expert-level ablation with actual model NLL, preserving exact binary-size budget. Original AF4 relative BF16 weight RMSE ~9.14%, symmetric Q4 roles ~15.30%, Q5 roles ~7.11%, current Q45 overall ~11.93%; coefficient-space error is not a proxy for gold NLL with guaranteed monotonicity. Define selection/thresholds on calibration only, then independently validate after locking the plan.

**Safety:** original AF4 checkpoint unchanged, research Draft PR #94 unmerged, no production release activation or default model change. Neither OOD128 nor same-source validation128 labels accessed. Full 256 calibration NLL NOT measured for Q45. No holdout performance or quality improvement claims.

**Source metadata:** Tailnet Commander EOE current/live `g6-0.9.0-operator.900500`; initial GPU-default admission required 4 CPUs and blocked despite empirical native peak 1.025 cores, so the prior waiting queue was cancelled while WAITING. Accurate mixed CPU+Metal reservations (2 CPU cores, 0.95 host GPU, 24 GiB RAM) passed formal `resource_plan` and `resource_submit`; no direct admin/exec bypass or other jobs were killed.