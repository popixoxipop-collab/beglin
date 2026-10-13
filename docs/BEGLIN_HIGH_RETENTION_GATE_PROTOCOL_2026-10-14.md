# Beglin — Micro-Expert high-retention checkpoint (90/95/99%)

## Scope: avoid the 99%-equals-100% trap

Experiment branch `research/micro-retention-90-95-99-20261014` starts from `research/trainable-micro-router-120prompt-20261014`. GitHub Actions XOX self-hosted runner is used only for a separate research binary and read-only local checkpoint; production, default `main`, and precision policy are not modified.

Generalized Tupper 2-D grids have `N=1408/c` rows and `M=64` routed macro experts. The bitmask records activated neuron groups; the integer is **not weight compression**.

| Channels per Micro-Expert | Total Micro-Experts per macro | Requested keep 90% | Keep 95% | Keep 99% |
|---:|---:|---:|---:|---:|
| 16 | 88 | 80 (90.91%) | 84 (95.45%) | **88 (100%, zero saving)** |
| 64 | 22 | 20 (90.91%) | 21 (95.45%) | **22 (100%, zero saving)** |

The 50% setting is a repeatability reference, not a newly optimized candidate.

## Preregistered dataset and selection
- Use the exact same 120 unique questions across 12 fields (72 training, 24 validation, 24 final holdout; fresh DeepSeek verification process per prompt), and keep the original 24-token bound.
- Inputs are actual DeepSeek quantized-AF layer-1 pre-FFN hidden vectors and actual top-K routed expert IDs; offline FFN teacher and selected Micro-Experts are computed using BF16 checkpoint experts, a distinct representation.
- All methods and all retention settings are predeclared. The method for each configuration is locked on validation only. Holdout estimates use 24 prompt blocks, with per-prompt bootstrap uncertainty, not token-resampling.
- Candidate comparison: static, static+mean, learned-only, learned+mean, learned+dense Ridge, and learned+rank8 correction. Compute identical full-channel BF16 local macro-output reference for each heldout prompt.
- Approximate active-channel ratio excludes teacher computation, gate/corrector, memory traffic, all other model layers, and packed kernels.

## Exact source and fail-closed gate
- XOX workload: `.github/workflows/beglin-xox-micro-high-retention.yml`. Only this workflow triggers the slow independent real-prompt rerun. The separate `.github/workflows/beglin-high-retention-unit.yml` verifies coordinate and evidence contracts.
- Final `high_retention_gate.json` rejects any row that lacks the correct prompt count/split, full Tupper roundtrip, correct *integer-quantized* kept-group fraction, finite holdout local output L2, and prompt-block CI.
- Conservative local-screening threshold: **upper endpoint of the locked method's 95% prompt-bootstrap output relative L2 must be <=0.05**, with real group savings >0. This is stricter than a point-estimate-only screen.
- A 99% row has all channels, so it can never qualify for sparsity or acceleration regardless of zero L2; never call it a "99% sparse inference" improvement.
- Even a row that passes this **local FFN** screen is only a possible next isolated shadow target. Full native end-to-end logits, token identity, actual qNg64 precision parity, task metrics, and measured total latency + memory **remain unverified**.
- No BEVAL approval or production promotion permitted. Any failure/unknown quality screen stops before scratch-only whole-model hooks.

## Evidence and decisions
- Previous actual 120-prompt study: [GitHub Actions #38430695333](https://github.com/popixoxipop-collab/beglin/actions/runs/38430695333), at width16 keep50% the validation-locked dense-Ridge output-relative-L2 was 0.411714.
- New high-retention study: [GitHub Actions #38434359681](https://github.com/popixoxipop-collab/beglin/actions/runs/38434359681). It must complete and seal its own SHA-bound JSON evidence before any final numeric claim.
- Strict gate unit suite: [GitHub Actions #38434502691](https://github.com/popixoxipop-collab/beglin/actions/runs/38434502691).

## Future follow-on only if screen passes
Evaluate one accepted configuration in a separately built and tagged XOX scratch end-to-end model using **the same source precision and frozen router/corrector** with full reference, logits, next-token agreement, full-model wall time and RSS/Metal memory. The present BF16/AF mismatch prohibits inference-quality claims from the offline study alone. Gate failure blocks this phase; do not relax threshold to manufacture a success.
