# JE01 — original DeepSeek 32/384 full-logit parity and real first-gold NLL
Date: 2026-11-17 KST. Draft PR #94. Research-only; production unchanged.

## Real original pretrained compressed BEGLIN model execution

Two new scratch-only C/MLX arm64 binaries on XOX were built from the identical previously SHA-attested BEGLIN original native AF source, with only 32 versus 384 position bounds differing, plus an identical research-only full-logit emission hook. Neither live worker, operating binary, original source, nor checkpoint was modified.

- Original weights: XOX DeepSeek-V2-Lite five-file AF bundle, 9,828,513,716 bytes, fully SHA-attested in JE01_ACTUAL_XOX_ORIGINAL_AF_NATIVE384_5_RUNS_20261117.json.
- New isolated 32 model binary SHA-256: 361eced42d72d2a4b9be342188d20d6ff2e7f25a493a633aec1c82581420e7bb
- New isolated 384 model binary SHA-256: 166e5480acef41cc9d153a9479fb3f09de98a912ca52e43b19b540e80b980c1b
- Inputs: the same 16 REAL preregistered WikiText train calibration samples, native BOS-token sequences of lengths 15–25 positions. NOT heldout or OOD.
- Both binaries completed actual 27-layer GPU forwards with exit code 0, original weights, one-slot B1, 269 bound native tensors and original MLX ABI.
- Each run emitted all 102,400 float32 next-token logits for all 16 cases, 1,638,400 paired float32 comparisons. Maximum absolute difference 0.0; exact differing values 0; whole raw logit file SHA-256 IDENTICAL across both arms: 428d2f985d1eca80ab299a7ab8cbf3d1330391efe46791f87d013a670f5d387a. Each raw result file is 6,553,886 bytes in the isolated XOX research scratch (not published as public model assets).
- A SECOND independent read-only script reopened both complete raw files, checked their complete SHA-256 and byte equality, decoded every request and 102,400-element logit row, checked every value is finite and all original argmax outputs were preserved, and rechecked both production worker identities. Independent audit PASS.
- Both arms were bounded by the system memory pressure and worker PID guard; minimum free memory was 48% (32) and 47% (384), above the 35%-free termination threshold. Healthz remained ok, generation 6, live route beglin-candidate-shared-up-l3-n6, original worker PIDs 35481/35493 unchanged.

[Native XOX 16x102400 full logit evidence](JE01_ACTUAL_XOX_ORIGINAL32_VS384_FULL102400_LOGITS_16_REAL_CAL_20261117.json) — SHA 34da06b2441bd1cfa8c8593ff20e8c2d3e40850e1c0ef6d6c8a127d70cb75d14.
Source executable reproducibility: tools/je01_xox_original32_vs384_full_logits_shadow.py.

## FIRST real gold-token NLL from the original AF source

Separate GitHub-hosted, explicitly calibration-only public WikiText reconstruction fetched all 20 prior SHA-pinned train pages, the original SHA-pinned model tokenizer and the 16 predetermined calibration source rows, then validated each prompt and target SHA and each original native input byte SHA. All 16 full-string tokenizations preserved the original prompt token prefix. NO same-source heldout or HellaSwag labels accessed.

[Calibration16 gold first token source SHA receipt](JE01_REAL_CAL16_ORIGINAL_WIKITEXT_NEXT_GOLD_TOKEN_SHA_20261117.json) — SHA 9d81479d2772c8e76f8ba9a7e2d6ef3eb97597ccf515d12186e969e037ca3b76. Hosted real source data job: https://github.com/popixoxipop-collab/beglin/actions/runs/44102398123

The native scratch raw logits were read again and a full 102,400-vocabulary stable log-sum-exp minus the real gold-next-token logit was calculated for each of the 16 cases, once per arm.

| REAL first-gold-token metric | Old 32 | New 384 |
|---|---:|---:|
| Verified original train calibration targets | 16 | 16 |
| Mean NLL | 2.7294751068138448 | 2.7294751068138448 |
| Max per-case NLL difference | — | 0.0 |

[Native real calibration next-gold-token NLL receipt](JE01_ACTUAL_XOX_ORIGINAL_CAL16_REAL_FIRST_GOLD_NLL_20261117.json) — SHA 272a5272e7af5408e89c38366dc71906ed9bfab4af6537dd14f8eece04b2920a.
Reproducible scorer: tools/je01_actual_original_cal16_next_gold_nll.py.

## Limits / decision

These are genuinely executed original native compressed DeepSeek GPU logits and real calibration first-token NLL, not synthetic-model/FFN-output proxies. HOWEVER, the results do not score the full 512 frozen prompts or their full target continuations, do not establish native original unquantized BF16 parity, do not evaluate Bayesian tier alternatives, do not compare exact packed-bytes, and make NO independent same-source/OOD quality or latency improvement claim. This is a successful *original runtime output and scoring pipeline calibration pilot*, not a model-accuracy gain.

The original 512-case P2 paired NLL, actual routed Expert posterior, exact budget and OOD gates remain HOLD. PR #94 remains Draft/unmerged; no release activation or production route/weight change.

Independent Python receipt/score tests: 98/98 PASS at https://github.com/popixoxipop-collab/beglin/actions/runs/44103262572.

VERDICT: ORIGINAL_REAL_GPU_FULL_LOGIT_PARITY_PASS / 16_CAL_REAL_FIRST_GOLD_NLL_PASS / 512_FULL_GOLD_NLL_NOT_MEASURED / BAYESIAN_EXPERT_GAIN_NOT_MEASURED / NO_PRODUCTION_PROMOTION.
