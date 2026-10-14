# Beglin HCP P17 — 12 first-use internal questions, 31 gold tokens, native A/B quality gate

Date: 2026-10-15 KST. Research-only; no operation activation or main merge.

Parent sealed P16 source commit: `577e0fae6d03d04882aa346a8e75d132b2850cd6`.

## First-use corpus and tokenizer precision

- Initial handcrafted internal bank SHA: `1d31f16faca1704bc79761703d12506f713868cb2e37c68fcc8509f37be8e77a`.
- Before any model evaluation, the original `Answer: ` ending was shown by the *original tokenizer* to merge its last-space token with the first target answer token. For valid prefix-conditional probabilities, it was amended to `Answer:\n` (questions and gold labels unchanged). Corrected bank SHA: `dee6f0741f7e4b9e47fcd68c87cf69296add7a30df457ffc36c06908efce5744`. Retain both original and amended files.
- Original DeepSeek-V2-Lite tokenization completed in XOX with exact model index SHA `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
- **Frozen 12 prompt + answer examples, total 31 gold tokens**, input length 34–43, 11/12 examples with 2–4 target tokens. The single-token reference is retained and accounted for honestly, not padded with artificial output. Gold context length <=112, avoiding KV128 overflow.
- Frozen token-id source `experiments/hcp_p17/P17_FROZEN_TOKENS.json`, SHA `a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521`.
- All 12 base prompts and 31 teacher-forced prefixes must have SHA disjoint from original P4/P6, P10 and P15 checked prompts. The set is **new internal first-use**, not an independent external benchmark and not provably absent from original model training.

## Real teacher-forced token loss and multi-token generation

1. Evaluate 12 complete prompt-plus-gold-answer sequences in source order through all 27 original decoder layers and 5,291 source BF16 checkpoint tensors in **BF16 CPU arithmetic**, with original P15 SHA-pinned streaming implementation and original DeepSeek computation classes. One layer resident; process RSS hard limit 6GiB; XOX storage >=20GB. At each of the 31 gold positions, collect original 102,400 logits, finite/weight-hash and source layer receipts. This is a **full-model teacher**, not local weight matrix inference.
2. Each heldout sample's teacher logits cover exactly the same gold prefixes later used by native. Prompt score = mean of gold token NLL, not NLL of generated continuation. Sum the gold-token log-loss by sample; no token-level bootstrap pseudoreplication.
3. Native original P9 A/B policy in a separate run-owned GPU worker, two original tensors with 5/6-bit precision and original P6 ACK. Evaluate 31 gold prefixes × 2 policies (62 one-token next-token full logit captures) plus 12 new prompts ×2 policies (24 greedy generations with a bounded 8-token output budget). Compare native gold-token NLL, KL(BF16 teacher||native), JS and strict gold token prefix accuracy; no semantic score invention. The native capture must see the exact frozen token IDs and EOS/greedy token from its source; missing or malformed captures are FAIL.
4. The native greedy long-generation path uses the ordinary original P9 worker without observer dump. Every A/B-generated sequence will be inspected for validity and retained by SHA and token-prefix correctness, not assumed correct. The P17 frozen answer token ids remain the only objective scoring labels.

## Precommitted research-only B noninferiority threshold

Policy A incumbent, B challenger. Let `d_i = mean_gold_NLL_B(i)-mean_gold_NLL_A(i)` at question level. One-sided paired **95%** bootstrap upper confidence bound (10,000 resamples, seed 20261015) must be <= +0.05 nats/target token. The lower one-sided 95% bound for policy B−A strict gold-token prefix accuracy must be >= −0.05. Both gates must pass, all 12 questions complete, and source/ACK/finite requirements pass. Report teacher-BF16 JS/KL per prefix separately. BF16 CPU is the authoritative teacher reference; FP32 CPU is optional sensitivity if measured and never substituted.

Even if these tiny **internally authored** gates pass, label result `P17_INTERNAL_RESEARCH_GATE_PASS` only, not external quality noninferiority or production readiness. Do not infer a global winner from 12 likely training-known questions. Unregistered errors, missing logits, semantic prompt mismatch or model instability => BLOCKED.

## Safety, evidence and previous achievements

Do not change original serving binary SHA `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`, route SHA `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`, original worker PID list [35481, 35493], XOX 20GB storage reserve, current Tailnet release plane, or main. Archive all raw BF16 full-logit arrays and native .f32 privately or in private SHA-linked artifacts; do not copy model checkpoint weights. Upon failure, record exact blocker and don't fabricate pass.

Previous P15/P16 succeeded in collecting full BF16 reference 2 prompts and 8 native A/B first-token vectors. This P17 uses newly frozen source questions, long target completions and a first-use empirical pilot before any actual quality certification.
