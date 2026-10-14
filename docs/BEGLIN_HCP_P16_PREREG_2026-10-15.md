# Beglin HCP P16 — Identical-token, full-vocabulary native A/B vs full BF16 teacher logits

Date: 2026-10-15 KST · Research-only / **No live promotion**

Parent P15 commit: `fb8e4bea935f7f0b3511e1fac3b87c1bf54f604d`
Research branch: `research/hcp-p16-native-vs-fullteacher-logits-20261015`.

## Frozen original model and two exact inputs

`experiments/hcp_p16/FROZEN_TEACHER_VS_NATIVE_V1.json` was created BEFORE P16 native logit instrumentation or inference. Its source is the full original DeepSeek-V2-Lite P15 BF16 and FP32 teacher forward, each 27 decoder layers, 5,291 original BF16 checkpoint tensors and final 102,400-dimensional float32-exported logits, for these EXACT tokenized inputs:

- prompt0: seven-times-eight English stem, 7 original tokenizer ids, token SHA f3b3029dbb5ecfe92bba4fd4922b6b4557b90aff65eaadc3e9bd34c485866ff4.
- prompt1: capital-of-France English stem, 9 original tokenizer ids, token SHA ee9f8b6e784080ed0150ae0979b46051675e77234bc6661ef5b13e7bd5566238.

Teacher original model index SHA `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`. P15 frozen source input SHA `c85d16c113780d73692fd0bdb9b1afaacb40bda9a998876420f361899d3cbb8b`. P15 BF16 and FP32 are deliberately different arithmetic references and will NEVER be combined into an unnamed teacher.

## First native logit capture

Use the original, isolated DeepSeek GPU persistent native inference stack with exact P5 CB128 C SHA `bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612` and exact P9 prepared MLX KV128 source SHA `ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f`; the original native P9 binary SHA is `103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd`.

Apply a SHA-pinned, **reversible C-only opt-in** observer. Capture the exact `gpu_logits` row used by greedy decoding at the **last input prefill position**, with `R=1`, `max_new=1`, `pass=1`, and `MOE_VOCAB=102400`. Write 409,600 raw float32 little-endian bytes per request into a run-owned directory with `O_CREAT|O_EXCL|O_NOFOLLOW`, SHA-256 proof, and a verified request-ID/prompt-input digest. Strictly refuse stale, unsupported geometry or ambiguous multiple captures. No modification of model math, quantization, GPU binding, ACK, policy change, or output greedy selection.

Eight native requests, using the frozen plan `A0,B0,A1,B1,B0,A0,B1,A1`, allow duplicate per-policy stability checks. A/B refer to P6 original roles L3 shared_up and L26 shared_down and exact expected 5/6-bit policies. Each request requires actual native terminal precision ACK where the policy switches, after policy selection and BEFORE the measured first prefill. Include source worker PID, policy SHA/epoch, request input token SHA and result token SHA. EOE re-reads each source-array SHA and checks the native argmax matches generated first token. Cache and performance speed claims are explicitly excluded; SHA dumping is observational overhead.

## Compare and certify within scope

Independently compare native A/B logits against BOTH frozen BF16-CPU and FP32-CPU teacher arrays of the SAME prompt token IDs: Top-1, Top-5 overlap, logit L2/RMSE, KL(teacher || native), JS divergence, teacher NLL of native-selected next token. Record per-prompt A-vs-B paired differences, finite-mask, any input alignment/geometry drift, and stable duplicate trials. Do not present approximate layer computation as a full model result, or 2 short English engineering prompts as an independent benchmark or statistically proven quality noninferiority. If numbers disagree or source is missing, preserve FAIL and blocker instead of claiming parity.

## Resource and operational isolation

Use a fresh XOX research root under `.tailnet-tmp/hcp-p16-native-logits-20261015`, estimated disk <=0.16GiB with >=20.0GB physical reserve; run a single native GPU worker at a time, RSS <14GiB. No full BF16 checkpoint load on GPU. No P15 teacher rerun needed. EOE original teacher logits already copied and SHA-verified. Original stage binary SHA `8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9`, route SHA `c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47`, stage PIDs 35481, 35493 must remain unchanged. Protect original BF16 checkpoint, serving, queue and Tailnet release.

GitHub research branch only; never merge `main`, release_activate or automatic promotion.

## Promotion blockers remain

The actual P16 paired logit data alone do not prove end-to-end task quality or noninferiority. An independent unseen corpus, longer contexts/generations, tokenizer alignment, quality and error-rate thresholds with confidence intervals, robust multi-worker production restart and immutable release certification remain mandatory.
