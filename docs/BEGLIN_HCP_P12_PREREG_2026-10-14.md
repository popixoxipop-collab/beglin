# Beglin HCP P12 — Startup Reconciliation and Local BF16 Tensor Oracle (preregistered)

2026-10-14 / isolated research, no production changes.

Base P11 immutable commit: e2271d6fe758c6d09877130637dccb193427ea15.
Branch: research/hcp-p12-prestart-journal-bf16-local-20261014.

## Gate A — restart-before-service durable reconciliation

- Run-owned XOX .tailnet-tmp/hcp-p11-inflight-p12-* scratch only. Never install into live serving.
- Acquire nonblocking POSIX flock before startup or admissions, and retain it until process ends. Concurrent owner blocks.
- Prior command with genuine matching REBIND_SET_APPLIED ACK: independently validate both targets, before/after policy SHA, old/new epochs, then archive command, ACK, INTENT and VERIFIED SHA with fsync before any new native start. Never resubmit old command.
- If command has no matching terminal ACK, declare UNKNOWN; prohibit automatic retry and startup. P11's separately verified crash-marker and escrow/quarantine must be validated first. Partial journal, symlink, stale model identity or simultaneous owner blocks startup.
- No exactly-once claim for interrupted original request.

## Gate B — original DeepSeek-V2-Lite BF16 local weight reference

- Do not load entire 29.256GiB BF16 model or stop live workers. XOX safe_open loads at most one local L3 shared_up or L26 shared_down tensor (about 11MiB BF16 each) at once with hard process RSS under 1.5GiB. Model index SHA d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf.
- Compile isolated original C gguf_quantize_qNg64 and qng64_matvec_f32; quantize BF16-to-FP32 weight matrices n=5 and 6. Compare 8 deterministic projection vectors against original BF16 float32 projections; report weight RMSE and projection errors.
- These local metrics are NOT full BF16 model logits, task correctness, or exact live A/B model agreement. No tokenizer, KV attention, expert routing or entire transformer.
- Hash exact C source and readonly checkpoint identity; seal output JSON; no model tensors in GitHub evidence.

## Gates for promotion

- Research pass cannot authorize main merge or live activation.
- Full BF16 teacher logits and independent end-to-end quality, multiworker concurrent crashes, source hot reload, long context and duration, system boot recovery, immutable release and fresh expected-current CAS remain blockers.
