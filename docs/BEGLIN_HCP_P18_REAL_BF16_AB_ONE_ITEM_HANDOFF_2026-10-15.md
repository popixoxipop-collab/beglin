# Beglin P18 — Original BF16 / Native A-B real NLL pilot handoff
2026-10-15 KST · Repository: popixoxipop-collab/beglin
Research branch: research/hcp-p18-external-multimetric-20261015
Verdict: **P18_REAL_ONE_ITEM_BF16_AB_NLL_PASS_NOT_480_NONINFERIORITY**.

## 1. Current true scope (do not overclaim)
- Frozen external test corpus: MMLU/ARC-Challenge/GSM8K ×160 = 480 question units, 673 gold-token positions.
- Exact token bank SHA-256: c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df.
- Frozen preregistration SHA-256: 77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c.
- Real complete original BF16 27L/5291 tensor teacher: **1 / 480 question forward**.
- Real native A full-vocab logits: **1 / 673** forced gold positions.
- Real native B full-vocab logits: **1 / 673** forced gold positions.
- 480-task exact correctness and warm 3× A/B fixed-work latency: **not measured**.
- The one engineering-pilot question, p18-mmlu-000, has been exposed during research qualification. Retain its preregistered membership and report its exposure transparently; do not retroactively change the cohort or noninferiority margin.

## 2. Real original and quantized logits — one question
p18-mmlu-000 has 107 prompt tokens and 1 gold-answer token. All three runs use the frozen original prompt token IDs and the same 102,400-way vocabulary.

| Source | independently recalculated gold NLL (nats/token) | full raw logits SHA-256 |
|---|---:|---|
| BF16 whole 27-layer original teacher | 1.017819917192373 | 7025aed3a8c23fdc024f9a2f4e4a9c4944129f09d1bc3c32633c57c21a1249b6 |
| Policy A qNg64 | 0.5151753750563053 | ea5dea070b577e9afcd1dadd12b41e92ec46d7b25f87d145e9935e8e69bc612e |
| Policy B qNg64 | 0.6301864903956584 | 252ffc12d1a76bd139c97d9eb8fe94d38259fb23e4ca29ae738d61a59d37b209 |

- Paired B−A = **+0.11501111533935315** nats/token. Positive denotes greater B loss on **this one question**; it does not establish P18 population inferiority or noninferiority.
- BF16, A and B greedy argmax each match the one gold token. This is not full 480-task accuracy.
- BF16 27 full layers executed, exactly 5,291 original source tensors loaded/verified, peak RSS ~2.25GiB, model run ~87.96s, unchanged production image.
- Original 5,291-tensor value SHA manifest: 88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda, **identical to sealed P17 source value manifest**.
- BF16 teacher receipt SHA: c7628194d2e601f64525b1c05e5c59be5848eb40c68420dbd5c935e6ec148c3b.
- Native paired raw receipt SHA: d41531836ad8fd1fc38fb3ff798894d618365998d431ab3987140a753609eb04.
- EOE independent raw recomputation receipt SHA: **c8684066c22140d154aba3837f35b9b0f00e6c6e35f956199ae8b2a45d88cd0c**.
- Independent auditor recalculated logsumexp(-gold_logit) directly from original .npy and 2× original native float32 .f32. Compared native C capture log RID/token hash/position/409600 byte count, A/B ACK and policy B epoch transition, source model SHA, P17 original stage and PID before/after.

## 3. P18 native full-logit observer security contract

The initial attempt to reuse P17 native observer with a P18 directory failed closed with P16_CAPTURE_SCOPE_INVALID because P17 was explicitly restricted to its own original 31 prefixes, directory, and 112-position cap. This **was not overridden in the P17 source**.

- Created a distinct native 1024 research source, generated from the SHA-frozen 480-question P18 bank. Its allowlist contains **exactly 673 token-prefix SHA-256 values**; 31 original P17 prefixes are excluded.
- Research capture path is strictly the XOX run-owned hcp-p18-capture1024-20261015/capture directory. Exclusive O_CREAT|O_EXCL|O_NOFOLLOW, 409600 bytes per capture, finite logit checks, 1024 position cap, P18 scratch-only + one-slot guard.
- New C source SHA: 82475145a4351b934cfccda6763112cdf25751e2ca470ac8417a16e6dc8fdb1c.
- Source MLX SHA: a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b.
- XOX-linked experimental binary SHA: ed2af527c1291567b0e9e67a9940cb5f13b298d4a645d11742ff1cc34248bc6e.
- C/MLX syntax and XOX CMake compile/link PASS; no research flag -> exit 89 (refuses launch).
- Source allowlist independent audit: 673/673 SHA matches, P17 31 excluded, 10/10 capture security properties PASS. Audit SHA-256: 9a419cf124c58ed3eff10e36567755c4469623837d1de1715d9e81b69c9f2a5f.
- The *new* observer prototype may be used only in the isolated research binary, never release/persistent-stage.

## 4. Extended context native smoke qualification already checked

- Existing P18 independent A/B GPU evidence confirms 203-token development-only B and 962-input+1 generation A/B both returned one finite next-token logit with genuine Metal worker logs, A/B ACK/epoch, production PID 35481/35493 unchanged.
- Independent boundary readback SHA: f5eb51e26edf7ae1e2155f6f1908870d2c794f032bdcdc4e62b8c738603f3eb0.
- Existing scratch dynamic KV arena prototype passed 458 standalone ASan/UBSan checks; this is a prototype plus GPU smoke, not universal 1024-context correctness certification.

## 5. Preservation and capacity
- All private raw BF16 .npy, source 5291 tensor manifest, A/B .f32 and full worker logs are stored under:
  /Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/native1024_real_evidence
- Frozen raw 480 question texts/token IDs and dev examples: D50 only. **Do not commit raw items, logits, checkpoint tensors, model files, private source manifest, secrets or credentials to GitHub.**
- XOX working paths:
  /Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-native1024-20261015 (base 1024 smoke and original teacher pilot)
  /Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015 (new exact 673-token research logit observer)
- XOX SSD free ~20.25GiB (protected reserve 20GiB) as observed after pilot. **Bulk 673× BF16/A/B logits must NOT accumulate on XOX.** Registered per-item transport to EOE D50 with bytes+SHA verification and explicit per-run cleanup receipts needed.
- WD archive copy and hash verification of one cold XOX HF cache revision freed ~3.1GB earlier; other models/P17 sealed evidence untouched.
- Latest fresh Tailnet release check: current=live=runtime=g6-0.9.0-operator.900350; converged=true, active_operation=null; stage SHA 8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9; route SHA c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47; original stage PIDs 35481,35493. No activation or rollback.

## 6. Remaining actual work, not marked complete

1. Freeze/validate a resumable P18 **per-item chunked original BF16 teacher** (27 complete layers with 5,291 checkpoint source tensors) for 479 remaining questions. Current P17 teacher has a hard 112-token context cap; the P18 >112 extension must separately prove correct context/RSS/attention position semantics before full 480 run. Original BF16 full greedy task output is additionally required for BF16 task-positive-control score floors.
2. Use exact P18 observer allowlist to capture remaining 672×2 A/B full vocab gold-prefix logits with true SHA/ACK, per-item D50 transport and cleanup that never threatens XOX 20GiB protected reserve. Verify native GPU prompt lengths/position boundaries through 963, fail on byte/log/epoch mismatch.
3. Full 480 question task exact accuracy, independent BF16 positive-control floors, 3 genuine fixed-work warm decode latency trials per policy; precision transition overhead recorded separately. Correct EOS, cap, answer-scoring and attribution rules; no synthetic or signaled-only pass.
4. Real raw full-vocab auditor to recompute all 673 original teacher positions, 1,346 A/B native positions, per-item NLLs and task correctness, then 10,000 question-cluster stratified bootstrap and Bonferroni familywise one-sided bounds using the **already immutable P18 protocol**: B−A NLL upper <=+0.05, B−A exact task accuracy lower >=−0.05, B/A latency geometric ratio upper <=1.10, empirical p95 ratio <=1.15.
5. Report training contamination and one-engineering-pilot-question-exposed limitations. Never use one-question results to change B's policy, cohort, thresholds, or selection. No automated live promotion, and no production policy B activation without separate explicit human approval.

### Verdict
**P18 original BF16 + native A/B raw logits and independently verified NLL are proven for ONE preregistered question. 479 questions, full task correctness and latency gates remain unfinished. A stays incumbent; B stays blocked.**
