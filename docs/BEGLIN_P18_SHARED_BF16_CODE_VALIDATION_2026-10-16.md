# P18 shared-original-BF16 code audit — 2026-10-16

**Decision: DEV-ONLY. DO NOT USE FOR FROZEN 480 QUALITY YET.**

## Executed on independent isolated CPU

The precise GitHub original source blobs were checked against the distributed prior review archive:
- `tools/hcp_p18_shared_bf16_core.py`: Git blob `ab3f9ea13ea6ec0f2faff787f3f06a9c21cbb9a2`
- `tools/test_hcp_p18_shared_bf16_core.py`: Git blob `7ae2a4fbb935c56836a2b82f0aa03c6aa23b1a9a`

Test environment: CPython 3.13.5, PyTorch 2.10.0+cpu. **45/45 synthetic and static tests PASS**: GitHub v1 10; v2 explicit gold-window toy 19; actual Q/K/V causal-mask toy 6; dev-only qualifier static source-contract 10. No original model weights were loaded in this verification.

## Review findings

1. v1's `F.linear(h[:, -2:-1, :], head)` scores only the **one token before the last**, given a prompt+one pseudo-gold input. v1 cannot be used as the BF16 teacher for P18 multi-token GSM8K answers: gold positions 2..8 would be missing.
2. v1's prior toy test checked mask *shape*, not causal masking arithmetic. Six new independent masked-attention toy tests check future-token isolation, sequence independence, and swapped order.
3. v1 caller-controlled `max_length` did not itself enforce P18's 1024 context ceiling. Proposed v2 verifies `max_length <= 1024`, explicit per-row prompt length and 1..8 gold tokens, same original independent unpadded causal masks, one tensor stream and full-vocab float32 gold spans.
4. v2 toy/contract source SHA256 `c160e79ecb527ccfcc7c908e7fc64f67abf8938a4186c2f7033eac753c7cde2f`; real EOE development-only qualification script SHA256 is recorded in the downloadable QA artifact, not executed.
5. The proposal is sequential per-row decoder computation **under a shared layer**: it saves loading model weights twice, but does not halve arithmetic. **No actual model speedup has been measured.**

## Required EOE original-model admission before any promotion

EOE connected access was unavailable: its Desktop Commander device was offline; another remote control path was usage-limited. No 27-layer shared BF16 real run occurred and no sealed 480 results were altered.

When an authorized EOE environment is available, run only the SHA-pinned **203-token development-only** original-model qualification with 3- and 5-token pseudo-gold windows. Require **both processing orders** (short/long and long/short) to match 2 separate original 27-layer/5291-tensor BF16 full-vocab logits **byte for byte** for all gold positions. Measure the wall-time ratio, 9GiB RSS cap, 1GiB new-swap cap, and source tensor SHA once-only proof. Then independently audit the result.

Even if these requirements pass, the complete P18 480-item BF16/A/B research gate, task accuracy, fixed-work latency and statistics remain separate, with incumbent policy A active and B live promotion disabled.

**Verdict: code design plausible, 45/45 synthetic/static QA PASS, real DeepSeek bit-exact parity 0, measured speedup unknown, NOT certified for official 480 evaluation.**
