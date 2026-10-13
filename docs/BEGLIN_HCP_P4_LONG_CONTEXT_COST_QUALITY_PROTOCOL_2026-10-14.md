# Beglin HCP P4 — Matched Contextual Routing Risk and Native Precision Cost/Agreement Data

## Scope

P3 proved that true full-Softmax rank-6 / rank-7 risk can be captured without affecting original final logits and Top-6 routing but **its two-level PCA+Ridge predictor was worse than flat** on 120 short 8–24-token prompts. P4 now collects *longer 104–112-token sequences* and observes the same prompt under **two existing qNg64 mixed-precision policies** in separate scratch workers.

This is a **medium-context pilot**, not a 4K-token long-context study. Original `qwen_infer.c` has `MOE_MAXPOS=160`, while the `QWEN_MOE_PROMPT_IDS` comma-string parser copies into `char buf[1024]` without failing on overflow/truncation. No production code changes are authorized to raise those limits. Therefore P4 caps input at 112 IDs and under 900 bytes, and explicitly verifies actual processed positions. The persistent serving supervisor advertises a 4096-token ingress cap, but this does **not** certify that the separate original verification path or native KV implementation supports 4096.

## Immutable corpus and independent split

- Frozen independent-question source: `research/trainable-micro-router-120prompt-20261014` commit `1b4606eef596e10f3003f7d9f07b607616e990d6`.
- 12 domains, 3 unique paired-question notes per domain, **36 distinct tokenized prompts**. Train, validation and final holdout have 12 different prompts each, with no constituent source question reused across splits.
- Each text is context-expanded, not a natural multi-turn transcript. Do not claim real conversation-distribution generalization.
- Real DeepSeek tokenizer: offline only, **104–112 tokens**, SHA-bound and no duplicate inputs. [Actual XOX tokenizer preregistration run #38465712995](https://github.com/popixoxipop-collab/beglin/actions/runs/38465712995).
- This is a probe of token horizons 4, 16 and 32. The future label is the minimum true Top-6/Top-7 relative gap across the 26 routed layers and future tokens; a dataset feature at time t cannot use those future labels as input.

## Real 26-layer AF-route dataset

- The immutable original checkpoint index SHA-256 is `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`.
- A scratch-only copy of the C inference file is instrumented *after full-64 Softmax and Top-6 ranking but before Top-K renormalization*, while layer1 h2 capture covers all <=112 positions. It does **not** alter the router scores, model operations or permanent source.
- Initial 112-token hook-on vs hook-off smoke must show **byte-identical original full logits and Top-6 routing**, and full 26-layer rank6/rank7 plus 112 h2 positions.
- 36 fresh processes isolate each prompt's attention KV history; no shared input state.
- Per prompt, compute causal 8-feature current state, 4-token and 16-token exponentially smoothed states, current true route gap, and observable complete-horizon 4/16/32 future labels.
- Original full logits and raw hidden vectors are processed and deleted per case in the scratch root. The durable artifact retains only derived scalar features (not vectors, original logits, raw 64-way router scores or token IDs), SHA digests, prompt splits and safe metadata.

## Real native qNg64 policy A/B experiment

Comparison is descriptive and **not a qNg64 precision promotion**:

- A startup: shared_up_proj at layer 3 n=6, shared_down_proj at layer 26 n=5.
- B combined: shared_up_proj layer 3 n=5 and shared_down_proj layer 26 n=6.
- qNg64 precision quality is nonmonotonic; do not call a larger `n` safer without intervention evidence.
- Use a new isolated `AdaptivePersistentRouteWorker` under `RUNNER_TEMP`, never the live active-route supervisor. Demand exact stage-binary hash, checkpoint identity, live route-file before/after SHA equality and each native worker ACK matching A or B.
- Match each original prompt exactly by tokenized-input SHA. Run AB or BA alternating by prompt to reduce order confounds, request four new tokens in each native policy. Compare first generated token with the **original AF verification forward's last-position argmax**, plus generated A/B sequence agreement. All original token IDs/response IDs remain in per-run scratch.
- Record native per-admission engine wall time, full roundtrip time, policy transition cost, cache bytes added, worker RSS, inference passes, and relevant identity SHA. Report p50/p95 and paired differences; separate time spent rebinding from model computation.
- **AF-reference agreement is not BF16 ground truth, perplexity, factual accuracy or human task success**. Report missing true task-quality labels, not fictitious correctness.
- The first two matched native inputs are a smoke test; only after both policies succeed are all 36 run.

## Stop and authenticity gates

Any missing exact input hash, 26-layer rank6/7 entry, future horizon label, runtime ACK, reused prompt identity, nonfinite result or live route modification blocks the entire experiment. If native comparison fails, retain the already finished **route-risk-only** artifact with `PARTIAL_BLOCKED_NOT_EVALUATED` status and explicitly distinguish it from full A/B completion. All per-run scratch files are deleted after any attempt, regardless of test outcome.

The independent `tools/hcp_p4_evidence_gate.py` (research branch) verifies the 36 original prompt groups, complete future32 labels, qNg64 A/B response and measured p50/p95 cost, and computes paired **domain-cluster** bootstrap intervals. Its only possible result is research evidence; policy change allowed and auto-promotion remain false.

## Execution links

- [XOX source and MAxPOS contract #38465002727](https://github.com/popixoxipop-collab/beglin/actions/runs/38465002727).
- [XOX independent prompt length preregistration #38465712995](https://github.com/popixoxipop-collab/beglin/actions/runs/38465712995).
- [Source-only regressions #38466641860](https://github.com/popixoxipop-collab/beglin/actions/runs/38466641860).
- [Full data collection and matched policy A/B experiment](https://github.com/popixoxipop-collab/beglin/actions/runs/38467501959) — current run; numerical results require a complete SHA-bound receipt.
- Research branch: `research/hcp-p4-longcontext-costquality-20261014`, never merge or activate automatically.

## Interpretation gate

Even if mixed policy B is faster than A on these inputs, no production inference speedup is claimed without multiple seeds/time blocks, actual task labels, a nonempty exact backend/trigger certification set, and full end-to-end policy lifecycle amortization. P7's archived first transition (~615ms) and warm restore (~307ms) were much larger than one archived policy's per-request steady advantage (~1.5ms). A multi-token horizon alone cannot pay back that cost. Performance must be compared on matched context, clock-time and full model cost including the planner.
