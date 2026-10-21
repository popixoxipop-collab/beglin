# BEGLIN Q v2 — activation-aware bit allocation and measured asynchronous CPU dispatch

Protocol locked before BF16 activation calibration or any new test-window NLL.

Source-only quality estimate uses 196 original Qwen projection inputs collected via BF16 forward hooks on Alienware (one GPU job). Estimate per-tensor 4bit-versus-8bit **diagonal activation-weighted local output error reduction** from original BEGLIN C packed encoders. It is **not** whole-model causal output sensitivity or Fisher/Hessian. No blocked per-tensor full-model-logit experiment is retried.

Independent output calibration CPU tasks are allowed to run asynchronously across bounded CPU workers **after** the single source BF16 forward capture. Q/K/V and gate/up are potentially independent on a decoder layer, but on shared CPU their 4bit six-thread kernels may contend; any latency DAG credit requires observed actual concurrency tests. No unverified parallel speedup will be assigned.

Select exactly 28/196 upgraded n8 at unchanged 830,570,496-byte maximum with the original 100ms cost bound, using per-item weighted quality proxy and native thread-aware cost. Compare against role-balanced RNG73, former source-weight-only kernel-aware, all4. Source public 12 prompts and held evaluation public 8 prompts are exactly those prelisted in adjacent JSON; evaluation prompts cannot enter selection. Keep Q hypothesis NO_VERDICT without independently replicated real task NLL/AUROC and prior V6 use-record audit.

References: HAWQ-V3 (Yao et al. 2021) uses hardware-aware mixed precision ILP; AWQ (Lin et al. 2024) motivates activation sensitivity, but current estimator, kernel layout and tests are BEGLIN-specific and make no novelty claim over these ideas.
