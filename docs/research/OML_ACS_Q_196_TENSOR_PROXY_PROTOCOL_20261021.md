# Q 196 tensor real BEGLIN allocation protocol (source-only)

**Lock timing:** before per-tensor distortion scan, any new PPL/decoding benchmark, or mixed-policy performance observations.

All 196 Qwen2.5-1.5B 28-layer attention/FFN projection matrices are read from a revision-pinned BF16 model file. Original *native* BEGLIN C encoders, **packed Q4G64** and **qNg64 n8 bit-planes**, encode each tensor using group=64. Unlike the earlier one-tensor result, the entire projection stack is considered.

**Important:** tensor weight relative error is a *surrogate* of downstream model sensitivity, not measured task sensitivity. It is allowed to select candidate precision only, not to declare layer importance in real LLM behavior.

For each of seven roles, select **4 of 28 layers** where `relative_error_4^2 - relative_error_8^2` is highest. Promote 28 tensors to true n8, keep the other 168 tensors packed4. Build a seeded **role-matched random control** with four layers per role, NumPy PCG64 seed=73, so logical memory cost is identical. Compare both to full F32 (source BF16 widened), all4 and all8 on the *same native BEGLIN* process and original HF BF16 checkpoint. Output head, biases, norms remain unchanged across modes (this Qwen has a tied FP32 head).

Evaluation: public synthetic token windows, source only; **no old 300 anchors, 600 labels, 200 C1D test rows or previous V6 24**. Preserve candidate selection before seeing any mixed-policy outputs. Compute full logits top1, KL/relative error, teacher-forced per-token NLL, and actual timed throughput/peak RSS if feasible. Never fabricate missing metrics. An observed one prompt or NLL-only result is development evidence, **not** an independently confirmed accuracy drop/AUROC/NLL final criterion.

No changes to production, main, XOX dirty worktree, operating release or source model. The code must keep a separate immutable source SHA and SHA-bound research copy.