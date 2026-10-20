# G0 – 300-anchor acquisition receipt (2026-10-21)

**Status:** `PROVISIONAL_300_NOT_FROZEN`. Technical acquisition succeeded. Not original-author row-identical ACS replication and not permission to open final evaluations.

## Confirmed collection

- Job: `job_9869b799a6b9403be1b384478e7499ba`, `SUCCEEDED`, exit code `0`.
- Exactly 15 source scenarios, 20 prompts each = **300 distinct exact normalized text hashes**; configured seed 42 and 10–300 character filter.
- Sources: MMLU, NaturalQA, NarrativeQA, BoolQ, HellaSwag, TruthfulQA, GSM8K, MATH, IMDB, CivilComments, RealToxicityPrompts, HarmBench, SimpleSafetyTests, XSTest, Anthropic Red Team. Per-scenario count = 20.
- Source collector: `research/oml_acs/build_anchors_300.py`; reported execution input SHA-256 `9fa6875c522e52b27f330b4ad17984840f32acc0acec45201fc02f700ab64bac`.
- **Raw 300-row file SHA-256:** `47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b`; size 228,835 bytes.
- Raw prompts moved to WSL ext4 user-owned `/home/alienware-r13/.local/share/beglin/oml_acs/20261021/anchors_300_local.jsonl` (mode 0600). Raw G-drive source removed after byte-level SHA match. Safety and toxicity prompts are **not committed**.
- Local SHA-only full-source manifest SHA-256 `ea054e0eb7961940121208bba996a79ee9184e8dee04fb0c7789e17d4c2e47ca`.
- Local separately redacted 300-row digest file SHA-256 `70d15d773f3079d4e91502194bf495cd795650b46cd2ba1dece79428cea8279c` (89,705 bytes). Automated transfer of that larger row-level digest through the connected host/GitHub boundary was not permitted; it is **not represented as published here**.

## Source and methodology deviations

- Original: Kim & Han, *Cross-Family Universality of Behavioral Axes via Anchor-Projected Representations*, arXiv:2605.09875v1, Appendix A Table 5, 15 HELM scenarios ×20 anchors. Author-selected exact IDs are unknown; pinned Hugging Face dataset mirrors were used.
- Large NarrativeQA source was limited to pinned Parquet shard `train-00000-of-00024.parquet` to avoid multi-GB downloads. This changes its candidate sampling frame.
- Anthropic Red Team was read from the separately pinned `red-team-attempts/red_team_attempts.jsonl.gz` artifact, *not* the chosen/rejected RLHF training examples.
- Exact text uniqueness passed. No independent semantic near-duplicate audit or license-by-source authorization review has passed.
- Earlier canceled/failed acquisition jobs are **not** counted as success.
- **V6 HCP/P3 previously consumed 24 prompt IDs/hashes remain unavailable**; unrelated NCLT “holdout_v6” manifests are not evidence for this LLM experiment. Hence final evaluation remains SEALED and never promoted to unused holdout status.

## Equation-fidelity correction

Paper’s displayed Eq. (4) returns the source mean and Eq. (5) returns the adjoint `Ahat.T @ c`. Its prose also says unit-normalize directions. Both literal, scale-preserving functions and normalized steering variants are now separately represented in `research/oml_acs/acs_core.py`. EOE NumPy source-level equation tests: **15/15 PASS**. Previous single-/two-family engineering results are preserved and explicitly marked *normalized variant*; no revised scientific result is claimed.

## Next development-only step

Extract 300×1536 BF16 Qwen2.5-1.5B final-token activations at zero-based layer 14 from the *private* row file, keep activations private, and commit aggregate SHA/cost evidence only. This is an extraction sanity gate, not detection AUROC, generation steering effectiveness, or A/B/C/Q hypothesis acceptance.

**Production / main / existing V6 verdict untouched.**
