# C1-D — Input-conditional ACS dynamic gating (protocol freeze before new prospective-development evaluation)

**2026-10-21, development-only.** The previous source-only fixed OOF co-error weights improved average AUROC by +0.0022 on *already used* SST-2/CARER development sets, and achieved 0/6 of the +0.02 gain threshold. No final C1 claim is supported. Do not tune this gate on those labels.

This next gate uses **only source-model input features and historical source-OOF binary errors**: for each new prompt, compute the two source models' 300-d canonical point projections; average their separately normalized cosine similarities to each of 200 paired historical source train examples; choose a deterministic top-32 subset and estimate its 2×2 source error co-occurrence matrix. Shrink local co-error matrix 50% toward the global 200-row matrix, add ridge 1e-6, and solve the 2-source nonnegative simplex weighting in closed form. The weights vary **per incoming prompt**, without observing target-model label, target-model hidden state, final verdict, or target-dev performance.

Comparator controls are **equal weights**, previous **static source-OOF correlated error weights**, and the **new input-local conditional gate**. The weighted two-source raw projected axes are normalized once for scalar classification; source training fixes polarity. No online learning, no hyperparameter search on target validation.

Before seeing prospective evaluation labels, this protocol fixes:
- **50 positive/50 negative each from two different public pinned train-only sources:** IMDB train (sentiment; short 10–1000 character full reviews) and GoEmotions simplified train (single-label joy=17 / anger=2; 10–300 characters). Source repository revisions and exact Parquet file paths are recorded in JSON. RNG=73.
- Only nonduplicate source rows, excluding exact canonical text overlaps with all old 300 development anchors and 600 SST-2/CARER training/development rows. Independent near-duplicate screening for superficial overlap is mandatory; full semantic plagiarism prevention is not asserted.
- **Three target families × two axes** and 999 paired stratified bootstraps, AUROC delta `conditional minus global_static`; heuristic +0.02 threshold per case, all six cases reported without selection.
- No original test/final data accessed, no old V6 24 holdouts reused, no code changes to production or main, no paid API.

This is an actual input-dependent **behavioral source-axis allocation** gate, not per-tensor BEGLIN 8-/4-bit dynamic quantization. The source model activations must be computed for every new input to obtain the two-feature-space similarity, so per-token latency benefit is *not* presumed.
