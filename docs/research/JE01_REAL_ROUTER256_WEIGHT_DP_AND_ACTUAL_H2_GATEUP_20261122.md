# JE01 — actual Router256 native precision DP and original27 routed `h2` activation-weighted gate/up GEMV

**2026-11-22 KST. Research-only, true pretrained model, signed D50 evidence. NO model promotion, NO heldout/OOD use.**

## 1. Full 256 original router frequency × native pretrained coefficient-space precision costs

The already SHA-attested original BEGLIN AF4 model supplied 256 unlabeled WikiText TRAIN last-token inputs and 26×64 softmax router ranks. The selected eight routed Expert IDs were independently SHA checked. Full original BF16 DeepSeek-V2-Lite-Chat four shards and original AF4 9.8 GB were SHA revalidated. True BEGLIN native original q4g64 and qNg64 C encoders reconstructed 8 Experts ×3 FFN roles = **24** roles / **69,206,016** coefficients. The objective is **Router count × BF16-normalized weight MSE**, explicitly not activations, Fisher, Hessian, downstream NLL, nor 256 prompt hidden vectors.

Under the exact original eight-Expert **43,253,760 bytes** physical role budget (AF4 role = 1,802,240 B, Q4 symmetric = 1,622,016 B, Q5 symmetric = 1,982,464 B), a Q4 replacement saves 180,224 B and a Q5 replacement spends 180,224 B.

Exact dynamic program over all 3^24 assignments under exact budget found **unchanged original AF4 on all 24 roles optimal**, score difference 0. The best changed policy (one L13/E22 `gate_proj` Q4, one L21/E61 `up_proj` Q5) has strictly worse weighted proxy score **+0.8527184929021485**. Across 24 true source matrices Q4 MSE > original AF4 MSE > Q5 MSE; minimum per-role Router-weighted Q4 penalty **1.1704497916862855** versus maximum per-role Q5 benefit **0.31773129878413703**. This proves no strict benefit of any pairwise exact-byte promotion/demotion **for this measured proxy**, not for actual NLL.

- [Exact 256 router×24-role source evidence](https://github.com/popixoxipop-collab/beglin/blob/research/je01-moe-bayesian-bit-allocation-20261117/docs/research/JE01_REAL_ROUTER256_NATIVE_AF4_Q4_Q5_EXACTBYTE_DP_20261122.json), canonical receipt `74e6c4c7a37b8172eb2a9c2f7b7dd9a14af1849c4897077c7d0d622b02b14337`.
- Actual D50 admitted CPU job `job_f713e67824d9f868a84b514beb832947`, SUCCEEDED exit0.
- [Independent 256-router×real-weight DP re-computation CI 44961659641 — PASS](https://github.com/popixoxipop-collab/beglin/actions/runs/44961659641).

## 2. Actual 27-layer routed hidden-state capture, with original gold-output identity

Scratch-only `mlx_moe.cpp` instrument emits **true pretrained model post-attention normalized FFN input h2** (2048 float32 dimensions) and selected Router scores at the original **last PROMPT token**, never at the end of the teacherforced target. A first technical attempt was rejected because it used full target `rq_plen−1`; a read-only forensic audit found the mismatch. Revised native C capture uses frozen `gold_meta.first_gold_start−1` and the identical original C++ forward path. The instrumentation is only in a separately compiled scratch Metal binary.

Revised original27 Metal executable SHA `ec703e0267d11a99caaa49b60ee906c63e0544bb72363d2beca3f84d1da788d9`; actual EOE Job `job_6dc2a2672119afac510622da7e476bc1` **SUCCEEDED**. It collected **45 actual routed h2 records** across 16 TRAIN prompts and eight frozen Experts. Each record contains request ID, layer/expert ID, last prompt position, actual Router weight, h2 L2/RMS/max and the SHA256 of the actual 2048-d vector.

**Critical parity:** original 656 gold-target output file SHA is **bit-identical** to the previously certified uninstrumented native original AF4 output `b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c` with original NLL **2.291799702354307**. Actual 45-record h2 trace binary SHA `238642649f68d111df3b60f60083a47d3a6074ada881ccb36f813dd606f9a5b4`; canonical receipt `42213c246c488dcef8176c468811fe516d542dd367532ea191c21cfd1d17d6ca`. No heldout/OOD labels read.

These are **45 routed events**, not 45 independent training prompts. Per-Expert routed observations vary from 4 to 8. Actual h2 RMS mean spans 0.589 (L12/E13) to 1.334 (L24/E26) in this tiny pilot.

- [Signed original27 real-h2 per-observation receipt](https://github.com/popixoxipop-collab/beglin/blob/research/je01-moe-bayesian-bit-allocation-20261117/docs/research/JE01_TRUE_ORIGINAL27_LASTPROMPT_ROUTED_H2_TRAIN16_20261122.json).
- [Revised native h2 probe compile on Apple Metal, CI 44962335940 — PASS](https://github.com/popixoxipop-collab/beglin/actions/runs/44962335940).

## 3. True FFN-input activation-weighted BF16/AF4/Q4/Q5 gate/up GEMV

Using each **actual, raw physical 2048-d original `h2`** captured above and the same independently pinned original BF16 / AF4 source tensors, apply native original BEGLIN quantizers Q4 and Q5, then evaluate full 1408-dimensional expert `gate_proj` and `up_proj` outputs with real original router weights. Experimental metric per role, with weight (p_t) and activation (h_t), is

\[\mathcal{E}_{b,r}=\frac{\sum_{t\in\mathrm{routed}(r)}p_t\lVert(W_{b,r}-W_{\mathrm{BF16},r})h_t\rVert_2^2}{\sum_t p_t\lVert W_{\mathrm{BF16},r}h_t\rVert_2^2}.\]

This is a measurable **activation-conditioned output-space squared perturbation under the IDENTITY output metric**, which is strictly more operational than naked weight MSE but **NOT** neural NLL loss curvature / Hessian / gradient Fisher. True `down_proj` SwiGLU input is unavailable and not substituted with `h2`.

| 16 gate/up pretrained roles ×45 total actual routed h2 | Root mean normalized output MSE |
| --- | ---: |
| Original affine AF4 | **0.06549721022313063** (6.55%) |
| Symmetric native Q4 | **0.10883960251207181** (10.88%) |
| Symmetric native Q5 | **0.05096151463578087** (5.10%) |

Each observed role: native Q4 output MSE > original AF4 > native Q5. For the exact-byte Q4/Q5 pair exchange among these **16 measured roles only**, multiplying the per-role normalized MSE change by observed sum of true Router scores gives minimum Q4 cost **0.0017831582121775665** (L24/E60 `up_proj`), versus maximum Q5 recovery **0.00143161569163315** (L20/E17 `up_proj`); minimal positive shortfall **+0.0003515425205444164** under this **local proxy**. This is NOT a global 24-role budget optimum; no `down_proj` or loss-gradient evidence exists.

- Actual EOE native CPU trace/weight GEMV Job `job_6784049c99a434ce842a05854ee9798e` **SUCCEEDED**, 16 actual gate/up roles evaluated from 45 actual h2 records, native Q4/Q5 physical encoder and original true source SHA rechecked.
- [Signed actual original27 activation-weighted gate/up source output receipt](https://github.com/popixoxipop-collab/beglin/blob/research/je01-moe-bayesian-bit-allocation-20261117/docs/research/JE01_TRUE_ORIGINAL_ROUTED_H2_GATEUP_ACTIVATION_WEIGHTED_GEMV_20261122.json), canonical SHA `d55f4b506076a3a30f73c582ede270c38d41ac12b06190e959ac5a7383157a89`.
- [Independent h2/canonical GEMV audit CI](https://github.com/popixoxipop-collab/beglin/actions/workflows/je01-original27-h2-real-gemv-audit.yml) checks SHA, 45 records, 16 roles and algebra (does not re-run physical GEMV because the raw h2 and 9.8/31GB weights remain on EOE).

## 4. Status and forward experimental protocol

**Measured source-output proxy favors original AF4** over forced equal-byte symmetric Q4/Q5 precision exchange on all 24 source-weight roles and on the 16 observed gate/up real h2 roles. This is consistent with observed actual original27 pilot NLL original AF4 2.291799702354307 vs 3/5bit 2.3067548780457776, 12Q4/12Q5 2.2975364339707807 and 12 affineAF4+6Q4+6Q5 2.295162638977357. These pilots reused the same 16 train sentences and were selected iteratively, so no external generalization evidence exists.

**Not measured, must NOT be invented or passed as H3/JE01 performance:** full 256 training actual hidden activations, down_proj SwiGLU inputs, per-token loss gradients, Jacobian-output importance metric, Fisher/Gauss–Newton or empirical Hessian; end-to-end new NLL under an actual Fisher plan, independent same-source128/OOD128, p95 throughput. The 256 router samples provide ONLY last-token routing counts. The real FFN h2 is currently only 16 training samples.

**Recommended next physically grounded order:** collect disjoint additional training-only h2 and true `down_proj` SwiGLU activations with output-bit parity; compute activation covariance by expert/role with same original model; condition output error by router score and covariance (\mathrm{tr}(\Delta W^T\Delta W\Sigma_x)); when feasible obtain native NLL output gradients and compute actual empirical Fisher (or explicitly call the result a proxy), then choose precision/exact bytes before sealed independent validation. Keep existing AF4 as default and PR #94 Draft. Current Tailnet release current/live `g6-0.9.0-operator.900500`, converged, unchanged.