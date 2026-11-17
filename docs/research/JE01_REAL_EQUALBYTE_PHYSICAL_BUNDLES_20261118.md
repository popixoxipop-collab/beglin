# JE01 — actual source-BF16 8 Expert exact original-AF-budget candidate blobs

Input is the *physically created* original Chat BF16 72 native n2/n6/q8 expert-role artifacts, independent SHA/byte audited in GitHub Actions run 44172058412, original same source complete SHA identity proven (DeepSeek-V2-Lite-Chat immutable revision 85864749cd611b4353ce1decdb286193298f64c7).

A 2bit/6bit assignment of exactly 3/5 out of frozen eight slots exactly matches actual baseline AF original 8 Expert payload budget **43,253,760 bytes**, independent of changed q4 affine bias metadata. All 56 `C(8,3)` alternatives physically feasible, no model/heldout labels used.

Create three physically contiguous SHA-sealed files with exactly the same byte size: (a) pre-label routing-frequency heuristic, protecting five most frequently routed Experts with 6bit; (b) true pretrained BF16 weight reconstruction proxy minimizing per-role mean relative MSE (NOT NLL and NOT Bayesian NIG); (c) independent deterministic Expert permutation. Every file concatenates actual source-artifact packed-weights+f32 scale regions in original fixed (layer,expert,role) order and writes an offset/sha ledger as sidecar. The total 43,253,760 budget measures ONLY weights+scales, not manifest/zip/envelope overhead, exactly matching existing original AF byte ledger scope.

These three real files are **8 selected Expert payload bundles**, NOT a directly loadable 64-Expert runtime AF model, so source-to-native original q4 parity, real routed full-model integration, Bayesian NLL effects, paired 256 calibration, untouched 128/128 and p95 latency remain NOT RUN. Keep PR Draft/no promotion.
