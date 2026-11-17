# JE01 — native AF eight-Expert exact-budget PRE-SOLVE (2026-11-18 KST)

Status: ENGINEERING PRESOLVE ONLY. Bayesian quality HOLD. No running service, model checkpoint, or Tailnet release changed.

## New finding

The original eight selected (layer, Expert) slots contain 43,253,760 real bytes: each slot has gate/up/down, each role has 1,441,792 original q4 weight bytes and 180,224 bytes each of f32 scale and f32 bias. These source intervals are already audited in the immutable JE01 AF ledger (receipt SHA f47bbccbc3878a70dccb9f4b7f5f7f5bd43087aeec58c948924ef85bb428ca4c).

Existing BEGLIN codecs have DIFFERENT per-slot metadata:
- unchanged native AF q4: original packed 4bit + f32 scale + f32 bias = 5,406,720 B per slot.
- qNg64 n=2: group64 bitplanes + f32 scale only = projected 2,703,360 B per slot.
- qNg64 n=6: group64 bitplanes + f32 scale only = projected 7,028,736 B per slot.
- signed q8g64: int8 codes + f32 scale only = projected 9,191,424 B per slot. The current MLX binder explicitly rejects qNg64 bitplane n=8; the 8bit path expects q8g64. Do not substitute the incompatible codec.

With eight slots and choices {2,4,6,8}, the formula-level EXACT 43,253,760 B combinatorial filter admits 1,317 assignments, including untouched all4 baseline. The only count patterns are (n2,n4,n6,n8):
- (2,4,1,1): 840 assignments
- (4,0,2,2): 420 assignments
- (3,0,5,0): 56 assignments
- (0,8,0,0): 1 baseline

The code in tools/je01_native_exact_byte_presolve.py validates the original signed ledger, computes the codec-aware byte projections, enumerates all 65,536 assignments, retains only the 1,317 formula-matched candidates and emits a deterministic catalog SHA. Tests check the true source byte ledger, bit tier metadata, counts, source tampering, and no false claims.

## Mandatory unresolved gates

1. The eight chosen slots were selected from real native original-AF route events. The older EOE pretrained BF16 Layer11 proxy selected DIFFERENT Expert IDs and therefore cannot supply this study's effects.
2. Before calling any higher-bit alternative a precision upgrade, prove source BF16 tensor identity vs the actual original AF checkpoint. Dequantizing original q4 then re-encoding it to 6/8bit cannot recover missing high-precision information. Original BF16->native AF q4 re-encoding parity must be measured, with any mismatch documented rather than silently calling checkpoints identical.
3. Extract 24 SHA-attested BF16 tensors for selected slots; generate actual native qNg64 n2/n6 and q8g64 n8 weight bytes, plus scale arrays. Compile and bit-exact/dequant parity-test isolated kernels, produce actual artifact SHA, physical byte ledger, and source-to-output provenance. A formula match alone is not an actual same-byte artifact.
4. Keep the original q4 uniform model intact. Apply Expert overlays only to an isolated MLX/Metal 384 scratch runner. Record 256 paired calibration full-gold NLL (at least 32 routed observations per alternative), no topK changes, NIG posterior; re-evaluate complete mixed candidate vs original baseline AND matched-byte permutation/frequency controls.
5. Before any success declaration: untouched WikiText128 and HellaSwag128, paired 20k bootstrap with lower bounds, 0 hard topK mismatch, p95 latency regression <=5%, and independent audit under unchanged preregistered thresholds.
6. Resource admission observed during this session: XOX has no eligible scratch pool (26.5GB disk free vs 36GB storage reserve); EOE D50 disk is eligible, but free allocatable CPU is 0 after existing reservations. Do not circumvent scheduling, cancel other jobs, alter serving workers, merge Draft PR #94, or activate Tailnet release.

This is byte-accounting feasibility research, NOT Bayesian NLL improvement. Final gate: PRESOLVE_FORMULA_FEASIBLE / REAL_NATIVE_ALTERNATIVE_FILE_HOLD / PAIRWISE_NLL_HOLD / NO_PROMOTION.
