# JE01 — original affine AF4 preservation versus symmetric Q4/Q5: real original27 gold16

## Result: source and real native GPU PASS; improvement over original AF4 NOT observed

Actual EOE native Apple Metal 27-layer real-model paired gold-token experiment, 16 WikiText train cases and **656 teacher-forced original gold tokens**, all original-source full SHA pinned. This follows 3/5bit NIG and 12Q4/12Q5 pilot experiments; candidate selected after seeing prior pilot results, **exploratory only**. No independent validation/OOD labels were accessed.

| Native true original 27-layer model | Gold-token mean NLL | Delta from original | Decision |
| --- | ---: | ---: | --- |
| Original AF4 (OFF), verified exact prior output SHA | **2.291799702354307** | 0 | Reference PASS |
| Prior full BF16 derived NIG 6Q3+18Q5 | 2.3067548780457776 | +0.014955175691470757 | Prior pilot worse |
| Prior full BF16 derived 12Q4sym+12Q5sym | 2.2975364339707807 | +0.005736731616473811 | Prior pilot worse |
| **Current original AF4 12 role pass-through + 6Q4sym+6Q5sym** | **2.295162638977357** | **+0.0033629366230503166** (**+0.14673780696%** relative NLL) | Better than prior mix, still worse than original |

The new 12/6/6 mixed candidate has **NLL 0.002373794993423495 lower than prior 12Q4/12Q5** and reduces its observed original-relative loss by ~41.38%, but the *precision assignments change too*: the comparison does NOT isolate a causal AF4-pass-through-only effect. The candidate differs from original AF4 in four Expert FFN weight triplets only.

## Frozen expert assignment (training routing counts only; no posterior/Fisher claimed)

- **Original E64 affine AF4 preserved (unchanged native GPU flow):** L21/E61 (calibration last-token routes 97), L24/E26 (91), L13/E28 (87), L20/E17 (86). All gate/up/down, **12 actual original roles** pass through the original 64-Expert AF4 tensors; they are not rebound into singleton GPU correction tensors.
- **BF16 original Chat-derived symmetric nibble Q4:** L13/E22 (79), L24/E60 (81); 6 roles.
- **BF16 original Chat-derived symmetric bitplane Q5:** L12/E13 (81), L14/E63 (80); 6 roles.

Actual strict binary package **43,253,760 bytes**, equal to original eight-Expert selected-source logical AF4 budget: 12×1,802,240 original native AF4 role bytes (nibble + signed scale + bias), 6×1,622,016 native Q4 (nibble + scale), 6×1,982,464 native Q5 (bitplanes + scale). Original 12 pass-through roles are physically present in the alternative package for source/byte accounting but not rebound; this is NOT a claim that scratch-runtime additional GPU memory is zero or that entire operating model size is reduced.

## Genuine experimental evidence

- Original full AF4 source file SHA-256 `9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4`; native original role packed+scales+real signed biases independently reread and byte-compared on EOE against all 12 physically copied source roles.
- Real equal-byte 12/6/6 payload file SHA-256 **`24cb38797cddcf6e6a3363df54a24d5d4665a39a3be03739eadbf837830e291a`**. Source full physical file 24/24 roles, 60/60 nonoverlapping complete packed/scale/bias byte spans. Signed package receipt **`4f3fa3ebc443bd9974b53f4ed2f804a9c31994e316fa59d099c9955e167f767b`**.
- Scratch-only compiled original27 Metal binary SHA **`856749a39a393a9a2aff011a5ac590a1a82382a3bea209a75839b10cd4eb9a15`**. [macOS compilation and original-AF4 pass-through source contract CI run 44960127705 — PASS](https://github.com/popixoxipop-collab/beglin/actions/runs/44960127705).
- EOE on-host independent true AF4 passthrough byte-comparison research stage receipt **`119297f4a07d808e1563ef575320e4cbff559801ce59da3d19808a34e440dc08`**.
- Real EOE two-mode native GPU Job **`job_27bf8d5ba8ff92035047b2d5cf7fcfd8`** = SUCCEEDED exit0. OFF original NLL **2.291799702354307** and output SHA **`b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c`**, bit-identical to previous source reference. OFF receipt **`adb71aed0abed7de389d0b0bd0918014ed389fdfadd1ae5018f14cf35b31f551`**.
- ON actual native original27 mix NLL **2.295162638977357**, output SHA **`c689432e4405f0ac0a2c2b0669cb628ffa4dd071ca066078ae6c9e20dc591e4e`**, ON receipt **`54b360884f3658b418068779999285402330ff3ae7979e98fd5135648acc916f`**.
- Per-case **10/16 sentences worsened**, 6/16 improved. Added NLL sum across 656 gold tokens **+2.206086424720816**. Largest positive deltas: row12 **+0.66357** (66 tokens), row2 **+0.64362** (35), row7 **+0.57760** (34). No heldout/OOD accessed.
- Exploratory after-observing sentence-cluster paired bootstrap (seed 20261122, n=50,000), 95% interval for new candidate minus original per-gold-token mean NLL: approximately **[-0.001635,+0.007663]**, includes zero. Bootstrap share ≤0 about 8.46%; this is neither preregistered nor strong statistical/generalization evidence.
- Actual OFF wall 61.926s, ON wall 62.042s, minimum reported memory free 70% ON, 71% OFF. This is NOT a throughput benchmark. Previous full Q45/GPU jobs were properly admitted using 2 CPU, 24 GB unified memory and Metal 0.95 share from observed process CPU peak ~1.025; no resource gate bypass.
- [Actual full per-token OFF receipt](https://github.com/popixoxipop-collab/beglin/blob/research/je01-moe-bayesian-bit-allocation-20261117/docs/research/JE01_REAL_ORIGINAL27_AFFINEKEEP4_MIX4_GOLD16_OFF_20261122.json) and [ON receipt](https://github.com/popixoxipop-collab/beglin/blob/research/je01-moe-bayesian-bit-allocation-20261117/docs/research/JE01_REAL_ORIGINAL27_AFFINEKEEP4_MIX4_GOLD16_ON_20261122.json). [Independent source SHA/60-segment, per-gold target NLL/pairing GitHub CI 44960716195 — PASS](https://github.com/popixoxipop-collab/beglin/actions/runs/44960716195).

## Research decision and next gate

**NO-GO to replace the original AF4** based on this pilot: despite progressively lower NLL than failed NIG and failed full 12Q4/12Q5 candidates, it still produces worse true native NLL on current 16 development sentences. Small sample and trial-after-trial selection mean quality comparisons cannot be called a pre-registered test. The loss is not statistically established out of sample; nor is there any evidence of gain.

Next high-value intervention is not another arbitrary precision shuffle. Calculate original-affine-vs-symmetric **activation-weighted groupwise quantization error** and approximate **Hessian/Fisher-weighted perturbation** on the **existing 256 training calibration** only. Freeze an Expert×role precision or scale/bias-preserving allocation under the exact 43,253,760B logical budget before any sealed independent128+OOD128 test. Use real native full27 paired NLL on disjoint training/development segments and explicit abstention/no-promotion controls. Avoid choosing policies from repeated train16 cases. Baseline AF4 remains default.

**Operational guarantee:** Research-only D50 scratch code and data; original native AF4 checkpoint, production release and service unchanged. PR #94 remains Draft/unmerged, Tailnet current/live `g6-0.9.0-operator.900500` converged. No independent holdout metrics asserted.