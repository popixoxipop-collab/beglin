# BEGLIN OML–ACS G0 300-anchor BF16 capture — 2026-10-21

**Verdict:** `G0_QWEN_300_CAPTURE_AND_PROJECTOR_PASS`. This is a technical extraction and coordinate construction result, **not** an A/B/C/Q research hypothesis result. `main`, BEGLIN serving, V6 prior negative outcomes, and Tailnet releases were not changed.

## Exact observed execution

- Input: 15 public benchmark scenarios × 20 distinct development anchors = **300**, selected from revision-locked Hugging Face sources rather than original-author HELM sample IDs.
- Private source SHA-256: `47d5ddcb37b81212e3f43a34441662062b42a038d4cc606d9d7081c31131f83b`; private WSL ext4 mode `0600`.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`, HF revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, **BF16 weights**, zero-based decoder layer `14` of 28.
- Model snapshot local, no download; RTX 5070 Ti 16GB, CUDA.
- Code: `research/oml_acs/extract_anchors_qwen.py` (SHA-256 `786ffb7320a6d656c6b81aa516acb9358eb277ea70dab34b0ed7491fc55393b1`) checked out from *this research branch*, pinned **commit `47b72a453af4fe656e26f3795f67c8bbfd8297ca`** in a **new isolated Git worktree**, leaving XOX's unrelated dirty worktree untouched.
- Durable execution `job_de3b19e9455efdf5db4e019f20acfa1e`: **`SUCCEEDED`**, exit code `0`, worker time `94.324s`, in-process recorded elapsed `43.48s`. These timings cover differing scopes and are **not a throughput benchmark or speedup claim**.
- Six progress checkpoints printed at 50,100,150,200,250,300; successful final receipt.
- True post-decoder-block final-token activations, stored as float32 NumPy after reading BF16 model:
  - Shape `(300,1536)`, `finite=True`.
  - Array SHA-256 `cbd5535b5247ea7b153a7835e5c27d9b8cc680cc6d031108b87db1a49136f692`.
  - Private `qwen25_15layer_300_anchors_bf16.npz` 874,714 bytes; SHA-256 `4892dd4e1a7ccfafcf7710373b29009702c37bbcb900d4e764eb65198ed8dad1`.
  - GPU peak torch-allocated `3016.51 MiB`. Not end-to-end GPU memory.
- Independent post-run verifier checked file mode, file SHA, float32 shape, finiteness, input-source matching **300 row hashes in order**, scenario counts, and each raw array SHA. All checks passed.
- ACS anchored projection: `mu=mean(A,axis=0)`, `Ahat[i]=normalize(A[i]-mu)` with float64 mathematics, saved privately as `qwen25_300_anchor_projector_v1.npz`:
  - Shape `(300,1536)`; every row norm `1.0`.
  - Gram trace `300.0`; algebraic/numerical rank `299` at `1e-6`; entropy effective rank `31.29`.
  - Private projector archive `3,533,740` bytes; SHA-256 `664f24c3aed82e9dcc15e895d695fcdfc90215dd6ba17e2d708421bb4f0c9722`, mode `0600`.

## Meaning and limitations

1. The model, data, and actual GPU mechanism have now run through the complete **300-anchor Qwen extraction**, proving the technical input/activation/projection path.
2. Rank `299` does not mean `299` independent/useful behavioral axes. Effective rank ~`31.29` implies concentrated spectrum in this chosen collection; this is a **diagnostic observation only**, not proof of cross-model alignment.
3. No behavior labels, steering effectiveness, held-out detection AUROC, calibrated output combination NLL, BEGLIN 8/4-bit native Q test, or third model execution was performed as part of this capture.
4. Dataset variants are original-paper-method **reimplementations**, not row-identical reproduction. NarrativeQA is sampled from one pinned small Parquet shard; source license review and independent semantic near-duplicate audit remain incomplete.
5. The previously evaluated **24 BEGLIN HCP/V6 holdout records are not independently bound to document identities yet**. They cannot be declared unused, and neither new final IID nor final OOD evaluation has been opened.
6. The completed output is private in the authorized WSL user home. GitHub contains only **aggregate evidence**; raw benchmark prompts and activation matrices are not published.

## Next bounded actions

- Verify exact same 300 anchor order on a **second architecture** (Mistral 7B in exploratory slot or Phi-4-mini after pinned public weights and architecture review), producing a separate `300 × hidden_dim` activation matrix and source-matched hash receipt; no target labels.
- Report literal ACS Eq. (4)/(5) and normalized-axis variants separately, including their scaling under steering; no post-hoc hidden evaluation tuning.
- Then evaluate model-family detection with disjoint behavior data and strong random/shuffled/control baselines, and only after preregistration + V6 identity audit open final splits.
- Handle A, B, C1, C2, Q **as five independently judged questions**. Any observed no-effect or degradation must be retained.

## Evidence

- Machine-readable capture + independent projector: `docs/research/OML_ACS_G0_QWEN_300_ACTIVATION_RECEIPT_20261021.json`.
- Source and anchors: `docs/research/OML_ACS_G0_300_ANCHOR_RECEIPT_20261021.md`.
- Earlier G1 technical-only, two-family 12-anchor experiment: `docs/research/OML_ACS_G1_TWO_FAMILY_TECHNICAL_20261021.json`. Do not merge its exploratory ranking into the 300-anchor extraction verdict.
