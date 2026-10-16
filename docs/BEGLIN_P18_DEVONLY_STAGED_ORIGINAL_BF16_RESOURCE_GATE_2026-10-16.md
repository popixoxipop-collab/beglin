# Beglin P18 — developer-only shared BF16 full-model execution gate (2026-10-16 KST)

**Current decision:** dev-only code prepared; actual original 27-layer/5,291-tensor parity and runtime speedup **NOT certified**. The baseline P17 original BF16 evaluator remains authoritative. Preserve production policy A and `automatic_live_promotion=false`.

## Verified source identities

- Dev-only fixed 203-token source SHA-256: `b628ec3247cfc7a94785a93e539562a7a2bc7726f133c21162479749b8f0ac2c`. This is expressly **not** a frozen 480-test item.
- Reviewed two-gold-window v2 BF16 core: `c160e79ecb527ccfcc7c908e7fc64f67abf8938a4186c2f7033eac753c7cde2f`.
- Original standalone dev qualifier: `5b50cd59f7b056177a5ef696e8603299c722081ae00f6d21ba634d15af2620a1`.
- Staged resumable EOE/D50 source, exact latest SHA at last local read: `cf5b20128bfa93d151d24eb1969397fa9e1def35025c13f295b392a953f6f7ff`.
- Original P17 torch model code `40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3`; 5,291 original BF16 tensor-value list `88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda`.
- EOE D50 root: `/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015`; source script `hcp_p18_devonly_staged_shared_bf16_parity.py`.

## Actual qualification completed here

- CPython 3.13.5, PyTorch 2.10.0+cpu: reran original v1, v2, causal-attention and dev contract unittest groups; **45/45 PASS**, exit code 0. This is **toy/static** testing: not a real DeepSeek reference run.
- V2 original BF16 core, dev-only one-shot qualifier, staged interrupt-resumable source: source SHA pins and Python AST checks passed on EOE earlier. The staged script's gold-shape and source/index/order receipt checks were strengthened, but the 27-layer stages **have not been executed**.
- EOE resource-planner check for CPU2/RAM9GiB reported `insufficient_free_cpu`. Subsequent check for CPU1/RAM9GiB reported `offline`, `insufficient_free_ram`, `insufficient_free_cpu`. EOE resource inventory when it was reachable showed 10 logical CPUs with load1 about 17.6, and a later host request timed out. Do **not** bypass the planner or assume a model job completed.
- No new genuine BF16 logits or NLL were obtained in this developer-only shared-batch qualification.
- Separate preexisting original P18 independent BF16 coverage was **10/480 unique questions** with **6/480 BF16+native A/B paired** at the latest fully audited checkpoint. These are earlier results and do not increase by this QA.

## Next valid controlled execution

1. Read fresh EOE `release_status`/`runtime_identity`/`resource_inventory` and `process_list`. Require release convergence, no competing research run, enough CPU, RAM, swap and D50 reserve.
2. Read SHA of staged script and its two imported modules, the P17 original and the fixed 203-token dev-only source. Require exact source hashes above. Do not access or modify any official P18 frozen 480 test question during optimization.
3. Call `resource_plan` for a single EOE CPU2, RAM9GiB isolated stage; proceed **only** when eligible. Execute stages `standalone_short`, `standalone_long`, `shared_short_long`, `shared_long_short` sequentially with no overlap. Each forward must consume 27 original layers and 5,291 genuine BF16 tensors; source value SHA must match the original P17 manifest. All 3+5 gold positions must produce finite full-vocab 102,400-element float32 logits.
4. Run a **read-only independent audit** that checks raw .npy SHA against both standalone baselines for both shared processing orders, absolute float32 difference zero, source once-only loading, peak RSS <=9GiB, new swap <=1GiB, and actual wall-time. Never treat a failed or missing stage as a PASS.
5. Only after full real-model numerical parity, compare measured wall-time gain to standalone BF16 and decide whether to admit the optimization to a later independent BF16 evaluation. The frozen 480-question A/B quality, latency, full generation and statistical gates remain separate and blocked until complete.

**No production change or B promotion was performed.**
