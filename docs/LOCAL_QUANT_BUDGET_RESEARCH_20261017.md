# BEGLIN local qNg64 quantization budget research (2026-10-17)

Scope: **engine algorithms only**. No Tailnet, AWS, model-copy certification, live routing, or automatic quantization promotion.

## Fix: serialized qNg64 weight bits

The original `tools/quant_sim_n.py` used `n + 32/(64*n)` as its claimed effective bits/weight, although qNg64 group-64 has **64 signed n-bit codes plus one 32-bit scale** per group. The correct amount is **`n + 0.5` bits per weight** for code + scale only. The simulator now imports the exact storage helper and rejects n outside 2..15, **n=4** (which uses the distinct q4g64 format), and malformed shapes.

- The distinct **q4g64** representation costs 4.5 bpw under the same one-F32-scale-per-64 convention, but it is **not qNg64(n=4)** and is not selectable in this planner.
- 5-bit qNg64 -> 5.5 bpw, not 5.1.
- 7-bit qNg64 -> 7.5 bpw, not ~7.0714.

The existing XOX local `quant_sim_n.py` has uncommitted **error feedback** not present in this GitHub main variant, so do not overwrite that file. Apply the same storage formula selectively later.

## New functionality

1. `tools/qng64_storage.py`: exact packed size and CPU/Metal format capability. **CPU qNg64(n=4) and GPU qNg64(n=4/8) are rejected**, matching the native `qng64_dense.c` width gate; q4g64/q8g64 are distinct formats.
2. `tools/precision_budget_search.py`: bounded exact Pareto DP minimizing weighted per-tensor reconstruction proxy under serialized payload byte cap; no monotonic-accuracy assumption.
3. `tools/precision_budget_fixture.py`: deterministic **synthetic** BF16 tensor calibration for tests.
4. `tools/precision_budget_collect.py`: SHA-pinned `quant_sim_n.py` summary ingestion -> offline candidate policy; rejects stale/wrong-bpw and wrong/unaligned tensor identity.
5. `tools/test_precision_budget_search.py` and `tools/test_precision_budget_collect.py`: 24 offline test cases, including 75 brute-force exact-optimum comparisons within a randomized case.

To run in the repository:
```sh
python3 -m unittest discover -s tools -p 'test_precision_budget*.py' -v
python3 tools/precision_budget_fixture.py --backend mlx_metal --budget-frac 0.5 --output /tmp/quant_profile.json
python3 tools/precision_budget_search.py --input /tmp/quant_profile.json --output /tmp/quant_candidate.json
```

**Demonstration only:** Six generated BF16-truncated tensors (12,288 total elements) with arbitrarily assigned importance weights `[8,3,0.5,14,4,0.5]`, an 11,520-byte qNg64 payload budget (vs 24,576 BF16 bytes). Search yields `[7,7,6,9,7,6]` vs fixed `[7,7,7,7,7,7]` at exactly the same 11,520-byte budget and 33.31% lower **synthetic weighted reconstruction-error proxy**. This is not a measured accuracy, throughput, GPU resident-memory, or production result.

All outputs stay `OFFLINE_RESEARCH_ONLY_NOT_QUALIFIED_FOR_LIVE`, with `quality_validated=false`, `latency_measured=false`, `gpu_resident_memory_measured=false` and `automatic_live_promotion=false`. The optimizer never modifies model weights or serves requests.

Next native work: obtain real per-tensor calibration from an authorized sandbox; compare original BF16, uniform qNg64 and mixed qNg64 end-to-end logits, tokens/s, GPU peak memory, and weight-epoch/KV cache correctness separately for CPU and Apple MLX/Metal.

Native C kernel regression: `tests/test_qng64_dense_contract.c` compiles the repository's real `qng64_dense.c` encoder/decoder/GEMV without model files, verifying n=4 reject, n=8 distinct support, packed code plus scale bytes and numerical matvec agreement. See `.github/workflows/beglin-local-quant-budget.yml` for Linux/macOS tests across Python 3.11/3.13. This is a kernel contract test, not a BF16-vs-mixed production model benchmark.\n\nLocal development verification is in this chat artifact `BEGLIN_LOCAL_QUANT_ENGINE_20261017.zip`; the XOX workspace forbade python test execution (`EXEC_DENIED`), so Apple Silicon inference benchmarking was not run.
