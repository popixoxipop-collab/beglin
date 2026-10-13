# Hierarchical Tupper Grid-of-Micro-Experts for Beglin

Experimental, isolated Python implementation. Origin: Beglin `popixoxipop-collab/beglin` mixed per-tensor qNg64 and MLX MoE paths, but no native engine changes.

**Goal:** an ordinary routed MoE expert is represented as an exact sum of internal neuron-channel micro-experts. For a DeepSeek-V2-Lite-like `hidden=2048`, `intermediate=1408`, `experts=64`, channel width 16 produces an `88×64` routing mask (and each 64-channel parent is assembled from 4 × 16-channel children). This grid can be encoded by a generalized Tupper binary integer, but this is bit indexing, **not weight compression**.

The study deliberately separates three claims:

1. **Exact functional assembly** (proved numerically for synthetic tensors, regardless of channel size and joint neuron permutation).
2. **Shared-law compression** (tested only on structured and independent synthetic weights at the full dimensions; real LLM structure unknown).
3. **Second-stage sparse micro-routing** (quality currently fails: even the calibration-selected half of micro channels exhibits ~70% output relative error in the synthetic probe).

## Files

- `hierarchical_moe_probe_v1_synthetic.py`: immutable, SHA-bound source used for the published synthetic full-shape JSON results.
- `hierarchical_moe_probe.py`: successor with `--checkpoint-index` read-only real-Safetensors mode.
- `real_checkpoint_reader.py`: mmap-backed BF16/F16/F32 per-layer unfused expert reader; validates index tensor shapes, sizes and shard containment.
- `test_hierarchical_moe_probe.py`, `test_real_checkpoint_reader.py`: currently 11/11 local tests PASS including a generated fake Safetensors end-to-end fixture.
- `fullshape_*.json`, `fullshape_exact_swiglu.json`: measurement artifacts; they do **not** represent a pretrained checkpoint.
- `RESULTS_2026-10-13.md`: full methodology, numeric results, cost accounting, limitations and real-run command.

## Run

```bash
python -m pip install numpy scipy
OPENBLAS_NUM_THREADS=1 python -m unittest discover -v
OPENBLAS_NUM_THREADS=2 python hierarchical_moe_probe.py --mode structured --output /tmp/structured.json
OPENBLAS_NUM_THREADS=2 python hierarchical_moe_probe.py --mode independent --output /tmp/independent.json
```

The full-size experiment generates and processes the **whole** `64×1408×(3×2048)` synthetic expert layer in tiles; it does not require a >2 GB monolithic weight stack. Source data exists only in a deterministic RNG stream and is discarded tile-by-tile. Peak RAM was ~1.4 GB per process.

## Real checkpoint prerequisite

Access to an actual local checkpoint plus full checkpoint hash/identity verification is required. The model and weight files were unavailable in this session: neither the Tailnet nor Desktop Commander connection could be used to read XOX. Once accessible, use the read-only real checkpoint command in `RESULTS_2026-10-13.md` and **do not merge results with the existing synthetic JSON**. No files in this experiment perform inference-engine activation or production mutation.
