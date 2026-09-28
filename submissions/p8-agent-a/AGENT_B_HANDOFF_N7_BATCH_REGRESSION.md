# Agent B handoff — kv_a/L11 n=7 cbatch regression

Agent A did not modify `mlx_moe.cpp/.h`.

## Reproduced P7 failure

- target: `kv_a_proj_with_mqa/L11`
- n: 7
- qNg64 quantization completed
- GPU promotion completed with 0 bind failures
- contract-005 activation emitted:
  - sample_count=32768
  - mean_abs=0.345896677
  - rms=0.978154826
- failure immediately after activation:
  - `FATAL: [moe gpu cb online] mlx_gpu_cbatch_layer_step_lazy failed at layer 11 step 0 (pass 0)`

This is not an n=7 unsupported case.

## Read-only diagnosis of current mlx_moe.cpp

Current `qng64_gemv_e0()` assumes the single-row attention path:

- custom kernel indexes `x[g*64+p]` with no batch offset
- output shape is fixed to `{1, out}`
- dispatch grid is `{64, out, 1}`

But `mlx_gpu_cbatch_layer_step_lazy()` calls `lazy_matvec_e0()` with `x` shaped `{A,in}`; P7 uses multiple active columns. For the promoted kv_a at L11 this returns a one-row qNg64 result into a batched graph, which throws inside the cbatch layer step.

This matches the previously observed class of bug: qNg64 GEMV batch>1 computes/returns only the first row.

## Required B-owned fix

Make the qNg64 attention GEMV batch-aware without changing Agent A files:

1. derive B from `x.shape(0)` for rank-2 `{B,in}`
2. custom kernel gets a batch/z index
3. read x with batch offset: `x[z*(ng*64)+g*64+p]`
4. write output with batch offset: `out[z*out_dim+row]`
5. output shape `{B,out}`
6. dispatch grid z dimension B
7. preserve B=1 exact behavior
8. validate B=1, B=2/4 multi-request and prefill
9. rerun `kv_a_proj_with_mqa/L11 n=7` under canonical P8 sweep

Do not classify n=7 as unsupported: the current binding map and P7 activation prove the tensor was promoted and bound before the cbatch failure.
