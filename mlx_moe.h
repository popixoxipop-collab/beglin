// V5a: MLX GPU backend, vendor boundary. Plain C header, mirrors sme2_kai.h's
// shape exactly -- no qwen_infer.c type crosses this boundary, every function
// re-checks availability internally so a caller that forgets to check cannot
// reach a guarded code path. Implemented in mlx_moe.cpp (C++17, links MLX);
// qwen_infer.c stays a plain-C, plain-compiled TU (D-gpu-1/D-gpu-2, plan
// PLAN_v5_v6_gpu_backend_and_role_device.md).
//
// This file declares an interface only -- it contains no fp32->intN encoding
// of its own (it binds/dequantizes bytes an existing quantizer already
// produced), so D18's residual-vs-stochastic-rounding tradeoff does not
// apply to anything here.
#ifndef MLX_MOE_H
#define MLX_MOE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 1 if MLX linked in and a usable Metal device is present, 0
// otherwise. Every other function below re-checks this internally.
int mlx_gpu_available(void);

// Registers one AF-family tensor (E experts, out x in, group size ng) with
// MLX, reading directly out of `blob` at the given byte offsets -- same
// mmap the CPU arm (moe_decode_af/moe_matvec_af) reads, zero-copy where
// possible (D-gpu-3). `bits` mirrors MoeAFTensor.bits (4/8/32); this AF-blob
// fixture registers everything at bits=4 (F-13), but the parameter is real
// so a future caller pointed at a mixed-precision tensor doesn't silently
// assume int4. Returns 1 on success, 0 on failure (unsupported bits, shape
// mismatch, MLX unavailable).
int mlx_gpu_bind_af(const uint8_t *blob, long blob_bytes, const char *name,
                     long E, long out, long in, long ng,
                     long packed_off, long scale_off, long bias_off, int bits);

// Reports how many previously-bound tensors got true zero-copy vs an
// explicit-copy fallback, and total bytes copied (should be near 0 -- only
// the F-10 stragglers with unaligned offsets fall back). Returns the total
// number of tensors bound.
int mlx_gpu_zerocopy_count(int *zero_copy, int *copied, size_t *bytes_copied);

// Gate 3: dequantizes ncols values starting at (row,col0) of expert `e` of
// tensor `name`, via MLX's own mx::dequantize, into `out`. For direct
// comparison against moe_decode_af() at the same coordinates. Returns 1 on
// success, 0 if `name`/`e`/`col0` is invalid.
int mlx_gpu_dequant_probe(const char *name, long e, long row, long col0,
                           int ncols, float *out);

// Gate 4: y = quantized_matmul(x, w_e) for tensor `name`'s expert `e`,
// against a caller-supplied dense fp32 x[in], written to y[out]. For direct
// comparison against moe_matvec_af() on the same expert/input. Returns 1 on
// success.
int mlx_gpu_matvec_probe(const char *name, long e, const float *x, float *y);

// Reports MLX's own active/peak/cache memory counters (mx::get_active_memory
// etc.) for the residency gate.
void mlx_gpu_report_memory(size_t *active, size_t *peak, size_t *cache);

// V5b: layer-0 MLA attention config -- call once, after V5a's mlx_gpu_bind_af()
// calls, before any mlx_gpu_mla_layer0() call. yarn_freqs_half has qk_rope_hd/2
// entries (moe_init_yarn()'s g_moe_yarn_freqs, the real per-model YaRN table --
// distinct values, not synthetic, per the plan's RoPE-sanity requirement).
// Returns 1 on success, 0 if MLX is unavailable.
int mlx_gpu_mla_config(int n_heads, int q_head_dim, int qk_nope_hd, int qk_rope_hd,
                       int v_hd, int kv_lora_rank, double rope_mscale, double attn_scale,
                       const double *yarn_freqs_half, double rms_eps);

// V5b: layer-0 MLA attention for one token position, on GPU. Looks up
// "model.layers.0.self_attn.{q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj}" from
// tensors already bound by mlx_gpu_bind_af() (V5a) -- mlx_gpu_mla_config() must
// have been called first. Maintains its own internal layer-0 K/V cache
// (position-indexed, separate from qwen_infer.c's own CPU-side g_moe_K/V) --
// call sequentially with pos=0,1,2,... exactly like the CPU moe_mla_attention()
// path. Writes o_proj's raw output (NOT accumulated into a residual -- the
// caller decides how to combine it) into o_out[hidden]. Returns 1 on success,
// 0 if config wasn't called, pos is out of range, or a required tensor is
// missing.
int mlx_gpu_mla_layer0(const float *h, int pos, const float *kv_a_ln_w, float *o_out);

// V5c: full-layer config -- call once, after mlx_gpu_mla_config(), before any
// mlx_gpu_layer_step() call.
int mlx_gpu_layer_config(int hidden, int im_dim, int dense_im, int n_experts, int n_shared,
                          int top_k, int group_size);

// V5c: full transformer block (attention + FFN) for layer `l`, one token position, on GPU.
// Mirrors moe_forward_token()'s per-layer loop body. w_inln/w_postln/w_kvaln/w_gate are raw
// F32 weight pointers (same convention as mlx_gpu_mla_layer0's kv_a_ln_w -- read directly from
// qwen_infer.c's own F32 blob, not bound into MLX). is_dense selects the dense MLP path
// (l < first_dense_layers) vs the routed+shared-expert path; w_gate/out_top_idx/out_top_wgt
// are ignored (may be NULL) when is_dense=1. x_in/x_out are the residual stream, [hidden]
// floats each (x_out may alias x_in). Returns 1 on success, 0 if config wasn't called or a
// required tensor is missing.
int mlx_gpu_layer_step(int l, int pos, int is_dense,
                        const float *x_in, const float *w_inln, const float *w_postln,
                        const float *w_kvaln, const float *w_gate,
                        float *x_out, int *out_top_idx, float *out_top_wgt);

// TEMPORARY debug aid: identical to mlx_gpu_layer_step() above, but also writes
// the post-attention residual (dbg_xmid_out, [hidden] floats) and the combined
// FFN output before the final residual add (dbg_routed_out, [hidden] floats) --
// either pointer may be NULL to skip that write. Used to bisect exactly where
// the per-layer lazy rewrite first diverges from this proven eager path.
int mlx_gpu_layer_step_dbg(int l, int pos, int is_dense,
                            const float *x_in, const float *w_inln, const float *w_postln,
                            const float *w_kvaln, const float *w_gate,
                            float *x_out, int *out_top_idx, float *out_top_wgt,
                            float *dbg_xmid_out, float *dbg_routed_out);

// V5c-fused: true one-eval-per-TOKEN rewrite (no residual/error-feedback
// applicable here -- pure control-flow/lifetime fix, not a quantization
// change; same D18 exemption as this file's own header). K/V history is a
// device-side FIXED-SHAPE window (constant shape+pointer every call, a
// boolean mask encoding validity instead of array length -- the same
// "device-side, fixed-shape" principle production CUDA-graph-capturable
// decode paths use, e.g. FreeToken's moe.py, though MLX's lazy graph has no
// literal capture/replay of its own), stored as a genuine persistent
// mx::array per layer (not a raw host buffer) and updated via
// mx::slice_update() so a layer's K/V write and the next layer's read live
// in the SAME lazy graph. STATUS: verified correct against the golden
// eager path across all 8 tested positions (rel-L2 ~7.5e-07 worst case,
// exact match), KILL-GATE PASSES (~52.9 tok/s vs the 48.34 tok/s
// llama.cpp+Metal bar). Full account of four real bugs found+fixed across
// this rewrite's history (a cross-product gather_qmm composition, a bare
// mx::slice() not forced contiguous, a switch_down cross-product
// eliminated outright by matching mlx_lm's own reference SwitchLinear
// convention, and a stack-lifetime use-after-free in a RoPE frequency
// array) is in mlx_moe.cpp's comments above
// mlx_gpu_layer_step_lazy()/mlx_gpu_forward_finalize() and in RESULTS.md.
// Call mlx_gpu_layer_step_lazy() once per layer, in order (l=0..NL-1), for
// the SAME token/pos -- each call builds that layer's ops into a single,
// still-growing lazy graph (nothing is evaluated inside the call itself
// anymore); x_in_host is only read on l==0 (the token's embedding), for
// l>0 the previous call's own (still-lazy) x_out is used instead. After
// all layers, call mlx_gpu_forward_finalize() exactly once -- this is
// where the ENTIRE token's graph (every layer's K/V update, the final
// norm, lm_head) actually evaluates, writing logits_out. Returns 1 on
// success, 0 on failure (aborts the pending state on failure -- next l==0
// call starts fresh).
int mlx_gpu_layer_step_lazy(int l, int pos, int is_dense,
                             const float *x_in_host, const float *w_inln, const float *w_postln,
                             const float *w_kvaln, const float *w_gate);
int mlx_gpu_forward_finalize(const float *w_finalnorm, float *logits_out);

// V5d: set the batch size B for subsequent mlx_gpu_layer_step_lazy()/
// mlx_gpu_forward_finalize() calls. B tokens are LOCKSTEP -- every call processes B
// sequences that all share the SAME `pos` (this does not hold for V5e's later ragged
// design). Default B=1, matching every call site before V5d -- omitting this call
// entirely reproduces the exact single-token behavior this file shipped with when the
// V5c-fused KILL-GATE first passed. x_in_host becomes B*hidden floats (one row per
// sequence, row-major) and logits_out becomes B*vocab floats. Bounded to [1,64] to match
// the CPU/SME2 arm's own MOE_BATCH_MAX ceiling (qwen_infer.c). Returns 1 on success, 0 if
// MLX is unavailable or B is out of range.
int mlx_gpu_set_batch(int B);

// F-4/D-sort: gate the routed-FFN's gather_qmm calls onto a global-sort (mlx_lm's
// _gather_sort/_scatter_unsort) code path when B*top_k >= threshold, instead of the
// unsorted per-row form. Default threshold is effectively infinite (sorted path never
// taken) -- callers that never call this get byte-identical behavior to the KILL-GATE
// build. Pass a real measured crossover (see RESULTS.md's F-4 in-situ sweep) once one
// exists; there is no compile-time "right" default, this is a runtime-measured value.
// Returns 1 on success, 0 if MLX is unavailable.
int mlx_gpu_set_sort_threshold(int threshold);

// V5e: ragged multi-step GPU decode -- generalizes mlx_gpu_layer_step_lazy()'s single
// shared `pos` to A independent (slot,pos) pairs, one per active column, covering BOTH
// prefill (a slot advancing one more prompt position) and decode (a slot generating a new
// token) with the SAME call shape. Mirrors moe_cbatch_step()'s own (token_ids, slot, spos,
// A) naming (qwen_infer.c). Reuses the SAME persistent per-slot K/V arrays
// mlx_gpu_set_batch(N)/mlx_gpu_layer_step_lazy() use -- call mlx_gpu_set_batch(N_SLOTS)
// once before the first step of a cbatch run to size them; unlike V5d's B (fully rewritten
// every call), these persist across every step of the run. Call
// mlx_gpu_cbatch_layer_step_lazy() once per layer (l=0..NL-1) for the SAME step's A
// columns, then mlx_gpu_cbatch_forward_finalize() once per step (mirrors
// mlx_gpu_forward_finalize()'s own eval-everything-at-once contract). slot[]/spos[] are
// read once per call (each call's own array(It,shape,dtype) COPYING ctor -- not the 4-arg
// no-copy wrap wrap_host_f32() uses for weights -- so their host lifetime need not extend
// past the call, unlike Bug 1's freqs_f32 mistake). Returns 1 on success, 0 on failure.
int mlx_gpu_cbatch_layer_step_lazy(int l, int A, const int *slot, const int *spos, int is_dense,
                                    const float *x_in_host, const float *w_inln,
                                    const float *w_postln, const float *w_kvaln,
                                    const float *w_gate);
int mlx_gpu_cbatch_forward_finalize(const float *w_finalnorm, float *logits_out);

#ifdef __cplusplus
}
#endif

#endif // MLX_MOE_H
