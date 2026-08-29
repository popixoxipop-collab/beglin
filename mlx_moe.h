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

#ifdef __cplusplus
}
#endif

#endif // MLX_MOE_H
