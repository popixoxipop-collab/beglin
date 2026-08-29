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

#ifdef __cplusplus
}
#endif

#endif // MLX_MOE_H
