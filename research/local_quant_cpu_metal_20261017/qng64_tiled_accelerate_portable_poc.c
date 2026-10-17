/* BEGLIN isolated CPU qNg64 group-64 tiled CPU GEMM proof-of-concept.
 *
 * Never alters qng64_dense.c or live model. Uses compact n-bit bitplanes as
 * authoritative weight storage, converts at most tile_rows output rows into a
 * transient float32 tile, then cblas_sgemm(x, tile^T). This trades decode
 * scratch (<= 64 * input_width * 4 bytes) for amortized batch GEMM.
 *
 * Returns -1 on invalid shape, n, arithmetic overflow, or allocation failure.
 * The reference engine path qng64_matmul_f32() accumulates double; our BLAS
 * accumulates FP32 and therefore does NOT promise bit-exact float output.
 * Test numerical tolerances separately. No acceleration/quality claims until
 * first-decoder model measurements PASS.
 */
#include "qng64_dense.h"

#include <Accelerate/Accelerate.h>
#if defined(__ARM_NEON) && !defined(BEGLIN_FORCE_SCALAR_UNPACK)
#include <arm_neon.h>
#endif
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BEGLIN_TILE_MAX 64
#define BEGLIN_G64 64

#if defined(__ARM_NEON) && !defined(BEGLIN_FORCE_SCALAR_UNPACK)
static const uint8_t k_masks[8] = {1,2,4,8,16,32,64,128};

/* Decode 64 signed bitplane codes as F32 with one F32 scale. */
static void decode64_neon(const uint8_t *p, int n, float scale, float *out) {
    const uint8x8_t mask = vld1_u8(k_masks);
    const int bias = 1 << (n-1);
    for (int byte = 0; byte < 8; ++byte) {
        int16x8_t acc = vdupq_n_s16((int16_t)-bias);
        for (int bit = 0; bit < n; ++bit) {
            uint8x8_t v = vdup_n_u8(p[bit*8 + byte]);
            uint8x8_t hit = vceq_u8(vand_u8(v,mask),mask);
            uint8x8_t one = vshr_n_u8(hit,7);
            uint16x8_t wide = vmovl_u8(one);
            uint16x8_t weighted = vmulq_n_u16(wide,(uint16_t)(1u<<bit));
            acc = vaddq_s16(acc,vreinterpretq_s16_u16(weighted));
        }
        float32x4_t lo = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(acc))),scale);
        float32x4_t hi = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(acc))),scale);
        vst1q_f32(out + byte*8,lo);
        vst1q_f32(out + byte*8+4,hi);
    }
}

#else
static void decode64_neon(const uint8_t *p, int n, float scale, float *out) {
    const int bias = 1 << (n - 1);
    for (int index = 0; index < 64; ++index) {
        int value = 0;
        for (int b = 0; b < n; ++b) {
            value |= ((p[b*8+(index/8)] >> (index%8)) & 1) << b;
        }
        out[index] = (float)(value-bias)*scale;
    }
}
#endif

int beglin_qng64_tiled_cblas_matmul(
    const uint8_t *packed, const float *scales, int n,
    const float *x, const float *bias, float *y,
    int out, int in, int M, int tile_rows) {

    if (!packed || !scales || !x || !y ||
        out <= 0 || in <= 0 || M <= 0 || (in % BEGLIN_G64) != 0 ||
        tile_rows <= 0 || tile_rows > BEGLIN_TILE_MAX ||
        qng64_group_bytes(n) == 0 || in > INT_MAX/4 || out > INT_MAX/4) {
        return -1;
    }
    const size_t input=(size_t)in;
    const size_t nr=(size_t)(in/BEGLIN_G64);
    const size_t row_count=(size_t)out;
    const size_t batch_count=(size_t)M;
    if (row_count > SIZE_MAX/input || batch_count > SIZE_MAX/row_count ||
        (size_t)tile_rows > SIZE_MAX/input) return -1;
    const size_t elems=(size_t)tile_rows*input;
    if (elems > SIZE_MAX/sizeof(float)) return -1;
    float *tile=(float*)malloc(elems*sizeof(float));
    if (!tile) return -1;

    const size_t bytes_per_group=qng64_group_bytes(n);
    for (int row0=0;row0<out;row0+=tile_rows) {
        int current=(row0+tile_rows <= out)?tile_rows:(out-row0);
        for (int tr=0;tr<current;++tr) {
            const size_t global_row=(size_t)row0+(size_t)tr;
            float *dst=tile+(size_t)tr*input;
            for (size_t group=0; group<nr; ++group) {
                const size_t index=global_row*nr+group;
                const uint8_t *src=packed+index*bytes_per_group;
                float scale=scales[index];
                if (!isfinite(scale) || scale<=0.f) {
                    free(tile);return -1;
                }
                decode64_neon(src,n,scale,dst+group*BEGLIN_G64);
            }
        }
        /* X: M*in row major, tile W: current*in. We need X*W^T. */
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                    M,current,in,1.0f,x,in,tile,in,
                    0.0f,y+row0,out);
        if (bias) {
            for (int m=0;m<M;++m) {
                for (int r=0;r<current;++r)
                    y[(size_t)m*row_count+(size_t)row0+(size_t)r] += bias[row0+r];
            }
        }
    }
    free(tile);
    return 0;
}
