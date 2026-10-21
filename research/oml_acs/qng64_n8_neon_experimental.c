/*
 * Experimental BEGLIN qNg64 n=8 Apple NEON dense GEMV. NOT integrated in the
 * production/runtime engine. This is a full-byte-precision bit-plane decode,
 * not a Q8G64 renamed buffer, and is incompatible with n=4 by contract.
 *
 * Same source layout as gguf_quantize_qNg64 and qng64_dense.c:
 * 64 codes/group; 8 planes of 8 bytes; each plane byte has 8 element bits,
 * code is biased by 128, n=8 range [-128,127]. Scales are F32 per group.
 *
 * This adds a 16-code NEON bit transpose + vector multiply path rather than
 * qng64_decode_code's scalar 8-plane loop per element.
 *
 * Input: current native BEGLIN packed qNg64 n8 planes, scales, FP32 input.
 * Output: FP32 GEMV of the quantized weights. Returns 0 on success, -1 on
 * malformed geometry. Must be independently tested before any live bind.
 */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

static inline uint8x16_t beglin_n8_codes16(const uint8_t *restrict planes,
                                            int byte_offset) {
    const uint8_t masks_arr[16] = {
        1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128
    };
    const uint8x16_t masks=vld1q_u8(masks_arr);
    const uint8x16_t zeros=vdupq_n_u8(0);
    uint8x16_t codes=vdupq_n_u8(0);
    for(int bit=0;bit<8;bit++) {
        const uint8_t a=planes[(size_t)bit*8+byte_offset];
        const uint8_t b=planes[(size_t)bit*8+byte_offset+1];
        const uint8x16_t bytes=vcombine_u8(vdup_n_u8(a),vdup_n_u8(b));
        const uint8x16_t active=vmvnq_u8(vceqq_u8(vandq_u8(bytes,masks),zeros));
        codes=vorrq_u8(codes,vandq_u8(active,vdupq_n_u8((uint8_t)(1u<<bit))));
    }
    return codes;
}

static inline float32x4_t beglin_float4_from_int8_low(int8x8_t x) {
    return vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(x))));
}
static inline float32x4_t beglin_float4_from_int8_high(int8x8_t x) {
    return vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(x))));
}

int beglin_qng64_n8_matvec_neon(
    const uint8_t *restrict packed,
    const float *restrict scales,
    const float *restrict x,
    const float *restrict bias,
    float *restrict y,
    int out, int in
) {
    if(!packed || !scales || !x || !y || out<=0 || in<=0 || (in&63))
        return -1;
    const int ng=in/64;
    for(int r=0;r<out;r++) {
        double sum=bias ? (double)bias[r]:0.;
        for(int g=0;g<ng;g++) {
            const uint8_t *group=packed+((size_t)r*ng+g)*64u;
            const float scale=scales[(size_t)r*ng+g];
            if(!isfinite(scale))return -1;
            const float *xg=x+(size_t)g*64u;
            float32x4_t acc0=vdupq_n_f32(0),acc1=vdupq_n_f32(0);
            float32x4_t acc2=vdupq_n_f32(0),acc3=vdupq_n_f32(0);
            for(int block=0;block<4;block++) {
                const uint8x16_t biased=beglin_n8_codes16(group,block*2);
                const int8x16_t codes=vreinterpretq_s8_u8(
                    veorq_u8(biased,vdupq_n_u8(0x80))
                );
                const int8x8_t lo=vget_low_s8(codes),hi=vget_high_s8(codes);
                const float *chunk=xg+block*16;
                // The original scalar path multiplies code by scale before
                // the dot. Preserve that ordering to reduce arithmetic drift.
                acc0=vfmaq_f32(acc0,vmulq_n_f32(beglin_float4_from_int8_low(lo),scale),vld1q_f32(chunk));
                acc1=vfmaq_f32(acc1,vmulq_n_f32(beglin_float4_from_int8_high(lo),scale),vld1q_f32(chunk+4));
                acc2=vfmaq_f32(acc2,vmulq_n_f32(beglin_float4_from_int8_low(hi),scale),vld1q_f32(chunk+8));
                acc3=vfmaq_f32(acc3,vmulq_n_f32(beglin_float4_from_int8_high(hi),scale),vld1q_f32(chunk+12));
            }
            // F32 SIMD reduction is not guaranteed bit-for-bit identical to
            // the original scalar double accumulation; compare quantitatively.
            sum+=(double)vaddvq_f32(acc0)+(double)vaddvq_f32(acc1)
                +(double)vaddvq_f32(acc2)+(double)vaddvq_f32(acc3);
        }
        y[r]=(float)sum;
    }
    return 0;
}
