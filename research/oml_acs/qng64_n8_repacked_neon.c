/*
 * Research-only BEGLIN qNg64 n8 in-place bitplane -> signed code repack.
 * Original bitplane group: [8 planes][8 bytes] (64 bytes = 64 weights).
 * Repacked group: [64 signed int8 codes] (same 64 bytes, same group scale).
 * MUST tag repacked pointers separately; not valid original bitplane data.
 * Do not invoke qng64_matmul_f32 or original n8 matvec on repacked memory.
 */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "research/oml_acs/qng64_n8_repacked_neon.h"

static inline uint8x16_t bitplane_to_16(const uint8_t *group,int at) {
    const uint8_t masks[16]={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    uint8x16_t m=vld1q_u8(masks);
    uint8x16_t code=vdupq_n_u8(0);
    for(int bit=0;bit<8;bit++) {
        uint8x16_t two=vcombine_u8(
             vdup_n_u8(group[(size_t)bit*8+at]),
             vdup_n_u8(group[(size_t)bit*8+at+1]));
        uint8x16_t nz=vmvnq_u8(vceqq_u8(vandq_u8(two,m),vdupq_n_u8(0)));
        code=vorrq_u8(code,vandq_u8(nz,vdupq_n_u8((uint8_t)(1u<<bit))));
    }
    return veorq_u8(code,vdupq_n_u8(0x80));
}

/* The only mutation: overwrite complete group after decoding into scratch. */
int beglin_qng64_n8_repack_inplace(uint8_t *packed,int out,int in) {
    if(!packed||out<=0||in<=0||(in&63))return -1;
    const size_t groups=(size_t)out*(in/64);
    for(size_t g=0;g<groups;g++) {
        uint8_t *block=packed+g*64u;
        uint8_t codes[64];
        for(int j=0;j<4;j++)
            vst1q_u8(codes+j*16,bitplane_to_16(block,2*j));
        memcpy(block,codes,64);
    }
    return 0;
}
static inline float32x4_t lo4(int8x8_t q) {
    return vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(q))));
}
static inline float32x4_t hi4(int8x8_t q) {
    return vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(q))));
}

/* Expects ALREADY REPACKED int8 codes (not the original on-disk bitplanes). */
int beglin_qng64_n8_repacked_matvec_neon(
    const int8_t *restrict repacked_codes,
    const float *restrict scales,
    const float *restrict x,
    const float *restrict bias,
    float *restrict y,
    int out,int in
) {
    if(!repacked_codes||!scales||!x||!y||out<=0||in<=0||(in&63))return -1;
    int ng=in/64;
    for(int r=0;r<out;r++) {
        double sum=bias?(double)bias[r]:0.;
        for(int g=0;g<ng;g++) {
            const int8_t *p=repacked_codes+((size_t)r*ng+g)*64u;
            const float scale=scales[(size_t)r*ng+g];
            if(!isfinite(scale))return -1;
            const float *a=x+(size_t)g*64u;
            float32x4_t acc0=vdupq_n_f32(0),acc1=vdupq_n_f32(0);
            float32x4_t acc2=vdupq_n_f32(0),acc3=vdupq_n_f32(0);
            for(int block=0;block<4;block++) {
                int8x16_t q=vld1q_s8(p+block*16);
                int8x8_t low=vget_low_s8(q),high=vget_high_s8(q);
                const float *c=a+block*16;
                acc0=vfmaq_f32(acc0,vmulq_n_f32(lo4(low),scale),vld1q_f32(c));
                acc1=vfmaq_f32(acc1,vmulq_n_f32(hi4(low),scale),vld1q_f32(c+4));
                acc2=vfmaq_f32(acc2,vmulq_n_f32(lo4(high),scale),vld1q_f32(c+8));
                acc3=vfmaq_f32(acc3,vmulq_n_f32(hi4(high),scale),vld1q_f32(c+12));
            }
            sum+=(double)vaddvq_f32(acc0)+(double)vaddvq_f32(acc1)
                +(double)vaddvq_f32(acc2)+(double)vaddvq_f32(acc3);
        }
        y[r]=(float)sum;
    }
    return 0;
}
