#ifndef BEGLIN_RESEARCH_QNG64_N8_REPACKED_NEON_H
#define BEGLIN_RESEARCH_QNG64_N8_REPACKED_NEON_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int beglin_qng64_n8_repack_inplace(uint8_t *original_bitplanes,int out,int in);
int beglin_qng64_n8_repacked_matvec_neon(
    const int8_t *repacked_codes,const float *scales,
    const float *x,const float *bias,float *y,int out,int in);
#ifdef __cplusplus
}
#endif
#endif
