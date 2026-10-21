#ifndef BEGLIN_RESEARCH_N8_NEON_H
#define BEGLIN_RESEARCH_N8_NEON_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int beglin_qng64_n8_matvec_neon(const uint8_t *packed,
    const float *scales, const float *x, const float *bias,
    float *y, int out, int in);
#ifdef __cplusplus
}
#endif
#endif
