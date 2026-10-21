/*
 * Research-only concurrency-safe BEGLIN native q4 kernel benchmark adapter.
 * Each Q/K/V or gate/up stage role uses its OWN q4pool (1t or 6t) so
 * concurrent ctypes calls never race through the shared q4pool.job/cv.
 * Original q4gemv.h untouched; this wrapper does not enable serving concurrency.
 */
#include "q4gemv.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static q4pool p1[3],p6[3];
static int ready=0;
int beglin_async_stage_profile_init(int max_in) {
    if(ready || max_in<64 || (max_in&63))return -1;
    for(int r=0;r<3;r++) {
        q4pool_init(&p1[r],1,max_in);q4pool_start(&p1[r]);
        q4pool_init(&p6[r],6,max_in);q4pool_start(&p6[r]);
    }
    ready=1;return 0;
}
int beglin_async_stage_profile_shutdown(void) {
    if(!ready)return -1;
    for(int r=0;r<3;r++) {
        q4pool_destroy(&p6[r]);q4pool_destroy(&p1[r]);
    }
    ready=0;return 0;
}
int beglin_async_stage_profile_q4(int r,int threads,
        const uint8_t *codes,const float *scales,
        const float *x,float *y,int out,int in) {
    if(!ready||r<0||r>=3||(threads!=1&&threads!=6)||
       !codes||!scales||!x||!y||out<=0||in<=0||(in&63))return -1;
    q4pool *pool=threads==6?&p6[r]:&p1[r];
    gemv_q4g64_mt_fused(pool,codes,scales,x,NULL,y,out,in);
    return 0;
}
