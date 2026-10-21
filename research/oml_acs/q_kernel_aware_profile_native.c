/* Research-only native BEGLIN CPU kernel cost profiling interface.
 * Original Q4G64 1-thread/6-thread fused NEON pool vs source-qNg64 n8
 * in-place repacked NEON for the same real BF16 Qwen tensor geometry.
 * Never exported from or linked into the production BEGLIN runtime.
 */
#include "q4gemv.h"
#include "gguf_transcode.h"
#include "qng64_dense.h"
#include "research/oml_acs/qng64_n8_repacked_neon.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static q4pool research_q4_pool_1, research_q4_pool_6;
static int research_live=0;
int beglin_kernel_profile_init(int max_in) {
    if(research_live || max_in<64 || max_in%64) return -1;
    q4pool_init(&research_q4_pool_1,1,max_in);
    q4pool_start(&research_q4_pool_1);
    q4pool_init(&research_q4_pool_6,6,max_in);
    q4pool_start(&research_q4_pool_6);
    research_live=1;
    return 0;
}
int beglin_kernel_profile_release(void) {
    if(!research_live)return -1;
    q4pool_destroy(&research_q4_pool_6);
    q4pool_destroy(&research_q4_pool_1);
    research_live=0;
    return 0;
}
int beglin_kernel_profile_q4(int threads,
    const uint8_t *packed,const float *scales,
    const float *x,float *y,int out,int in) {
    if(!research_live || (threads!=1 && threads!=6) || !packed || !scales
       || !x || !y || out<=0 || in<=0 || (in&63))return -1;
    q4pool *pool=threads==6?&research_q4_pool_6:&research_q4_pool_1;
    gemv_q4g64_mt_fused(pool,packed,scales,x,NULL,y,out,in);
    return 0;
}
