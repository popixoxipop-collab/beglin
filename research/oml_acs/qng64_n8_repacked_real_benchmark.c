/*
 * BEGLIN Q real Qwen L0 q_proj: native scalar bitplanes vs on-the-fly
 * Apple NEON decode vs one-time in-place signed-code NEON GEMV.
 * Isolated CPU single tensor experiment, no model/prompt/logit data touched.
 */
#define _POSIX_C_SOURCE 200809L
#include "qng64_dense.h"
#include "research/oml_acs/qng64_n8_neon_experimental.h"
#include "research/oml_acs/qng64_n8_repacked_neon.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

enum { D=1536,N=96,NTEST=24,REPEAT=9 };
static volatile double sink=0.0;
static double elapsed(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}
static int ordered(const void *aa,const void *bb) {
    const double a=*(const double*)aa,b=*(const double*)bb;
    return (a>b)-(a<b);
}
static int read_exact(const char *dir,const char *name,void *dst,size_t bytes) {
    char p[4096];if(snprintf(p,sizeof p,"%s/%s",dir,name)>=(int)sizeof(p))return 0;
    FILE *f=fopen(p,"rb");if(!f)return 0;
    int ok=fread(dst,1,bytes,f)==bytes && fgetc(f)==EOF;
    fclose(f);return ok;
}
static int write_exact(const char *dir,const char *name,const void *src,size_t bytes) {
    char p[4096];if(snprintf(p,sizeof p,"%s/%s",dir,name)>=(int)sizeof(p))return 0;
    FILE *f=fopen(p,"wbx");if(!f)return 0;
    int ok=fwrite(src,1,bytes,f)==bytes && fflush(f)==0;
    fclose(f);return ok;
}
static int gemv(int kind,const uint8_t *bitplanes,const int8_t *repacked,
    const float *scales,const float *x,float *y) {
    if(kind==0) {
        qng64_matvec_f32(bitplanes,scales,8,x,NULL,y,D,D);return 0;
    } else if(kind==1) {
        return beglin_qng64_n8_matvec_neon(bitplanes,scales,x,NULL,y,D,D);
    }
    return beglin_qng64_n8_repacked_matvec_neon(repacked,scales,x,NULL,y,D,D);
}
int main(int argc,char **argv) {
    umask(0077);
    if(argc!=3){fprintf(stderr,"usage: %s <original_Qwen_fixture> <fresh_private_output>\n",argv[0]);return 2;}
    const char *dir=argv[1],*outdir=argv[2];
    const size_t weights=(size_t)D*D;
    const size_t vectors=(size_t)N*D;
    uint8_t *original=malloc(weights),*copy=malloc(weights);
    float *scales=malloc((weights/64)*sizeof(float));
    float *inputs=malloc(vectors*sizeof(float));
    float *ref=malloc(vectors*sizeof(float));
    float *out=malloc(vectors*sizeof(float));
    float *scratch=malloc(D*sizeof(float));
    if(!original||!copy||!scales||!inputs||!ref||!out||!scratch)return 3;
    if(!read_exact(dir,"native_qng64_n8.planes",original,weights)||
       !read_exact(dir,"native_qng64_n8.scales.f32",scales,weights/64*sizeof(float))||
       !read_exact(dir,"layer0_token_norm_inputs.f32",inputs,vectors*sizeof(float))||
       !read_exact(dir,"native_qng64_n8_outputs.f32",ref,vectors*sizeof(float))) {
       fprintf(stderr,"unsecured/missing pinned fixture bytes\n");return 4;
    }
    memcpy(copy,original,weights);
    if(beglin_qng64_n8_repack_inplace(copy,D,D+1)==0||
       beglin_qng64_n8_repack_inplace(NULL,D,D)==0) return 5;
    double t0=elapsed();
    if(beglin_qng64_n8_repack_inplace(copy,D,D)!=0)return 6;
    const double one_time_repack_ms=(elapsed()-t0)*1000.0;
    // Independent spot assertions on both end groups/bit boundaries.
    const size_t chunks[4]={0,37,weights/64/2,weights/64-1};
    for(int j=0;j<4;j++) {
        const uint8_t *bp=original+chunks[j]*64;
        const int8_t *packed=(const int8_t*)copy+chunks[j]*64;
        for(int k=0;k<64;k++) {
            unsigned int code=0;
            for(int bit=0;bit<8;bit++) code|=((bp[bit*8+k/8]>>(k%8))&1u)<<bit;
            if((int8_t)(code-128)!=packed[k]) {
                fprintf(stderr,"bitplane / repacked int8 mismatch group=%zu col=%d\n",chunks[j],k);
                return 7;
            }
        }
    }
    double e2=0,den=0,mx=0;
    for(int i=0;i<N;i++) {
        if(gemv(2,original,(int8_t*)copy,scales,inputs+(size_t)i*D,out+(size_t)i*D))return 8;
        for(int k=0;k<D;k++) {
            size_t idx=(size_t)i*D+k;
            if(!isfinite(ref[idx])||!isfinite(out[idx]))return 9;
            double e=(double)out[idx]-(double)ref[idx];
            e2+=e*e;den+=(double)ref[idx]*(double)ref[idx];
            if(fabs(e)>mx)mx=fabs(e);
        }
    }
    const double rel=sqrt(e2/den),rmse=sqrt(e2/vectors);
    if(mx>.0005||rel>1e-5){fprintf(stderr,"numerical mismatch max %.12g rel %.12g\n",mx,rel);return 10;}
    double times[3][REPEAT];
    for(int k=0;k<3;k++)if(gemv(k,original,(int8_t*)copy,scales,inputs,scratch))return 11;
    for(int round=0;round<REPEAT;round++) {
        // Rotation balances temperature and cache order between 3 kernels.
        for(int step=0;step<3;step++) {
            const int kind=(round+step)%3;
            double t=elapsed();
            for(int i=0;i<NTEST;i++) {
                const int idx=i*(N-1)/(NTEST-1);
                if(gemv(kind,original,(int8_t*)copy,scales,inputs+(size_t)idx*D,scratch))return 12;
                sink+=(double)scratch[(i*53)%D];
            }
            times[kind][round]=(elapsed()-t)*1000/NTEST;
        }
    }
    for(int k=0;k<3;k++)qsort(times[k],REPEAT,sizeof(double),ordered);
    if(!write_exact(outdir,"repacked_n8_full_outputs.f32",out,vectors*sizeof(float)))return 13;
    double oldms=times[0][REPEAT/2],flyms=times[1][REPEAT/2],repackms=times[2][REPEAT/2];
    printf("{\"schema\":\"beglin-q-native-n8-repack-real-l0/1\",\"status\":\"PASS_ISOLATED\","
       "\"model\":\"Qwen2.5-1.5B\",\"shape\":[%d,%d],\"num_source_inputs\":%d,"
       "\"original_bitplane_weight_bytes\":%zu,\"repacked_code_weight_bytes\":%zu,"
       "\"peak_temp_conversion_bytes_per_group\":64,"
       "\"one_time_repack_ms\":%.8g,"
       "\"numeric_max_abs_vs_original\":%.12g,\"numeric_rel_l2_vs_original\":%.12g,"
       "\"numeric_rmse_vs_original\":%.12g,"
       "\"old_scalar_median_ms\":%.9g,\"fly_neon_median_ms\":%.9g,"
       "\"predecoded_neon_median_ms\":%.9g,"
       "\"predecoded_speedup_vs_scalar\":%.8g,"
       "\"predecoded_speedup_vs_fly_neon\":%.8g,"
       "\"amortization_vectors_vs_fly_neon\":%.9g,"
       "\"rounds\":%d,\"timed_vectors_per_round\":%d,"
       "\"production_modified\":false,\"whole_model_tested\":false}\n",
       D,D,N,weights,weights,one_time_repack_ms,mx,rel,rmse,
       oldms,flyms,repackms,oldms/repackms,flyms/repackms,
       flyms>repackms?one_time_repack_ms/(flyms-repackms):-1.,REPEAT,NTEST);
    free(original);free(copy);free(scales);free(inputs);free(ref);free(out);free(scratch);
    return 0;
}
