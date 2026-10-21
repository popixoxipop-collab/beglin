/*
 * Isolated native BEGLIN n8 scalar vs standalone NEON GEMV benchmark.
 * Uses exact actual Qwen2.5-1.5B original BF16 L0 q_proj packed with
 * BEGLIN qNg64(n8), not a synthetic or dequantized-only matmul shortcut.
 * No live engine modifications.
 */
#define _POSIX_C_SOURCE 200809L
#include "qng64_dense.h"
#include "research/oml_acs/qng64_n8_neon_experimental.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

enum { IN=1536, OUT=1536, N=96, TIMED=24, ROUNDS=7 };
static volatile double keep_work=0.0;

static double stamp(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+(double)t.tv_nsec/1e9;
}
static int cmp(const void *a,const void *b) {
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}
static int read_data(const char *root,const char *name,void *buffer,size_t size) {
    char path[4096];
    if(snprintf(path,sizeof path,"%s/%s",root,name)>=(int)sizeof(path))return 0;
    FILE *f=fopen(path,"rb");
    if(!f)return 0;
    int ok=fread(buffer,1,size,f)==size && fgetc(f)==EOF;
    fclose(f);
    return ok;
}
static int write_data(const char *root,const char *name,const void *buffer,size_t size) {
    char path[4096];
    if(snprintf(path,sizeof path,"%s/%s",root,name)>=(int)sizeof(path))return 0;
    FILE *f=fopen(path,"wbx");
    if(!f)return 0;
    int ok=fwrite(buffer,1,size,f)==size && fflush(f)==0;
    fclose(f);
    return ok;
}
static void scalar(const uint8_t *packed,const float *scales,const float *x,float *y) {
    qng64_matvec_f32(packed,scales,8,x,NULL,y,OUT,IN);
}
static int neon(const uint8_t *packed,const float *scales,const float *x,float *y) {
    return beglin_qng64_n8_matvec_neon(packed,scales,x,NULL,y,OUT,IN);
}
int main(int argc,char **argv) {
    umask(0077);
    if(argc!=2) {
        fprintf(stderr,"usage: %s <private_qwen_fixture_directory>\n",argv[0]);return 2;
    }
    const char *root=argv[1];
    const size_t elements=(size_t)IN*OUT,outputs=(size_t)N*OUT;
    const size_t planes=elements;
    const size_t groups=elements/64;
    uint8_t *packed=malloc(planes);
    float *scales=malloc(groups*sizeof(float));
    float *inputs=malloc(outputs*sizeof(float));
    float *old=malloc(outputs*sizeof(float));
    float *newout=malloc(outputs*sizeof(float));
    float *scratch=malloc(OUT*sizeof(float));
    if(!packed||!scales||!inputs||!old||!newout||!scratch)return 3;
    if(!read_data(root,"native_qng64_n8.planes",packed,planes) ||
       !read_data(root,"native_qng64_n8.scales.f32",scales,groups*sizeof(float)) ||
       !read_data(root,"layer0_token_norm_inputs.f32",inputs,outputs*sizeof(float)) ||
       !read_data(root,"native_qng64_n8_outputs.f32",old,outputs*sizeof(float))) {
        fprintf(stderr,"pinned native Qwen qNg64 fixture missing/wrong size\n");return 4;
    }
    if(qng64_group_bytes(4)!=0||qng64_group_bytes(8)!=64) {
        fprintf(stderr,"format mismatch (cannot use qNg64 for n4)\n");return 5;
    }
    if(beglin_qng64_n8_matvec_neon(packed,scales,inputs,NULL,scratch,OUT,IN+1)==0 ||
       beglin_qng64_n8_matvec_neon(packed,scales,inputs,NULL,scratch,0,IN)==0) {
        fprintf(stderr,"invalid shape accepted\n");return 6;
    }
    double err2=0.0,ref2=0.0,maxerr=0.0;
    for(int i=0;i<N;i++) {
        if(neon(packed,scales,inputs+(size_t)i*IN,newout+(size_t)i*OUT)!=0)
            return 7;
        for(int j=0;j<OUT;j++) {
            size_t k=(size_t)i*OUT+j;
            if(!isfinite(old[k])||!isfinite(newout[k]))return 8;
            double err=(double)newout[k]-(double)old[k];
            err2+=err*err;ref2+=(double)old[k]*(double)old[k];
            if(fabs(err)>maxerr)maxerr=fabs(err);
        }
    }
    double rel=sqrt(err2/ref2),rmse=sqrt(err2/outputs);
    if(!isfinite(rel)||!isfinite(maxerr))return 9;
    const double max_tolerance=0.0005,rel_tolerance=1e-5;
    if(maxerr>max_tolerance || rel>rel_tolerance) {
        fprintf(stderr,"NEON numerical mismatch %.9g max %.9g rel\n",maxerr,rel);
        return 10;
    }
    double samples[2][ROUNDS];
    for(int kind=0;kind<2;kind++) {
        if(kind==0)scalar(packed,scales,inputs,scratch);
        else if(neon(packed,scales,inputs,scratch)!=0)return 11;
    }
    for(int rep=0;rep<ROUNDS;rep++) {
        // Alternate order to reduce page-temperature/order confounding.
        for(int step=0;step<2;step++) {
            const int kind=(rep&1) ? 1-step : step;
            double t0=stamp();
            for(int i=0;i<TIMED;i++) {
                int idx=(i*(N-1))/(TIMED-1);
                if(kind==0)scalar(packed,scales,inputs+(size_t)idx*IN,scratch);
                else if(neon(packed,scales,inputs+(size_t)idx*IN,scratch)!=0)return 12;
                keep_work += (double)scratch[(i*61)%OUT];
            }
            samples[kind][rep]=(stamp()-t0)*1000/TIMED;
        }
    }
    qsort(samples[0],ROUNDS,sizeof(double),cmp);
    qsort(samples[1],ROUNDS,sizeof(double),cmp);
    if(!write_data(root,"native_qng64_n8_neon_experimental_outputs.f32",
                   newout,outputs*sizeof(float))) {
        fprintf(stderr,"failed safe output file creation\n");return 13;
    }
    double scalar_med=samples[0][ROUNDS/2];
    double neon_med=samples[1][ROUNDS/2];
    printf("{\"schema\":\"beglin-q-native-neon-8bit-single-tensor/1\","
       "\"status\":\"PASS_NATIVE_ISOLATED_NOT_PRODUCTION\","
       "\"model\":\"Qwen/Qwen2.5-1.5B-Instruct\","
       "\"tensor\":\"model.layers.0.self_attn.q_proj.weight\","
       "\"shape\":[%d,%d],\"rows\":%d,"
       "\"packed_group_size\":64,\"bitplanes\":8,"
       "\"kernel_reference\":\"BEGLIN qng64_matvec_f32\","
       "\"kernel_candidate\":\"BEGLIN isolated NEON transpose16 n8\","
       "\"max_abs_error\":%.12g,\"relative_l2_error\":%.12g,"
       "\"rmse\":%.12g,\"max_abs_gate\":%.12g,\"relative_l2_gate\":%.12g,"
       "\"scalar_p50_ms_per_vector\":%.9g,\"neon_p50_ms_per_vector\":%.9g,"
       "\"measured_speedup\":%.8g,"
       "\"scalar_7round_min_ms\":%.9g,\"scalar_7round_max_ms\":%.9g,"
       "\"neon_7round_min_ms\":%.9g,\"neon_7round_max_ms\":%.9g,"
       "\"sample_count\":%d,\"rounds\":%d,"
       "\"production_touched\":false,\"full_model_tps_measured\":false}\n",
       OUT,IN,N,maxerr,rel,rmse,max_tolerance,rel_tolerance,
       scalar_med,neon_med,scalar_med/neon_med,
       samples[0][0],samples[0][ROUNDS-1],
       samples[1][0],samples[1][ROUNDS-1],
       TIMED,ROUNDS);
    free(packed);free(scales);free(inputs);free(old);free(newout);free(scratch);
    return 0;
}
