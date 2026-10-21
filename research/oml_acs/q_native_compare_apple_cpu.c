// Q: isolated Apple CPU real Qwen BF16 q_proj vs BEGLIN qNg64(n8) / Q4G64.
// Original BF16 weights widened to FP32 is the reference. This measures ONE
// layer0 Q projection, not a full LLM, task AUROC, or Apple Metal GPU path.
// qNg64 excludes n=4: Q4G64 is a distinct native packed nibble kernel.
#include "qng64_dense.h"
#include "gguf_transcode.h"
#include "q4gemv.h"
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>

enum { D=1536, N=96, TIMED_SAMPLES=24, ROUNDS=5 };

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + 1e-9*(double)t.tv_nsec;
}
static int read_binary(const char *root, const char *name, void *dest, size_t n) {
    char path[2048];
    if(snprintf(path,sizeof path,"%s/%s",root,name)>=sizeof(path)) return 0;
    FILE *fp=fopen(path,"rb");
    if(!fp) return 0;
    int ok=fread(dest,1,n,fp)==n && fgetc(fp)==EOF;
    fclose(fp);
    return ok;
}
static int write_binary(const char *root,const char *name,const void *ptr,size_t n) {
    char path[2048];
    if(snprintf(path,sizeof path,"%s/%s",root,name)>=sizeof(path)) return 0;
    FILE *fp=fopen(path,"wbx"); // new file only (Darwin supports x, prevents overwrite)
    if(!fp) return 0;
    int ok=fwrite(ptr,1,n,fp)==n && fflush(fp)==0;
    fclose(fp);
    return ok;
}
static int cmp_double(const void *a,const void *b) {
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}
static void run_one(int kind, const float *w,const uint8_t *p8,const float *s8,
                  const uint8_t *p4,const float *s4,const float *x,float *out) {
    if(kind==0) {
        cblas_sgemv(CblasRowMajor,CblasNoTrans,D,D,1.f,w,D,x,1,0.f,out,1);
    } else if(kind==1) {
        qng64_matvec_f32(p8,s8,8,x,NULL,out,D,D);
    } else {
        gemv_q4g64_fused(p4,s4,x,NULL,out,D,D);
    }
}
static void analyze(const float *ref,const float *a, size_t n,
                    double *rel_l2,double *rmse,double *mae,double *max_abs) {
    long double sq_err=0.,sq_ref=0.,abs_err=0.;double mx=0.;
    for(size_t i=0;i<n;i++) {
        if(!isfinite(a[i])||!isfinite(ref[i])) { *rel_l2=NAN;return; }
        double delta=(double)a[i]-(double)ref[i];
        sq_err+=delta*delta; sq_ref+=(double)ref[i]*ref[i];
        abs_err+=fabs(delta);
        if(fabs(delta)>mx) mx=fabs(delta);
    }
    *rel_l2=sqrt((double)(sq_err/sq_ref));
    *rmse=sqrt((double)(sq_err/n));
    *mae=(double)(abs_err/n);
    *max_abs=mx;
}
int main(int argc,char **argv) {
    // Enforce private output files. The first run disclosed a default 0644
    // mode inside a protected 0700 lab folder; fix the creator, not just
    // chmod post hoc, so every future benchmark is fail-closed.
    umask(0077);
    if(argc!=2) {fprintf(stderr,"usage: %s fixture_dir\n",argv[0]);return 2;}
    const char *dir=argv[1];
    const size_t count=(size_t)D*D;
    const size_t total=(size_t)N*D;
    const size_t ng=(size_t)D*(D/64);
    const size_t p8bytes=qng64_packed_bytes(D,D,8);
    const size_t p4bytes=count/2;
    if(p8bytes!=count || qng64_group_bytes(8)!=64 || qng64_group_bytes(4)!=0)
      {fprintf(stderr,"unexpected qNg64 format or n4 rejection\n");return 3;}
    float *w=malloc(count*sizeof(float)),*x=malloc(total*sizeof(float));
    float *ref=malloc(total*sizeof(float)),*out8=malloc(total*sizeof(float)),
          *out4=malloc(total*sizeof(float));
    uint8_t *p8=calloc(p8bytes,1),*p4=calloc(p4bytes,1);
    float *s8=malloc(ng*sizeof(float)),*s4=malloc(ng*sizeof(float));
    if(!w||!x||!ref||!out8||!out4||!p8||!p4||!s8||!s4) {
      fprintf(stderr,"allocation failed\n");return 4;
    }
    if(!read_binary(dir,"q_proj_weight.f32",w,count*sizeof(float)) ||
       !read_binary(dir,"layer0_token_norm_inputs.f32",x,total*sizeof(float))) {
      fprintf(stderr,"source fixture missing or unexpected length\n");return 5;
    }
    for(size_t i=0;i<count;i++) if(!isfinite(w[i]))return 6;
    for(size_t i=0;i<total;i++) if(!isfinite(x[i]))return 7;
    double t0=seconds();
    gguf_quantize_qNg64(w,D,D,8,p8,s8);
    double t8=seconds()-t0;
    t0=seconds();
    gguf_quantize_q4g64_error_feedback(w,D,D,p4,s4);
    double t4=seconds()-t0;
    for(int i=0;i<N;i++) {
        run_one(0,w,p8,s8,p4,s4,x+(size_t)i*D,ref+(size_t)i*D);
        run_one(1,w,p8,s8,p4,s4,x+(size_t)i*D,out8+(size_t)i*D);
        run_one(2,w,p8,s8,p4,s4,x+(size_t)i*D,out4+(size_t)i*D);
    }
    double r8,r4,rmse8,rmse4,mae8,mae4,max8,max4;
    analyze(ref,out8,total,&r8,&rmse8,&mae8,&max8);
    analyze(ref,out4,total,&r4,&rmse4,&mae4,&max4);
    if(!isfinite(r8)||!isfinite(r4)) {fprintf(stderr,"nonfinite outputs\n");return 8;}
    if(r8>1.0 || r4>1.0) {fprintf(stderr,"pathologically inaccurate packed outputs\n");return 9;}
    double values[3][ROUNDS]={0};
    for(int k=0;k<3;k++) {
        float scratch[D];
        run_one(k,w,p8,s8,p4,s4,x,scratch); // cache warmup
        for(int round=0;round<ROUNDS;round++) {
            double start=seconds();
            for(int i=0;i<TIMED_SAMPLES;i++) {
                int idx=(i*(N-1))/(TIMED_SAMPLES-1);
                run_one(k,w,p8,s8,p4,s4,x+(size_t)idx*D,scratch);
            }
            values[k][round]=(seconds()-start)*1000/TIMED_SAMPLES;
        }
    }
    double med[3]={0};
    for(int i=0;i<3;i++) {
        qsort(values[i],ROUNDS,sizeof(double),cmp_double);
        med[i]=values[i][ROUNDS/2];
    }
    if(!write_binary(dir,"native_bf16_reference_outputs.f32",ref,total*sizeof(float))||
       !write_binary(dir,"native_qng64_n8_outputs.f32",out8,total*sizeof(float))||
       !write_binary(dir,"native_packed4_outputs.f32",out4,total*sizeof(float))||
       !write_binary(dir,"native_qng64_n8.planes",p8,p8bytes)||
       !write_binary(dir,"native_qng64_n8.scales.f32",s8,ng*sizeof(float))||
       !write_binary(dir,"native_packed4.codes",p4,p4bytes)||
       !write_binary(dir,"native_packed4.scales.f32",s4,ng*sizeof(float))) {
        fprintf(stderr,"refusing to overwrite / write outputs\n");return 10;
    }
    const size_t bf16_weight_bytes=count*2;
    const size_t q8_weight_bytes=p8bytes+ng*sizeof(float);
    const size_t q4_weight_bytes=p4bytes+ng*sizeof(float);
    printf("{\"schema\":\"oml-acs-native-beglin-q-tensor/1\","
      "\"status\":\"PASS_NATIVE_1_TENSOR_ONLY\",\"device\":\"Apple Silicon CPU\","
      "\"tensor\":\"model.layers.0.self_attn.q_proj.weight\","
      "\"shape\":[%d,%d],\"group_size\":64,"
      "\"bf16_weight_bytes\":%zu,\"qng64_8_weight_bytes\":%zu,"
      "\"packed4_weight_bytes\":%zu,"
      "\"reference\":\"BF16 source widened FP32 Accelerate GEMV\","
      "\"n_inputs\":%d,\"quant8_encoder\":\"BEGLIN gguf_quantize_qNg64\","
      "\"quant4_encoder\":\"BEGLIN gguf_quantize_q4g64_error_feedback\","
      "\"quant8_kernel\":\"BEGLIN qng64_matvec_f32\","
      "\"quant4_kernel\":\"BEGLIN gemv_q4g64_fused NEON\","
      "\"n8_relative_l2_error\":%.10g,\"n4_relative_l2_error\":%.10g,"
      "\"n8_rmse\":%.10g,\"n4_rmse\":%.10g,"
      "\"n8_mae\":%.10g,\"n4_mae\":%.10g,"
      "\"n8_max_abs\":%.10g,\"n4_max_abs\":%.10g,"
      "\"quant8_build_ms\":%.8g,\"quant4_build_ms\":%.8g,"
      "\"steady_f32_accelerate_median_ms_per_vector\":%.9g,"
      "\"steady_qng64_8_median_ms_per_vector\":%.9g,"
      "\"steady_packed4_median_ms_per_vector\":%.9g,"
      "\"timed_vectors_per_round\":%d,\"timing_rounds\":%d,"
      "\"full_model_generation\":false,\"task_AUROC\":null,\"metal_GPU_tested\":false,"
      "\"production_touched\":false}\n",
      D,D,bf16_weight_bytes,q8_weight_bytes,q4_weight_bytes,N,
      r8,r4,rmse8,rmse4,mae8,mae4,max8,max4,t8*1000,t4*1000,
      med[0],med[1],med[2],TIMED_SAMPLES,ROUNDS);
    free(w);free(x);free(ref);free(out8);free(out4);free(p8);free(p4);free(s8);free(s4);
    return 0;
}
