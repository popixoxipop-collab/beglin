// BEGLIN research-only qNg64 n7 MLA latent absorption on Apple Metal.
//
// B=1, H=16, L<=64, Dlatent=512, Knope=128, Rope=64, V=128.
// Source kv_b_proj layout [H * (K+V)=4096, 512], packed group64 bitplanes,
// one signed bit7 scale per [row,g64]. We never materialize old expanded
// per-head K/V (nor dense dequantized GPU kv_b weight arrays).
//
// Four actual Metal kernels:
//   1. absorbed q = q_nope * Wk  (decode bitplanes directly)
//   2. source scaled dot-product scores against resident latent/RoPE keys
//   3. softmax-weighted latent aggregation (H,512)
//   4. source out = Wv * aggregated_latent (decode bitplanes directly)
//
// This is not the original BEGLIN production symbol, fused one-dispatch kernel,
// exact BF16 source attention, or measured serving speed. Source/model policy
// control and production remain untouched.
#include "mlx/mlx.h"
#include "mlx/scheduler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mx = mlx::core;
namespace {
constexpr int H=16, K=128, V=128, R=64, D=512;
constexpr int STRIDE=K+V, NG=D/64, BITS=7;
constexpr size_t PLANE_BYTES=(size_t)H*STRIDE*NG*BITS*8;
constexpr size_t SCALE_COUNT=(size_t)H*STRIDE*NG;

void noop_deleter(void*) {}

bool finite_f32(const float* values, size_t len) {
    if(!values) return false;
    for(size_t i=0;i<len;i++) if(!std::isfinite(values[i])) return false;
    return true;
}
bool valid(const uint8_t* planes, size_t plane_size,
           const float* scales, size_t scale_count,
           const float* q_nope, size_t q_nope_count,
           const float* q_rope, size_t q_rope_count,
           const float* latent, size_t latent_count,
           const float* k_pe, size_t k_pe_count,
           int T, float scale,
           float* out, size_t out_count,
           float* scores, size_t score_count) {
    if(!planes||plane_size!=PLANE_BYTES||
       !scales||scale_count!=SCALE_COUNT||
       !q_nope||q_nope_count!=(size_t)H*K||
       !q_rope||q_rope_count!=(size_t)H*R||
       !latent||T<=0||T>64||latent_count!=(size_t)T*D||
       !k_pe||k_pe_count!=(size_t)T*R||
       !out||out_count!=(size_t)H*V||
       !scores||score_count!=(size_t)H*T||
       !std::isfinite(scale)||scale<=0.f||scale>2.f)
        return false;
    if(!finite_f32(q_nope,q_nope_count)||!finite_f32(q_rope,q_rope_count)||
       !finite_f32(latent,latent_count)||!finite_f32(k_pe,k_pe_count))
        return false;
    for(size_t i=0;i<scale_count;i++)
        if(!std::isfinite(scales[i])||scales[i]<=0.f) return false;
    return true;
}
using Args=std::vector<std::pair<std::string,mx::fast::TemplateArg>>;

mx::fast::CustomKernelFunction& absorb_query_kernel() {
    static auto kernel=mx::fast::metal_kernel(
       "beglin_mla_q7_absorb_query_v1",
       {"planes","scales","q_nope"},{"q_abs"},
       R"(
          uint d = thread_position_in_grid.x;
          uint h = thread_position_in_grid.y;
          if (d>=512 || h>=16) return;
          uint group = d / 64;
          uint p = d % 64;
          uint byte_in_plane = p >> 3;
          uint bit_in_byte = p & 7;
          float acc = 0.0f;
          for (uint r=0;r<128;r++) {
              uint row=h*256+r;
              uint offset=(row*8+group)*56;
              uint code=0;
              for(uint j=0;j<7;j++) {
                 uint bit=(uint)(planes[offset+j*8+byte_in_plane]>>bit_in_byte)&1u;
                 code|=bit<<j;
              }
              float value=(float)((int)code-64)*scales[row*8+group];
              acc+=q_nope[h*128+r]*value;
          }
          q_abs[h*512+d]=acc;
       )");
    return kernel;
}
mx::fast::CustomKernelFunction& scores_kernel() {
    static auto kernel=mx::fast::metal_kernel(
      "beglin_mla_q7_latent_scores_v1",
      {"q_abs","q_rope","latent","k_pe","scalar"},{"scores"},
      R"(
          uint t = thread_position_in_grid.x;
          uint h = thread_position_in_grid.y;
          if(h>=16 || t >= (uint)t_len) return;
          float sum = 0.0f;
          for (uint d=0;d<512;d++)
              sum+=q_abs[h*512+d]*latent[t*512+d];
          for (uint r=0;r<64;r++)
              sum+=q_rope[h*64+r]*k_pe[t*64+r];
          scores[h*(uint)t_len+t]=sum*scalar[0];
      )");
    return kernel;
}
mx::fast::CustomKernelFunction& aggregate_kernel() {
    static auto kernel=mx::fast::metal_kernel(
      "beglin_mla_q7_latent_softmax_aggregation_v1",
      {"scores","latent"},{"agg"},
      R"(
          uint d = thread_position_in_grid.x;
          uint h = thread_position_in_grid.y;
          if (d>=512 || h>=16) return;
          float max_s=-INFINITY;
          for (uint t=0;t<(uint)t_len;t++)
              max_s=max(max_s,scores[h*(uint)t_len+t]);
          float sumexp=0.0f, weighted=0.0f;
          for (uint t=0;t<(uint)t_len;t++) {
              float e=exp(scores[h*(uint)t_len+t]-max_s);
              sumexp+=e;
              weighted+=e*latent[t*512+d];
          }
          agg[h*512+d]=weighted/sumexp;
      )");
    return kernel;
}
mx::fast::CustomKernelFunction& apply_value_kernel() {
    static auto kernel=mx::fast::metal_kernel(
      "beglin_mla_q7_absorbed_value_v1",
      {"planes","scales","agg"},{"result"},
      R"(
          uint v = thread_position_in_grid.x;
          uint h = thread_position_in_grid.y;
          if(v>=128 || h>=16) return;
          uint row=h*256+128+v;
          float acc=0.0f;
          for (uint g=0;g<8;g++) {
             uint offset=(row*8+g)*56;
             float scale=scales[row*8+g];
             for (uint p=0;p<64;p++) {
                uint code=0;
                uint byte_in_plane=p>>3;
                uint bit_in_byte=p&7;
                for(uint j=0;j<7;j++) {
                   uint bit=(uint)(planes[offset+j*8+byte_in_plane]>>bit_in_byte)&1u;
                   code|=bit<<j;
                }
                float value=(float)((int)code-64)*scale;
                acc+=value*agg[h*512+g*64+p];
             }
          }
          result[h*128+v]=acc;
      )");
    return kernel;
}
}

extern "C" {

// Buffer lengths must exactly match declared contract BEFORE constructing
// Metal arrays; do not use these raw pointer APIs with untrusted lengths.
int beglin_qng64_mla_absorbed_probe_v1(
    const uint8_t* planes, size_t plane_size,
    const float* scales, size_t scale_count,
    const float* q_nope, size_t q_nope_count,
    const float* q_rope, size_t q_rope_count,
    const float* latent, size_t latent_count,
    const float* k_pe, size_t k_pe_count,
    int T, float attn_scale,
    float* result, size_t result_count,
    float* scores, size_t scores_count,
    double* measured_pipeline_ms) {

    if(!valid(planes,plane_size,scales,scale_count,q_nope,q_nope_count,
              q_rope,q_rope_count,latent,latent_count,k_pe,k_pe_count,
              T,attn_scale,result,result_count,scores,scores_count))
       return 0;
    if(!mx::metal::is_available()) return 0;
    try {
       auto started=std::chrono::steady_clock::now();
       mx::array codes((void*)planes,{(int)PLANE_BYTES},mx::uint8,noop_deleter);
       mx::array s((void*)scales,{(int)SCALE_COUNT},mx::float32,noop_deleter);
       mx::array qn((void*)q_nope,{H,K},mx::float32,noop_deleter);
       mx::array qr((void*)q_rope,{H,R},mx::float32,noop_deleter);
       mx::array lh((void*)latent,{T,D},mx::float32,noop_deleter);
       mx::array kp((void*)k_pe,{T,R},mx::float32,noop_deleter);
       mx::array scalar((void*)&attn_scale,{1},mx::float32,noop_deleter);
       auto QA=absorb_query_kernel()(
         {codes,s,qn},{{H,D}},{mx::float32},
         {512,16,1},{64,1,1},{},std::nullopt,false,{})[0];
       Args template_scores={{"t_len",T}};
       auto S=scores_kernel()(
         {QA,qr,lh,kp,scalar},{{H,T}},{mx::float32},
         {T,16,1},{32,1,1},template_scores,std::nullopt,false,{})[0];
       auto A=aggregate_kernel()(
         {S,lh},{{H,D}},{mx::float32},
         {512,16,1},{64,1,1},{{"t_len",T}},std::nullopt,false,{})[0];
       auto VOUT=apply_value_kernel()(
         {codes,s,A},{{H,V}},{mx::float32},
         {128,16,1},{64,1,1},{},std::nullopt,false,{})[0];
       mx::eval(VOUT,S);
       std::memcpy(result,VOUT.data<float>(),result_count*sizeof(float));
       std::memcpy(scores,S.data<float>(),scores_count*sizeof(float));
       if(measured_pipeline_ms) {
          auto elapsed=std::chrono::duration<double,std::milli>(
               std::chrono::steady_clock::now()-started).count();
          *measured_pipeline_ms=elapsed;
       }
       return 1;
    }catch(const std::exception& e) {
       std::fprintf(stderr,"[BEGLIN qNg64 MLA Metal research] %s\n",e.what());
       return 0;
    }catch(...) {return 0;}
}

int beglin_qng64_mla_absorbed_available_v1() {
    return mx::metal::is_available()?1:0;
}
}
