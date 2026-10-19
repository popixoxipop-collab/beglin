// BEGLIN isolated research-only compressed qNg64 MLA latent absorption on Metal.
// This is NOT production mlx_moe.cpp, not a complete native attention forward,
// and does NOT implement a one-kernel fused whole attention. It is an actual
// native C++/MLX GPU staged attention pipeline with two custom packed-bitplane
// Metal kernels and device-resident MLX score/softmax/latent aggregation ops.
//
// Packed source: 4096 rows (16 heads x (128 KNoPE + 128 V)), 512 input,
// qNg64 bit width n7; planes row/group/bitplane layout from qng64_dense.c;
// packed bits/scales remain compressed on Metal. No expanded K/V or dense
// dequantized matrix is materialized on device, at any step.
// Research probe API uses pointer capacities controlled by caller; validating
// backing pointer lengths requires an upstream approved source manifest.
#include "mlx/mlx.h"
#include "mlx/fast.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace mx = mlx::core;
namespace {

constexpr int H=16;
constexpr int K=128;
constexpr int V=128;
constexpr int LAT=512;
constexpr int R=64;
constexpr int OUT=H*(K+V);
constexpr int N=7;
constexpr int NG=LAT/64;
constexpr int MAX_T=64;

void noop_deleter(void*){}

template<typename T>
bool finite_arr(const T* ptr,size_t n) {
    if(!ptr || n==0) return false;
    for(size_t i=0;i<n;i++)if(!std::isfinite((double)ptr[i]))return false;
    return true;
}

mx::fast::CustomKernelFunction& compressed_query_kernel() {
    static std::optional<mx::fast::CustomKernelFunction> kernel;
    if(!kernel) {
        // x is a latent index 0..511; y is head 0..15. Each lane loops
        // over the 128 packed Wk rows and decodes *one* 7-bit weight/code
        // from its group64 bitplane. Result = transpose(Wk[h]) @ q_nope[h].
        static const std::string src=R"(
            uint d = thread_position_in_grid.x;
            uint h = thread_position_in_grid.y;
            if(d >= 512u || h >= 16u) return;
            uint g = d >> 6;
            uint bit_index = d & 63u;
            uint byte_in_plane = bit_index >> 3;
            uint bit_in_byte = bit_index & 7u;
            float accum = 0.0f;
            for(uint r=0;r<128u;r++) {
                uint row = h*256u + r;
                uint start = (row*8u + g)*56u;
                uint code=0u;
                for(uint plane=0;plane<7u;plane++) {
                    uint8_t b=planes[start+plane*8u+byte_in_plane];
                    code |= ((uint)((b >> bit_in_byte)&1u) << plane);
                }
                float w=(float)((int)code-64)*scales[row*8u+g];
                accum += w*q_nope[h*128u+r];
            }
            qabs[h*512u+d]=accum;
        )";
        kernel=mx::fast::metal_kernel(
           "beglin_mla_qng64_direct_absorb_query_v1",
           {"planes","scales","q_nope"},{"qabs"},src);
    }
    return *kernel;
}

mx::fast::CustomKernelFunction& compressed_value_kernel() {
    static std::optional<mx::fast::CustomKernelFunction> kernel;
    if(!kernel) {
        // x=lane (one source group position); y=one V output row; z=head.
        // qNg64 packed planes and F32 scales read directly; 64-lane
        // threadgroup two-stage reduction matches BEGLIN's original fixed
        // qng64_gemv_kernel() 32+32-lane summation safety invariant.
        static const std::string src=R"(
            uint p=thread_position_in_grid.x;
            uint v=thread_position_in_grid.y;
            uint h=thread_position_in_grid.z;
            if(p>=64u || v>=128u || h>=16u) return;
            uint row=h*256u+128u+v;
            uint byte_in_plane=p>>3;
            uint bit_in_byte=p&7u;
            float accum=0.0f;
            for(uint g=0;g<8u;g++) {
                uint start=(row*8u+g)*56u;
                uint code=0u;
                for(uint bit=0;bit<7u;bit++) {
                    uint8_t b=planes[start+bit*8u+byte_in_plane];
                    code|=((uint)((b>>bit_in_byte)&1u)<<bit);
                }
                float w=(float)((int)code-64)*scales[row*8u+g];
                accum+=w*latent_aggregate[h*512u+g*64u+p];
            }
            threadgroup float shared_sums[2];
            uint lane=p%32u;
            uint group=p/32u;
            float subtotal=simd_sum(accum);
            if(lane==0u)shared_sums[group]=subtotal;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(p==0u) value_output[h*128u+v]=shared_sums[0]+shared_sums[1];
        )";
        kernel=mx::fast::metal_kernel(
           "beglin_mla_qng64_direct_compressed_value_v1",
           {"planes","scales","latent_aggregate"},{"value_output"},src);
    }
    return *kernel;
}

} // namespace

extern "C" {

// Return codes: 1=GPU result computed and source conditions valid;
// 0=malformed/inadmissible input; -1=GPU unavailable; -2=Metal/MLX error.
// API pointer lengths are *contracts*, not inferable from C pointers.
// All F32 host inputs must be contiguous and known-good buffer lengths.
// A test or real integration must pin source tensor dimensions/bytes first.
int beglin_mla_qng64_metal_absorb_v1(
    const uint8_t* planes, const float* scales, int bits,
    const float* q_nope, const float* q_rope,
    const float* latents, const float* k_pe,
    int tokens, float score_scale, float* out_hv) {

    if(!planes || !scales || !q_nope || !q_rope
       || !latents || !k_pe || !out_hv || bits!=N
       || tokens<1 || tokens>MAX_T || !std::isfinite(score_scale)
       || score_scale<=0.0f || score_scale>1.0f) return 0;
    if(!finite_arr(scales,(size_t)OUT*NG)
       || !finite_arr(q_nope,(size_t)H*K)
       || !finite_arr(q_rope,(size_t)H*R)
       || !finite_arr(latents,(size_t)tokens*LAT)
       || !finite_arr(k_pe,(size_t)tokens*R)) return 0;
    for(size_t i=0;i<(size_t)OUT*NG;i++)
        if(scales[i]<=0.0f)return 0;
    if(!mx::metal::is_available())return -1;
    try {
        mx::array compressed_planes((void*)planes,{OUT,NG*N*8},mx::uint8,noop_deleter);
        mx::array group_scales((void*)scales,{OUT,NG},mx::float32,noop_deleter);
        mx::array query_nonrot((void*)q_nope,{H,K},mx::float32,noop_deleter);
        mx::array query_rope((void*)q_rope,{H,R},mx::float32,noop_deleter);
        mx::array resident_latent((void*)latents,{tokens,LAT},mx::float32,noop_deleter);
        mx::array resident_rope((void*)k_pe,{tokens,R},mx::float32,noop_deleter);
        std::vector<mx::array> input_query={
            compressed_planes,group_scales,query_nonrot
        };
        auto absorbed_query=compressed_query_kernel()(
            input_query, {{H,LAT}}, {mx::float32},
            {LAT,H,1}, {64,1,1},{},std::nullopt,false,{}
        )[0]; // Only [16,512] F32, never expanded K/V.
        mx::array latent_logits=mx::matmul(
            absorbed_query,mx::transpose(resident_latent));
        mx::array rope_logits=mx::matmul(
            query_rope,mx::transpose(resident_rope));
        mx::array scores=(latent_logits+rope_logits)*score_scale;
        mx::array probs=mx::softmax(scores,std::vector<int>{-1},true);
        mx::array latent_aggregate=mx::matmul(probs,resident_latent);
        auto value_output=compressed_value_kernel()(
            {compressed_planes,group_scales,latent_aggregate},
            {{H,V}}, {mx::float32},
            {64,V,H}, {64,1,1},{},std::nullopt,false,{}
        )[0];
        mx::eval(value_output);
        const float* final_output=value_output.data<float>();
        if(!finite_arr(final_output,(size_t)H*V))return 0;
        std::memcpy(out_hv,final_output,sizeof(float)*(size_t)H*V);
        return 1;
    } catch(const std::exception& e) {
        std::fprintf(stderr,
           "BEGLIN_MLA_METAL_RESEARCH_EXCEPTION: %s\n",e.what());
        return -2;
    } catch(...) {
        std::fprintf(stderr,"BEGLIN_MLA_METAL_RESEARCH_UNKNOWN_EXCEPTION\n");
        return -2;
    }
}

int beglin_mla_qng64_metal_available_v1(){
    return mx::metal::is_available()?1:0;
}

} // extern C
