// BEGLIN research-only, SINGLE-Metal-kernel qNg64 n7 MLA attention absorption.
//
// NO expanded per-head K/V, NO device dense dequantized weights, NO model data.
// Inputs packed qNg64 n7 group64 and normalized 512D latent, rotated 64D k_pe.
// One Metal threadgroup per attention head: direct packed Wk^T qNoPE,
// latent scores+rotary scores, causal last-position softmax, latent weighted
// aggregation, and direct packed Wv output. This initial fused kernel is a
// CORRECTNESS BASELINE, not an optimized throughput or serving benchmark.
// Reuses staged probe's bounds/source helpers in one independent TU.
#include "mla_qng64_metal_absorbed_staged_v1.cpp"

namespace {

mx::fast::CustomKernelFunction& fused_mla_qng64_kernel() {
    static std::optional<mx::fast::CustomKernelFunction> k;
    if(!k) {
        static const std::string src=R"(
            uint lane=thread_position_in_grid.x;
            uint head=thread_position_in_grid.z;
            // Grid={64,1,16}, threadgroup={64,1,1}. All 64 lanes reach
            // every barrier; never return conditionally in this kernel.
            threadgroup float qabs[512];
            threadgroup float scores[64];
            threadgroup float probs[64];
            threadgroup float aggregation[512];

            // 1. q_absorb[h,d]=sum_r Wk[h,r,d] * qnope[h,r].
            // Weight rows are directly accessed from n7 raw compressed planes.
            for(uint chunk=0;chunk<8u;chunk++) {
                uint d=chunk*64u+lane;
                uint g=d>>6;
                uint p=d&63u;
                float sum=0.0f;
                for(uint r=0;r<128u;r++) {
                    uint row=head*256u+r;
                    uint offset=(row*8u+g)*56u;
                    uint8_t byte0=planes[offset+0u*8u+(p>>3)];
                    uint8_t byte1=planes[offset+1u*8u+(p>>3)];
                    uint8_t byte2=planes[offset+2u*8u+(p>>3)];
                    uint8_t byte3=planes[offset+3u*8u+(p>>3)];
                    uint8_t byte4=planes[offset+4u*8u+(p>>3)];
                    uint8_t byte5=planes[offset+5u*8u+(p>>3)];
                    uint8_t byte6=planes[offset+6u*8u+(p>>3)];
                    uint bit=p&7u;
                    int value = ((byte0>>bit)&1u)
                              |(((byte1>>bit)&1u)<<1)
                              |(((byte2>>bit)&1u)<<2)
                              |(((byte3>>bit)&1u)<<3)
                              |(((byte4>>bit)&1u)<<4)
                              |(((byte5>>bit)&1u)<<5)
                              |(((byte6>>bit)&1u)<<6);
                    float weight=float(value-64)*scales[row*8u+g];
                    sum+=q_nope[head*128u+r]*weight;
                }
                qabs[d]=sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 2. Compute one causal last-position attention score/token.
            // Each SIMD lane is responsible for one of up to 64 tokens.
            if(lane<(uint)cache_tokens) {
                float sum=0.0f;
                for(uint d=0;d<512u;d++)
                    sum+=qabs[d]*latents[lane*512u+d];
                for(uint d=0;d<64u;d++)
                    sum+=q_rope[head*64u+d]*k_pe[lane*64u+d];
                scores[lane]=sum*score_param[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 3. Stable softmax; lane0 performs small T<=64 reduction.
            if(lane==0u) {
                float m=scores[0];
                for(uint t=1u;t<(uint)cache_tokens;t++)
                    m=max(m,scores[t]);
                float sum=0.0f;
                for(uint t=0u;t<(uint)cache_tokens;t++) {
                    float p=exp(scores[t]-m);
                    probs[t]=p;
                    sum+=p;
                }
                for(uint t=0u;t<(uint)cache_tokens;t++)
                    probs[t]/=sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 4. Aggregate resident normalized latent, not expanded V.
            for(uint chunk=0;chunk<8u;chunk++) {
                uint d=chunk*64u+lane;
                float v=0.0f;
                for(uint t=0u;t<(uint)cache_tokens;t++)
                    v+=probs[t]*latents[t*512u+d];
                aggregation[d]=v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 5. Direct packed 7-bit Wv GEMV. One lane computes two
            // independently bounded value output coordinates.
            for(uint value_row=lane;value_row<128u;value_row+=64u) {
                uint row=head*256u+128u+value_row;
                float y=0.0f;
                for(uint g=0u;g<8u;g++) {
                    uint offset=(row*8u+g)*56u;
                    for(uint p=0u;p<64u;p++) {
                        uint u=0u;
                        uint bit=p&7u;
                        uint index=p>>3;
                        for(uint plane=0u;plane<7u;plane++) {
                            uint8_t b=planes[offset+plane*8u+index];
                            u|=((uint)((b>>bit)&1u)<<plane);
                        }
                        y+=((float)((int)u-64))*scales[row*8u+g]
                             *aggregation[g*64u+p];
                    }
                }
                value_output[head*128u+value_row]=y;
            }
        )";
        k=mx::fast::metal_kernel(
          "beglin_mla_qng64_single_kernel_fused_absorbed_v1",
          {"planes","scales","q_nope","q_rope","latents","k_pe","score_param"},
          {"value_output"},src);
    }
    return *k;
}

} // namespace

extern "C" {

// Same source buffer contracts as the staged probe: 16 heads, qNg64 n7,
// 4096 by 512 source kv_b packed matrix, 1 <= tokens <= 64, F32 vectors.
// Unlike the staged variant: ONE physical MLX custom Metal kernel dispatch
// contains ALL source key/score/softmax/value calculations.
int beglin_mla_qng64_fused_metal_v1(
    const uint8_t* planes, const float* scales, int bits,
    const float* q_nope, const float* q_rope,
    const float* latents, const float* k_pe,
    int tokens, float score_scale, float* out_hv) {

    if(!planes || !scales || !q_nope || !q_rope ||
       !latents || !k_pe || !out_hv || bits != N ||
       tokens<1 || tokens>MAX_T ||
       !std::isfinite(score_scale) || score_scale<=0.0f || score_scale>1.0f)
        return 0;
    if(!finite_arr(scales,(size_t)OUT*NG)
       || !finite_arr(q_nope,(size_t)H*K)
       || !finite_arr(q_rope,(size_t)H*R)
       || !finite_arr(latents,(size_t)tokens*LAT)
       || !finite_arr(k_pe,(size_t)tokens*R))return 0;
    for(size_t i=0;i<(size_t)OUT*NG;i++)
        if(scales[i]<=0.0f)return 0;
    if(!mx::metal::is_available())return -1;
    try{
        mx::array packed_arr((void*)planes,{OUT,NG*N*8},mx::uint8,noop_deleter);
        mx::array scales_arr((void*)scales,{OUT,NG},mx::float32,noop_deleter);
        mx::array qn_arr((void*)q_nope,{H,K},mx::float32,noop_deleter);
        mx::array qr_arr((void*)q_rope,{H,R},mx::float32,noop_deleter);
        mx::array lat_arr((void*)latents,{tokens,LAT},mx::float32,noop_deleter);
        mx::array rope_arr((void*)k_pe,{tokens,R},mx::float32,noop_deleter);
        mx::array score_parameter((void*)&score_scale,{1},mx::float32,noop_deleter);
        auto out=fused_mla_qng64_kernel()(
          {packed_arr,scales_arr,qn_arr,qr_arr,lat_arr,rope_arr,score_parameter},
          {{H,V}},{mx::float32},{64,1,H},{64,1,1},
          {{"cache_tokens",tokens}},
          std::nullopt,false,{}
        )[0];
        mx::eval(out);
        const float* data=out.data<float>();
        if(!finite_arr(data,(size_t)H*V))return 0;
        std::memcpy(out_hv,data,sizeof(float)*(size_t)H*V);
        return 1;
    }catch(const std::exception& e){
        std::fprintf(stderr,"BEGLIN_MLA_FUSED_METAL_RESEARCH_EXCEPTION:%s\n",e.what());
        return -2;
    }catch(...){
        std::fprintf(stderr,"BEGLIN_MLA_FUSED_METAL_UNHANDLED_EXCEPTION\n");
        return -2;
    }
}

int beglin_mla_qng64_fused_metal_available_v1(){
    return mx::metal::is_available()?1:0;
}

} // extern C
