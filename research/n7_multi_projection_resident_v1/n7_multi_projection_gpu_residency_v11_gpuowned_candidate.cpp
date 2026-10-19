// BEGLIN isolated research-only n7/group64 DEVICE-RESIDENT GEMV/GEMM.
// This implements the original qng64_dense.c bitplane contract as a native
// MLX C++ Metal CustomKernel. It owns packed uint8 and F32 group scales in
// evaluated GPU allocations, and reuses the exact handles across requests.
// Supports independent E=1 linear matrices and batch 1..16. No dense
// dequantized weights, no deployment, no original model source modifications.
//
// RESEARCH GATES: bounded n7 shape/byte count and resident capacity;
// explicit materialization + byte-perfect GPU->CPU verification; exact
// source metadata must be checked separately by trusted attested loader.
// API raw pointers cannot discover caller-provided capacity.
#include "mlx/mlx.h"
#include "mlx/fast.h"
#include "mlx/memory.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mx = mlx::core;

namespace {
constexpr int BITS=7, GROUP=64;
constexpr int MAX_OUT=16384;
constexpr int MAX_IN=16384;
constexpr int MAX_BATCH=16;
constexpr uint64_t MAX_ONE_TENSOR_BYTES=96ULL*1024*1024;
constexpr uint64_t REGISTRY_BUDGET_BYTES=256ULL*1024*1024;
constexpr size_t MAX_ACTIVE_HANDLES=128;
void noop(void*){}

bool good(int out,int in,int bits,uint64_t* planes_bytes,uint64_t* scales_bytes) {
    if(bits!=BITS||out<1||out>MAX_OUT||in<GROUP||in>MAX_IN||in%GROUP!=0)
        return false;
    uint64_t ng=(uint64_t)in/GROUP;
    uint64_t p=(uint64_t)out*ng*BITS*8;
    uint64_t s=(uint64_t)out*ng*sizeof(float);
    if(p==0||s==0||p+s>MAX_ONE_TENSOR_BYTES
       ||ng>UINT32_MAX||(uint64_t)out*ng>UINT32_MAX/56)return false;
    *planes_bytes=p;*scales_bytes=s;
    return true;
}
bool finite(const float* values,size_t n,bool positive=false) {
    if(!values)return false;
    for(size_t j=0;j<n;++j){
        if(!std::isfinite(values[j]) || (positive && values[j]<=0.0f))
            return false;
    }
    return true;
}

mx::fast::CustomKernelFunction& resident_kernel() {
    static std::optional<mx::fast::CustomKernelFunction> kernel;
    if(!kernel) {
        // Two 32-lane simdgroup reductions and a 2-item threadgroup sum are
        // required (one simd_sum() over 64 lanes is incorrect).
        static const std::string source=R"(
            uint lane=thread_position_in_grid.x;
            uint row=thread_position_in_grid.y;
            uint batch=thread_position_in_grid.z;
            if(lane>=64u || row>=(uint)out_dim || batch>=(uint)batch_count)return;
            float partial=0.0f;
            const uint lane_byte=lane>>3;
            const uint lane_bit=lane&7u;
            for(uint g=0;g<(uint)ng;g++){
                const uint base=(row*(uint)ng+g)*56u;
                uint code=0u;
                for(uint bit=0;bit<7u;bit++){
                    uint8_t data=planes[base+bit*8u+lane_byte];
                    code|=((uint)((data>>lane_bit)&1u)<<bit);
                }
                const float scale=scales[row*(uint)ng+g];
                const float w=(float)((int)code-64)*scale;
                partial += w*x[batch*((uint)ng*64u)+g*64u+lane];
            }
            threadgroup float scratch[2];
            const uint simd_lane=lane%32u;
            const uint simd_group=lane/32u;
            const float sum=simd_sum(partial);
            if(simd_lane==0u)scratch[simd_group]=sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane==0u){
                y[batch*(uint)out_dim+row]=scratch[0]+scratch[1];
            }
        )";
        kernel=mx::fast::metal_kernel(
            "beglin_qng64_n7_device_resident_gemv_v1",
            {"planes","scales","x"},{"y"},source);
    }
    return *kernel;
}
struct Packed {
    mx::array planes;
    mx::array scales;
    // V11 GPU-only ownership CANDIDATE: source CPU buffers are retained only
    // inside registration and explicitly consumed by a GPU elementwise
    // materialization. The registry must never hold source host array views.
    // This file is NOT approved until separate after-scope lifetime/poison,
    // GC pressure and real-source model parity independent GPU QA all pass.
    int out, in, ng;
    uint64_t bytes;
    Packed(mx::array p,mx::array s,int o,int i,uint64_t n)
      :planes(std::move(p)),scales(std::move(s)),
       out(o),in(i),ng(i/GROUP),bytes(n){}
};
std::mutex guard;
std::unordered_map<uint64_t,std::shared_ptr<Packed>> registry;
uint64_t next_id=1,total_bytes=0;

std::shared_ptr<Packed> lookup(uint64_t id) {
    std::lock_guard<std::mutex> lock(guard);
    auto it=registry.find(id);
    return it==registry.end()?nullptr:it->second;
}
}

extern "C" {

int beglin_multi_n7_gpu_available_v1() {
    return mx::metal::is_available()?1:0;
}

uint64_t beglin_multi_n7_register_v1(
    const uint8_t* planes,const float* scales,int out,int in,int bits) {
    uint64_t pb=0,sb=0;
    if(!planes||!scales||!mx::metal::is_available()||
       !good(out,in,bits,&pb,&sb)||
       !finite(scales,(size_t)(sb/4),true))return 0;
    {
        std::lock_guard<std::mutex> lock(guard);
        if(registry.size()>=MAX_ACTIVE_HANDLES
           ||total_bytes+pb+sb>REGISTRY_BUDGET_BYTES)return 0;
    }
    try {
        // Deep copies are necessary until mx::copy completes; MLX input
        // tensors below borrow temporary heap data, never caller pointers.
        std::vector<uint8_t> ownp(planes,planes+pb);
        std::vector<float> owns(scales,scales+sb/4);
        mx::array hostp((void*)ownp.data(),{out,(in/GROUP)*56},
                        mx::uint8,noop);
        mx::array hosts((void*)owns.data(),{out,in/GROUP},
                        mx::float32,noop);
        mx::Device gpu(mx::Device::gpu);
        // MLX copy alone was empirically insufficient to discard host
        // zero-copy buffers safely in previous prototypes. Force an actual
        // independent *computed GPU output* by elementwise addition with
        // device zeros. The two compact output tensors, not original views,
        // become the sole registry-owned source weight representations.
        auto staging_planes=mx::copy(hostp,gpu);
        auto staging_scales=mx::copy(hosts,gpu);
        mx::array gp=mx::add(staging_planes,
            mx::zeros({out,(in/GROUP)*56},mx::uint8,gpu));
        mx::array gs=mx::add(staging_scales,
            mx::zeros({out,in/GROUP},mx::float32,gpu));
        mx::eval(gp,gs);
        // Source transfer authority: physical round-trip of compact bytes,
        // not just equality of pointer-type declarations.
        mx::array backp=mx::copy(gp,mx::Device(mx::Device::cpu));
        mx::array backs=mx::copy(gs,mx::Device(mx::Device::cpu));
        mx::eval(backp,backs);
        if(std::memcmp(backp.data<uint8_t>(),ownp.data(),(size_t)pb)!=0||
           std::memcmp(backs.data<float>(),owns.data(),(size_t)sb)!=0)
           return 0;
        auto object=std::make_shared<Packed>(gp,gs,out,in,pb+sb);
        std::lock_guard<std::mutex> lock(guard);
        if(next_id==0||next_id>UINT64_C(9223372036854775807)||
           registry.size()>=MAX_ACTIVE_HANDLES||
           total_bytes+pb+sb>REGISTRY_BUDGET_BYTES)return 0;
        auto id=next_id++;
        registry.emplace(id,std::move(object));
        total_bytes+=pb+sb;
        return id;
    }catch(const std::exception& e){
        std::fprintf(stderr,"BEGLIN_MULTI_GPU_REGISTER:%s\n",e.what());
        return 0;
    }catch(...){return 0;}
}

// No bias, batch [B,in], output [B,out]. All inputs/outputs F32 in native ABI.
// Return 1 success, 0 invalid shape/handle/data, -2 internal MLX failure.
int beglin_multi_n7_forward_v1(
    uint64_t id,const float* x,int B,float* output) {
    auto weights=lookup(id);
    if(!weights||!x||!output||B<1||B>MAX_BATCH||
       !finite(x,(size_t)B*weights->in))return 0;
    try {
        mx::array hX((void*)x,{B,weights->in},mx::float32,noop);
        mx::array gpuX=mx::copy(hX,mx::Device(mx::Device::gpu));
        // Synchronize dynamic host input -> GPU copy before dispatch. The
        // original host x is zero-copy wrapped and must not remain an
        // asynchronous dependency past its C ABI ownership boundary.
        mx::eval(gpuX);
        std::vector<std::pair<std::string,mx::fast::TemplateArg>> tags={
            {"out_dim",weights->out},
            {"ng",weights->ng},
            {"batch_count",B}
        };
        auto z=resident_kernel()(
            {weights->planes,weights->scales,gpuX},
            {{B,weights->out}},{mx::float32},
            {64,weights->out,B},{64,1,1},tags,std::nullopt,false,{})[0];
        mx::eval(z);
        mx::array cpu=mx::copy(z,mx::Device(mx::Device::cpu));
        mx::eval(cpu);
        auto val=cpu.data<float>();
        if(!finite(val,(size_t)B*weights->out))return 0;
        std::memcpy(output,val,sizeof(float)*(size_t)B*weights->out);
        return 1;
    }catch(const std::exception& e){
        std::fprintf(stderr,"BEGLIN_MULTI_GPU_FORWARD:%s\n",e.what());
        return -2;
    }catch(...){return -2;}
}

int beglin_multi_n7_unregister_v1(uint64_t id) {
    if(!id)return 0;
    std::lock_guard<std::mutex> lock(guard);
    auto it=registry.find(id);
    if(it==registry.end())return 0;
    total_bytes-=it->second->bytes;
    registry.erase(it);
    return 1;
}
size_t beglin_multi_n7_active_count_v1() {
    std::lock_guard<std::mutex> lock(guard);
    return registry.size();
}
uint64_t beglin_multi_n7_payload_bytes_v1() {
    std::lock_guard<std::mutex> lock(guard);
    return total_bytes;
}
uint64_t beglin_multi_n7_handle_payload_v1(uint64_t id) {
    auto ptr=lookup(id);return ptr?ptr->bytes:0;
}
uint64_t beglin_multi_n7_allocator_active_v1() {
    return mx::get_active_memory();
}
uint64_t beglin_multi_n7_allocator_cached_v1() {
    return mx::get_cache_memory();
}
uint64_t beglin_multi_n7_registry_budget_v1() {
    return REGISTRY_BUDGET_BYTES;
}
}
