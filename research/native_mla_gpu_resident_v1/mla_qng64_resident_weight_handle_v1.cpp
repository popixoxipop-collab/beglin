// BEGLIN EOE-only C++/Metal qNg64 MLA resident compressed-weight handle.
// Includes the already independently tested single physical fused Metal
// attention kernel, but binds packed n7 planes and FP32 scales ONCE and
// materializes them to GPU-owned MLX buffers. No dense dequantization and
// no expanded 16-head KV. This is a research cache ABI, not production.
//
// Thread-safety: process-local nonzero handles in a mutex-protected registry;
// shared_ptr lifetime permits in-flight eval while invalidating handles.
// Caller buffer capacity and untrusted ownership are checked by upstream test
// contracts; pointer bounds cannot be inferred from a raw C pointer.
#include "mla_qng64_metal_single_fused_v1.cpp"
#include "mlx/memory.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <exception>

namespace {
struct DevicePacked {
    mx::array planes;
    mx::array scales;
    // MLX async graphs may retain dependencies on imported host arrays
    // beyond the immediate eval call. Hold the deep-copied source bytes
    // until GPU handle destruction; never alias the caller's host memory.
    std::vector<uint8_t> backing_planes;
    std::vector<float> backing_scales;
    size_t source_bytes;
    explicit DevicePacked(mx::array&& p, mx::array&& s,
                          std::vector<uint8_t>&& bp,
                          std::vector<float>&& bs)
      : planes(std::move(p)), scales(std::move(s)),
        backing_planes(std::move(bp)),backing_scales(std::move(bs)),
        source_bytes((size_t)1835008+131072) {}
};
std::mutex store_mutex;
std::unordered_map<uint64_t,std::shared_ptr<DevicePacked>> registry;
uint64_t sequence=1;
constexpr size_t SOURCE_N7_BYTES=1835008;
constexpr size_t SOURCE_SCALE_CELLS=4096*8;

std::shared_ptr<DevicePacked> resolve(uint64_t id) {
    std::lock_guard<std::mutex> lock(store_mutex);
    auto it=registry.find(id);
    return it==registry.end()?nullptr:it->second;
}
}

extern "C" {

// Returns nonzero process-local opaque ID on success. Returns 0 on failure.
// Source input pointers need at least 1835008 uint8 and 32768 FP32 cells.
uint64_t beglin_mla_qng64_resident_register_v1(
    const uint8_t* planes,const float* scales,int bits) {
    if(!planes||!scales||bits!=7||!mx::metal::is_available())return 0;
    for(size_t i=0;i<SOURCE_SCALE_CELLS;i++)
        if(!std::isfinite(scales[i])||scales[i]<=0)return 0;
    try {
        // Own a temporary deep copy of caller buffers until MLX finishes
        // GPU placement. Neither caller memory nor test-only expanded BF16
        // references are retained after function exit.
        std::vector<uint8_t> own(planes,planes+SOURCE_N7_BYTES);
        std::vector<float> own_s(scales,scales+SOURCE_SCALE_CELLS);
        mx::array host_planes((void*)own.data(),{OUT,NG*N*8},
                              mx::uint8,noop_deleter);
        mx::array host_scales((void*)own_s.data(),{OUT,NG},
                              mx::float32,noop_deleter);
        auto gpu=mx::Device(mx::Device::gpu);
        auto device_planes=mx::copy(host_planes,gpu);
        auto device_scales=mx::copy(host_scales,gpu);
        mx::eval(device_planes,device_scales);
        // Strict GPU->CPU source byte roundtrip, so successful registration
        // proves the device copy is not just a pointer-shaped allocation.
        auto back_planes=mx::copy(device_planes,mx::Device(mx::Device::cpu));
        auto back_scales=mx::copy(device_scales,mx::Device(mx::Device::cpu));
        mx::eval(back_planes,back_scales);
        if(std::memcmp(back_planes.data<uint8_t>(),own.data(),SOURCE_N7_BYTES)!=0 ||
           std::memcmp(back_scales.data<float>(),own_s.data(),
                       SOURCE_SCALE_CELLS*sizeof(float))!=0) {
            std::fprintf(stderr,"BEGLIN_RESIDENT_WEIGHT_CPU_GPU_CPU_BYTES_MISMATCH\n");
            return 0;
        }
        auto p=std::make_shared<DevicePacked>(
            std::move(device_planes),std::move(device_scales),
            std::move(own),std::move(own_s));
        std::lock_guard<std::mutex> lock(store_mutex);
        if(sequence==0 || sequence>UINT64_C(9223372036854775807))
            return 0;
        uint64_t id=sequence++;
        registry.emplace(id,std::move(p));
        return id;
    }catch(const std::exception& e) {
        std::fprintf(stderr,"BEGLIN_MLA_RESEARCH_REGISTER_EXCEPTION:%s\n",e.what());
        return 0;
    }catch(...){return 0;}
}

// Compute with GPU-persistent ORIGINAL packed source weight arrays.
// Inputs: [16,128] qNoPE, [16,64] qRoPE, [T,512] latent,
//         [T,64] kPe, 1<=T<=64. Output: [16,128] F32.
int beglin_mla_qng64_resident_forward_v1(
    uint64_t id,const float* q_nope,const float* q_rope,
    const float* latents,const float* k_pe,int tokens,
    float score_scale,float* out_hv) {
    if(!id||!q_nope||!q_rope||!latents||!k_pe||!out_hv
       ||tokens<1||tokens>64||!std::isfinite(score_scale)
       ||score_scale<=0||score_scale>1)return 0;
    auto weight=resolve(id);
    if(!weight)return 0;
    if(!finite_arr(q_nope,(size_t)H*K)
       ||!finite_arr(q_rope,(size_t)H*R)
       ||!finite_arr(latents,(size_t)tokens*LAT)
       ||!finite_arr(k_pe,(size_t)tokens*R))return 0;
    try {
        mx::array host_qn((void*)q_nope,{H,K},mx::float32,noop_deleter);
        mx::array host_qr((void*)q_rope,{H,R},mx::float32,noop_deleter);
        mx::array host_lat((void*)latents,{tokens,LAT},mx::float32,noop_deleter);
        mx::array host_kr((void*)k_pe,{tokens,R},mx::float32,noop_deleter);
        mx::array host_score((void*)&score_scale,{1},mx::float32,noop_deleter);
        // The qNg64 weight inputs are GPU-owned, so place dynamic inputs
        // explicitly on the SAME GPU stream before fused custom dispatch.
        auto dev=mx::Device(mx::Device::gpu);
        auto qn=mx::copy(host_qn,dev);
        auto qr=mx::copy(host_qr,dev);
        auto lat=mx::copy(host_lat,dev);
        auto kr=mx::copy(host_kr,dev);
        auto score_param=mx::copy(host_score,dev);
        mx::eval(qn,qr,lat,kr,score_param);
        auto res=fused_mla_qng64_kernel()(
            {weight->planes,weight->scales,qn,qr,lat,kr,score_param},
            {{H,V}},{mx::float32},{64,1,H},{64,1,1},
            {{"cache_tokens",tokens}},std::nullopt,false,{})[0];
        mx::eval(res);
        auto result=res.data<float>();
        if(!finite_arr(result,(size_t)H*V)){
            std::fprintf(stderr,
              "BEGLIN_RESIDENT_SOURCE_NONFINITE first=%g second=%g third=%g\n",
              (double)result[0],(double)result[1],(double)result[2]);
            return 0;
        }
        std::memcpy(out_hv,result,(size_t)H*V*sizeof(float));
        return 1;
    }catch(const std::exception& e) {
        std::fprintf(stderr,"BEGLIN_MLA_RESEARCH_FORWARD_EXCEPTION:%s\n",e.what());
        return -2;
    }catch(...){return -2;}
}

// 1=existed + removed, 0=invalid. GPU memory can remain in MLX allocator
// CACHE after resident object destruction; active-vs-cached statistics differ.
int beglin_mla_qng64_resident_unregister_v1(uint64_t id) {
    if(!id)return 0;
    std::lock_guard<std::mutex> lock(store_mutex);
    return registry.erase(id)==1?1:0;
}
size_t beglin_mla_qng64_resident_payload_bytes_v1(uint64_t id) {
    auto ptr=resolve(id);
    return ptr?ptr->source_bytes:0;
}
size_t beglin_mla_qng64_resident_active_weights_v1(){
    std::lock_guard<std::mutex> lock(store_mutex);
    return registry.size();
}
size_t beglin_mla_qng64_resident_mlx_active_memory_v1(){
    return mx::get_active_memory();
}
size_t beglin_mla_qng64_resident_mlx_allocator_cache_v1(){
    return mx::get_cache_memory();
}
int beglin_mla_qng64_resident_gpu_available_v1(){
    return mx::metal::is_available()?1:0;
}

} // extern C
