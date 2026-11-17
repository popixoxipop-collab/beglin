// JE01 isolated 8-Expert mixed-bit GPU matvec proof, no pretrained model.
// Synthetic 64x64 three-role FFNs, real BEGLIN q4/qNg64/q8 quantizers,
// real Apple Metal mlx_gpu_bind_af and mlx_gpu_matvec_probe.
// Native Expert-ID-by-Expert-ID host dispatch is a proof only, NOT full
// routed-MoE MLX gather integration or model gold NLL.
#include "mlx_moe.h"
extern "C" {
#include "gguf_transcode.h"
}
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int E = 64, N = 64, GROUPS = N / 64, TOP = 6;
const std::array<const char *, 3> ROLES = {"gate", "up", "down"};
const std::array<int, TOP> IDS = {7, 21, 55, 3, 18, 44};
const std::array<double, TOP> ROUTE_WEIGHTS = {.27, .22, .19, .14, .10, .08};

struct Tensor {
    int bits=0, experts=0, out=0, in=0;
    size_t packed_bytes=0, scales_bytes=0, biases_bytes=0;
    std::string name;
    std::vector<uint8_t> blob;
};
std::map<std::string, Tensor> registry;

void need(bool condition, const char *msg) {
    if (!condition) throw std::runtime_error(msg);
}
void put_float(std::vector<uint8_t> &dst, size_t off, float v) {
    need(off+4 <= dst.size(), "JE01_WRITE_FLOAT_OVERFLOW");
    std::memcpy(dst.data()+off,&v,4);
}
float get_float(const std::vector<uint8_t> &src, size_t off) {
    need(off+4 <= src.size(), "JE01_READ_FLOAT_OVERFLOW");
    float value=0; std::memcpy(&value,src.data()+off,4); return value;
}
std::vector<float> dense_for_role(int role) {
    std::vector<float> w((size_t)E*N*N);
    for(int e=0;e<E;e++) for(int r=0;r<N;r++) for(int c=0;c<N;c++) {
        int code = ((e*11+r*5+c*7+role*17)%37)-18;
        // Contains every signed value and nonuniform group scales.
        w[((size_t)e*N+r)*N+c] =
            (float)(.032 * code + .006 * std::sin((e+1)*.13+r*.07+c*.11+role));
    }
    return w;
}

Tensor make_tensor(const std::string &name, const std::vector<float> &data,
                   int experts, int bits) {
    need(experts==1 || experts==E, "JE01_E_COUNT");
    need(data.size()==(size_t)experts*N*N, "JE01_INPUT_SHAPE");
    need(bits==2 || bits==4 || bits==6 || bits==8, "JE01_BITS_UNSUPPORTED");
    Tensor t; t.name=name; t.bits=bits; t.experts=experts; t.out=N; t.in=N;
    size_t count=(size_t)experts*N*N;
    t.packed_bytes = count*bits/8;
    t.scales_bytes = (size_t)experts*N*GROUPS*4;
    t.biases_bytes = bits==4 ? t.scales_bytes : 0;
    t.blob.assign(t.packed_bytes + t.scales_bytes + t.biases_bytes, 0);
    std::vector<float> scales((size_t)experts*N*GROUPS);
    if (bits==4) {
        gguf_quantize_q4g64_error_feedback(data.data(), experts*N, N,
                                          t.blob.data(), scales.data());
    } else if (bits==8) {
        gguf_quantize_q8g64(data.data(), experts*N, N,
                            (int8_t *)t.blob.data(), scales.data());
    } else {
        gguf_quantize_qNg64(data.data(), experts*N, N, bits,
                            t.blob.data(), scales.data());
    }
    for(size_t g=0;g<scales.size();g++){
        need(std::isfinite(scales[g]) && scales[g]>0, "JE01_NONFINITE_SCALE");
        put_float(t.blob,t.packed_bytes+g*4,scales[g]);
        if(bits==4) put_float(t.blob,t.packed_bytes+t.scales_bytes+g*4,
                               -8.0f*scales[g]);
    }
    return t;
}
void bind(const std::string &key, Tensor &&src) {
    need(!registry.count(key), "JE01_BIND_DUPLICATE");
    auto result=registry.emplace(key,std::move(src));
    Tensor &t=result.first->second;
    const long bias_off = t.biases_bytes
        ? (long)(t.packed_bytes+t.scales_bytes) : -1L;
    need(mlx_gpu_bind_af(t.blob.data(),(long)t.blob.size(),t.name.c_str(),
                         t.experts,t.out,t.in,GROUPS,0,
                         (long)t.packed_bytes,bias_off,t.bits)==1,
         "JE01_NATIVE_MLX_BIND_FAILED");
    int observed=0;
    need(mlx_gpu_binding_kind(t.name.c_str(),&observed)==1 &&
         observed==t.bits, "JE01_GPU_ACTIVE_BINDING_BITS_DRIFT");
}
float cpu_code(const Tensor &t,int e,int row,int col) {
    int idx=(e*N+row)*N+col;
    int group=(e*N+row)*GROUPS+col/64;
    float scale=get_float(t.blob,t.packed_bytes+(size_t)group*4);
    if(t.bits==4) {
        uint8_t raw=t.blob[(size_t)idx/2];
        int c=(raw >> ((idx%2)*4))&15;
        float bias=get_float(t.blob,t.packed_bytes+t.scales_bytes+(size_t)group*4);
        return c*scale+bias;
    }
    if(t.bits==8) return (float)(int8_t)t.blob[(size_t)idx]*scale;
    size_t off=(size_t)group*t.bits*8;
    int c=0;
    int p=col%64;
    for(int j=0;j<t.bits;j++)
        c |= ((t.blob[off+j*8+p/8] >> (p%8))&1)<<j;
    return (c-(1<<(t.bits-1)))*scale;
}
std::vector<float> cpu_matvec(const Tensor &t,int e,const std::vector<float> &x) {
    need(e>=0 && e<t.experts && x.size()==N, "JE01_REFERENCE_SHAPE");
    std::vector<float> y(N,0);
    for(int r=0;r<N;r++){
        double sum=0;
        for(int c=0;c<N;c++) sum+=(double)cpu_code(t,e,r,c)*x[c];
        y[r]=(float)sum;
    }
    return y;
}
std::vector<float> gpu_matvec(const Tensor &t,int e,const std::vector<float> &x) {
    need(e>=0 && e<t.experts, "JE01_GPU_EXPERT_OUT_OF_RANGE");
    std::vector<float> y(N,0);
    need(mlx_gpu_matvec_probe(t.name.c_str(),e,x.data(),y.data())==1,
         "JE01_GPU_MATVEC_FAILED");
    for(float v:y) need(std::isfinite(v),"JE01_NONFINITE_GPU");
    return y;
}
const Tensor &choose(int role,int expert,bool mixed) {
    const std::string role_name=ROLES[role];
    int bits=0;
    if(mixed && expert==7) bits=2;
    if(mixed && expert==21) bits=6;
    if(mixed && expert==55) bits=8;
    std::string key=bits
        ? std::string("je01.")+role_name+".expert"+std::to_string(expert)+".n"+std::to_string(bits)
        : std::string("je01.")+role_name+".all4";
    auto it=registry.find(key);
    need(it!=registry.end(),"JE01_SELECTED_EXPERT_BINDING_MISSING");
    return it->second;
}
std::vector<float> ffn(int expert,const std::vector<float> &x,bool mixed,bool gpu) {
    auto apply=[&](int role,const std::vector<float>&input){
        const Tensor &t=choose(role,expert,mixed);
        int idx=t.experts==1?0:expert;
        return gpu?gpu_matvec(t,idx,input):cpu_matvec(t,idx,input);
    };
    std::vector<float> gate=apply(0,x), up=apply(1,x);
    std::vector<float> swi(N);
    for(int i=0;i<N;i++) {
        float silu=gate[i] / (1.0f+std::exp(-gate[i]));
        swi[i]=silu*up[i];
    }
    return apply(2,swi);
}
std::vector<float> routed(const std::vector<float>&x,bool mixed,bool gpu) {
    std::vector<float> out(N,0);
    for(int k=0;k<TOP;k++) {
        std::vector<float> y=ffn(IDS[k],x,mixed,gpu);
        for(int j=0;j<N;j++) out[j]+=(float)(ROUTE_WEIGHTS[k]*y[j]);
    }
    return out;
}
double max_error(const std::vector<float>&a,const std::vector<float>&b) {
    need(a.size()==b.size(),"JE01_OUTPUT_SIZE");
    double mx=0;
    for(size_t i=0;i<a.size();i++) mx=std::max(mx,std::abs((double)a[i]-b[i]));
    return mx;
}
} // namespace

int main() {
    try {
        need(mlx_gpu_available()==1,"JE01_APPLE_METAL_UNAVAILABLE");
        need(mlx_gpu_layer_config(N,N,N,E,0,TOP,64,0)==1,
             "JE01_NATIVE_MLX_GROUP_CONFIG_FAILED");
        // Immutable baseline q4 E=64 per FFN role; overlays are E=1.
        for(int role=0;role<3;role++) {
            auto dense=dense_for_role(role);
            std::string base=std::string("je01.")+ROLES[role]+".all4";
            bind(base,make_tensor(base,dense,E,4));
            for(const auto &[expert,bits] : std::array<std::pair<int,int>,3>{
                    {{7,2},{21,6},{55,8}}}) {
                const size_t a=(size_t)expert*N*N;
                std::vector<float> one(dense.begin()+a,dense.begin()+a+N*N);
                std::string key=std::string("je01.")+ROLES[role]+".expert"
                    +std::to_string(expert)+".n"+std::to_string(bits);
                bind(key,make_tensor(key,one,1,bits));
            }
        }
        if (registry.size()!=12) {
            std::fprintf(stderr, "JE01_ACTUAL_GPU_PROBE_BINDING_COUNT=%zu\\n", registry.size());
            for(const auto &item: registry)
                std::fprintf(stderr, "JE01_GPU_PROBE_BOUND_KEY=%s\\n", item.first.c_str());
        }
        need(registry.size()==12,"JE01_MIXED_BINDING_COUNT");
        std::vector<float> x(N);
        for(int i=0;i<N;i++) x[i]=(float)(std::sin((i+1)*.29)+.17*std::cos((i+1)*.13));
        const auto original_gpu = routed(x,false,true);
        const auto original_cpu = routed(x,false,false);
        const auto mixed_gpu = routed(x,true,true);
        const auto mixed_cpu = routed(x,true,false);
        const double baseline_cpu_gpu_max=max_error(original_gpu,original_cpu);
        const double mixed_cpu_gpu_max=max_error(mixed_gpu,mixed_cpu);
        const double actual_change=max_error(mixed_gpu,original_gpu);
        double cpu_scale=0.0;
        for(float z:mixed_cpu)cpu_scale=std::max(cpu_scale,std::abs((double)z));
        const double tolerance=0.02 + .02*cpu_scale;
        need(baseline_cpu_gpu_max<tolerance,"JE01_Q4_REFERENCE_PARITY_FAIL");
        need(mixed_cpu_gpu_max<tolerance,"JE01_MIXED_QUANT_REFERENCE_PARITY_FAIL");
        need(actual_change>1e-6,"JE01_MIXED_GPU_OUTPUT_IDENTICAL_TO_BASELINE");
        // Selected top6 contains 3 untouched Expert IDs. Their outputs must
        // retain bit-exact all4 fallback, not accidentally take an overlay.
        double fallback_max=0;
        for(int e : {3,18,44}) {
            auto orig=ffn(e,x,false,true), mix=ffn(e,x,true,true);
            fallback_max=std::max(fallback_max,max_error(orig,mix));
        }
        need(fallback_max==0.0,"JE01_UNTOUCHED_EXPERT_GPU_PARITY_FAIL");
        size_t active=0,peak=0,cache=0;
        mlx_gpu_report_memory(&active,&peak,&cache);
        std::printf(
            "{\"schema\":\"beglin-je01-synthetic-native-mixed-expert-metal-proof/1\","
            "\"status\":\"PASS_REAL_METAL_SYNTHETIC_WEIGHTS\","
            "\"original_E64_q4_bindings\":3,"
            "\"per_expert_E1_n2_n6_n8_bindings\":9,"
            "\"selected_route_expert_count\":6,"
            "\"unchanged_q4_fallback_expert_count\":3,"
            "\"fallback_max_abs_diff\":%.9g,"
            "\"baseline_vs_cpu_max_abs_diff\":%.9g,"
            "\"mixed_vs_cpu_max_abs_diff\":%.9g,"
            "\"native_mixed_vs_original_max_abs_diff\":%.9g,"
            "\"GPU_active_bytes\":%zu,"
            "\"GPU_peak_bytes\":%zu,"
            "\"actual_pretrained_BF16_or_AF_weights_loaded\":false,"
            "\"full_routed_model_forward\":\"NOT_RUN\","
            "\"gold_token_NLL\":\"NOT_MEASURED\","
            "\"production_modified\":false}\n",
            fallback_max,baseline_cpu_gpu_max,mixed_cpu_gpu_max,actual_change,active,peak);
        return 0;
    } catch(const std::exception &e) {
        std::fprintf(stderr,"JE01_MIXED_GPU_PROBE_BLOCKED: %s\n",e.what());
        return 2;
    }
}
