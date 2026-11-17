// JE01: real BF16-origin 8-Expert n=2/6 native Metal FFN role matvec proof.
// Does not load original 27-layer AF model, run route logits or score NLL.
#include "mlx_moe.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Region {
    int layer, expert, bits, out, in;
    std::string role,name;
    size_t packed_begin,packed_end,scale_begin,scale_end;
};
void require(bool v,const std::string &s){if(!v)throw std::runtime_error(s);}
std::vector<uint8_t> file_bytes(const char *path) {
    std::ifstream f(path,std::ios::binary);
    require(bool(f),"JE01_REAL_BUNDLE_FILE_MISSING");
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>());
}
std::vector<Region> regions(const char *path) {
    std::ifstream f(path);
    require(bool(f),"JE01_REAL_TSV_MISSING");
    std::vector<Region> entries;
    std::string line;
    while(std::getline(f,line)){
        if(line.empty())continue;
        Region v{};
        std::istringstream iss(line);
        require(bool(iss>>v.layer>>v.expert>>v.role>>v.bits>>v.out>>v.in
                      >>v.packed_begin>>v.packed_end>>v.scale_begin>>v.scale_end),
                "JE01_REAL_TSV_PARSE_FAILURE");
        require(v.layer>=1 && v.layer<=26 && v.expert>=0 && v.expert<64 &&
                (v.bits==2 || v.bits==6) && v.in%64==0 &&
                ((v.out==1408 && v.in==2048) || (v.out==2048 && v.in==1408)) &&
                (v.role=="gate_proj" || v.role=="up_proj" || v.role=="down_proj"),
                "JE01_REAL_TSV_VALUES_WRONG");
        v.name="je01_real_L"+std::to_string(v.layer)+"_E"+std::to_string(v.expert)+"_"+v.role;
        entries.push_back(v);
    }
    require(entries.size()==24,"JE01_REAL_24_BINDINGS_REQUIRED");
    return entries;
}
float read_f32(const std::vector<uint8_t>&v,size_t pos) {
    require(pos+4<=v.size(),"JE01_REAL_SCALE_OVERFLOW");
    float a;std::memcpy(&a,v.data()+pos,4);return a;
}
float qng64_decode_real(const std::vector<uint8_t>&v,const Region&r,int row,int c){
    int group=row*(r.in/64)+c/64;
    size_t off=r.packed_begin+(size_t)group*r.bits*8;
    const int index=c%64,byte=index/8,bit=index%8;
    int code=0;
    for(int b=0;b<r.bits;b++) code|=((v[off+b*8+byte]>>bit)&1)<<b;
    float scale=read_f32(v,r.scale_begin+(size_t)group*4);
    return (float)(code-(1<<(r.bits-1)))*scale;
}
std::vector<float> reference(const std::vector<uint8_t>&v,const Region&r,
                             const std::vector<float>&x){
    std::vector<float> y(r.out);
    for(int i=0;i<r.out;i++){
        double sum=0;
        for(int j=0;j<r.in;j++)sum+=(double)qng64_decode_real(v,r,i,j)*x[j];
        y[i]=(float)sum;
    }
    return y;
}
std::vector<float> input(int n,int layer,int expert,const std::string&role){
    std::vector<float> x(n);
    for(int j=0;j<n;j++)x[j]=(float)(std::sin((j+1)*.021+(layer+expert)*.13)+
                                    .21*std::cos((j+3)*.09+role.size()));
    return x;
}
int main(int argc,char**argv){
    try{
        require(argc==3,"JE01_REAL_GPU_EXPECT_BUNDLE_AND_TSV");
        auto blob=file_bytes(argv[1]);
        require(blob.size()==43253760,"JE01_REAL_BUNDLE_NOT_EXACT_BYTE");
        auto entries=regions(argv[2]);
        require(mlx_gpu_available()==1,"JE01_REAL_METAL_MISSING");
        double worst=0,worst_relative_rms=0;
        int success=0,bits2=0,bits6=0;
        for(const auto &r:entries){
            require(r.packed_begin<r.packed_end && r.packed_end<=blob.size() &&
                    r.scale_begin<r.scale_end && r.scale_end<=blob.size(),"JE01_REAL_RANGE_INVALID");
            require(r.packed_end-r.packed_begin==(size_t)r.out*r.in*r.bits/8 &&
                    r.scale_end-r.scale_begin==(size_t)r.out*(r.in/64)*4,
                    "JE01_REAL_PROJECTION_SHAPE_BYTE_CONTRACT");
            require(mlx_gpu_bind_af(blob.data(),(long)blob.size(),r.name.c_str(),
                        1,r.out,r.in,r.in/64,
                        (long)r.packed_begin,(long)r.scale_begin,-1,r.bits)==1,
                    "JE01_REAL_MLX_TIER_BIND_FAILED:"+r.name);
            int active=0;
            require(mlx_gpu_binding_kind(r.name.c_str(),&active)==1 && active==r.bits,
                    "JE01_REAL_GPU_REGISTERED_BITS_NOT_REQUESTED");
            auto x=input(r.in,r.layer,r.expert,r.role);
            auto ref=reference(blob,r,x);
            std::vector<float> gpu(r.out);
            require(mlx_gpu_matvec_probe(r.name.c_str(),0,x.data(),gpu.data())==1,
                    "JE01_REAL_GPU_MATVEC_FAILED:"+r.name);
            double err2=0,base2=0;
            for(int k=0;k<r.out;k++){
                require(std::isfinite(gpu[k]),"JE01_REAL_GPU_NONFINITE");
                double diff=(double)gpu[k]-ref[k];
                worst=std::max(worst,std::abs(diff));
                err2+=diff*diff;
                base2+=(double)ref[k]*ref[k];
            }
            double rel=std::sqrt(err2/std::max(base2,1e-25));
            worst_relative_rms=std::max(worst_relative_rms,rel);
            require(rel<0.05,"JE01_REAL_GPU_VS_CPU_REL_RMSE_TOO_HIGH:"+r.name);
            if(r.bits==2)bits2++;else bits6++;
            success++;
            std::printf("JE01_REAL_WEIGHT_ROLE_PASS L%d E%d %s n%d rel_rmse=%.9g\n",
                        r.layer,r.expert,r.role.c_str(),r.bits,rel);
        }
        require(success==24 && bits2==9 && bits6==15,"JE01_REAL_ROLE_OR_BIT_COUNTS_DRIFT");
        size_t active=0,peak=0,cache=0;
        mlx_gpu_report_memory(&active,&peak,&cache);
        std::printf("{\"schema\":\"beglin-je01-real-bf16-eight-expert-native-metal-ffn-matvec/1\","
            "\"status\":\"PASS_ACTUAL_PRETRAINED_BF16_DERIVED_N2_N6_METAL\","
            "\"real_pretrained_expert_roles_gpu\":%d,"
            "\"real_n2_roles\":%d,\"real_n6_roles\":%d,"
            "\"physical_original_AF_budget_bytes\":%zu,"
            "\"worst_gpu_cpu_absolute_error\":%.10g,"
            "\"worst_gpu_cpu_relative_rmse\":%.10g,"
            "\"gpu_active_bytes\":%zu,\"gpu_peak_bytes\":%zu,"
            "\"actual_full_27layer_model_forward\":\"NOT_RUN\","
            "\"full_gold_token_NLL\":\"NOT_MEASURED\","
            "\"production_changed\":false}\n",
            success,bits2,bits6,blob.size(),worst,worst_relative_rms,active,peak);
        return 0;
    } catch(const std::exception&e){
        std::fprintf(stderr,"JE01_REAL_WEIGHT_METAL_FAIL: %s\n",e.what());
        return 2;
    }
}
