#include "mlx_moe.h"
#include "qng64_dense.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void compact_to_bitplanes(const uint8_t *compact, uint8_t *planes, long out, long in, int n) {
    const long ng=in/64; const size_t cgb=(size_t)(64*n+7)/8; const size_t pgb=(size_t)n*8;
    std::memset(planes,0,(size_t)out*ng*pgb);
    for(long r=0;r<out;r++) for(long g=0;g<ng;g++) {
        const uint8_t *src=compact+((size_t)r*ng+g)*cgb;
        uint8_t *dst=planes+((size_t)r*ng+g)*pgb;
        for(int p=0;p<64;p++) {
            uint32_t raw=0;
            for(int b=0;b<n;b++) {
                size_t bit=(size_t)p*n+b;
                raw|=(uint32_t)((src[bit>>3]>>(bit&7))&1u)<<b;
            }
            for(int b=0;b<n;b++) if((raw>>b)&1u) dst[(size_t)b*8+(p>>3)]|=(uint8_t)(1u<<(p&7));
        }
    }
}

int main(int argc,char **argv) {
    if(argc!=2){std::fprintf(stderr,"usage: %s raw_f32\n",argv[0]);return 2;}
    const long out=896,in=896,ng=in/64; const int n=5;
    FILE *fp=std::fopen(argv[1],"rb"); if(!fp)return 3;
    std::vector<float>w((size_t)out*in); if(std::fread(w.data(),sizeof(float),w.size(),fp)!=w.size())return 4; std::fclose(fp);
    std::vector<uint8_t> compact(qng64_packed_bytes(out,in,n)); std::vector<float> scales((size_t)out*ng);
    if(qng64_quantize_f32(w.data(),out,in,n,compact.data(),scales.data())!=0)return 5;
    std::vector<uint8_t> planes((size_t)out*ng*n*8); compact_to_bitplanes(compact.data(),planes.data(),out,in,n);
    std::vector<float>x(in),cpu(out),gpu(out),restored(out);
    for(long i=0;i<in;i++)x[i]=((i*17)%101-50)/53.0f;
    qng64_matvec_f32(compact.data(),scales.data(),n,x.data(),nullptr,cpu.data(),out,in);
    const char *name="qwen2.5.L0.q_proj.real";
    if(!mlx_gpu_available())return 6;
    if(!mlx_gpu_bind_qng64_dense_probe(planes.data(),scales.data(),name,out,in,n))return 7;
    if(!mlx_gpu_matvec_probe(name,0,x.data(),gpu.data()))return 8;
    double max_abs=0,max_rel=0;
    for(long i=0;i<out;i++){double d=std::fabs((double)gpu[i]-cpu[i]);max_abs=std::max(max_abs,d);max_rel=std::max(max_rel,d/(std::fabs((double)cpu[i])+1e-6));}
    uint64_t sid=0;if(!mlx_gpu_snapshot_binding(name,&sid)||!sid)return 9;
    if(!mlx_gpu_restore_binding_snapshot(sid))return 10;
    if(!mlx_gpu_matvec_probe(name,0,x.data(),restored.data()))return 11;
    double restore=0;for(long i=0;i<out;i++)restore=std::max(restore,std::fabs((double)restored[i]-gpu[i]));
    mlx_gpu_drop_binding_snapshot(sid);
    std::printf("{\"status\":\"%s\",\"n\":5,\"out\":896,\"in\":896,\"max_abs_error\":%.9g,\"max_rel_error\":%.9g,\"restore_max_abs_diff\":%.9g}\n",(max_abs<5e-4&&restore==0.0)?"PASS":"FAIL",max_abs,max_rel,restore);
    return (max_abs<5e-4&&restore==0.0)?0:12;
}
