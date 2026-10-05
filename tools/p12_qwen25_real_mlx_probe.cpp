#include "mlx_moe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

template <typename T>
static bool read_vec(const std::string &path, std::vector<T> &v) {
    FILE *fp=std::fopen(path.c_str(),"rb"); if(!fp) return false;
    bool ok=std::fread(v.data(),sizeof(T),v.size(),fp)==v.size();
    std::fclose(fp); return ok;
}

int main(int argc,char **argv) {
    if(argc!=2 && argc!=3){std::fprintf(stderr,"usage: %s fixture_dir [n]\n",argv[0]);return 2;}
    const long out=896,in=896,ng=in/64;
    const int n=(argc==3)?std::atoi(argv[2]):5;
    if(n<3 || n>8){std::fprintf(stderr,"unsupported n=%d\n",n);return 14;}
    const std::string root=argv[1];
    std::vector<uint8_t> planes((size_t)out*ng*n*8);
    std::vector<float> scales((size_t)out*ng),x(in),cpu(out),gpu(out),restored(out);
    if(!read_vec(root+"/planes.bin",planes)) return 3;
    if(!read_vec(root+"/scales.bin",scales)) return 4;
    if(!read_vec(root+"/input.bin",x)) return 5;
    if(!read_vec(root+"/cpu_expected.bin",cpu)) return 6;
    const std::string name="qwen2.5.L0.q_proj.real.n"+std::to_string(n);
    if(!mlx_gpu_available())return 7;
    if(!mlx_gpu_bind_qng64_dense_probe(planes.data(),scales.data(),name.c_str(),out,in,n))return 8;
    if(!mlx_gpu_matvec_probe(name.c_str(),0,x.data(),gpu.data()))return 9;
    double max_abs=0,max_rel=0;
    for(long i=0;i<out;i++){double d=std::fabs((double)gpu[i]-cpu[i]);max_abs=std::max(max_abs,d);max_rel=std::max(max_rel,d/(std::fabs((double)cpu[i])+1e-6));}
    uint64_t sid=0;if(!mlx_gpu_snapshot_binding(name.c_str(),&sid)||!sid)return 10;
    if(!mlx_gpu_restore_binding_snapshot(sid))return 11;
    if(!mlx_gpu_matvec_probe(name.c_str(),0,x.data(),restored.data()))return 12;
    double restore=0;for(long i=0;i<out;i++)restore=std::max(restore,std::fabs((double)restored[i]-gpu[i]));
    mlx_gpu_drop_binding_snapshot(sid);
    const bool pass=max_abs<5e-4&&restore==0.0;
    std::printf("{\"status\":\"%s\",\"n\":%d,\"out\":896,\"in\":896,\"max_abs_error\":%.9g,\"max_rel_error\":%.9g,\"restore_max_abs_diff\":%.9g}\n",pass?"PASS":"FAIL",n,max_abs,max_rel,restore);
    return pass?0:13;
}
