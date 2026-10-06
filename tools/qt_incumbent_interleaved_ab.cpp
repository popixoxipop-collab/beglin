#include "mlx_moe.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
struct F{std::vector<uint8_t>p,b;std::vector<uint32_t>o;std::vector<float>s,x,y;};
template<class T>static bool rd(const std::string&p,std::vector<T>&v){FILE*f=fopen(p.c_str(),"rb");if(!f)return false;fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);if(n<0||n%(long)sizeof(T)){fclose(f);return false;}v.resize(n/sizeof(T));bool q=v.empty()||fread(v.data(),sizeof(T),v.size(),f)==v.size();fclose(f);return q;}
static bool load(const std::string&r,F&f){f.y.resize(896);return rd(r+"/planes.bin",f.p)&&rd(r+"/offsets.bin",f.o)&&rd(r+"/bits.bin",f.b)&&rd(r+"/scales.bin",f.s)&&rd(r+"/input.bin",f.x);}
static double bench(F&f,int it){auto t=std::chrono::steady_clock::now();for(int k=0;k<it;k++)if(!mlx_gpu_qng64_mixed_dense_probe(f.p.data(),f.p.size(),f.o.data(),f.b.data(),f.s.data(),896,896,f.x.data(),1,f.y.data()))return -1;mlx_gpu_synchronize();return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/it;}
int main(int ac,char**av){if(ac!=3)return 2;F a,b;if(!load(av[1],a)||!load(av[2],b))return 3;bench(a,20);bench(b,20);std::vector<double>rat;double sa=0,sb=0;for(int r=0;r<7;r++){double x,y;if((r&1)==0){x=bench(a,200);y=bench(b,200);}else{y=bench(b,200);x=bench(a,200);}if(x<0||y<0)return 4;sa+=x;sb+=y;rat.push_back(x/y);}std::sort(rat.begin(),rat.end());size_t active=0,peak=0,cache=0;mlx_gpu_report_memory(&active,&peak,&cache);printf("{\"status\":\"PASS\",\"pairs\":7,\"iterations_each\":1400,\"ratio_mixed_over_uniform_speed_median\":%.9g,\"uniform_mean_seconds\":%.9g,\"mixed_mean_seconds\":%.9g,\"mlx_active_bytes\":%zu,\"mlx_peak_bytes\":%zu,\"mlx_cache_bytes\":%zu}\n",rat[3],sa/7,sb/7,active,peak,cache);return 0;}
