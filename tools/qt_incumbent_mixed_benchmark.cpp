#include "mlx_moe.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
template<class T> static bool rd(const std::string&p,std::vector<T>&v){FILE*f=fopen(p.c_str(),"rb");if(!f)return false;fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);if(n<0||n%(long)sizeof(T)){fclose(f);return false;}v.resize(n/sizeof(T));bool ok=v.empty()||fread(v.data(),sizeof(T),v.size(),f)==v.size();fclose(f);return ok;}
int main(int ac,char**av){if(ac!=2)return 2;constexpr long O=896,I=896,NG=14;size_t C=O*NG;std::string r=av[1];std::vector<uint8_t>p,b;std::vector<uint32_t>o;std::vector<float>s,x,y(O);if(!rd(r+"/planes.bin",p)||!rd(r+"/offsets.bin",o)||!rd(r+"/bits.bin",b)||!rd(r+"/scales.bin",s)||!rd(r+"/input.bin",x))return 3;if(b.size()!=C||s.size()!=C||o.size()!=C+1)return 4;
auto bench=[&](int it){auto t=std::chrono::steady_clock::now();for(int k=0;k<it;k++)if(!mlx_gpu_qng64_mixed_dense_probe(p.data(),p.size(),o.data(),b.data(),s.data(),O,I,x.data(),1,y.data()))return -1.0;mlx_gpu_synchronize();return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/it;};
bench(10);std::vector<double> runs;for(int r=0;r<7;r++){double v=bench(200);if(v<0)return 5;runs.push_back(v);}std::sort(runs.begin(),runs.end());double sec=runs[runs.size()/2];size_t active=0,peak=0,cache=0;mlx_gpu_report_memory(&active,&peak,&cache);double avg=0;for(auto n:b)avg+=n;avg/=b.size();printf("{\\\"status\\\":\\\"PASS\\\",\\\"iterations\\\":1400,\\\"statistic\\\":\\\"median_of_7x200\\\",\\\"seconds_per_matvec\\\":%.9g,\\\"matvec_per_second\\\":%.9g,\\\"average_bits\\\":%.9g,\\\"planes_bytes\\\":%zu,\\\"mlx_active_bytes\\\":%zu,\\\"mlx_peak_bytes\\\":%zu,\\\"mlx_cache_bytes\\\":%zu}\\n",sec,1.0/sec,avg,p.size(),active,peak,cache);return 0;}
