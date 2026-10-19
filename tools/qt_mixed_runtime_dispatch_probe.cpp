#include "mlx_moe.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
static void enc(std::vector<uint8_t>&p,size_t b,int n,int seed){int bias=1<<(n-1);for(int i=0;i<64;i++){int u=((i*7+seed)% (1<<n));for(int j=0;j<n;j++)if((u>>j)&1)p[b+(size_t)j*8+(i>>3)]|=(uint8_t)(1u<<(i&7));}}
int main(){constexpr long OUT=8,IN=128,NG=2;size_t cells=OUT*NG;std::vector<uint8_t> bits(cells);std::vector<uint32_t> off(cells+1);std::vector<float> sc(cells);std::vector<uint8_t> planes;std::vector<float>x(IN),y(OUT);
for(long i=0;i<IN;i++)x[i]=std::sin((float)(i+1)*.07f);
for(size_t c=0;c<cells;c++){int n=4+(int)(c%3);bits[c]=(uint8_t)n;sc[c]=.001f*(1+(c%5));off[c]=(uint32_t)planes.size();size_t b=planes.size();planes.resize(b+n*8);enc(planes,b,n,(int)c+3);off[c+1]=(uint32_t)planes.size();}
const char*name="model.layers.0.self_attn.q_proj.weight";if(!mlx_gpu_bind_qng64_mixed_dense_probe(planes.data(),planes.size(),off.data(),bits.data(),sc.data(),name,OUT,IN))return 2;int k=-1;if(mlx_gpu_binding_kind(name,&k)!=4)return 3;if(!mlx_gpu_matvec_probe(name,0,x.data(),y.data()))return 4;for(float v:y)if(!std::isfinite(v))return 5;std::puts("QT_MIXED_RUNTIME_DISPATCH_PASS");return 0;}
