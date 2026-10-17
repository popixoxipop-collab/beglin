// Actual BEGLIN mlx_moe.cpp real Metal gather kernel, independent CPU oracle.
// Research-only synthetic tensors, with malformed ID and length-shape guards.
// This must be run on a physical Apple Metal host (hosted CI only compiles).
#include "mlx_moe.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr int E = 4, OUT = 8, IN = 128, NG = 2, COUNT = 9;

struct Result {
  double rel_l2=0.0;
  double max_abs=0.0;
  bool guards=false;
};

bool probe(int bits, const std::vector<uint8_t>& planes,
           const std::vector<float>& scales,
           const std::vector<float>& x,
           const std::vector<int32_t>& ids,
           std::vector<float>& y,
           int E_=E, int OUT_=OUT, int IN_=IN, int NG_=NG,
           int COUNT_=COUNT) {
  return mlx_gpu_qng64_gather_probe(planes.data(),E_,OUT_,IN_,NG_,bits,
              scales.data(),x.data(),COUNT_,ids.data(),y.data()) == 1;
}

bool run_one(int n,Result& r) {
  const size_t cells=(size_t)E * OUT * NG;
  std::vector<uint8_t> planes(cells * (size_t)n * 8u,0);
  std::vector<float> scales(cells);
  std::vector<float> x(COUNT * IN);
  const std::vector<int32_t> ids={0,3,1,2,0,3,2,1,3};
  std::vector<float> y(COUNT * OUT,-999.0f);
  std::vector<double> cpu(COUNT * OUT,0.0);
  for(int z=0;z<COUNT;z++)for(int j=0;j<IN;j++)
    x[(size_t)z * IN + j] = std::sin((j+1)*0.0283f*(z+1)) +
                           std::cos((j+3)*0.016f);
  for(int e=0;e<E;e++)for(int row=0;row<OUT;row++)
    for(int g=0;g<NG;g++) {
      const size_t c=((size_t)e * OUT + row) * NG + g;
      const float scale=(float)(1+e+row+g)*0.00041f;
      scales[c]=scale;
      for(int p=0;p<64;p++) {
        const int u=(13*p + 7*row + 23*e + 17*g + 5) % (1<<n);
        for(int j=0;j<n;j++)
          if((u>>j)&1)
            planes[c*n*8u + (size_t)j*8u+(size_t)(p>>3)] |=
              uint8_t(1u<<(p&7));
        const float weight=float(u-(1<<(n-1)))*scale;
        for(int z=0;z<COUNT;z++)
          if(ids[z]==e)
            cpu[(size_t)z*OUT+row]+=
              double(weight) * x[(size_t)z*IN + g*64+p];
      }
    }
  if(!probe(n,planes,scales,x,ids,y)) return false;
  double e2=0.0,ref2=0.0;
  for(size_t k=0;k<y.size();k++){
    if(!std::isfinite(y[k])) return false;
    const double d=double(y[k])-cpu[k];
    e2+=d*d;
    ref2+=cpu[k]*cpu[k];
    r.max_abs=std::max(r.max_abs,std::fabs(d));
  }
  r.rel_l2=std::sqrt(e2/std::max(ref2,1e-30));
  if(r.rel_l2>5e-5 || r.max_abs>0.0001) return false;

  // Reject malformed caller-controlled shapes/IDs before touching Metal.
  int blocked=0,tests=0;
  auto check=[&](bool pass){tests++;if(pass)blocked++;};
  std::vector<int32_t> wrong=ids;
  wrong[0]=-1;
  check(!probe(n,planes,scales,x,wrong,y));
  wrong[0]=E;
  check(!probe(n,planes,scales,x,wrong,y));
  check(!probe(n,planes,scales,x,ids,y,E,OUT,IN,NG+1));
  check(!probe(n,planes,scales,x,ids,y,E,OUT,IN-1,NG));
  check(!probe(n,planes,scales,x,ids,y,E,OUT,IN,NG,0));
  check(!probe(6,planes,scales,x,ids,y));
  check(!probe(16,planes,scales,x,ids,y));
  check(!probe(n,planes,scales,x,ids,y,2147483647,2147483647,IN,NG));
  check(mlx_gpu_qng64_gather_probe(nullptr,E,OUT,IN,NG,n,scales.data(),
             x.data(),COUNT,ids.data(),y.data())==0);
  check(mlx_gpu_qng64_gather_probe(planes.data(),E,OUT,IN,NG,n,nullptr,
             x.data(),COUNT,ids.data(),y.data())==0);
  check(mlx_gpu_qng64_gather_probe(planes.data(),E,OUT,IN,NG,n,scales.data(),
             nullptr,COUNT,ids.data(),y.data())==0);
  check(mlx_gpu_qng64_gather_probe(planes.data(),E,OUT,IN,NG,n,scales.data(),
             x.data(),COUNT,nullptr,y.data())==0);
  check(mlx_gpu_qng64_gather_probe(planes.data(),E,OUT,IN,NG,n,scales.data(),
             x.data(),COUNT,ids.data(),nullptr)==0);
  r.guards=blocked==tests && tests==13;
  return r.guards;
}
} // namespace

int main(){
  if(!mlx_gpu_available()) {
    std::fprintf(stderr,"NO_METAL_GPU_CANNOT_CERTIFY\n");
    return 2;
  }
  Result a,b;
  if(!run_one(7,a)){std::fprintf(stderr,"FAILED_NATIVE_GATHER_N7_OR_GUARDS\n");return 3;}
  if(!run_one(9,b)){std::fprintf(stderr,"FAILED_NATIVE_GATHER_N9_OR_GUARDS\n");return 4;}
  std::printf(
    "{\"status\":\"PASS_CPP_REAL_METAL_GATHER_SYNTHETIC_AND_NEGATIVE\","
    "\"n7_relative_L2\":%.10g,\"n9_relative_L2\":%.10g,"
    "\"n7_max_abs\":%.10g,\"n9_max_abs\":%.10g,"
    "\"expert_count\":%d,\"routed_pairs\":%d,\"invalid_cases_per_width\":13,"
    "\"source_model_used\":false,\"automatic_live_promotion\":false}\n",
    a.rel_l2,b.rel_l2,a.max_abs,b.max_abs,E,COUNT);
  return 0;
}
