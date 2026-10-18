// BEGLIN qNg64 n7 compressed-query absorption — independent CPU synthetic oracle.
//
// This is a future-MLA fused-attention building block, NOT the model's C++
// Metal kernel and NOT an actual DeepSeek quality/speed measurement.
// K_NoPE key projection Wk[h]: [128, 512], packed n7 group64 bitplanes;
// compute absorbed_query[h,512] = Wk[h]^T * q_nope[h,128] with **no
// permanent dense dequantized matrix**. Values use the same native compressed
// qng64_matvec_f32 on latent probability-aggregated [512] vectors.
// Compare all results against an independently expanded *test-only* dense
// dequantized oracle for synthetic tensors and multiple sequence lengths.

#include "qng64_dense.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
constexpr int H=16, D=512, K=128, V=128, S=256, OUT=H*S, N=7, NG=D/64;
constexpr size_t GROUPBYTES=56;
void require(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
bool qng64_query_absorb_packed(const uint8_t *packed,const float *scales,
                               int num_heads,int head,int bits,int in,
                               const float *q_nope,float *latent_q) {
    if(!packed||!scales||!q_nope||!latent_q||bits!=7||num_heads!=16||
       head<0||head>=num_heads||in!=512||in%64||NG!=8)return false;
    std::fill(latent_q,latent_q+D,0.0f);
    std::array<double,D> acc{};
    for(int r=0;r<K;r++){
        size_t row=(size_t)head*S+r;
        double q=(double)q_nope[r];
        for(int g=0;g<NG;g++){
            const uint8_t* group=packed+(row*NG+(size_t)g)*GROUPBYTES;
            float scale=scales[row*NG+g];
            if(!std::isfinite(scale)||scale<=0)return false;
            for(int p=0;p<64;p++){
                const float code=qng64_decode_code(group,bits,p);
                acc[(size_t)g*64+p]+=q*(double)(code*scale);
            }
        }
    }
    for(int d=0;d<D;d++)latent_q[d]=(float)acc[d];
    return true;
}
void calculate_reference(const std::vector<float>& dense,
                         int head,const std::vector<float>& qnope,
                         std::array<float,D>& result) {
    std::array<double,D> acc{};
    for(int r=0;r<K;r++){
        const double q=qnope[(size_t)head*K+r];
        for(int d=0;d<D;d++){
            acc[d]+=q*(double)dense[((size_t)head*S+r)*D+d];
        }
    }
    for(int d=0;d<D;d++)result[d]=(float)acc[d];
}
std::vector<float> softmax(const std::vector<double>& score) {
    const double maxv=*std::max_element(score.begin(),score.end());
    std::vector<float> out(score.size());
    double denom=0;
    for(size_t t=0;t<score.size();t++){
        out[t]=(float)std::exp(score[t]-maxv);
        denom+=out[t];
    }
    require(denom>0 && std::isfinite(denom),"softmax norm invalid");
    for(float& x:out)x=(float)(x/denom);
    return out;
}
struct Metrics {double relL2=0,maxAbs=0;};
Metrics compare(const std::vector<float>& a,const std::vector<float>& b){
    require(a.size()==b.size()&&!a.empty(),"compare sizes");
    double d=0,den=0,maxv=0;
    for(size_t i=0;i<a.size();i++){
        require(std::isfinite(a[i])&&std::isfinite(b[i]),"numerical finite");
        double dif=(double)a[i]-b[i];d+=dif*dif;den+=(double)b[i]*b[i];
        maxv=std::max(maxv,std::abs(dif));
    }
    return {std::sqrt(d/std::max(den,1e-30)),maxv};
}
}
int main(){
  try{
    auto begin=std::chrono::steady_clock::now();
    std::vector<float> w((size_t)OUT*D);
    for(size_t i=0;i<w.size();i++)w[i]=(float)(0.02*std::sin((double)(i+1)*0.00973));
    auto bytes=qng64_packed_bytes(OUT,D,N);
    require(bytes==(size_t)OUT*NG*GROUPBYTES,"actual n7 bytes");
    std::vector<uint8_t> packed(bytes);
    std::vector<float> scales((size_t)OUT*NG);
    require(qng64_quantize_f32(w.data(),OUT,D,N,packed.data(),scales.data())==0,
            "actual native C packed source n7 quantization");
    std::vector<float> dense(w.size());
    // Dequantized matrix is a *test-only independent reference*, never
    // consumed by qng64_query_absorb_packed nor the fused value algorithm.
    for(int r=0;r<OUT;r++)for(int d=0;d<D;d++){
        int g=d/64,p=d%64;
        const uint8_t *ptr=packed.data()+((size_t)r*NG+g)*GROUPBYTES;
        dense[(size_t)r*D+d]=qng64_decode_code(ptr,N,p)*scales[(size_t)r*NG+g];
    }
    std::vector<float> qnope((size_t)H*K),qrope((size_t)H*64);
    for(size_t i=0;i<qnope.size();i++)qnope[i]=(float)(0.4*std::sin(0.043*(i+1)));
    for(size_t i=0;i<qrope.size();i++)qrope[i]=(float)(0.25*std::cos(0.087*(i+1)));
    std::vector<float> qabs((size_t)H*D),qabs_ref((size_t)H*D);
    for(int h=0;h<H;h++){
        require(qng64_query_absorb_packed(packed.data(),scales.data(),H,h,N,D,
                qnope.data()+(size_t)h*K,qabs.data()+(size_t)h*D),
                "actual compressed transposed query");
        std::array<float,D> rr{};
        calculate_reference(dense,h,qnope,rr);
        std::copy(rr.begin(),rr.end(),qabs_ref.begin()+(size_t)h*D);
    }
    auto qgap=compare(qabs,qabs_ref);
    require(qgap.relL2<1e-7,"compressed transpose source oracle parity");
    std::array<int,5> lengths={1,5,7,16,64};
    std::vector<float> outcomes, references;
    double max_softmax_delta=0;
    double max_output_relative=0;
    for(int T:lengths){
        std::vector<float> lat((size_t)T*D),kpe((size_t)T*64);
        for(size_t i=0;i<lat.size();i++)lat[i]=(float)(0.25*std::sin(0.0034*(i+17)));
        for(size_t i=0;i<kpe.size();i++)kpe[i]=(float)(0.3*std::cos(0.017*(i+6)));
        for(int h=0;h<H;h++){
            std::vector<double> score(T),score_ref(T);
            for(int t=0;t<T;t++){
                double x1=0,x2=0;
                for(int d=0;d<D;d++)x1+=(double)qabs[(size_t)h*D+d]*lat[(size_t)t*D+d];
                for(int r=0;r<K;r++){
                    double kvalue=0;
                    for(int d=0;d<D;d++)
                        kvalue+=(double)dense[((size_t)h*S+r)*D+d]*lat[(size_t)t*D+d];
                    x2+=(double)qnope[(size_t)h*K+r]*kvalue;
                }
                for(int r=0;r<64;r++){
                    double rope=(double)qrope[(size_t)h*64+r]*kpe[(size_t)t*64+r];
                    x1+=rope;x2+=rope;
                }
                score[t]=x1*0.073;
                score_ref[t]=x2*0.073;
            }
            auto prob=softmax(score),prob_ref=softmax(score_ref);
            for(int t=0;t<T;t++){
                max_softmax_delta=std::max(max_softmax_delta,(double)std::abs(prob[t]-prob_ref[t]));
            }
            std::array<float,D> aggregated{};
            for(int t=0;t<T;t++)for(int d=0;d<D;d++)
                aggregated[d]+=(float)(prob[t]*lat[(size_t)t*D+d]);
            std::array<float,V> out{};
            const size_t start=(size_t)(h*S+K)*NG;
            qng64_matvec_f32(packed.data()+start*GROUPBYTES,
                               scales.data()+start,N,aggregated.data(),
                               nullptr,out.data(),V,D);
            std::vector<float> explicit_ref(V);
            for(int r=0;r<V;r++){
                double val=0;
                for(int t=0;t<T;t++){
                    double dot=0;
                    for(int d=0;d<D;d++)dot+=(double)dense[((size_t)h*S+K+r)*D+d]*lat[(size_t)t*D+d];
                    val+=(double)prob_ref[t]*dot;
                }
                explicit_ref[r]=(float)val;
            }
            std::vector<float> got(out.begin(),out.end());
            auto compared=compare(got,explicit_ref);
            outcomes.insert(outcomes.end(),got.begin(),got.end());
            references.insert(references.end(),explicit_ref.begin(),explicit_ref.end());
            max_output_relative=std::max(max_output_relative,compared.relL2);
        }
    }
    Metrics total=compare(outcomes,references);
    require(total.relL2<1e-4 && max_output_relative<5e-4 &&
            max_softmax_delta<1e-5,"compressed source query+latent output parity failed");
    std::array<float,D> invalid{};
    require(!qng64_query_absorb_packed(packed.data(),scales.data(),H,16,N,D,
            qnope.data(),invalid.data()),"malformed expert/head not rejected");
    require(!qng64_query_absorb_packed(packed.data(),scales.data(),H,0,6,D,
            qnope.data(),invalid.data()),"unsupported n6 hidden format not rejected");
    require(!qng64_query_absorb_packed(nullptr,scales.data(),H,0,N,D,
            qnope.data(),invalid.data()),"null packed data not rejected");
    auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<"{\"status\":\"PASS_NATIVE_COMPRESSED_QNG64_MLA_ABSORPTION_CPU_ORACLE\","
             <<"\"model_checkpoint_read\":false,\"actual_model_tokens_generated\":0,"
             <<"\"native_qng64_C_encoder_used\":true,\"fused_native_Metal_KV_attention_implemented\":false,"
             <<"\"no_dense_dequantized_weights_in_candidate_function\":true,"
             <<"\"q_absorbed_relative_L2\":"<<qgap.relL2<<","
             <<"\"source_output_relative_L2\":"<<total.relL2<<","
             <<"\"max_output_per_head_relative_L2\":"<<max_output_relative<<","
             <<"\"max_softmax_abs_diff\":"<<max_softmax_delta<<","
             <<"\"input_lengths_tested\":[1,5,7,16,64],"
             <<"\"native_packed_bytes\":"<<bytes<<","
             <<"\"elapsed_seconds\":"<<elapsed<<","
             <<"\"production_modified\":false}\n";
    return 0;
  }catch(const std::exception& e){
    std::cerr<<"BEGLIN_MLA_COMPRESSED_ORACLE_FAILURE:"<<e.what()<<"\n";
    return 2;
  }
}
