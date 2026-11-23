// Isolated generated-data test. This does not load JE01 weights or claim a
// full-model gradient. Compile against the unmodified MLX 0.32.1 installation.
#include "je01_mlx_native_jvp_adapters.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace mx = mlx::core;
namespace fs = std::filesystem;
using je01_native_jvp::Arrays;
using je01_native_jvp::ArrayFn;
using je01_native_jvp::need;

static std::string json_quote(const std::string &s) {
    std::ostringstream o;
    o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (c < 32) o << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(c) << std::dec;
        else o << c;
    }
    o << '"';
    return o.str();
}

static std::vector<float> host(const mx::array &a) {
    need(a.dtype() == mx::float32, "SMOKE_EXPECTED_FLOAT32");
    auto flat = mx::contiguous(mx::reshape(a, {-1}));
    mx::eval(flat);
    return {flat.data<float>(), flat.data<float>() + flat.size()};
}

static mx::array f32(const std::vector<float> &v, const mx::Shape &shape) {
    return mx::array(v.data(), shape, mx::float32);
}

static mx::array index1(int value) {
    int32_t v = value;
    return mx::array(&v, {1}, mx::int32); // Iterator constructor copies here.
}

struct Error {
    double max_abs = 0, rms = 0, reference_rms = 0;
    std::string json() const {
        std::ostringstream o;
        o << std::setprecision(17) << "{\"max_abs\":" << max_abs
          << ",\"rms\":" << rms << ",\"reference_rms\":" << reference_rms << '}';
        return o.str();
    }
};

static Error error(const std::vector<float> &a, const std::vector<float> &b) {
    need(a.size() == b.size() && !a.empty(), "SMOKE_COMPARE_SIZE");
    Error e;
    double squared = 0, reference = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        need(std::isfinite(a[i]) && std::isfinite(b[i]), "SMOKE_NONFINITE_OUTPUT");
        double d = static_cast<double>(a[i]) - b[i];
        e.max_abs = std::max(e.max_abs, std::abs(d));
        squared += d*d;
        reference += static_cast<double>(b[i])*b[i];
    }
    e.rms = std::sqrt(squared / a.size());
    e.reference_rms = std::sqrt(reference / a.size());
    return e;
}

static bool exact(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static void numeric_gate(const Error &e, double atol, double rtol, const char *label) {
    double bound = atol + rtol*e.reference_rms;
    need(e.rms <= bound && e.max_abs <= 8*bound, label);
}

struct Dumps {
    fs::path root;
    std::vector<std::string> entries;
    std::string save(const std::string &name, const mx::array &array) {
        need(array.dtype() == mx::float32 || array.dtype() == mx::uint32,
             "SMOKE_DUMP_DTYPE");
        auto a = mx::contiguous(mx::reshape(array, {-1}));
        mx::eval(a);
        std::string dtype = a.dtype() == mx::float32 ? "float32" : "uint32";
        std::string filename = name + (dtype == "float32" ? ".f32" : ".u32");
        fs::path file = root / filename;
        need(!fs::exists(file), "SMOKE_REFUSES_EXISTING_DUMP");
        std::ofstream out(file, std::ios::binary | std::ios::out);
        const char *data = a.dtype() == mx::float32
            ? reinterpret_cast<const char *>(a.data<float>())
            : reinterpret_cast<const char *>(a.data<uint32_t>());
        out.write(data, a.size()*4);
        out.close();
        need(out.good(), "SMOKE_DUMP_WRITE_FAILED");
        std::ostringstream meta;
        meta << "{\"file\":" << json_quote(filename) << ",\"dtype\":" << json_quote(dtype)
             << ",\"elements\":" << a.size() << ",\"shape\":[";
        for (size_t j = 0; j < array.shape().size(); ++j) {
            if (j) meta << ',';
            meta << array.shape()[j];
        }
        meta << "]}";
        entries.push_back(meta.str());
        return filename;
    }
};

// Every case emits ordinary-native, custom primal, JVP primal, tangent,
// independent reference, and two native finite differences. The Python driver
// recomputes parity and numeric gates from these raw arrays, then hashes them.
static std::string report_case(Dumps &d, const std::string &name,
        const std::string &kind, const mx::array &native, const mx::array &wrapped,
        const mx::array &jvp_primal, const mx::array &tangent,
        const mx::array &reference, const mx::array &fd_h, const mx::array &fd_h2,
        double analytic_atol, double analytic_rtol,
        double fd_atol, double fd_rtol, const std::string &extra = "") {
    auto n=host(native), w=host(wrapped), p=host(jvp_primal), t=host(tangent);
    auto r=host(reference), f=host(fd_h), f2=host(fd_h2);
    need(exact(n,w), "SMOKE_NATIVE_CUSTOM_PRIMAL_BYTE_MISMATCH");
    need(exact(n,p), "SMOKE_NATIVE_JVP_PRIMAL_BYTE_MISMATCH");
    Error ar=error(t,r), er=error(f,t), er2=error(f2,t);
    numeric_gate(ar, analytic_atol, analytic_rtol, "SMOKE_ANALYTIC_JVP_MISMATCH");
    numeric_gate(er, fd_atol, fd_rtol, "SMOKE_COARSE_FD_JVP_MISMATCH");
    numeric_gate(er2, fd_atol, fd_rtol, "SMOKE_FINE_FD_JVP_MISMATCH");
    std::ostringstream o;
    o << std::setprecision(17) << "{\"name\":" << json_quote(name)
      << ",\"kind\":" << json_quote(kind) << ",\"elements\":" << n.size()
      << ",\"native\":" << json_quote(d.save(name+".native",native))
      << ",\"wrapped\":" << json_quote(d.save(name+".wrapped",wrapped))
      << ",\"jvp_primal\":" << json_quote(d.save(name+".jvp_primal",jvp_primal))
      << ",\"tangent\":" << json_quote(d.save(name+".tangent",tangent))
      << ",\"reference\":" << json_quote(d.save(name+".reference",reference))
      << ",\"fd_coarse\":" << json_quote(d.save(name+".fd_coarse",fd_h))
      << ",\"fd_fine\":" << json_quote(d.save(name+".fd_fine",fd_h2))
      << ",\"native_primal_exact\":true,\"jvp_primal_exact\":true"
      << ",\"analytic_error\":" << ar.json()
      << ",\"fd_coarse_error\":" << er.json()
      << ",\"fd_fine_error\":" << er2.json()
      << ",\"analytic_atol\":" << analytic_atol << ",\"analytic_rtol\":" << analytic_rtol
      << ",\"fd_atol\":" << fd_atol << ",\"fd_rtol\":" << fd_rtol
      << extra << '}';
    return o.str();
}

static std::string unsupported(const ArrayFn &f, const Arrays &p,
                               const Arrays &t, const std::string &expected) {
    try {
        auto result=mx::jvp(f,p,t);
        mx::eval(result.second);
    } catch (const std::exception &e) {
        need(std::string(e.what()).find(expected) != std::string::npos,
             "SMOKE_UNEXPECTED_RAW_JVP_FAILURE");
        return e.what();
    }
    throw std::runtime_error("SMOKE_EXPECTED_RAW_JVP_UNSUPPORTED_IN_MLX_0_32_1");
}

struct TransformGuardFailure : std::runtime_error {
    std::string transform, expected, actual;
    bool returned_without_exception;
    TransformGuardFailure(const std::string &name, const std::string &wanted,
                          const std::string &observed, bool returned = false)
        : std::runtime_error(std::string(returned ? "SMOKE_TRANSFORM_GUARD_DID_NOT_REFUSE"
                                                : "SMOKE_TRANSFORM_GUARD_WRONG_FAILURE")
                             + ":transform=" + name + ":expected=" + wanted
                             + ":actual=" + observed),
          transform(name), expected(wanted), actual(observed),
          returned_without_exception(returned) {}
};

static std::string require_refusal(const std::function<void()> &attempt,
                                    const std::string &transform,
                                    const std::string &expected) {
    try {
        attempt();
    } catch (const std::exception &e) {
        // Only the deliberately installed guard is a passing negative control.
        // Preserve an unexpected native exception verbatim instead of hiding it.
        if (std::string(e.what()) != expected)
            throw TransformGuardFailure(transform, expected, e.what());
        return e.what();
    }
    throw TransformGuardFailure(transform, expected, "NO_EXCEPTION", true);
}

static void transform_guard_cases(const std::string &kind,const ArrayFn &wrapped,
                                  const Arrays &primals,std::vector<std::string> &guards) {
    auto cotangent=mx::ones_like(wrapped(primals)[0]);
    auto reverse_error=require_refusal([&]() {
        auto result=mx::vjp(wrapped,primals,{cotangent});
        mx::eval(result.second);
    },kind+"_vjp","JE01_JVP_ADAPTER_REVERSE_MODE_NOT_SUPPORTED");
    Arrays batched;
    for(const auto &a:primals)batched.push_back(mx::stack({a,a},0));
    auto batch_error=require_refusal([&]() {
        auto mapped=mx::vmap(wrapped,std::vector<int>(primals.size(),0),{0});
        mx::eval(mapped(batched));
    },kind+"_direct_vmap","JE01_JVP_ADAPTER_DIRECT_VMAP_NOT_SUPPORTED");
    guards.push_back(json_quote(kind+"_vjp")+":"+json_quote(reverse_error));
    guards.push_back(json_quote(kind+"_direct_vmap")+":"+json_quote(batch_error));
}

static void gather_cases(Dumps &d, std::vector<std::string> &cases,
                         std::string &raw_error,std::vector<std::string> &guards) {
    constexpr int E=8, OUT=64, IN=128, K=6;
    const std::vector<int32_t> ids{7,0,3,5,1,6};
    std::vector<float> wh(E*OUT*IN);
    for (size_t i=0;i<wh.size();++i)
        wh[i] = static_cast<float>(static_cast<int>((i*17+i/IN*11)%67)-33)/37.0f;
    auto weights=f32(wh,{E,OUT,IN});
    for (int bits : {4,5}) {
        auto q=mx::quantize(weights,64,bits,"affine");
        need(q.size()==3 && q[0].dtype()==mx::uint32,"SMOKE_QUANTIZE_AFFINE_TRIPLE");
        auto decoded=mx::dequantize(q[0],q[1],q[2],64,bits,"affine");
        auto dw=host(decoded);
        for (bool down : {false,true}) {
            std::string name=std::string("gather_")+(down?"down_q":"gate_q")+std::to_string(bits);
            std::vector<float> xv((down?K:1)*IN), dxv(xv.size());
            for (size_t i=0;i<xv.size();++i) {
                xv[i]=static_cast<float>(static_cast<int>((i*13+5)%43)-21)/31.0f;
                dxv[i]=static_cast<float>(static_cast<int>((i*7+3)%37)-18)/29.0f;
            }
            auto x=f32(xv,down?mx::Shape{K,1,IN}:mx::Shape{1,1,1,IN});
            auto dx=f32(dxv,x.shape());
            auto rhs=mx::array(ids.data(),down?mx::Shape{K}:mx::Shape{1,K},mx::int32);
            ArrayFn native=[=](const Arrays &a) {
                return Arrays{mx::gather_qmm(a[0],q[0],q[1],q[2],std::nullopt,rhs,
                                            true,64,bits,"affine",false)};
            };
            auto wrapped=je01_native_jvp::frozen_gather_qmm(q[0],q[1],q[2],
                                                          std::nullopt,rhs,64,bits);
            if (raw_error.empty()) {
                raw_error=unsupported(native,{x},{dx},"GatherQMM::jvp NYI");
                transform_guard_cases("gather_qmm",wrapped,{x},guards);
            }
            auto result=mx::jvp(wrapped,{x},{dx});
            auto native_tangent=native({dx})[0];
            need(exact(host(result.second[0]),host(native_tangent)),
                 "SMOKE_GATHER_TANGENT_NOT_NATIVE_GATHER_DX");
            std::vector<float> ref(K*OUT);
            for (int k=0;k<K;++k) for (int row=0;row<OUT;++row) {
                double sum=0;
                for (int j=0;j<IN;++j)
                    sum+=static_cast<double>(dxv[(down?k:0)*IN+j])*dw[(ids[k]*OUT+row)*IN+j];
                ref[k*OUT+row]=static_cast<float>(sum);
            }
            auto native_out=native({x})[0];
            auto fd=[&](float h) {return (native({x+h*dx})[0]-native({x-h*dx})[0])/(2*h);};
            std::ostringstream extra;
            extra << ",\"bits\":" << bits << ",\"group_size\":64,\"native_tangent_exact\":true"
                  << ",\"native_tangent\":" << json_quote(d.save(name+".native_tangent",native_tangent))
                  << ",\"packed\":" << json_quote(d.save(name+".packed",q[0]))
                  << ",\"scales\":" << json_quote(d.save(name+".scales",q[1]))
                  << ",\"biases\":" << json_quote(d.save(name+".biases",q[2]))
                  << ",\"decoded_weights\":" << json_quote(d.save(name+".decoded",decoded))
                  << ",\"activation\":" << json_quote(d.save(name+".x",x))
                  << ",\"activation_tangent\":" << json_quote(d.save(name+".dx",dx));
            cases.push_back(report_case(d,name,"gather_qmm",native_out,wrapped({x})[0],
                result.first[0],result.second[0],f32(ref,native_out.shape()),fd(0.0625f),
                fd(0.03125f),1e-5,2e-5,2e-4,2e-4,extra.str()));
        }
    }
}

static void scatter_cases(Dumps &d,std::vector<std::string> &cases,std::string &raw_error,
                          std::vector<std::string> &guards) {
    constexpr int H=2,L=4,D=8,P=2;
    std::vector<float> cv(H*L*D),uv(H*D),dcv(cv.size()),duv(uv.size());
    for(size_t i=0;i<cv.size();++i) {cv[i]=static_cast<float>(i)/64;dcv[i]=(static_cast<int>(i%17)-8)/32.0f;}
    for(size_t i=0;i<uv.size();++i) {uv[i]=-static_cast<float>(i)/32;duv[i]=(static_cast<int>(i%7)-3)/16.0f;}
    auto cache=f32(cv,{1,H,L,D}),update=f32(uv,{1,1,H,1,D});
    auto dc=f32(dcv,cache.shape()),du=f32(duv,update.shape());
    auto slot=index1(0),pos=index1(P);
    ArrayFn native=[=](const Arrays &a){return Arrays{mx::scatter(a[0],{slot,pos},a[1],{0,2})};};
    auto wrapped=je01_native_jvp::fixed_b1_kv_scatter(slot,pos);
    raw_error=unsupported(native,{cache,update},{dc,du},"[scatter] JVP not yet implemented");
    transform_guard_cases("scatter",wrapped,{cache,update},guards);
    for(int mode=0;mode<3;++mode) {
        // Closures deliberately make one CustomTransforms input non-differentiable
        // in modes 1 and 2, exercising its subset argnums contract.
        std::string name=mode==0?"scatter_both":mode==1?"scatter_cache_only":"scatter_update_only";
        ArrayFn fn = mode==0 ? wrapped : mode==1 ? ArrayFn([=](const Arrays &a){return wrapped({a[0],update});})
                                                 : ArrayFn([=](const Arrays &a){return wrapped({cache,a[0]});});
        auto result=mode==0?mx::jvp(fn,{cache,update},{dc,du}):
                    mode==1?mx::jvp(fn,{cache},{dc}):mx::jvp(fn,{update},{du});
        auto cdir=mode==2?mx::zeros_like(cache):dc;
        auto udir=mode==1?mx::zeros_like(update):du;
        auto fd=[&](float h){return (native({cache+h*cdir,update+h*udir})[0]-
                                     native({cache-h*cdir,update-h*udir})[0])/(2*h);};
        std::vector<float> ref=mode==2?std::vector<float>(cv.size(),0.0f):dcv;
        for(int head=0;head<H;++head) for(int j=0;j<D;++j)
            ref[(head*L+P)*D+j]=mode==1?0.0f:duv[head*D+j];
        need(exact(host(result.second[0]),ref),"SMOKE_SCATTER_TANGENT_CPU_REPLACEMENT_MISMATCH");
        cases.push_back(report_case(d,name,"scatter",native({cache,update})[0],
            wrapped({cache,update})[0],result.first[0],result.second[0],f32(ref,cache.shape()),
            fd(0.0625f),fd(0.03125f),0,0,0,0,
            ",\"cpu_replacement_tangent_exact\":true"));
    }
}

static mx::array flatten_recurrence(const mx::array &cache,const mx::array &loss0,
                                    const mx::array &loss1) {
    return mx::concatenate({mx::reshape(cache,{-1}),mx::reshape(loss0,{1}),mx::reshape(loss1,{1})});
}

static void recurrence_case(Dumps &d,std::vector<std::string> &cases) {
    constexpr int H=2,L=4,D=8;
    std::vector<float> u0v(H*D),u1v(H*D);
    for(size_t i=0;i<u0v.size();++i) {
        u0v[i]=0.5f+(i%5)*0.0625f;
        u1v[i]=-0.25f+(i%7)*0.03125f;
    }
    auto u0=f32(u0v,{1,1,H,1,D}),u1=f32(u1v,{1,1,H,1,D});
    auto cache0=mx::zeros({1,H,L,D},mx::float32);
    auto alpha=mx::array(0.25f),one=mx::array(1.0f);
    auto slot=index1(0),pos0=index1(0),pos1=index1(1);
    auto scatter0=je01_native_jvp::fixed_b1_kv_scatter(slot,pos0);
    auto scatter1=je01_native_jvp::fixed_b1_kv_scatter(slot,pos1);
    auto history=[](const mx::array &cache) {
        return mx::reshape(mx::slice(cache,{0,0,0,0},{1,H,1,D}),{1,1,H,1,D});
    };
    ArrayFn step0=[=](const Arrays &a) {
        auto cache=scatter0({a[1],a[0]*u0})[0];
        return Arrays{cache,mx::sum(cache*cache)};
    };
    ArrayFn step1=[=](const Arrays &a) {
        auto cache=scatter1({a[1],a[0]*u1+0.5f*history(a[1])})[0];
        return Arrays{cache,mx::sum(cache*cache)};
    };
    ArrayFn whole=[=](const Arrays &a) {
        auto first=step0({a[0],cache0});
        auto second=step1({a[0],first[0]});
        return Arrays{flatten_recurrence(second[0],first[1],second[1])};
    };
    ArrayFn native=[=](const Arrays &a) {
        auto cache1=mx::scatter(cache0,{slot,pos0},a[0]*u0,{0,2});
        auto cache2=mx::scatter(cache1,{slot,pos1},a[0]*u1+0.5f*history(cache1),{0,2});
        return Arrays{flatten_recurrence(cache2,mx::sum(cache1*cache1),mx::sum(cache2*cache2))};
    };
    auto full=mx::jvp(whole,{alpha},{one});
    auto first=mx::jvp(step0,{alpha,cache0},{one,mx::zeros_like(cache0)});
    Arrays first_eval=first.first;
    first_eval.insert(first_eval.end(),first.second.begin(),first.second.end());
    mx::eval(first_eval); // Completed token boundary; explicitly carry dcache.
    auto second=mx::jvp(step1,{alpha,first.first[0]},{one,first.second[0]});
    auto detached=mx::jvp(step1,{alpha,first.first[0]},{one,mx::zeros_like(first.first[0])});
    auto streamed=flatten_recurrence(second.second[0],first.second[1],second.second[1]);
    auto wrong=flatten_recurrence(detached.second[0],first.second[1],detached.second[1]);
    Error stream_error=error(host(streamed),host(full.second[0]));
    numeric_gate(stream_error,1e-7,1e-6,"SMOKE_STREAMED_HISTORY_JVP_MISMATCH");
    auto right_host=host(full.second[0]),wrong_host=host(wrong);
    double detached_loss1_error=std::abs(static_cast<double>(right_host.back())-wrong_host.back());
    need(detached_loss1_error>0.1,"SMOKE_DETACHED_HISTORY_NEGATIVE_CONTROL_INSENSITIVE");
    std::vector<float> ref(H*L*D+2,0.0f);
    double norm0=0,norm1=0;
    for(int h=0;h<H;++h) for(int j=0;j<D;++j) {
        size_t i=h*D+j;
        float later=u1v[i]+0.5f*u0v[i];
        ref[(h*L)*D+j]=u0v[i];
        ref[(h*L+1)*D+j]=later;
        norm0+=static_cast<double>(u0v[i])*u0v[i];
        norm1+=static_cast<double>(later)*later;
    }
    ref[H*L*D]=static_cast<float>(2*0.25*norm0);
    ref[H*L*D+1]=static_cast<float>(2*0.25*(norm0+norm1));
    auto fd=[&](float h){return (native({alpha+h})[0]-native({alpha-h})[0])/(2*h);};
    std::ostringstream extra;
    extra << std::setprecision(17) << ",\"streamed_tangent\":"
          << json_quote(d.save("kv_two_step.streamed_tangent",streamed))
          << ",\"detached_tangent\":" << json_quote(d.save("kv_two_step.detached_tangent",wrong))
          << ",\"streaming_error\":" << stream_error.json()
          << ",\"detached_loss1_error\":" << detached_loss1_error
          << ",\"loss1_derivative_analytic\":" << ref.back()
          << ",\"loss_is_toy_quadratic_not_token_nll\":true";
    cases.push_back(report_case(d,"kv_two_step","kv_recurrence",native({alpha})[0],
        whole({alpha})[0],full.first[0],full.second[0],f32(ref,{H*L*D+2}),
        fd(0.0625f),fd(0.03125f),1e-7,1e-6,1e-7,1e-6,extra.str()));
}

int main(int argc,char **argv) {
    try {
        need(argc==2,"SMOKE_USAGE_OUTPUT_DIRECTORY");
        fs::path root=argv[1];
        need(root.is_absolute() && fs::is_directory(root) && !fs::is_symlink(root),
             "SMOKE_OUTPUT_MUST_BE_EXISTING_ABSOLUTE_REAL_DIRECTORY");
        need(mx::is_available(mx::Device(mx::Device::gpu)),"SMOKE_GPU_UNAVAILABLE");
        mx::set_default_device(mx::Device(mx::Device::gpu));
        Dumps dumps{root,{}};
        std::vector<std::string> cases,guards;
        std::string raw_gather,raw_scatter;
        gather_cases(dumps,cases,raw_gather,guards);
        scatter_cases(dumps,cases,raw_scatter,guards);
        recurrence_case(dumps,cases);
        std::cout << "{\"schema\":\"beglin-je01-mlx-native-custom-jvp-smoke/2\","
                     "\"status\":\"PASS\",\"expected_mlx_version\":\"0.32.1\","
                     "\"device\":\"gpu\",\"fd_h\":0.0625,\"fd_h2\":0.03125,"
                     "\"proof_scope\":\"tiny-native-primitives-and-two-step-KV-only\","
                     "\"raw_jvp_errors\":{\"gather_qmm\":" << json_quote(raw_gather)
                  << ",\"scatter\":" << json_quote(raw_scatter) << "},\"transform_guard_errors\":{";
        for(size_t i=0;i<guards.size();++i) {if(i)std::cout<<',';std::cout<<guards[i];}
        std::cout << "},\"cases\":[";
        for(size_t i=0;i<cases.size();++i) {if(i)std::cout<<',';std::cout<<cases[i];}
        std::cout << "],\"arrays\":[";
        for(size_t i=0;i<dumps.entries.size();++i) {if(i)std::cout<<',';std::cout<<dumps.entries[i];}
        std::cout << "]}\n";
        return 0;
    } catch(const std::exception &e) {
        std::cerr << "JE01 native JVP smoke failed: " << e.what() << '\n';
        std::cout << "{\"schema\":\"beglin-je01-mlx-native-custom-jvp-smoke/2\","
                     "\"status\":\"FAIL\",\"error\":" << json_quote(e.what());
        if (const auto *guard = dynamic_cast<const TransformGuardFailure *>(&e)) {
            std::cout << ",\"failure_context\":{\"phase\":\"unsupported_transform_control\","
                         "\"transform\":" << json_quote(guard->transform)
                      << ",\"expected_exception\":" << json_quote(guard->expected)
                      << ",\"actual_exception\":" << json_quote(guard->actual)
                      << ",\"returned_without_exception\":"
                      << (guard->returned_without_exception ? "true" : "false") << '}';
        }
        std::cout << "}\n";
        return 2;
    }
}
