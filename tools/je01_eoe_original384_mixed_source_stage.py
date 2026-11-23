#!/usr/bin/env python3
"""JE01 isolated original384 real 27-layer native-Q3Q5 Expert overlay source.

Requires disposable original384 stage, original TF gold patch, JE01 cbatch
overlay. Strictly modifies that scratch source, never root source/model.
"""
import argparse,hashlib,json
from pathlib import Path

C_START="static int run_moe_gpu_cbatch_online_gate(int argc, char **argv) {"
C_END="// V5i Phase B:"
C_ANCHOR="    mlx_gpu_set_sort_threshold(sort_thr);\n"
CPP_ANCHOR="int mlx_gpu_layer_step_lazy(int l, int pos, int is_dense,"
H_ANCHOR="#ifdef __cplusplus\n}\n#endif"
HEADER_API="""
// JE01 scratch-only real BF16-derived q3/q5 8 Expert full-model hook.
int mlx_gpu_je01_overlay_load_bundle(const char *bundle_path,const char *tsv_path);
"""
C_HOOK=r'''
    // Original full-model B1 isolated research only. No env => no Expert edits.
    const char *je01_bundle=getenv("JE01_REAL_Q35_BUNDLE");
    const char *je01_tsv=getenv("JE01_REAL_Q35_TSV");
    if ((je01_bundle && *je01_bundle)!=(je01_tsv && *je01_tsv)) {
        fprintf(stderr,"FATAL: JE01 real bundle/TSV must be paired\n");exit(1);
    }
    if (je01_bundle && *je01_bundle) {
        if (B!=1 || pfB!=1 || !mlx_gpu_je01_overlay_load_bundle(je01_bundle,je01_tsv) ||
            mlx_gpu_je01_overlay_count()!=8) {
            fprintf(stderr,"FATAL: JE01 real native original model mixed loader failed\n");exit(1);
        }
        fprintf(stderr,"JE01_REAL_Q35_EIGHT_EXPERTS_FULL_MODEL_REGISTERED\n");
    }
'''
CPP_LOADER=r'''
// JE01 scratch-only real original BF16 native Q3/Q5 override.
// Real physical blob SHA and TSV signed source are ALSO checked in the EOE
// launch driver and Python preflight. This only validates the physical
// contract, never changes the baseline E64 q4 registry or router.
static std::vector<uint8_t> g_je01_real_native_bundle;
int mlx_gpu_je01_overlay_load_bundle(const char *bundle_path,const char *tsv_path) {
    if(!mlx_gpu_available() || !bundle_path || !tsv_path ||
       mlx_gpu_je01_overlay_count()!=0 || g_layer_n_experts!=64 ||
       g_layer_top_k!=6 || g_layer_group!=64 ||
       g_layer_hidden!=2048 || g_layer_im_dim!=1408 ||
       !g_je01_real_native_bundle.empty()) return 0;
    try {
        namespace fs=std::filesystem;
        fs::path bin(bundle_path),tsv(tsv_path);
        if(fs::is_symlink(bin) || fs::is_symlink(tsv) ||
           !fs::is_regular_file(bin) || !fs::is_regular_file(tsv) ||
           fs::file_size(bin)!=43253760 || fs::file_size(tsv)>4096)
            return 0;
        std::ifstream fb(bin,std::ios::binary);
        if(!fb) return 0;
        g_je01_real_native_bundle.resize(43253760);
        if(!fb.read((char*)g_je01_real_native_bundle.data(),43253760) ||
           fb.peek()!=std::char_traits<char>::eof()) return 0;
        struct RealRole {
            int layer=0,expert=0,bits=0,out=0,in=0;
            size_t pb=0,pe=0,sb=0,se=0;
            std::string name;
        };
        std::map<std::pair<int,int>,std::array<RealRole,3>> roles;
        std::map<std::pair<int,int>,int> seen;
        std::vector<std::pair<size_t,size_t>> spans;
        int n3=0,n5=0,count=0;
        std::ifstream fd(tsv);
        if(!fd) return 0;
        std::string line;
        while(std::getline(fd,line)) {
            if(line.empty()) return 0;
            std::istringstream row(line);
            RealRole r;std::string role,extra;
            if(!(row>>r.layer>>r.expert>>role>>r.bits>>r.out>>r.in>>
                 r.pb>>r.pe>>r.sb>>r.se) || row>>extra) return 0;
            int idx=role=="gate_proj"?0:(role=="up_proj"?1:(role=="down_proj"?2:-1));
            if(!je01_slot_frozen(r.layer,r.expert) || idx<0 ||
               (r.bits!=3 && r.bits!=5) ||
               r.out!=(idx==2?2048:1408) || r.in!=(idx==2?1408:2048) ||
               r.pb>=r.pe || r.sb>=r.se ||
               r.pe>43253760 || r.se>43253760 ||
               r.pe-r.pb!=(size_t)r.out*r.in*r.bits/8 ||
               r.se-r.sb!=(size_t)r.out*(r.in/64)*4) return 0;
            auto k=std::make_pair(r.layer,r.expert);
            if(seen[k] & (1<<idx)) return 0;
            seen[k]|=(1<<idx);
            r.name="je01.real.L"+std::to_string(r.layer)+".E"+
                    std::to_string(r.expert)+"."+role;
            std::string original="model.layers."+std::to_string(r.layer)+
                   ".mlp.switch_mlp."+role;
            auto native=g_tensors.find(original);
            if(native==g_tensors.end() || native->second.E!=64 ||
               native->second.bits!=4 || native->second.out!=r.out ||
               native->second.in!=r.in) return 0;
            roles[k][idx]=r;
            spans.emplace_back(r.pb,r.pe);
            spans.emplace_back(r.sb,r.se);
            if(r.bits==3)n3++;else n5++;
            if(++count>24)return 0;
        }
        if(count!=24 || roles.size()!=8 || seen.size()!=8 ||
           n3!=6 || n5!=18 || spans.size()!=48) return 0;
        for(const auto &v:seen) if(v.second!=7)return 0;
        std::sort(spans.begin(),spans.end());
        if(spans.front().first!=0 || spans.back().second!=43253760)
            return 0;
        for(size_t i=1;i<spans.size();i++)
            if(spans[i-1].second!=spans[i].first) return 0;
        for(const auto &group:roles) {
            const auto &v=group.second;
            for(const auto &r:v) {
                if(mlx_gpu_bind_af(g_je01_real_native_bundle.data(),43253760,
                    r.name.c_str(),1,r.out,r.in,r.in/64,(long)r.pb,
                    (long)r.sb,-1,r.bits)!=1) return 0;
                int active=0;
                if(mlx_gpu_binding_kind(r.name.c_str(),&active)!=1 ||
                   active!=r.bits)return 0;
            }
            if(!mlx_gpu_je01_overlay_register(
                group.first.first,group.first.second,
                v[0].name.c_str(),v[1].name.c_str(),v[2].name.c_str()))
                return 0;
        }
        return mlx_gpu_je01_overlay_count()==8 ? 1:0;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"JE01_EIGHT_NATIVE_LOADER: %s\n",e.what());
        return 0;
    } catch(...) {return 0;}
}
'''

def exactly_one(s,a,b,scope):
    if s.count(a)!=1:raise ValueError("JE01_UNPINNED_SOURCE_"+scope+":"+str(s.count(a)))
    return s.replace(a,b,1)

def patch(cpp,c,header):
    if cpp.count("routed_sum=je01_scratch_adjust_routed_sum(")!=1 or \
       c.count("JE01_ORIGINAL_GOLD_OUTPUT")<1 or \
       header.count("mlx_gpu_je01_overlay_register")!=1:
        raise ValueError("JE01_ORIGINAL384_REQUIRED_TWO_STAGE_GUARDS_MISSING")
    cpp=exactly_one(cpp,"#include <array>","#include <array>\n#include <fstream>\n#include <sstream>\n#include <map>\n#include <filesystem>","CPP_INCLUDES")
    cpp=exactly_one(cpp,CPP_ANCHOR,CPP_LOADER+"\n"+CPP_ANCHOR,"CPP_NATIVE_ENTRY")
    header=exactly_one(header,H_ANCHOR,HEADER_API+"\n"+H_ANCHOR,"C_HEADER")
    start=c.index(C_START)
    end=c.index(C_END,start)
    scope=exactly_one(c[start:end],C_ANCHOR,C_ANCHOR+C_HOOK,"EXACT_ORIGINAL_GPU_ONLINE_CBATCH")
    c=c[:start]+scope+c[end:]
    if c.count("JE01_REAL_Q35_EIGHT_EXPERTS_FULL_MODEL_REGISTERED")!=1:
        raise ValueError("JE01_ONLINE_GATE_NOT_UNIQUE")
    return cpp,c,header

def stage(folder):
    if folder.is_symlink() or not folder.is_dir():raise ValueError("JE01_SCRATCH_SOURCE_MISSING")
    paths=[folder/"mlx_moe.cpp",folder/"qwen_infer.c",folder/"mlx_moe.h"]
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("JE01_SOURCE_FILES_MISSING")
    original=[p.read_text() for p in paths]
    changed=patch(*original)
    for p,s in zip(paths,changed):p.write_text(s)
    return {"status":"JE01_REAL_ORIGINAL384_EIGHT_EXPERT_Q3Q5_RESEARCH_HOOK_STAGED",
            "original_source_sha256":{p.name:hashlib.sha256(s.encode()).hexdigest()
                for p,s in zip(paths,original)},
            "patched_source_sha256":{p.name:hashlib.sha256(s.encode()).hexdigest()
                for p,s in zip(paths,changed)},
            "source_root":str(folder),"model_forward":"NOT_RUN",
            "gold_token_NLL":"NOT_MEASURED","production_modified":False}
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--scratch",type=Path,required=True)
    opts=a.parse_args()
    try:print(json.dumps(stage(opts.scratch),sort_keys=True))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:290]}))
        raise SystemExit(2)
