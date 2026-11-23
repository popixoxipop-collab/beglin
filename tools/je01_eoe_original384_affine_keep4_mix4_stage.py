#!/usr/bin/env python3
"""JE01 original384 scratch-only four-important-Expert AFFINE AF4 pass-through.

Depends on pinned original384 + original true-gold + previous scratch-only 8
Expert source hook + q4 symmetric sentinel binder. Replaces *only* disposable
E1-loader and original cbatch overlay-count gate. Four important original
Experts are kept as original E64 affine native tensors in original 27-layer
GPU flow; only four less-frequently-routed Experts are E1-sym Q4/Q5 corrected.

The 43,253,760B budgeted file physically contains 12 original AF4 FFN roles
(packed+scale+bias) + 6 Q4 packed+scale + 6 Q5 packed+scale. Its AF4 sections
are verified from full-hash original 9.8GB model but never rebound into GPU.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

CPP_BEG="int mlx_gpu_je01_overlay_load_bundle("
CPP_END="int mlx_gpu_layer_step_lazy("
MARK_OLD="JE01_REAL_Q35_EIGHT_EXPERTS_FULL_MODEL_REGISTERED"
MARK_NEW="JE01_REAL_AF4_KEEP4_Q4Q5_MIXED4_REGISTERED"
COUNT_OLD="mlx_gpu_je01_overlay_count()!=8"
COUNT_NEW="mlx_gpu_je01_overlay_count()!=4"
SOURCE_MARK="routed_sum=je01_scratch_adjust_routed_sum("
EXPECTED_FROZEN={(21,61),(24,26),(13,28),(20,17),
                 (13,22),(24,60),(12,13),(14,63)}
LOADER=r'''
// JE01 scratch-only true-source 12 AFFINE AF4 roles passthrough,
// 6 BF16 symmetric Q4 roles and 6 BF16 symmetric Q5 roles.
// Original AF4's original E64 top-K routed tensors are NEVER rebound
// in the four important Experts; only lower-routed 4 Experts override.
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
        struct Role {
            int layer=0,expert=0,bits=0,out=0,in=0;
            size_t pb=0,pe=0,sb=0,se=0,bb=0,be=0;
            std::string codec,name;
        };
        std::map<std::pair<int,int>,std::array<Role,3>> roles;
        std::map<std::pair<int,int>,int> seen;
        std::vector<std::pair<size_t,size_t>> spans;
        int affine=0,sym4=0,sym5=0,all=0;
        std::ifstream fd(tsv);
        if(!fd) return 0;
        std::string line;
        while(std::getline(fd,line)) {
            if(line.empty())return 0;
            std::istringstream row(line);
            Role r;std::string role,extra;
            if(!(row>>r.layer>>r.expert>>role>>r.bits>>r.out>>r.in>>
                 r.pb>>r.pe>>r.sb>>r.se>>r.bb>>r.be>>r.codec) ||
               row>>extra) return 0;
            int idx=role=="gate_proj"?0:(role=="up_proj"?1:(role=="down_proj"?2:-1));
            const bool high=((r.layer==21 && r.expert==61) ||
                             (r.layer==24 && r.expert==26) ||
                             (r.layer==13 && r.expert==28) ||
                             (r.layer==20 && r.expert==17));
            const bool low4=((r.layer==13 && r.expert==22) ||
                             (r.layer==24 && r.expert==60));
            const bool low5=((r.layer==12 && r.expert==13) ||
                             (r.layer==14 && r.expert==63));
            if(!je01_slot_frozen(r.layer,r.expert) || idx<0 ||
               (int(high)+int(low4)+int(low5))!=1 ||
               (high && (r.codec!="AF4" || r.bits!=4)) ||
               (low4 && (r.codec!="Q4" || r.bits!=4)) ||
               (low5 && (r.codec!="Q5" || r.bits!=5)) ||
               r.out!=(idx==2?2048:1408) || r.in!=(idx==2?1408:2048) ||
               r.pb>=r.pe || r.sb>=r.se ||
               r.pe>43253760 || r.se>43253760 ||
               r.pe-r.pb!=(size_t)r.out*r.in*r.bits/8 ||
               r.se-r.sb!=(size_t)r.out*(r.in/64)*4) return 0;
            if(high) {
                if(r.bb>=r.be || r.be>43253760 ||
                   r.be-r.bb!=(size_t)r.out*(r.in/64)*4) return 0;
                affine++;spans.emplace_back(r.bb,r.be);
            } else {
                if(r.bb!=0 || r.be!=0) return 0;
                if(low4)sym4++;else sym5++;
            }
            auto pair=std::make_pair(r.layer,r.expert);
            if(seen[pair] & (1<<idx)) return 0;
            seen[pair]|=(1<<idx);
            r.name="je01.real.L"+std::to_string(r.layer)+".E"+
                   std::to_string(r.expert)+"."+role;
            std::string original="model.layers."+std::to_string(r.layer)+
                    ".mlp.switch_mlp."+role;
            auto native=g_tensors.find(original);
            if(native==g_tensors.end() || native->second.E!=64 ||
               native->second.bits!=4 || native->second.out!=r.out ||
               native->second.in!=r.in) return 0;
            roles[pair][idx]=r;
            spans.emplace_back(r.pb,r.pe);
            spans.emplace_back(r.sb,r.se);
            if(++all>24)return 0;
        }
        if(all!=24 || roles.size()!=8 || seen.size()!=8 ||
           affine!=12 || sym4!=6 || sym5!=6 || spans.size()!=60)
            return 0;
        for(const auto &s:seen)if(s.second!=7)return 0;
        std::sort(spans.begin(),spans.end());
        if(spans.front().first!=0 || spans.back().second!=43253760)return 0;
        for(size_t i=1;i<spans.size();i++)
            if(spans[i-1].second!=spans[i].first)return 0;
        // High-frequency original affine Expert routes are byte-preserved
        // and computation-preserved as the existing E64 native path.
        for(const auto &group:roles) {
            const auto &v=group.second;
            if(v[0].codec=="AF4") {
                for(const auto &r:v)if(r.codec!="AF4" || r.bits!=4)return 0;
                continue;
            }
            if(v[0].codec!="Q4" && v[0].codec!="Q5")return 0;
            for(const auto &r:v) {
                if(r.codec!=v[0].codec || (r.bits!=4 && r.bits!=5))return 0;
                if(mlx_gpu_bind_af(g_je01_real_native_bundle.data(),43253760,
                   r.name.c_str(),1,r.out,r.in,r.in/64,(long)r.pb,
                   (long)r.sb,-1,r.bits)!=1)return 0; // Q4 implicit -8sc
                int active=0;
                if(mlx_gpu_binding_kind(r.name.c_str(),&active)!=1 ||
                   active!=r.bits)return 0;
            }
            if(!mlx_gpu_je01_overlay_register(
                   group.first.first,group.first.second,
                   v[0].name.c_str(),v[1].name.c_str(),v[2].name.c_str()))
                   return 0;
        }
        return mlx_gpu_je01_overlay_count()==4?1:0;
    } catch(const std::exception &e) {
        std::fprintf(stderr,"JE01_REAL_ORIGINAL_AF4_PASS_Q45_LOADER: %s\n",e.what());
        return 0;
    } catch(...) {return 0;}
}
'''

def single(s,old,new,name):
    count=s.count(old)
    if count!=1:
        raise ValueError("JE01_AFFINE_MIX4_SOURCE_ANCHOR_"+name+"_COUNT_"+str(count))
    return s.replace(old,new,1)

def patch(cpp,c):
    if cpp.count(SOURCE_MARK)!=1 or cpp.count("static std::vector<uint8_t> g_je01_real_native_bundle;")!=1 or \
       "if (bias_off==-1)" not in cpp or \
       "mlx_gpu_je01_overlay_register" not in cpp:
        raise ValueError("JE01_MIX4_NATIVE_Q45_CXX_BASE_NOT_VERIFIED")
    if cpp.count(CPP_BEG)!=1 or cpp.count(CPP_END)!=1:
        raise ValueError("JE01_ORIGINAL27_NATIVE_LOADER_SCOPE_DRIFT")
    start=cpp.index(CPP_BEG)
    end=cpp.index(CPP_END,start)
    cpp=cpp[:start]+LOADER+"\n"+cpp[end:]
    c=single(c,COUNT_OLD,COUNT_NEW,"ONLY_REAL_GPU_CBATCH_OVERLAY_COUNT")
    c=single(c,MARK_OLD,MARK_NEW,"ONLY_REAL_GPU_CBATCH_OVERLAY_MARK")
    if cpp.count("return mlx_gpu_je01_overlay_count()==4?1:0;")!=1 or \
       c.count(MARK_NEW)!=1 or c.count(COUNT_NEW)!=1:
        raise ValueError("JE01_ORIGINAL_REAL27_AFFINE_MIX4_CODE_NOT_UNIQUE")
    if cpp.count("if (bias_off==-1)")!=1:
        raise ValueError("JE01_Q4_SYMMETRIC_REAL_NIBBLE_BIAS_LOST")
    return cpp,c

def run(scratch):
    if not scratch.is_dir() or scratch.is_symlink():
        raise ValueError("JE01_REAL_ORIGINAL384_SOURCE_ROOT_INVALID")
    files=[scratch/"mlx_moe.cpp",scratch/"qwen_infer.c"]
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("JE01_REAL_MIX4_INPUT_SOURCE_MISSING")
    old=[p.read_text() for p in files]
    result=patch(*old)
    for p,c in zip(files,result):p.write_text(c)
    record={"status":"JE01_ORIGINAL27_AFFINE4_KEEP4_Q4Q5_MIX4_NATIVE_CODE_STAGED",
       "source_original_code_sha256":{p.name:hashlib.sha256(s.encode()).hexdigest() for p,s in zip(files,old)},
       "mixed_source_code_sha256":{p.name:hashlib.sha256(s.encode()).hexdigest() for p,s in zip(files,result)},
       "registered_experts":4,"native_original_AF4_preserved_experts":4,
       "original_E64_affine_GPU_path_preserved":True,
       "symmetric_Q4_bias_sentinel_preserved":True,
       "logical_eight_Expert_source_byte_budget":43253760,
       "production_modified":False,"model_forward":"NOT_RUN","NLL":"NOT_MEASURED"}
    print(json.dumps(record,sort_keys=True),flush=True)
    return record

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scratch",required=True,type=Path)
    try:run(p.parse_args().scratch)
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED","reason":str(exc)[:350]},sort_keys=True),flush=True)
        raise SystemExit(2)
