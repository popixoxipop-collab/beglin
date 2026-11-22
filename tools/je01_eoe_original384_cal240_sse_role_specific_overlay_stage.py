#!/usr/bin/env python3
"""Scratch-only original27 MLX overlay for frozen JE01 CAL240 *role-specific* tiers.

Existing AFFINEKEEP4 loader assumes gate/up/down all share a tier per Expert;
that cannot execute the signed SSE-DP policy. This patch binds each changed
Expert's three roles individually (native AF4 real bias or Q4/Q5 symmetric),
while leaving fully unchanged L14/E63 on native original E64 path.
No original source/model/service patch; NLL and numerical parity NOT RUN.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

WORK = Path("/Volumes/D50/TailnetCommander/work")
CPP_BEG = "int mlx_gpu_je01_overlay_load_bundle("
CPP_END = "int mlx_gpu_layer_step_lazy("
OLD_COUNT = "mlx_gpu_je01_overlay_count()!=4"
NEW_COUNT = "mlx_gpu_je01_overlay_count()!=7"
OLD_MARK = "JE01_REAL_AF4_KEEP4_Q4Q5_MIXED4_REGISTERED"
NEW_MARK = "JE01_CAL240_FROZEN_ROLEWISE_SEVEN_EXPERTS_REGISTERED"
SCHEMA = "beglin-je01-original27-scratch-only-cal240-frozen-role-specific-af4-q4-q5-overlay/1"

LOADER = r"""
// JE01 source-frozen actual TRAIN240 activation policy: 12 original AF4
// roles, six native Q4 roles, six native Q5 roles, exact 43,253,760 bytes.
// One wholly original Expert L14/E63 never overlaid. Seven others use three
// role-specific E1 tensors, including actual source AF4 real signed biases.
int mlx_gpu_je01_overlay_load_bundle(const char *bundle_path,const char *tsv_path) {
    if(!mlx_gpu_available() || !bundle_path || !tsv_path ||
       mlx_gpu_je01_overlay_count()!=0 || g_layer_n_experts!=64 ||
       g_layer_top_k!=6 || g_layer_group!=64 || g_layer_hidden!=2048 ||
       g_layer_im_dim!=1408 || !g_je01_real_native_bundle.empty()) return 0;
    try {
        namespace fs=std::filesystem;
        fs::path bin(bundle_path),tsv(tsv_path);
        if(fs::is_symlink(bin) || fs::is_symlink(tsv) ||
           !fs::is_regular_file(bin) || !fs::is_regular_file(tsv) ||
           fs::file_size(bin)!=43253760 || fs::file_size(tsv)>4096)
            return 0;
        std::ifstream fb(bin,std::ios::binary);
        if(!fb)return 0;
        g_je01_real_native_bundle.resize(43253760);
        if(!fb.read((char*)g_je01_real_native_bundle.data(),43253760) ||
           fb.peek()!=std::char_traits<char>::eof())return 0;
        struct Role {
            int layer=0,expert=0,bits=0,out=0,in=0;
            size_t pb=0,pe=0,sb=0,se=0,bb=0,be=0;
            std::string codec,name;
        };
        std::map<std::pair<int,int>,std::array<Role,3>> roles;
        std::map<std::pair<int,int>,int> seen;
        std::vector<std::pair<size_t,size_t>> spans;
        int nAF4=0,nQ4=0,nQ5=0,total=0;
        std::ifstream manifest(tsv);
        if(!manifest)return 0;
        std::string line;
        while(std::getline(manifest,line)) {
            if(line.empty())return 0;
            std::istringstream input(line);
            Role r;std::string role,extra;
            if(!(input>>r.layer>>r.expert>>role>>r.bits>>r.out>>r.in>>
                 r.pb>>r.pe>>r.sb>>r.se>>r.bb>>r.be>>r.codec) ||
                 input>>extra)return 0;
            int idx=role=="gate_proj"?0:(role=="up_proj"?1:(role=="down_proj"?2:-1));
            const bool q4=((r.layer==12 && r.expert==13) ||
                           (r.layer==13 && r.expert==22) ||
                           (r.layer==13 && r.expert==28)) && idx>=0 && idx<2;
            const bool q5=((r.layer==20 && r.expert==17 && idx==2) ||
                           (r.layer==21 && r.expert==61 && idx==2) ||
                           (r.layer==24 && r.expert==60 && idx==2) ||
                           (r.layer==24 && r.expert==26 && idx>=0));
            const bool af4=!q4 && !q5;
            if(!je01_slot_frozen(r.layer,r.expert) || idx<0 ||
               (q4 && (r.codec!="Q4" || r.bits!=4)) ||
               (q5 && (r.codec!="Q5" || r.bits!=5)) ||
               (af4 && (r.codec!="AF4" || r.bits!=4)) ||
               r.out!=(idx==2?2048:1408) || r.in!=(idx==2?1408:2048) ||
               r.pb>=r.pe || r.sb>=r.se ||
               r.pe>43253760 || r.se>43253760 ||
               r.pe-r.pb!=(size_t)r.out*r.in*r.bits/8 ||
               r.se-r.sb!=(size_t)r.out*(r.in/64)*4)return 0;
            if(af4) {
                if(r.bb>=r.be || r.be>43253760 ||
                   r.be-r.bb!=(size_t)r.out*(r.in/64)*4)return 0;
                nAF4++;spans.emplace_back(r.bb,r.be);
            } else {
                if(r.bb!=0 || r.be!=0)return 0;
                if(q4)nQ4++;else nQ5++;
            }
            auto pair=std::make_pair(r.layer,r.expert);
            if(seen[pair] & (1<<idx))return 0;
            seen[pair]|=1<<idx;
            r.name="je01.real.L"+std::to_string(r.layer)+".E"+
                   std::to_string(r.expert)+"."+role;
            std::string original="model.layers."+std::to_string(r.layer)+
                    ".mlp.switch_mlp."+role;
            auto native=g_tensors.find(original);
            if(native==g_tensors.end() || native->second.E!=64 ||
               native->second.bits!=4 || native->second.out!=r.out ||
               native->second.in!=r.in)return 0;
            roles[pair][idx]=r;
            spans.emplace_back(r.pb,r.pe);
            spans.emplace_back(r.sb,r.se);
            if(++total>24)return 0;
        }
        if(total!=24 || roles.size()!=8 || seen.size()!=8 ||
           nAF4!=12 || nQ4!=6 || nQ5!=6 || spans.size()!=60)return 0;
        for(const auto &s:seen)if(s.second!=7)return 0;
        std::sort(spans.begin(),spans.end());
        if(spans.front().first!=0 || spans.back().second!=43253760)return 0;
        for(size_t i=1;i<spans.size();i++)
            if(spans[i-1].second!=spans[i].first)return 0;
        int altered_experts=0;
        for(const auto &group:roles) {
            const auto &rr=group.second;
            const bool wholly_original=(rr[0].codec=="AF4" &&
                                         rr[1].codec=="AF4" &&
                                         rr[2].codec=="AF4");
            if(wholly_original) {
                if(group.first!=std::make_pair(14,63))return 0;
                continue; // preserve native original E64 GPU path
            }
            for(const auto &r:rr) {
                long bias=(r.codec=="AF4") ? (long)r.bb : -1L;
                // For original AF4, real positive bias offset is mandatory.
                // For Q4, bias==-1 synthesizes -8*scale; for Q5 no bias.
                if(mlx_gpu_bind_af(g_je01_real_native_bundle.data(),43253760,
                   r.name.c_str(),1,r.out,r.in,r.in/64,(long)r.pb,
                   (long)r.sb,bias,r.bits)!=1)return 0;
                int active=0;
                if(mlx_gpu_binding_kind(r.name.c_str(),&active)!=1 ||
                   active!=r.bits)return 0;
            }
            if(!mlx_gpu_je01_overlay_register(
                   group.first.first,group.first.second,
                   rr[0].name.c_str(),rr[1].name.c_str(),rr[2].name.c_str()))
                   return 0;
            altered_experts++;
        }
        return altered_experts==7 && mlx_gpu_je01_overlay_count()==7 ? 1:0;
    } catch(const std::exception &e) {
        std::fprintf(stderr,"JE01_REAL_CAL240_ROLEWISE_LOADER: %s\n",e.what());
        return 0;
    } catch(...) {return 0;}
}
"""


def replace_once(text, old, new, label):
    seen = text.count(old)
    if seen != 1:
        raise ValueError("JE01_CAL240_SCRATCH_ANCHOR_" + label + "_" + str(seen))
    return text.replace(old, new, 1)


def patch(cpp, c):
    need = (
        "static std::vector<uint8_t> g_je01_real_native_bundle;",
        "if (bias_off==-1)", "mlx_gpu_je01_overlay_register",
        "(t.bits!=4 && t.bits!=5)", "routed_sum=je01_scratch_adjust_routed_sum("
    )
    if any(s not in cpp for s in need):
        raise ValueError("JE01_CAL240_SCRATCH_NATIVE_Q4_Q5_AND_AF4_BINDER_NOT_READY")
    if cpp.count(CPP_BEG) != 1 or cpp.count(CPP_END) != 1:
        raise ValueError("JE01_ROLE_LOADER_CPP_SECTION_AMBIGUOUS")
    start = cpp.index(CPP_BEG)
    end = cpp.index(CPP_END, start + len(CPP_BEG))
    cpp = cpp[:start] + LOADER + "\n" + cpp[end:]
    c = replace_once(c, OLD_COUNT, NEW_COUNT, "SEVEN_CHANGED_EXPERT_COUNT")
    c = replace_once(c, OLD_MARK, NEW_MARK, "SEVEN_EXPERT_SUCCESS_MARK")
    if cpp.count("altered_experts==7") != 1 or c.count(NEW_MARK) != 1:
        raise ValueError("JE01_CAL240_PER_ROLE_OVERLAY_UNIQUENESS_FAILED")
    return cpp, c


def stage(root):
    resolved = root.resolve()
    if not root.is_dir() or root.is_symlink() or not resolved.is_relative_to(WORK.resolve()) or \
       "je01" not in str(resolved).lower():
        raise ValueError("JE01_CAL240_DISPOSABLE_SCRATCH_ROOT_REQUIRED")
    paths = [root / "mlx_moe.cpp", root / "qwen_infer.c"]
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("JE01_CAL240_NATIVE_SOURCE_FILES_MISSING")
    original = [p.read_text() for p in paths]
    changed = patch(*original)
    for p, s in zip(paths, changed):
        p.write_text(s)
    report = {
        "schema": SCHEMA,
        "status": "JE01_CAL240_ROLE_SPECIFIC_7_EXPERT_SCRATCH_ONLY_SOURCE_PATCHED",
        "original": {p.name: hashlib.sha256(s.encode()).hexdigest()
                     for p, s in zip(paths, original)},
        "patched": {p.name: hashlib.sha256(s.encode()).hexdigest()
                    for p, s in zip(paths, changed)},
        "changed_expert_overlays": 7, "fully_original_E64_experts": 1,
        "physical_selected_budget_bytes": 43_253_760,
        "physical_role_tiers": {"AF4": 12, "Q4": 6, "Q5": 6},
        "native_binder_compilation": "NOT_RUN",
        "original_E1_AF4_vs_E64_exact_gold_parity": "NOT_RUN",
        "candidate_full_27layer_NLL": "NOT_RUN",
        "production_modified": False
    }
    print(json.dumps(report, sort_keys=True))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", type=Path, required=True)
    stage(p.parse_args().scratch)
