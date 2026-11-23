#!/usr/bin/env python3
"""Research-only original 384 B1 cbatch Expert GPU overlay source stage.

Modifies ONLY an already staged disposable JE01 source tree.
"""
import argparse,hashlib,json
from pathlib import Path

PAIRS=((12,13),(13,22),(13,28),(14,63),(20,17),(21,61),(24,26),(24,60))
HEADER_ANCHOR="#ifdef __cplusplus\n}\n#endif"
CXX_ANCHOR="int mlx_gpu_layer_step_lazy(int l, int pos, int is_dense,"
CB_ANCHOR="            mx::array routed_sum = mx::sum(weighted, std::vector<int>{1}, false);    // {A,HIDDEN}"
CXX_ADDITION=r'''
// JE01 isolated scratch-only overlay (never committed into root MLX backend).
struct JE01ScratchOverlay {std::array<std::string,3> role;};
static std::unordered_map<int,std::unordered_map<int,JE01ScratchOverlay>> g_je01_scratch_overlays;
static bool je01_slot_frozen(int l,int e) {
    static constexpr int slots[8][2]={{12,13},{13,22},{13,28},{14,63},
                                       {20,17},{21,61},{24,26},{24,60}};
    for(const auto &p:slots) if(p[0]==l && p[1]==e) return true;
    return false;
}
int mlx_gpu_je01_overlay_register(int l,int e,const char *gate,
                                   const char *up,const char *down) {
    if(!mlx_gpu_available() || !je01_slot_frozen(l,e) ||
       !gate || !up || !down || !*gate || !*up || !*down ||
       g_layer_n_experts!=64 || g_layer_top_k!=6 || g_layer_group!=64 ||
       g_layer_hidden<=0 || g_layer_im_dim<=0) return 0;
    std::array<std::string,3> names={gate,up,down};
    if(names[0]==names[1] || names[0]==names[2] || names[1]==names[2] ||
       g_je01_scratch_overlays[l].count(e)) return 0;
    for(int role=0;role<3;role++) {
        auto it=g_tensors.find(names[role]);
        if(it==g_tensors.end()) return 0;
        const QTensor &t=it->second;
        long out=role==2 ? g_layer_hidden : g_layer_im_dim;
        long in=role==2 ? g_layer_im_dim : g_layer_hidden;
        if(t.E!=1 || t.out!=out || t.in!=in || t.ng!=in/64 ||
           (t.bits!=3 && t.bits!=5) ||
           g_dtensors.count(names[role]) || g_qng64_tensors.count(names[role]))
            return 0;
    }
    g_je01_scratch_overlays[l].emplace(e,JE01ScratchOverlay{names});
    return 1;
}
int mlx_gpu_je01_overlay_count(void) {
    int n=0;
    for(const auto &layer:g_je01_scratch_overlays) n+=(int)layer.second.size();
    return n;
}
void mlx_gpu_je01_overlay_clear(void) {g_je01_scratch_overlays.clear();}

static mx::array je01_scratch_adjust_routed_sum(
    int layer,const mx::array &h2,const mx::array &top_idx,
    const mx::array &top_wgt,const mx::array &down_flat,
    const mx::array &routed_sum) {
    auto it=g_je01_scratch_overlays.find(layer);
    if(it==g_je01_scratch_overlays.end()) return routed_sum; // immutable baseline path
    if(h2.ndim()!=2 || h2.shape(0)!=1 || h2.shape(1)!=g_layer_hidden ||
       top_idx.ndim()!=2 || top_idx.shape(0)!=1 || top_idx.shape(1)!=6 ||
       top_wgt.ndim()!=2 || top_wgt.shape(0)!=1 || top_wgt.shape(1)!=6 ||
       down_flat.ndim()!=3 || down_flat.shape(0)!=1 ||
       down_flat.shape(1)!=6 || down_flat.shape(2)!=g_layer_hidden)
        throw std::runtime_error("JE01_SCRATCH_OVERLAY_B1_SHAPE_FENCE");
    mx::array changed=routed_sum;
    for(const auto &row:it->second) {
        int expert=row.first;
        const auto &names=row.second.role;
        // Preserve original topk and top-weight; correct ONLY selected outputs.
        auto mask=mx::astype(mx::equal(top_idx,mx::array(expert,mx::int32)),
                             mx::float32);
        auto weight=mx::sum(top_wgt*mask,std::vector<int>{1},true);
        auto base=mx::sum(down_flat*mx::expand_dims(mask,std::vector<int>{-1}),
                          std::vector<int>{1},false);
        auto g=lazy_matvec_e0(names[0].c_str(),h2);
        auto u=lazy_matvec_e0(names[1].c_str(),h2);
        auto alt=lazy_matvec_e0(names[2].c_str(),lazy_silu(g)*u);
        changed=changed+weight*(alt-base);
    }
    return changed;
}
int mlx_gpu_je01_overlay_probe(int layer,
    const float *x,const int32_t *idx,const float *w,
    const float *baseline_down,const float *baseline_sum,float *result) {
    if(!mlx_gpu_available() || !x || !idx || !w || !baseline_down ||
       !baseline_sum || !result || g_layer_hidden<=0 ||
       g_layer_top_k!=6) return 0;
    try {
        const int h=g_layer_hidden;
        mx::array h2((void*)x,{1,h},mx::float32,noop_deleter);
        mx::array ids((void*)idx,{1,6},mx::int32,noop_deleter);
        mx::array weights((void*)w,{1,6},mx::float32,noop_deleter);
        mx::array base((void*)baseline_down,{1,6,h},mx::float32,noop_deleter);
        mx::array sum((void*)baseline_sum,{1,h},mx::float32,noop_deleter);
        auto out=je01_scratch_adjust_routed_sum(layer,h2,ids,weights,base,sum);
        mx::eval(out);
        std::memcpy(result,out.data<float>(),sizeof(float)*(size_t)h);
        return 1;
    } catch(const std::exception &ex) {
        std::fprintf(stderr,"JE01_OVERLAY_PROBE_FAILED: %s\n",ex.what());
        return 0;
    } catch(...) {return 0;}
}
'''
HEADER_ADDITION=r'''
// JE01 research-only, disposable original384 cbatch overlay API.
int mlx_gpu_je01_overlay_register(int layer,int expert,const char *gate,
                                   const char *up,const char *down);
int mlx_gpu_je01_overlay_count(void);
void mlx_gpu_je01_overlay_clear(void);
int mlx_gpu_je01_overlay_probe(int layer,const float *x,const int32_t *idx,
   const float *w,const float *baseline_down,const float *baseline_sum,float *out);
'''

def once(s,needle,replacement,label):
    n=s.count(needle)
    if n!=1:raise ValueError("JE01_UNPINNED_SOURCE_"+label+":"+str(n))
    return s.replace(needle,replacement,1)

def patch(cpp,header):
    cpp=once(cpp,"#include <algorithm>","#include <algorithm>\n#include <array>","ARRAY_INCLUDE")
    cpp=once(cpp,CXX_ANCHOR,CXX_ADDITION+"\n"+CXX_ANCHOR,"ONLY_CPP_LAZY_ENTRY")
    cpp=once(cpp,CB_ANCHOR,CB_ANCHOR+"\n"+
             "            routed_sum=je01_scratch_adjust_routed_sum(l,h2,top_idx,top_wgt,down_flat,routed_sum);",
             "ONLY_ORIGINAL_ONLINE_CBATCH_HOOK")
    header=once(header,HEADER_ANCHOR,HEADER_ADDITION+"\n"+HEADER_ANCHOR,"C_HEADER_API")
    if cpp.count("routed_sum=je01_scratch_adjust_routed_sum(")!=1:
        raise ValueError("JE01_SCRATCH_OVERLAY_NOT_UNIQUE")
    return cpp,header

def stage(folder):
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("JE01_SCRATCH_FOLDER_NOT_EXIST")
    paths=[folder/"mlx_moe.cpp",folder/"mlx_moe.h"]
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("JE01_SCRATCH_SOURCE_NOT_FOUND")
    originals=[p.read_text() for p in paths]
    patched=patch(*originals)
    for p,data in zip(paths,patched):p.write_text(data)
    return {"status":"ONLY_ISOLATED_ORIGINAL384_GPU_CPP_AND_HEADER_PATCHED",
            "original_sha256":{p.name:hashlib.sha256(s.encode()).hexdigest() for p,s in zip(paths,originals)},
            "patched_sha256":{p.name:hashlib.sha256(s.encode()).hexdigest() for p,s in zip(paths,patched)},
            "frozen_slots":[list(x) for x in PAIRS],
            "cbatch_hook_count":patched[0].count("routed_sum=je01_scratch_adjust_routed_sum("),
            "root_repository_files_modified":False,"real_pretrained_model_forward":"NOT_RUN",
            "real_gold_NLL":"NOT_MEASURED"}
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--staged-source",type=Path,required=True)
    v=a.parse_args()
    try:print(json.dumps(stage(v.staged_source),sort_keys=True))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:260]}))
        raise SystemExit(2)
