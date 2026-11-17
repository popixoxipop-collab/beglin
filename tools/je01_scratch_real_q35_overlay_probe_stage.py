#!/usr/bin/env python3
"""JE01 isolated Metal source probe of real-weight original384 correction algebra.

After context384 + overlay scratch patches, instrument a second source-clone
of the existing real-weight FFN 24-role GPU comparator. No production edits.
"""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ANCHOR='        size_t active=0,peak=0,cache=0;\n        mlx_gpu_report_memory(&active,&peak,&cache);'
CMAKE_TAIL=r'''
# JE01 original384 cbatch scratch E1 overlay proof, never served.
add_executable(je01_real_q35_overlay_graph_probe
  tools/je01_real_weight_n3n5_metal_probe.cpp
  $<TARGET_OBJECTS:mlx_moe_obj>)
target_include_directories(je01_real_q35_overlay_graph_probe PRIVATE .)
target_link_libraries(je01_real_q35_overlay_graph_probe PRIVATE mlx)
'''
CPP_PROBE=r'''
        // JE01: original 64-Expert router-selected L13 E22 real q3 alternative.
        require(mlx_gpu_layer_config(2048,1408,1408,64,0,6,64,0)==1,
                "JE01_OVERLAY_LAYER_CONFIG_REFUSED");
        const Region *gate=nullptr,*up=nullptr,*down=nullptr;
        for(const auto &p:entries) if(p.layer==13 && p.expert==22) {
            if(p.role=="gate_proj")gate=&p;
            if(p.role=="up_proj")up=&p;
            if(p.role=="down_proj")down=&p;
        }
        require(gate && up && down && gate->bits==3 && up->bits==3 && down->bits==3,
                "JE01_OVERLAY_FROZEN_EXPERT_ROLES_MISSING");
        require(mlx_gpu_je01_overlay_count()==0,"JE01_OVERLAY_NOT_INITIAL_EMPTY");
        require(mlx_gpu_je01_overlay_register(99,99,gate->name.c_str(),
                up->name.c_str(),down->name.c_str())==0,
                "JE01_OVERLAY_ALLOWED_UNFROZEN_EXPERT");
        require(mlx_gpu_je01_overlay_register(13,22,gate->name.c_str(),
                up->name.c_str(),down->name.c_str())==1,
                "JE01_OVERLAY_REAL_Q3_REGISTER_FAIL");
        require(mlx_gpu_je01_overlay_register(13,22,gate->name.c_str(),
                up->name.c_str(),down->name.c_str())==0,
                "JE01_OVERLAY_DUPLICATE_NOT_BLOCKED");
        require(mlx_gpu_je01_overlay_count()==1,"JE01_OVERLAY_REGISTER_COUNT");
        auto x_in=input(2048,13,22,"gate_proj");
        auto CPU_gate=reference(blob,*gate,x_in);
        auto CPU_up=reference(blob,*up,x_in);
        std::vector<float> sw(CPU_gate.size());
        for(size_t i=0;i<sw.size();i++)
            sw[i]=CPU_gate[i]/(1.0f+std::exp(-CPU_gate[i]))*CPU_up[i];
        auto CPU_alt=reference(blob,*down,sw);
        std::array<int32_t,6> selected={22,1,2,3,4,5};
        std::array<float,6> prob={0.3f,0.2f,0.18f,0.14f,0.1f,0.08f};
        std::vector<float> old_down((size_t)6*2048),old_sum(2048,0),out_mixed(2048);
        for(int k=0;k<6;k++)for(int i=0;i<2048;i++) {
            float v=(float)(0.02*std::sin((k+1)*0.7+(i+1)*0.013));
            old_down[(size_t)k*2048+i]=v;
            old_sum[i]+=prob[k]*v;
        }
        require(mlx_gpu_je01_overlay_probe(13,x_in.data(),selected.data(),
                prob.data(),old_down.data(),old_sum.data(),out_mixed.data())==1,
                "JE01_REAL_SOURCE_GPU_OVERLAY_PROBE_FAILURE");
        double err2=0,ref2=0,max_abs=0;
        for(int i=0;i<2048;i++) {
            double expected=old_sum[i]+(double)prob[0]*
                         ((double)CPU_alt[i]-old_down[i]);
            double diff=(double)out_mixed[i]-expected;
            err2+=diff*diff;ref2+=expected*expected;
            max_abs=std::max(max_abs,std::abs(diff));
        }
        double rel_err=std::sqrt(err2/std::max(ref2,1e-25));
        require(rel_err<0.03,"JE01_OVERLAY_GPU_VS_INDEPENDENT_CPU_FFN_REL_ERROR");
        // Selected original Expert absent => bitwise unchanged baseline routed_sum.
        std::array<int32_t,6> not_selected={61,1,2,3,4,5};
        std::vector<float> inactive(2048);
        require(mlx_gpu_je01_overlay_probe(13,x_in.data(),not_selected.data(),
                prob.data(),old_down.data(),old_sum.data(),inactive.data())==1,
                "JE01_REAL_SOURCE_INACTIVE_PROBE_FAILURE");
        double no_route_max=0;
        for(int i=0;i<2048;i++)
            no_route_max=std::max(no_route_max,std::abs((double)inactive[i]-old_sum[i]));
        require(no_route_max<1e-7,"JE01_INACTIVE_EXPERT_FALLBACK_CHANGED");
        // Clearing registry must restore unaltered graph results.
        mlx_gpu_je01_overlay_clear();
        require(mlx_gpu_je01_overlay_count()==0,"JE01_OVERLAY_CLEAR_FAILED");
        std::vector<float> cleared(2048);
        require(mlx_gpu_je01_overlay_probe(13,x_in.data(),selected.data(),
                prob.data(),old_down.data(),old_sum.data(),cleared.data())==1,
                "JE01_CLEARED_OVERLAY_PROBE_FAILED");
        double cleared_max=0;
        for(int i=0;i<2048;i++)
            cleared_max=std::max(cleared_max,std::abs((double)cleared[i]-old_sum[i]));
        require(cleared_max<1e-7,"JE01_OVERLAY_CLEAR_BASELINE_NOT_RESTORED");
        std::printf(
            "{\"schema\":\"beglin-je01-original384-cbatch-real-BF16-E1-overlay-math/1\","
            "\"status\":\"PASS_REAL_Q3_GPU_CORRECTION_WITH_ORIGINAL_Q4_UNSELECTED_FALLBACK\","
            "\"selected_layer\":13,\"selected_expert\":22,"
            "\"q3_actual_pretrained_roles_bound\":3,"
            "\"relative_error_vs_independent_CPU_reference\":%.10g,"
            "\"max_abs_error\":%.10g,"
            "\"nonrouted_expert_baseline_max_abs_diff\":%.10g,"
            "\"cleared_baseline_max_abs_diff\":%.10g,"
            "\"full_original_27layer_routed_forward\":\"NOT_RUN\","
            "\"actual_gold_NLL\":\"NOT_MEASURED\","
            "\"production_modified\":false}\n",
            rel_err,max_abs,no_route_max,cleared_max);
'''

def once(s,a,b,label):
    if s.count(a)!=1:raise ValueError("JE01_SCRATCH_REAL_PROBE_ANCHOR_"+label)
    return s.replace(a,b,1)
def stage(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("JE01_SCRATCH_ROOT_INVALID")
    cpp=root/"tools/je01_real_weight_n3n5_metal_probe.cpp"
    cm=root/"CMakeLists.txt"
    if any(not p.is_file() or p.is_symlink() for p in (cpp,cm)):
        raise ValueError("JE01_SCRATCH_REAL_WEIGHT_SOURCE_MISSING")
    data=cpp.read_text()
    if "int success=0,bits3=0,bits5=0;" not in data:
        raise ValueError("JE01_EXPECTED_Q3Q5_REAL_WEIGHT_PROBE_NOT_PRESENT")
    cpp.write_text(once(data,ANCHOR,CPP_PROBE+"\n"+ANCHOR,"CPP"))
    cm.write_text(cm.read_text()+"\n"+CMAKE_TAIL)
    return {"status":"JE01_REAL_PRETRAINED_Q35_ONE_EXPERT_OVERLAY_TEST_STAGED_ONLY",
            "injected_real_weight_probe_count":cpp.read_text().count(
             "JE01_OVERLAY_GPU_VS_INDEPENDENT_CPU_FFN_REL_ERROR"),
            "original_27layer_forward":"NOT_RUN","model_gold_NLL":"NOT_MEASURED"}
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--staged-source",required=True,type=Path)
    v=parser.parse_args()
    try:print(json.dumps(stage(v.staged_source),sort_keys=True))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:200]}))
        raise SystemExit(2)
