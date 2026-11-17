#!/usr/bin/env python3
"""Retarget an already scratch-patched Q3Q5 Metal selected-Expert graph probe.

Select real mixed-role prelabel weight-MSE control L24 E26 (3/3/5) instead of
frequency plan L13 E22 (3/3/3). Never modifies tracked base engine source.
"""
import argparse,json
from pathlib import Path
CFILE="tools/je01_real_weight_n3n5_metal_probe.cpp"
def replace_exact(s,a,b,label,count=1):
    n=s.count(a)
    if n!=count:raise ValueError(f"JE01_SCRATCH_MIXED_ROLE_ANCHOR_{label}_DRIFT:{n}")
    return s.replace(a,b)
def stage(folder):
    p=folder/CFILE
    if not p.is_file() or p.is_symlink() or folder.is_symlink():
        raise ValueError("JE01_SCRATCH_REAL_PROBE_NOT_FOUND")
    s=p.read_text()
    s=replace_exact(s,"if(p.layer==13 && p.expert==22)","if(p.layer==24 && p.expert==26)","REAL_SELECTION")
    s=replace_exact(s,"gate->bits==3 && up->bits==3 && down->bits==3",
                    "gate->bits==3 && up->bits==3 && down->bits==5","MIXED_ROLE")
    s=replace_exact(s,"mlx_gpu_je01_overlay_register(13,22,",
                    "mlx_gpu_je01_overlay_register(24,26,","REGISTER",2)
    s=replace_exact(s,"input(2048,13,22,","input(2048,24,26,","INPUT")
    s=replace_exact(s,"std::array<int32_t,6> selected={22,1,2,3,4,5};",
                    "std::array<int32_t,6> selected={26,1,2,3,4,5};","SELECTED_ROUTING")
    s=replace_exact(s,"std::array<int32_t,6> not_selected={61,1,2,3,4,5};",
                    "std::array<int32_t,6> not_selected={61,1,2,3,4,5};","UNSELECTED")
    s=replace_exact(s,"mlx_gpu_je01_overlay_probe(13,","mlx_gpu_je01_overlay_probe(24,","PROBE",3)
    s=replace_exact(s,'"\\"selected_layer\\":13,\\"selected_expert\\":22,"',
                    '"\\"selected_layer\\":24,\\"selected_expert\\":26,"',"JSON_SELECTED")
    s=replace_exact(s,'"\\"q3_actual_pretrained_roles_bound\\":3,"',
                    '"\\"actual_pretrained_roles_bound\\":3,\\"q3_roles\\":2,\\"q5_roles\\":1,"',"ROLE_COUNTS")
    s=replace_exact(s,'"\\"status\\":\\"PASS_REAL_Q3_GPU_CORRECTION_WITH_ORIGINAL_Q4_UNSELECTED_FALLBACK\\","',
                    '"\\"status\\":\\"PASS_REAL_MIXED_Q3Q5_ROLE_GPU_CORRECTION_AND_BASELINE_FALLBACK\\","',"STATUS")
    p.write_text(s)
    return {"status":"PASS_SCRATCH_L24E26_REAL_SOURCE_MIXED_ROLE_GRAPH_PATCH_ONLY",
            "layers":[24],"experts":[26],"role_bits":[3,3,5],
            "real_model_NLL":"NOT_MEASURED","production_modified":False}
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--folder",type=Path,required=True)
    opt=a.parse_args()
    try:print(json.dumps(stage(opt.folder),sort_keys=True))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:260]}))
        raise SystemExit(2)
