#!/usr/bin/env python3
"""JE01 isolated original384 correction source adaptation for REAL Q4g64/Q5g64.

Run strictly AFTER existing original384 source, fullgold patch, JE01 E1 overlay
patch, and existing NIG n3/n5 loader patch on a disposable scratch source.
No changes to checked-in mlx_moe.cpp/h, original q4 AF checkpoint, or service.

Q4 source is BEGLIN native packed symmetric q4g64 (nibbles+scales only).
Existing MLX bits4 binder requires affine bias buffer; for bias_off==-1 ONLY
synthesize -8*scale to preserve exact q4g64 decode. Original AF4 ALWAYS
provides a real nonnegative bias_offset and follows unchanged branch.
"""
import argparse,hashlib,json
from pathlib import Path
CPP_ENTRY="int mlx_gpu_je01_overlay_load_bundle("
CPP_END="int mlx_gpu_layer_step_lazy("
REG_ENTRY="int mlx_gpu_je01_overlay_register("
REG_END="int mlx_gpu_je01_overlay_count("
NATIVE_BIND_ENTRY="int mlx_gpu_bind_af("
NATIVE_BIND_END="int mlx_gpu_snapshot_binding("
BITS_OLD="(r.bits!=3 && r.bits!=5)"
BITS_NEW="(r.bits!=4 && r.bits!=5)"
COUNT_OLD="int n3=0,n5=0,count=0;"
COUNT_NEW="int n4=0,n5=0,count=0;"
STAT_OLD="if(r.bits==3)n3++;else n5++;"
STAT_NEW="if(r.bits==4)n4++;else n5++;"
CHECK_OLD="n3!=6 || n5!=18"
CHECK_NEW="n4!=12 || n5!=12"
REGISTER_OLD="(t.bits!=3 && t.bits!=5)"
REGISTER_NEW="(t.bits!=4 && t.bits!=5)"
BIAS_PTR_OLD="void *bias_ptr = (void *)(blob + bias_off);"
BIAS_PTR_NEW="""// JE01 symmetric native q4g64 only; original AF4 real bias is unchanged.
            // bias_off==-1 is not a valid pointer and must NEVER be read.
            void *bias_ptr = bias_off==-1 ? nullptr : (void *)(blob + bias_off);"""
BIAS_LOAD_OLD="""biases = mx::array(bias_ptr, {(int)E, (int)out, (int)ng}, mx::float32,
                                 noop_deleter);"""
BIAS_LOAD_NEW="""if (bias_off==-1) {
                biases = mx::multiply(scales, mx::array(-8.0f)); // Q4 symmetric
            } else {
                biases = mx::array(bias_ptr, {(int)E, (int)out, (int)ng},
                                   mx::float32, noop_deleter);  // original affine AF4
            }"""

def replace_exact(text,old,new,label):
    n=text.count(old)
    if n!=1:raise ValueError("JE01_Q45_SCRATCH_ANCHOR_"+label+"_EXPECTED_ONCE_FOUND_"+str(n))
    return text.replace(old,new,1)

def section(text,a,b):
    if text.count(a)!=1:raise ValueError("JE01_Q45_SCOPE_ANCHOR_"+a+"_CHANGED")
    start=text.index(a)
    end=text.find(b,start+len(a))
    if end<0 or end<=start:raise ValueError("JE01_Q45_SCOPE_END_MISSING")
    return text[:start],text[start:end],text[end:]

def patch(cpp):
    before,loader,after=section(cpp,CPP_ENTRY,CPP_END)
    for label,old,new in [
        ("LOADER_4_5_BITS",BITS_OLD,BITS_NEW),
        ("LOADER_COUNTER",COUNT_OLD,COUNT_NEW),
        ("LOADER_COUNTER_INCR",STAT_OLD,STAT_NEW),
        ("LOADER_EXACT_COUNT",CHECK_OLD,CHECK_NEW),
    ]:
        loader=replace_exact(loader,old,new,label)
    cpp=before+loader+after
    before,register,after=section(cpp,REG_ENTRY,REG_END)
    register=replace_exact(register,REGISTER_OLD,REGISTER_NEW,"ROLE_BINDER_EXACT_BITS")
    cpp=before+register+after
    before,binder,after=section(cpp,NATIVE_BIND_ENTRY,NATIVE_BIND_END)
    binder=replace_exact(binder,BIAS_PTR_OLD,BIAS_PTR_NEW,"ORIGINAL_BINDER_BIAS_POINTER")
    binder=replace_exact(binder,BIAS_LOAD_OLD,BIAS_LOAD_NEW,"ORIGINAL_BINDER_BIAS_BRANCH")
    if binder.count("void *bias_ptr = bias_off==-1 ? nullptr")!=1 or \
       binder.count("if (bias_off==-1) {")!=1:
        raise ValueError("JE01_Q45_SYMMETRIC_SENTINEL_BRANCHED_INVALID")
    cpp=before+binder+after
    if cpp.count("Q4 symmetric")!=1 or cpp.count(CHECK_NEW)!=1:
        raise ValueError("JE01_Q45_SOURCE_PATCH_NOT_UNIQUE")
    return cpp

def stage(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("JE01_Q45_SCRATCH_ROOT_REQUIRED")
    f=root/"mlx_moe.cpp"
    if not f.is_file() or f.is_symlink():
        raise ValueError("JE01_Q45_SCRATCH_GPU_SOURCE_MISSING")
    original=f.read_text()
    if original.count("JE01_REAL_Q35_EIGHT_EXPERTS_FULL_MODEL_REGISTERED")!=0:
        # This is a C-file marker, not C++; scratch C++ must not carry it.
        raise ValueError("JE01_Q45_GPU_SOURCE_CORRUPT")
    if "routed_sum=je01_scratch_adjust_routed_sum(" not in original:
        raise ValueError("JE01_Q45_ORIGINAL_SCRATCH_FFN_CORRECTION_MISSING")
    altered=patch(original)
    f.write_text(altered)
    report={
       "status":"JE01_Q45_ORIGINAL384_B1_Q4SYMMETRIC_Q5_NATIVE_SCRATCH_ONLY_PATCH",
       "original_cpp_sha256":hashlib.sha256(original.encode()).hexdigest(),
       "adapted_cpp_sha256":hashlib.sha256(altered.encode()).hexdigest(),
       "original_affine_q4_bias_when_nonnegative_offset_preserved":True,
       "new_q4_symmetric_implicit_bias_only_when_bias_off_minus_one":True,
       "new_role_count":"12Q4_PLUS_12Q5",
       "production_code_modified":False,
       "native_full_model_NLL":"NOT_MEASURED"}
    print(json.dumps(report,sort_keys=True))
    return report

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--scratch",required=True,type=Path)
    v=a.parse_args()
    try:stage(v.scratch)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error":str(e)[:260]},sort_keys=True))
        raise SystemExit(2)
