#!/usr/bin/env python3
"""JE01 scratch-only actual MLX 2-direction *output-interpolation* pilot.

Adds a strictly registered scalar alpha to the real original27 Expert
routed-output correction, for finite difference of per-token NLL. This is
NOT weight-space Fisher, and does not backpropagate model parameters.
Source-verified physical B/C Q4/Q5 bundles and original E64 path stay frozen.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

CPP_PIN="640da6e8afcace5e5440177ef9953732bee1cf83927680992f809a7af3f6950a"
C_PIN="fe4b82b70a68ced39fa59357e3dba1f117ff831c503d528b5777602f6f5cd7ae"
ALLOWED_ALPHA={"0.0":0.0,"1.0":1.0,"-0.05":-0.05,"+0.05":0.05}
BEG="    auto it=g_je01_scratch_overlays.find(layer);\n"
OLD="        changed=changed+weight*(alt-base);"
NEW=r'''        if(je01_alpha==1.0f) {
            changed=changed+weight*(alt-base); // exact original ON expression
        } else {
            changed=changed+weight*(alt-base)*mx::array(je01_alpha,mx::float32);
        }'''
INSERT=r'''    const char *value=std::getenv("JE01_POSTNLL_ALPHA");
    if(!value)throw std::runtime_error("JE01_POSTNLL_ALPHA_MISSING");
    std::string v(value);
    float je01_alpha=0.0f;
    if(v=="0.0")je01_alpha=0.0f;
    else if(v=="1.0")je01_alpha=1.0f;
    else if(v=="-0.05")je01_alpha=-0.05f;
    else if(v=="+0.05")je01_alpha=0.05f;
    else throw std::runtime_error("JE01_POSTNLL_ALPHA_OUTSIDE_PREDECLARED_VALUES");
    if(je01_alpha==0.0f)return routed_sum; // bit-identity control
'''


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(cpp:str,c:str):
    if cpp.count(BEG)!=1 or cpp.count(OLD)!=1 or cpp.count("JE01_CAL240_POSTNLL_ARM")!=1:
        raise ValueError("JE01_ALPHA_SOURCE_ANCHOR_NOT_EXACT_ONE")
    if c.count("JE01_CAL240_POSTNLL_BC_SEVEN_EXPERTS_REGISTERED")!=1:
        raise ValueError("JE01_ALPHA_ROLEWISE_FINAL_MARKER_MISSING")
    patched=cpp.replace(BEG,BEG+INSERT,1).replace(OLD,NEW,1)
    if patched.count("JE01_POSTNLL_ALPHA")!=3:
        raise ValueError("JE01_ALPHA_STAGED_GUARD_MISMATCH")
    if patched.count("changed=changed+weight*(alt-base);")!=1:
        raise ValueError("JE01_ALPHA_ORIGINAL_ON_EXACT_EXPRESSION_LOST")
    if c!=c:
        raise AssertionError("unreachable")
    return patched,c


def stage(source:Path):
    if source.is_symlink() or not source.is_dir() or source.resolve().name!="source" or "je01_postnll_alpha2_" not in str(source.parent):
        raise ValueError("JE01_ALPHA_SCRATCH_ISOLATED_SOURCE_REQUIRED")
    cpp=source/"mlx_moe.cpp"
    c=source/"qwen_infer.c"
    if not cpp.is_file() or not c.is_file() or cpp.is_symlink() or c.is_symlink():
        raise ValueError("JE01_ALPHA_NATIVE_FILES_MISSING")
    if sha(cpp)!=CPP_PIN or sha(c)!=C_PIN:
        raise ValueError("JE01_ALPHA_PRIOR_REAL_BC_COMPILED_SOURCE_SHA_DRIFT")
    a,b=apply(cpp.read_text(),c.read_text())
    cpp.write_text(a)
    if c.read_text()!=b:
        raise ValueError("JE01_ALPHA_NATIVE_C_NOT_UNMODIFIED")
    return {"status":"JE01_REAL_ORIGINAL27_MLX_POSTNLL_ALPHA_SOURCE_STAGED",
            "original_cpp_sha256":CPP_PIN,
            "patched_cpp_sha256":sha(cpp),
            "unchanged_c_sha256":C_PIN,
            "allowed_alpha_values":sorted(ALLOWED_ALPHA),
            "alpha0_original_AF4_parity":"NOT_RUN",
            "alpha1_original_BC_ON_parity":"NOT_RUN",
            "model_fisher_score":"NOT_MEASURED",
            "model_weights_modified":False,
            "production_modified":False}

if __name__=="__main__":
    import argparse,json
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(stage(args.source),sort_keys=True))
