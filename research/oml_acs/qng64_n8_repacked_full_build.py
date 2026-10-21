#!/usr/bin/env python3
"""Build source-SHA-pinned, separate Apple native BEGLIN in-place n8 lab.

No modification to tracked qwen_infer.c, production plugin, or original
safetensors. Only change: immediately after BEGLIN native qNg64 n8 encoding,
repack SAME n8 allocation from bitplanes to signed codes; dispatch separate
SIMD GEMV + batch matmul on these buffers. q4 controls untouched.
"""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"lab_local"
N8SRC=BASE/"q_full_model_20261021_v2/qwen_infer_q_lab.c"
MIXSRC=BASE/"q_full_model_mixed_20261021/qwen_infer_q_mixed_lab.c"
SOURCE_SHAS={
    "all_n8":"c03099aed130a5fd9222f0ba4d0fdf00d3e9bd1b1c203bf145ccb8290a0ad131",
    "mixed":"1671afa05644d4d79d1932fca4de2554a4a324222a13f34c72c7f64f889df590",
}
TARGET=BASE/"qng64_n8_repacked_full_20261022"
SOURCE_QNG64_SHA="cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956"
POLICIES={
 "proxy":"f65f50421cd9e9bac32bc780805942d68c3d503075ce962ce25315fdb5914cb9",
 "random":"72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5",
}
N8_REPACK_SYMBOL="beglin_qng64_n8_repack_inplace"
ST_FN="static WT *st_register_qng64_as(const char *name, int bits) {"
ENCODE_LINE="if(!packed||!scales||qng64_quantize_f32(deq,out,in,bits,packed,scales)!=0) exit(1);"

def sha(x:bytes)->str:return hashlib.sha256(x).hexdigest()
def write_private(p:Path,blob:bytes):
    if p.exists():raise FileExistsError("will not overwrite "+str(p))
    with p.open("xb") as f:
        f.write(blob);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def run(args):
    cp=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if cp.returncode:
        raise RuntimeError("compile/link failure: "+" ".join(args[:4])+"\n"+cp.stderr[-5000:])

def patch(code:str)->str:
    if code.count(ST_FN)!=1 or code.count(ENCODE_LINE)!=1:
        raise RuntimeError("loaded previous experiment is not a safe single-patch target")
    declaration="extern int beglin_qng64_n8_repack_inplace(uint8_t *,int,int);\n"
    code=code.replace(ST_FN,declaration+ST_FN,1)
    added=ENCODE_LINE+"""
    // Research-only: bitplane n8 is one byte per weight. Decode the entire
    // group into a 64-byte scratch and overwrite packed allocation IN PLACE.
    // The corresponding experimental GEMV + GEMM dispatcher understands this
    // in-memory CODE format. Never use original scalar bitplane decoder now.
    if(bits==8 && beglin_qng64_n8_repack_inplace(packed,out,in)!=0) {
        fprintf(stderr,"FATAL: research in-place n8 code conversion failed\\n");
        exit(1);
    }
"""
    code=code.replace(ENCODE_LINE,added,1)
    if code.count(N8_REPACK_SYMBOL)!=2:
        raise RuntimeError("unsafe patched source references")
    return code

def main():
    if TARGET.exists():raise FileExistsError("repacked full model experiment exists")
    srcs={"all_n8":N8SRC,"mixed":MIXSRC}
    for key,src in srcs.items():
        if not src.is_file() or sha(src.read_bytes())!=SOURCE_SHAS[key]:
            raise RuntimeError("earlier isolated BEGLIN lab source SHA mismatch")
    if sha((ROOT/"qng64_dense.c").read_bytes())!=SOURCE_QNG64_SHA:
        raise RuntimeError("original scalar native codec drift")
    native=ROOT/"research/oml_acs/qng64_n8_repacked_neon.c"
    wrapper=ROOT/"research/oml_acs/qng64_n8_repacked_lab_dispatch.c"
    for src in (native,wrapper):
        if not src.is_file():raise RuntimeError("missing experimental predecoded code")
    original=(ROOT/".build")
    others=sorted(p for p in original.glob("*.o") if p.name not in ("qwen_infer.o","qng64_dense.o"))
    if len(others)!=20:raise RuntimeError("original 20 native dependencies unexpectedly changed")
    os.umask(0o077)
    TARGET.mkdir(parents=True,mode=0o700)
    objects={}
    for name,src in srcs.items():
        code=patch(src.read_text())
        file=TARGET/f"qwen_{name}_n8_repacked.c"
        write_private(file,code.encode("utf-8"))
        obj=TARGET/f"qwen_{name}_n8_repacked.o"
        run(["clang","-O3","-w","-I",str(ROOT),"-c",str(file),"-o",str(obj)])
        objects[name]=obj
    scalar=TARGET/"qng64_original_aliased.o"
    run(["clang","-O3","-c",str(ROOT/"qng64_dense.c"),
         "-Dqng64_matvec_f32=beglin_original_scalar_matvec_f32",
         "-Dqng64_matmul_f32=beglin_original_scalar_matmul_f32",
         "-o",str(scalar)])
    neon=TARGET/"n8_repacked_simd.o"
    run(["clang","-O3","-std=c11","-I",str(ROOT),"-c",str(native),"-o",str(neon)])
    dispatch=TARGET/"n8_repacked_dispatch.o"
    run(["clang","-O3","-std=c11","-I",str(ROOT),"-c",str(wrapper),"-o",str(dispatch)])
    binaries={}
    for name in ("all_n8","mixed"):
        binfile=TARGET/f"beglin_{name}_repacked"
        run(["clang","-O3",str(objects[name]),str(scalar),str(neon),str(dispatch)]+
             [str(x) for x in others]+["-o",str(binfile),"-framework","Accelerate","-lpthread","-lm"])
        binfile.chmod(0o700)
        binaries[name]=sha(binfile.read_bytes())
    for name in ("n8","mixed_proxy","mixed_random"):
        work=TARGET/name
        work.mkdir(mode=0o700)
        for sub in ("results","eval","eval/ppl_work"):
            (work/sub).mkdir(exist_ok=True,mode=0o700)
        originalroot=BASE/"q_full_model_20261021_v2/n8" if name=="n8" else BASE/f"q_full_model_mixed_20261021/{name}"
        for path in ("prompt_ids.i32","eval/ppl_work/w0.i32","eval/ppl_work/w1.i32"):
            sourcefile=originalroot/path
            if sourcefile.stat().st_mode&0o777!=0o600:
                raise RuntimeError("source evaluation inputs not private")
            write_private(work/path,sourcefile.read_bytes())
    for key,digest in POLICIES.items():
        originalpath=BASE/f"q_full_model_mixed_20261021/policy_{key}.txt"
        if sha(originalpath.read_bytes())!=digest:
            raise RuntimeError("original source-only mixed bit allocation SHA drift")
        write_private(TARGET/f"policy_{key}.txt",originalpath.read_bytes())
    receipt=dict(schema="beglin-n8-inplace-repacked-apple-full-model/1",
          status="BUILT_NOT_YET_FULL_MODEL_TESTED",
          research_files_source_shas={name:sha((TARGET/f"qwen_{name}_n8_repacked.c").read_bytes()) for name in srcs},
          baseline_experimental_source_SHA256=SOURCE_SHAS,
          original_BEG_LIN_scalar_c_sha256=SOURCE_QNG64_SHA,
          n8_inplace_source_SHA256=sha(native.read_bytes()),
          repacked_dispatch_source_SHA256=sha(wrapper.read_bytes()),
          native_binary_SHA256=binaries,
          preserved_source_tied_FP32_head=True,only_repacked_n8_at_load=True,
          disk_bitplane_checkpoint_unchanged=True,repacked_tensors_same_byte_size=True,
          mixed_source_only_selection_unchanged=True,policy_sha256=POLICIES,
          batched_n8_call_handler="repacked GEMV per token, not original bitplane matmul",
          model="Qwen/Qwen2.5-1.5B-Instruct",
          bf16_source_rev="4ae77cb209f06199b8df1c94e21ff341332a3a89",
          original_V6_holdout_untouched=True,main_untouched=True,production_touched=False)
    write_private(TARGET/"source_build_receipt.json",(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_REPACKED_N8_FULL_MODEL_BINARIES_READY",json.dumps(receipt,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
