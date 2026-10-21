#!/usr/bin/env python3
"""Build an isolated BEGLIN 196-tensor kernel-aware precision research worker.

Original source and model are SHA-pinned read-only. A COPY of the previously
independently audited in-place n8 Qwen lab is changed ONLY to accept any
role counts summing to 28, instead of mandatory 4 n8 per role.
Selection was frozen earlier using source weights + native kernel costs only.
"""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_kernel_aware_native_20261022"
PREVIOUS=ROOT/"lab_local/qng64_n8_repacked_full_20261022"
SOURCE=PREVIOUS/"qwen_mixed_n8_repacked.c"
SELECTED=ROOT/"lab_local/q_kernel_aware_allocator_20261022"
SEL=SELECTED/"kernel_aware_source_only_selection.json"
EXPECTED_SOURCE_SHA="990439f892f6bdf6c5eb86524133eb07b2d6ea868c76a8fa7cf8d16932c10b87"
EXPECTED_SELECTION_SHA="5841c6505cb4e5805b228e7af5609ecc11f38817fc7bccd34e5272f93f400df1"
CODE_TARGET="if (j<0 || layer<0 || layer>=g_cfg.nl || count[j]>=4 || q_lab_mix_n8[ids[j]][layer])"
CODE_REPLACE="if (j<0 || layer<0 || layer>=g_cfg.nl || count[j]>=g_cfg.nl || q_lab_mix_n8[ids[j]][layer])"
OLD_QUOTA="""        for(int i=0;i<7;i++) {
            if(count[i]!=4) {fprintf(stderr,"FATAL: expected 4 n8 projections per each role\\n");exit(1);}
        }"""
NEW_QUOTA="""        int total_n8=0;
        for(int i=0;i<7;i++) {
            total_n8 += count[i];
        }
        if(total_n8!=28) {
            fprintf(stderr,"FATAL: expected exactly 28 n8 tensor roles, got %d\\n",total_n8);
            exit(1);
        }"""
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def put(p:Path,b:bytes):
    if p.exists():raise FileExistsError("no overwrite previous kernel allocator experiment")
    with p.open("xb") as f:
        f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def build():
    if LAB.exists():raise FileExistsError("native allocator research experiment already exists")
    if sha(SOURCE.read_bytes())!=EXPECTED_SOURCE_SHA:
        raise RuntimeError("native BEGLIN mixed n8 source changed")
    if sha(SEL.read_bytes())!=EXPECTED_SELECTION_SHA or SEL.stat().st_mode&0o777!=0o600:
        raise RuntimeError("source-only policy selection SHA changed")
    selection=json.loads(SEL.read_text())
    candidate=selection["policies"]["kernel_aware"]
    chosen_file=SELECTED/"kernel_aware_n8_roles.txt"
    if sha(chosen_file.read_bytes())!=candidate["policy_sha256"]:
        raise RuntimeError("candidate tensor IDs changed after source-only selection")
    entries=[line.split() for line in chosen_file.read_text().splitlines()]
    if len(entries)!=28 or len({(role,idx) for role,idx in entries})!=28:
        raise RuntimeError("frozen allocator selected wrong number of tensor promotions")
    if [sum(1 for item in entries if item[0]==r) for r in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")] != candidate["counts"]:
        raise RuntimeError("frozen chosen role counts differ from candidate")
    source=SOURCE.read_text()
    for before,after in [(CODE_TARGET,CODE_REPLACE),(OLD_QUOTA,NEW_QUOTA)]:
        if source.count(before)!=1:raise RuntimeError("BEGLIN allocation research patch site not unique")
        source=source.replace(before,after,1)
    originals=list((ROOT/".build").glob("*.o"))
    preserved=sorted(p for p in originals if p.name not in ("qwen_infer.o","qng64_dense.o"))
    if len(preserved)!=20:raise RuntimeError("missing original 20 native build dependency objects")
    links=[
      PREVIOUS/"qng64_original_aliased.o",
      PREVIOUS/"n8_repacked_simd.o",
      PREVIOUS/"n8_repacked_dispatch.o",
    ]
    if any(not p.is_file() for p in links):raise RuntimeError("original repacked qng64 dependencies absent")
    os.umask(0o077);LAB.mkdir(parents=True,mode=0o700)
    srcfile=LAB/"qwen_kernel_aware_copy.c"
    put(srcfile,source.encode())
    obj=LAB/"qwen_kernel_aware_copy.o"
    subprocess.run(["clang","-O3","-w","-I",str(ROOT),"-c",str(srcfile),"-o",str(obj)],check=True)
    exe=LAB/"beglin_kernel_aware_native_Qwen"
    subprocess.run(["clang","-O3",str(obj)]+[str(p) for p in links+preserved]+
        ["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],check=True)
    exe.chmod(0o700)
    put(LAB/"chosen_n8_policy.txt",chosen_file.read_bytes())
    directory=LAB/"chosen"
    directory.mkdir(mode=0o700)
    (directory/"results").mkdir(mode=0o700)
    (directory/"eval").mkdir(mode=0o700)
    (directory/"eval/ppl_work").mkdir(mode=0o700)
    originals=PREVIOUS/"mixed_random"
    for rel in ("prompt_ids.i32","eval/ppl_work/w0.i32","eval/ppl_work/w1.i32"):
        f=originals/rel
        if not f.is_file() or (f.stat().st_mode&0o777)!=0o600:
            raise RuntimeError("public synthetic prompt not secure or absent")
        put(directory/rel,f.read_bytes())
    receipt=dict(schema="beglin-kernel-aware-precision-native-build/1",
         status="NATIVE_BUILT_NOT_YET_EVALUATED",
         model="Qwen2.5-1.5B-Instruct 28 layers",
         bf16_source_model_revision="4ae77cb209f06199b8df1c94e21ff341332a3a89",
         full_BEG_original_source_readonly=True,
         input_source_C_sha256=EXPECTED_SOURCE_SHA,
         output_research_C_sha256=sha(srcfile.read_bytes()),
         native_binary_sha256=sha(exe.read_bytes()),
         preselected_allocator_sha256=EXPECTED_SELECTION_SHA,
         preselected_28_tensor_policy_sha256=candidate["policy_sha256"],
         n8_count=28,n4_count=168,
         predicted_proxy_decode_ms=candidate["native_approx_total_decode_ms"],
         role_counts=candidate["counts"],
         policy_memory_bytes=candidate["packed_projection_memory_bytes"],
         original_BF16_and_original_200_C1D_and_V6_24_not_opened=True,
         prior_blocked_output_sensitivity_not_retried=True,
         production_modified=False,main_modified=False,immutable_release_activated=False)
    put(LAB/"build_receipt.json",(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
    print("KERNEL_AWARE_FULL_QWEN_BIN_READY",json.dumps(receipt,sort_keys=True),flush=True)
if __name__=="__main__":build()
