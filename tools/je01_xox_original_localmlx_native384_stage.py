#!/usr/bin/env python3
"""JE01 reproduce isolated B1 context384 from XOX's pinned CLEAN original source.

Exports tracked Git HEAD into a NEW research directory and patches exact
source markers only. Does not edit XOX original repo, server, model or MLX.
Local version ties the binary to XOX's own installed mlx C++ ABI, eliminating
the GitHub hosted-arm64's incompatible MLX ABI.
"""
import hashlib,json,subprocess,pathlib,os
from datetime import datetime, timezone
R=pathlib.Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original384_20261117")
SRC=pathlib.Path("/Users/xox/beglin")
OUT=R/"src_xox_mlxabi384"
HEAD="e827b522959a5bb01852686c76c4e65ac16b63ba"
ORIG={
"qwen_infer.c":"74e95c60fb2b4781041e3af1f2128df60c246afc0e064fe4466d5ea0bed34a1c",
"mlx_moe.cpp":"f06e977a93b26f68b70fb5ba4f6c17bb8e637fe6aff72510886f8ad67576ffe9",
"CMakeLists.txt":"bcb355ec3faaa73f6e05569c3183d79fc6302213427af5608fcfe7c62a8cfe70"}
def sha(b):return hashlib.sha256(b).hexdigest()
def patch_exact(s,old,new,n,name):
    if s.count(old)!=n:raise ValueError("JE01_SOURCE_ANCHOR_DRIFT:"+name+":"+str(s.count(old)))
    return s.replace(old,new)
head=subprocess.check_output(["/usr/bin/git","-C",str(SRC),"rev-parse","HEAD"],text=True,timeout=12).strip()
if head!=HEAD:raise SystemExit("JE01_XOX_SOURCE_COMMIT_DRIFT")
if subprocess.check_output(["/usr/bin/git","-C",str(SRC),"status","--porcelain","-uno"],text=True).strip():
    raise SystemExit("JE01_XOX_ORIGINAL_SOURCE_DIRTY")
if OUT.exists() or OUT.is_symlink():raise SystemExit("JE01_SOURCE_SCRATCH_ALREADY_EXISTS")
for k,v in ORIG.items():
    if sha((SRC/k).read_bytes())!=v:raise SystemExit("JE01_ORIGINAL_SOURCE_CONTENT_SHA_CHANGED:"+k)
OUT.mkdir(mode=0o700)
git=subprocess.Popen(["/usr/bin/git","-C",str(SRC),"archive","--format=tar","HEAD"],
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)
tar=subprocess.run(["/usr/bin/tar","-xf","-","-C",str(OUT)],
                   stdin=git.stdout,capture_output=True,timeout=60)
git.stdout.close()
rc=git.wait(timeout=12)
if rc or tar.returncode:raise SystemExit("JE01_ARCHIVE_EXTRACT_FAIL")
for k,v in ORIG.items():
    if sha((OUT/k).read_bytes())!=v:raise SystemExit("JE01_GIT_ARCHIVE_IS_NOT_HEAD:"+k)
q=(OUT/"qwen_infer.c").read_text()
q=patch_exact(q,"#define MOE_MAXPOS 160","#define MOE_MAXPOS 384",1,"cpu_pos")
q=patch_exact(q,"#define MOE_CBATCH_MAXPOS 32","#define MOE_CBATCH_MAXPOS 384",1,"cb_pos")
q=patch_exact(q,"static int g_moe_cb_slots_cap = MOE_BATCH_MAX;",
    "static int g_moe_cb_slots_cap = 1; // JE01 local original model, isolated B1",1,"slot_cap")
q=patch_exact(q,'if (B < 1 || B > MOE_BATCH_MAX) { fprintf(stderr, "FATAL: QWEN_MOE_CB_SLOTS',
    'if (B != 1) { fprintf(stderr, "FATAL: QWEN_MOE_CB_SLOTS',4,"all_cpu_gpu_cb_launch_gates")
m=(OUT/"mlx_moe.cpp").read_text()
m=patch_exact(m,"#define MLA_L0_MAXPOS 32","#define MLA_L0_MAXPOS 384",1,"metal_kv_pos")
m=patch_exact(m,"if (B < 1 || B > 64) return 0;",
    "if (B != 1) return 0; // JE01 isolated B1",1,"metal_batch_fence")
m=patch_exact(m,"if (A < 1 || A > 64 || l < 0 || l >= MLA_MAXLAYERS) return 0;",
    "if (A != 1 || !slot || slot[0] != 0 || !spos || l < 0 || l >= MLA_MAXLAYERS) return 0; // JE01 B1 one token per step",
    1,"ragged_layer_one_column")
c=(OUT/"CMakeLists.txt").read_text()
c=patch_exact(c,"target_link_libraries(qwen_infer_gpu PRIVATE mlx",
    "set_target_properties(qwen_infer_gpu PROPERTIES OUTPUT_NAME qwen_infer_gpu_je01_context384_xoxabi)\n"
    "target_link_libraries(qwen_infer_gpu PRIVATE mlx",1,"output_name")
for name,s in [("qwen_infer.c",q),("mlx_moe.cpp",m),("CMakeLists.txt",c)]:
    (OUT/name).write_text(s)
report={"schema":"JE01-XOX-PINNED-ORIGINAL-AF-NATIVE384-SOURCE-STAGE/1",
        "status":"STAGED_NO_MODEL_RUN","original_source_commit":HEAD,
        "original_file_sha256":ORIG,
        "isolated_file_sha256":{k:sha((OUT/k).read_bytes()) for k in ORIG},
        "local_mlx_lib":"/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/lib/libmlx.dylib",
        "source_original_modified":False,"production_route_modified":False,
        "native_positions":384,"native_max_slots":1,"prefill_budget_expected":1,
        "full_original_model_forward":"NOT_RUN","promotion_allowed":False}
report["receipt_sha256"]=sha(json.dumps(report,sort_keys=True,separators=(",",":")).encode())
with (R/"je01_xox_stage_receipt.json").open("x") as fd:json.dump(report,fd,sort_keys=True,indent=2)
print(json.dumps(report,sort_keys=True),flush=True)
