#!/usr/bin/env python3
"""Build strictly isolated 22-layer Llama native mixed BEGLIN Q4/n8 CPU worker.

Copies SHA-audited Qv5 mixed-capable C source, which already implements HF
llama architecture. Replaces ONLY 28-layer/28-n8 validations with 22/22.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
BEG=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
SOURCE=OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/qv5_mixed.c"
BUILD=OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/mixed_build_receipt.json"
OUT=ROOT/"lab_local/qv7_tinyllama_20261022/native_mixed_22"
PRE=ROOT/"configs/research/BEGLIN_QV7_CROSSFAMILY_CALIBRATION_PRELOCK_20261022.json"
def sha(b):return hashlib.sha256(b).hexdigest()
def put(path,data):
    with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def main():
    if OUT.exists():raise FileExistsError("Llama mixed 22 research build already created")
    if json.loads(PRE.read_text())["target_n8_count"]!=22:
        raise RuntimeError("not precommitted TinyLlama count")
    receipt=json.loads(BUILD.read_text())
    original=SOURCE.read_bytes()
    if sha(original)!=receipt["new_C_SHA256"] or not receipt["original_ppl_unchanged"]:
        raise RuntimeError("original audited Qv5 C source drift")
    decoded=original.decode()
    patches=(
      ("if (!spec || !*spec || g_cfg.nl!=28)", "if (!spec || !*spec || g_cfg.nl!=22)"),
      ("if(total_n8!=28)", "if(total_n8!=22)"),
      ("expected exactly 28 n8 tensor roles", "expected exactly 22 n8 tensor roles"),
    )
    for a,b in patches:
        if decoded.count(a)!=1:raise RuntimeError("unambiguous 22layer policy hardening location missing "+a)
        decoded=decoded.replace(a,b,1)
    if "BEGLIN_Q_LAB_N8_ROLES_FILE" not in decoded or '"masked_choice_qv5"' not in decoded:
        raise RuntimeError("original C model evaluation kernel source missing")
    os.umask(0o077);OUT.mkdir(mode=0o700,parents=True)
    file=OUT/"qv7_tinyllama_original_qv5_mixed_copy.c"
    put(file,decoded.encode())
    obj=OUT/"tinyllama_original_mixed.o"
    exe=OUT/"beglin_tinyllama_22layer_mixed"
    subprocess.run(["clang","-O3","-w","-I",str(BEG),"-c",str(file),"-o",str(obj)],check=True,timeout=120)
    original_links=[
        BEG/"lab_local/qng64_n8_repacked_full_20261022/qng64_original_aliased.o",
        BEG/"lab_local/qng64_n8_repacked_full_20261022/n8_repacked_simd.o",
        BEG/"lab_local/qng64_n8_repacked_full_20261022/n8_repacked_dispatch.o",
    ]
    extra=sorted(p for p in (BEG/".build").glob("*.o") if p.name not in ("qwen_infer.o","qng64_dense.o"))
    if len(extra)!=20 or any(not p.is_file() for p in original_links):
        raise RuntimeError("native Q4/n8 original dependency object closure drift")
    subprocess.run(["clang","-O3",str(obj)]+list(map(str,original_links+extra))+
        ["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],check=True,timeout=120)
    exe.chmod(0o700)
    receipt_out={
       "schema":"BEGLIN-Qv7-native-22-layer-Llama-repacked-n8-vs-Q4-build/1",
       "status":"SOURCE_PINNED_EXACTLY_22_N8_TIERS_NATIVE_C_BIN_BUILT",
       "model_family":"llama","target_model_revision":"fe8a4ea1ffedaf415f4da2f062534de366a451e6",
       "original_Qv5_audited_C_SHA256":sha(original),
       "only_3_source_anchor_changes":True,
       "new_native_Llama_C_SHA256":sha(decoded.encode()),
       "native_binary_SHA256":sha(exe.read_bytes()),
       "n8_target":22,"all_native_original_Q4_n8_kernel_dependencies_unchanged":True,
       "model_weights_not_mutated":True,"main_and_live_release_untouched":True}
    raw=(json.dumps(receipt_out,sort_keys=True,indent=2)+"\n").encode()
    put(OUT/"build_receipt.json",raw)
    print("QV7_TINYLLAMA_22N8_NATIVE_WORKER_BUILT",json.dumps({
      "exe_sha":receipt_out["native_binary_SHA256"],
      "parent_original_C_sha":sha(original),
      "new_C_sha":sha(decoded.encode()),
      "receipt_sha":sha(raw)},sort_keys=True),flush=True)
if __name__=="__main__":main()
