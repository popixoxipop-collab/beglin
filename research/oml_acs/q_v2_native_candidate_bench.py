#!/usr/bin/env python3
"""Actual BEGLIN Qv2 selected original Qwen decoder performance, CPU serial.

Bench the TWO source-only BF16 activation-aware policies after their hash
was frozen before held public PPL. This does not add asynchronous QKV stages
to the native engine: predicts async scheduling separately. Keep physical
measurement of serial native decoder separate from estimated DAG parallel.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,re,subprocess,time

ROOT=pathlib.Path(__file__).resolve().parents[2]
ALLOC=ROOT/"lab_local/q_v2_joint_activation_async_20261022"
EVAL=ROOT/"lab_local/q_v2_new_public_development_20261022"
RUNTIME=ROOT/"lab_local/q_kernel_aware_native_20261022/beglin_kernel_aware_native_Qwen"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
OUT=EVAL/"two_source_only_actual_cpu_decode_bench.json"
EXPECTED_FREEZE="8dd48956820d47073f7f415c7427d07f20e01db154cdd8c895ec5d6d134c38a4"
EXPECTED_BIN="172613dc69f79f38c8553ff7087e653ec9c0031ca80aec8efd2f76f948599f14"
MODES={
 "activation_serial100":("serial100",EVAL/"activation_serial100"),
 "activation_async80_candidate":("parallel80",EVAL/"activation_async80_candidate"),
}
def sha(data:bytes):return hashlib.sha256(data).hexdigest()
def private(file):
    if not file.is_file() or file.is_symlink() or (file.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private Qv2 source/evaluation artifact missing")
    return file.read_bytes()
def put(file,data:bytes):
    with file.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    file.chmod(0o600)
def main():
    os.umask(0o077)
    if OUT.exists():raise FileExistsError("cannot alter native decode evidence after run")
    if sha(RUNTIME.read_bytes())!=EXPECTED_BIN:raise RuntimeError("same verified native research Qwen binary changed")
    frozen=private(ALLOC/"frozen_v2_source_only_policies.json")
    if sha(frozen)!=EXPECTED_FREEZE:raise RuntimeError("source-only activation allocator freeze changed")
    policy=json.loads(frozen)
    eval_previous=json.loads(private(EVAL/"native_results_private.json"))
    if eval_previous["actual_new_public_eval_tokens"]!=131:
        raise RuntimeError("native development evaluation not completed")
    native={}
    for tag,(policy_key,folder) in MODES.items():
        pol=ALLOC/f"{policy_key}_source_only_roles.txt"
        if sha(private(pol))!=policy["selected_policies"][policy_key]["policy_sha256"]:
            raise RuntimeError("precision mask modified after new model eval")
        if not private(folder/"prompt_ids.i32") or not SNAP.is_dir():
            raise RuntimeError("source model or same synthetic prompt missing")
        env=os.environ.copy()
        for k in ("QWEN_GGUF","QWEN_DENSE_QNG64_OVERRIDE","QWEN_WT_PROMOTION_FILE"):
            if k in env:raise RuntimeError("unexpected source override")
        env.update({
         "VECLIB_MAXIMUM_THREADS":"1",
         "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
         "QWEN_HF_CONFIG":str(SNAP/"config.json"),
         "QWEN_SAFETENSORS_MAXSEQ":"256",
         "QWEN_BASE":str(folder.resolve()),
         "QWEN_PROMPT":str((folder/"prompt_ids.i32").resolve()),
         "BEGLIN_Q_LAB_PRECISION":"mixed",
         "BEGLIN_Q_LAB_N8_ROLES_FILE":str(pol.resolve())
        })
        t=time.monotonic()
        p=subprocess.run([str(RUNTIME.resolve()),"bench","1"],env=env,cwd=str(ROOT),
               capture_output=True,text=True,timeout=150)
        raw=(p.stdout+"\n"+p.stderr).encode()
        put(EVAL/f"{tag}_bench_native_log.txt",raw)
        if p.returncode:raise RuntimeError(tag+" native bench exit "+str(p.returncode))
        m=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",raw.decode())
        cnt=re.search(r"\[BEGLIN-N8-REPACKED-LAB\] fast_gemv=(\d+)",raw.decode())
        if not m or not cnt or int(cnt.group(1))<=0:
            raise RuntimeError("native BEGLIN scalar vs Q4/n8 dispatch not actually invoked")
        info=policy["selected_policies"][policy_key]
        native[tag]={
           "native_cpu_one_token_best_3_ms":float(m.group(2)),
           "native_cpu_reported_token_per_s":float(m.group(1)),
           "source_predicted_serial_ms":info["predicted_serial_ms"],
           "hypothetical_async_stage_ms_not_executed":info["hypothetical_parallel_ms"],
           "frozen_role_counts":info["counts"],
           "native_source_mask_sha256":info["policy_sha256"],
           "native_n8_GEMV_calls":int(cnt.group(1)),
           "original_preregistered_8public_PPL":eval_previous["conditions"][tag]["PPL"],
           "log_sha256":sha(raw),
           "elapsed_seconds":round(time.monotonic()-t,2)
        }
        print("BEGLIN_QV2_ACTIVATION_SOURCE_NATIVE_BENCH_PASS",json.dumps({tag:native[tag]},sort_keys=True),flush=True)
    final=dict(schema="BEGLIN-Qv2-native-source-activation-two-mixed-serial/1",
        status="PASS_ACTUAL_NATIVE_SERIAL_CPU_DECODE_BENCHMARK",
        source_policy_sha256=EXPECTED_FREEZE,
        source_native_binary_sha256=EXPECTED_BIN,
        test_conditions=native,
        actual_async_stage_execution=False,
        source_Qwen_unchanged=True,
        scientific_quality_superiority=False,
        main_or_production_unchanged=True)
    put(OUT,(json.dumps(final,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_TWO_NATIVE_BENCH_COMPLETE",json.dumps({
       "sha256":sha(private(OUT)),
       "times_ms":{tag:info["native_cpu_one_token_best_3_ms"] for tag,info in native.items()},
       "PPL":{tag:info["original_preregistered_8public_PPL"] for tag,info in native.items()}},sort_keys=True),flush=True)
if __name__=="__main__":main()
