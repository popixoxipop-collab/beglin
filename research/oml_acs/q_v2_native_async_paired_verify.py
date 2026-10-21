#!/usr/bin/env python3
"""BEGLIN Qv2 actual native pthread QKV/Gate-Up asynchronous paired evaluation.

Same already-compiled, SHA-verified, isolated research binary; runtime toggle:
 BEGLIN_QV2_NATIVE_ASYNC=0 (original serial) vs 1 (real persistent pthread)
Both use identical source-only globally MILP-frozen 28 n8 tensors, original
BF16 checkpoint, same 8 prereg public inputs, CPU and tied FP32 head.
Logs private. Model and production never modified.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,subprocess,time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_actual_native_async_retry1_20261022"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
EVAL=ROOT/"lab_local/q_v2_8public_evaluation_20261022"
OUT=LAB/"paired_native_async_serial_8public.json"
EXE=LAB/"beglin_Qv2_native_async"
WORK=LAB/"isolated"
POLICY=LAB/"frozen_selected_28_roles.txt"
EXPECTED_BIN="fb947372495962ed67bb00f91199030b2e41d50ab283f4a5bd81ab2e51cdc054"
EXPECTED_POLICY="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
EXPECTED_STAGE="c36e76aea0b051e85bf94f95fa6973ad8918342213067c45e805674bdc47bfc5"
EXPECTED_REFERENCE_DUMP="b46dc07b9b5ff32271a807d18c49444d8f0c33d0fb5533b17d83006de5d33d1e"

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private(p:pathlib.Path)->bytes:
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("missing/insecure private native model output: "+str(p))
    return p.read_bytes()
def create(p:pathlib.Path,b:bytes):
    with p.open("xb") as f:
        f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def source_preflight():
    if OUT.exists():raise FileExistsError("refuse to edit existing actual parallel receipt")
    meta=json.loads(private(LAB/"build_receipt.json"))
    if sha(EXE.read_bytes())!=EXPECTED_BIN or sha(EXE.read_bytes())!=meta["native_binary_SHA256"]:
        raise RuntimeError("native isolated async executable changed")
    if sha(private(POLICY))!=EXPECTED_POLICY or meta["original_q4_n8_tensor_policy_SHA256"]!=EXPECTED_POLICY:
        raise RuntimeError("selected source-only 28-tensor policy changed")
    sch=ROOT/"configs/research/BEGLIN_Q_V2_FROZEN_PARALLEL_STAGE_SCHEDULE_20261022.json"
    if sha(sch.read_bytes())!=EXPECTED_STAGE:raise RuntimeError("native pthread schedule drift")
    actual=WORK/"results/c_logits_last.f32"
    ref=EVAL/"v2_activation_async_MILP/results/c_logits_last.f32"
    if sha(private(actual))!=EXPECTED_REFERENCE_DUMP or private(actual)!=private(ref):
        raise RuntimeError("actual async native Qwen first input not bit-exact with former serial")
    for rel in ("prompt_ids.i32",)+tuple(f"eval/ppl_work/w{k}.i32" for k in range(8)):
        if private(WORK/rel)!=private(EVAL/"v2_activation_async_MILP"/rel):
            raise RuntimeError("same 8 public source-frozen model input bytes violated")
    return meta

def run_one(async_mode:int,task:str)->dict:
    if task not in ("bench","ppl") or async_mode not in (0,1):
        raise ValueError("unsupported research-only workload")
    name=("async" if async_mode else "serial")+"_"+task
    env=os.environ.copy()
    for k in ("QWEN_GGUF","QWEN_DENSE_QNG64_OVERRIDE","QWEN_WT_PROMOTION_FILE"):
        if k in env:raise RuntimeError("foreign precision env present")
    env.update({
     "VECLIB_MAXIMUM_THREADS":"1",
     "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
     "QWEN_HF_CONFIG":str(SNAP/"config.json"),
     "QWEN_SAFETENSORS_MAXSEQ":"256",
     "QWEN_BASE":str(WORK.resolve()),
     "QWEN_PROMPT":str((WORK/"prompt_ids.i32").resolve()),
     "BEGLIN_Q_LAB_PRECISION":"mixed",
     "BEGLIN_Q_LAB_N8_ROLES_FILE":str(POLICY.resolve()),
     "BEGLIN_QV2_NATIVE_ASYNC":str(async_mode),
    })
    args=[str(EXE.resolve()),"bench","1"] if task=="bench" else ["/usr/bin/time","-l",str(EXE.resolve()),"ppl","8"]
    start=time.monotonic()
    run=subprocess.run(args,env=env,cwd=str(ROOT),capture_output=True,
              text=True,timeout=180)
    duration=time.monotonic()-start
    payload=(run.stdout+"\n"+run.stderr).encode()
    create(LAB/(name+"_native_log.txt"),payload)
    if run.returncode:
        raise RuntimeError("native "+name+" failure, exit="+str(run.returncode)+" "+run.stderr[-1800:])
    text=payload.decode()
    if "[Q-NATIVE-RESEARCH] projection_role_tensors: f32=0 n8=28 q4=168" not in text:
        raise RuntimeError("actual native bitmask execution role count wrong")
    mode_stats=re.search(r"\[BEGLIN-QV2-NATIVE-ASYNC\] qkv_joined=(\d+) gateup_joined=(\d+) serial_stage=(\d+)",text)
    if async_mode==1:
        if not mode_stats or int(mode_stats.group(1))<=0 or int(mode_stats.group(2))<=0:
            raise RuntimeError("async native QKV/GU stages not observed")
    elif mode_stats and (int(mode_stats.group(1)) or int(mode_stats.group(2))):
        raise RuntimeError("serial control unexpectedly ran concurrent native path")
    repacked=re.search(r"\[BEGLIN-N8-REPACKED-LAB\] fast_gemv=(\d+) fast_batch_rows=(\d+)",text)
    if not repacked or int(repacked.group(1))<=0:
        raise RuntimeError("native repacked n8 GEMV dispatch absent")
    obs=dict(kind=name,exit_code=0,actual_async_enabled=bool(async_mode),
             all_n8_GEMV_calls=int(repacked.group(1)),
             genuine_pthread_qkv_join_count=int(mode_stats.group(1)) if mode_stats else 0,
             genuine_pthread_gateup_join_count=int(mode_stats.group(2)) if mode_stats else 0,
             process_log_sha256=sha(payload),elapsed_seconds=round(duration,3))
    if task=="bench":
        line=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",text)
        if not line:raise RuntimeError("native bench metrics missing")
        obs.update(single_decode_best_of_3_ms=float(line.group(2)),
                   tok_s=float(line.group(1)))
    else:
        line=re.search(r"\[ppl\].*ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",text)
        rss=re.search(r"(\d+)\s+maximum resident set size",text)
        if not line or int(line.group(2))!=131 or not rss:
            raise RuntimeError("preregistered 131-token PPL or native RSS missing")
        obs.update(PPL=float(line.group(1)),
                   NLL_nats_per_token=math.log(float(line.group(1))),
                   scored_tokens=131,max_rss_bytes=int(rss.group(1)))
    print("ACTUAL_BEG_NATIVE_ASYNC_COMPARISON",json.dumps(obs,sort_keys=True),flush=True)
    return obs

def main():
    os.umask(0o077)
    meta=source_preflight()
    observed={}
    for task in ("bench","ppl"):
        observed[task]={}
        for mode in (0,1):
            observed[task]["async" if mode else "serial"]=run_one(mode,task)
    b=observed["bench"];nll=observed["ppl"]
    drift=abs(nll["async"]["PPL"]-nll["serial"]["PPL"])/nll["serial"]["PPL"]
    if drift>1e-5:raise RuntimeError("actual concurrent full model PPL drift too high")
    speedup=b["serial"]["single_decode_best_of_3_ms"]/b["async"]["single_decode_best_of_3_ms"]
    receipts=dict(schema="BEGLIN-Qv2-actual-native-QKV-FFN-pthread-paired/1",
      status="PASS_NATIVE_ASYNC_FULL_MODEL_OUTPUT_AND_PPL_PARITY",
      native_binary_sha256=EXPECTED_BIN,policy_sha256=EXPECTED_POLICY,
      QKV_GU_stage_schedule_SHA256=EXPECTED_STAGE,
      serial_reference_8public_dump_logits_sha256=EXPECTED_REFERENCE_DUMP,
      two_cases_same_original_model=True,
      before_after=observed,relative_ppl_delta=drift,
      actual_one_token_native_speedup=speedup,
      real_pipeline_qkv_gateup_async=True,
      native_ASYNC_threads_join_before_attention_and_swiglu=True,
      extra_scratch_q4_pools_not_accounted_by_projection_weight_bytes=True,
      stage_microbench_cost_not_assumed_to_equal_full_model=True,
      production_or_main_touched=False,
      final_Q_qual_accuracy_verdict="NO_VERDICT",
      limitations=["8 public synthetic teacher-forced chat prompts 131 token positions only",
         "The source-selected global MILP allocation was frozen before prospective evaluation but fails quality vs Q4 and old random",
         "Single-token native bench fastest of 3, not sustained throughput or confidence interval",
         "Additional role-owned Q4 threadpools can increase process RSS and CPU oversubscription",
         "No Metal GPU, independent final IID/OOD or original V6 used 24 exclusions"])
    create(OUT,(json.dumps(receipts,sort_keys=True,indent=2)+"\n").encode())
    print("QV2_GENUINE_ASYNC_FULL_NATIVE_COMPLETE",json.dumps({
        "receipt_sha256":sha(private(OUT)),
        "actual_speedup":speedup,
        "serial_ms":b["serial"]["single_decode_best_of_3_ms"],
        "parallel_ms":b["async"]["single_decode_best_of_3_ms"],
        "native_ppl_equal":drift<=1e-5},sort_keys=True),flush=True)
if __name__=="__main__":
    main()
