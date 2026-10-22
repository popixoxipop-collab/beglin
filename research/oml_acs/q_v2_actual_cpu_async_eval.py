#!/usr/bin/env python3
"""Run real BEGLIN Qv2 single-token full decoder CPU async A/B, research only.

Same SHA-pinned native executable toggled between serial and actual pthread
Q/K/V + gate/up (3 isolated background Q4 pools). Identical 28 QNg64 n8 and
168 Q4G64 original BF16 weights, eval prompt files, bias/norm/tied head.
Preserves raw outputs privately; STOP if input integrity or output drifts.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,re,subprocess,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_actual_full_cpu_async_20261022"
PRIOR=ROOT/"lab_local/q_v2_activation_async_native_eval_20261022"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
BIN=LAB/"beglin_qwen_actual_async"
PRIOR_PPL=807.363
PPL_TOKEN_COUNT=131
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private(p:pathlib.Path)->bytes:
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private original input/output not secured "+str(p))
    return p.read_bytes()
def store(p:pathlib.Path,b:bytes):
    if p.exists():raise FileExistsError("refuse overwrite "+str(p))
    with p.open("xb") as f:
        f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def run_case(mode,phase):
    if mode not in ("serial","async") or phase not in ("dump","bench","ppl"):
        raise ValueError("unknown research condition")
    os.umask(0o077)
    lab=json.loads(private(LAB/"build_receipt.json"))
    if sha(BIN.read_bytes())!=lab["native_binary_sha256"]:
        raise RuntimeError("original research binary mutated after compile")
    if sha(private(LAB/"policy.txt"))!=lab["selected_role_mask_SHA256"]:
        raise RuntimeError("preselected bit role mask mutated")
    if not (SNAP/"model.safetensors").is_file():
        raise RuntimeError("pinned source HF BF16 snapshot missing")
    src=PRIOR/"v2_async_activation"
    dst=LAB/mode
    for x in ["prompt_ids.i32"]+[f"eval/ppl_work/w{i}.i32" for i in range(8)]:
        if private(src/x)!=private(dst/x):
            raise RuntimeError("same source public evaluation inputs not matched "+x)
    rec=LAB/f"{mode}_{phase}_execution_receipt.json"
    log=LAB/f"{mode}_{phase}_native_stdout_stderr.txt"
    if rec.exists() or log.exists():raise FileExistsError("native test already run")
    output=dst/"results/c_logits_last.f32"
    if phase=="dump" and output.exists():raise FileExistsError("one-prompt logits already exist")
    env=dict(os.environ)
    if any(k in env for k in ("QWEN_WT_PROMOTION_FILE","QWEN_GGUF","QWEN_DENSE_QNG64_OVERRIDE")):
        raise RuntimeError("unexpected original output precision override")
    env.update({
      "VECLIB_MAXIMUM_THREADS":"1",
      "QWEN_SAFETENSORS":str((SNAP/"model.safetensors").resolve()),
      "QWEN_HF_CONFIG":str((SNAP/"config.json").resolve()),
      "QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(dst.resolve()),
      "QWEN_PROMPT":str((dst/"prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":"mixed",
      "BEGLIN_Q_LAB_N8_ROLES_FILE":str((LAB/"policy.txt").resolve()),
      "QWEN_FUSED_DISPATCH":"0",
      "BEGLIN_QV2_FULL_ASYNC":"1" if mode=="async" else "0",
      "QWEN_PROF":"0",
    })
    cmd=[str(BIN.resolve()),phase]+(["1"] if phase=="bench" else ["8"] if phase=="ppl" else [])
    if phase=="ppl":cmd=["/usr/bin/time","-l"]+cmd
    start=time.monotonic()
    complete=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=330,
       cwd=str(ROOT))
    wall=time.monotonic()-start
    transcript=complete.stdout+"\n"+complete.stderr
    store(log,transcript.encode())
    if complete.returncode!=0:
        raise RuntimeError(f"{mode} {phase} exit {complete.returncode}: "+complete.stderr[-3500:])
    mix=re.search(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)",transcript)
    if mix is None or tuple(int(x) for x in mix.groups())!=(0,28,168):
        raise RuntimeError("actual 28 n8 + 168 q4 original model not loaded")
    a=re.search(r"\[BEGLIN-QV2-FULL-CPU-ASYNC\] qkv_tokens=(\d+) ffn_tokens=(\d+) K_q4=(\d+) K_n8=(\d+) V_q4=(\d+) V_n8=(\d+) U_q4=(\d+) U_n8=(\d+) threads=3 per_worker_Q4pool=6 done=1",transcript)
    if (mode=="async")!=(a is not None):
        raise RuntimeError("threaded async proof contradicts mode")
    if mode=="async" and (int(a.group(1))!=int(a.group(2)) or int(a.group(1))<=0):
        raise RuntimeError("QKV/FFN async real join counts do not match")
    args={
       "schema":"BEGLIN-Qv2-actual-full-CPU-async-native-run/1",
       "status":"PASS_NATIVE_SAME_SOURCE_RESEARCH_ONLY",
       "condition":mode,"phase":phase,
       "native_binary_sha256":lab["native_binary_sha256"],
       "source_BF16_snapshot_rev":SNAP.name,
       "role_policy_sha256":lab["selected_role_mask_SHA256"],
       "source_eval_identical_to_previously_locked_native_study":True,
       "elapsed_wall_seconds_including_BF16_model_load":round(wall,3),
       "native_process_log_sha256":sha(transcript.encode()),
       "worker_async_QKV_FFN_token_pairs":int(a.group(1)) if a is not None else 0,
       "worker_concurrency_verified":mode=="async",
       "production_changed":False}
    if phase=="dump":
        b=private(output)
        v=np.frombuffer(b,dtype="<f4").astype(np.float64)
        if len(v)!=151936 or not np.isfinite(v).all():
            raise RuntimeError("wrong full native Qwen logits")
        prior=private(src/"results/c_logits_last.f32")
        vv=np.frombuffer(prior,dtype="<f4").astype(np.float64)
        delta=v-vv
        maxabs=float(np.abs(delta).max())
        rel=float(np.linalg.norm(delta)/np.linalg.norm(vv))
        if maxabs>0.001 or rel>0.0001 or int(v.argmax())!=int(vv.argmax()):
            raise RuntimeError("native async vs original serial full-model output drift")
        args.update(native_vocab_count=151936,output_SHA256=sha(b),prior_native_serial_SHA256=sha(prior),
            original_native_logits_max_abs_error=maxabs,
            original_native_logits_relative_l2_error=rel,
            native_full_vocab_top1=int(v.argmax()),bitwise_equal_to_old_serial=b==prior)
    if phase=="bench":
        found=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",transcript)
        if not found:raise RuntimeError("native BEGLIN single-token timing absent")
        args.update(native_bench_best_of_three_tok_s=float(found.group(1)),
                    native_bench_best_of_three_ms_tok=float(found.group(2)))
    if phase=="ppl":
        ppl=re.search(r"\[ppl\].*ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",transcript)
        rss=re.search(r"(\d+)\s+maximum resident set size",transcript)
        if not ppl or not rss or int(ppl.group(2))!=PPL_TOKEN_COUNT:
            raise RuntimeError("PPL 131 position or resident memory absent")
        value=float(ppl.group(1))
        if abs(value-PRIOR_PPL)/PRIOR_PPL>1e-4:
            raise RuntimeError("source-only v2 PPL changed under research asynchronous scheduling")
        args.update(actual_same_source_PPL=value,nll_nats_per_token=math.log(value),
             original_131_development_positions_count=PPL_TOKEN_COUNT,
             relative_PPL_difference_vs_prior=abs(value-PRIOR_PPL)/PRIOR_PPL,
             maximum_RSS_bytes=int(rss.group(1)))
    store(rec,(json.dumps(args,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_ACTUAL_CPU_ASYNC_NATIVE_"+mode.upper()+"_"+phase.upper()+"_PASS",
          json.dumps(dict(mode=mode,phase=phase,log_sha256=sha(transcript.encode()),
             latency_ms=args.get("native_bench_best_of_three_ms_tok"),
             ppl=args.get("actual_same_source_PPL"),
             max_logit_abs=args.get("original_native_logits_max_abs_error"),
             QKV_join_pairs=args["worker_async_QKV_FFN_token_pairs"],
             receipt_sha256=sha(private(rec))),sort_keys=True),flush=True)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",required=True,choices=["serial","async"])
    parser.add_argument("--phase",required=True,choices=["dump","bench","ppl"])
    arg=parser.parse_args()
    run_case(arg.mode,arg.phase)
if __name__=="__main__":
    main()
