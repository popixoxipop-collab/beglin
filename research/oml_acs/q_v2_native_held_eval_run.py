#!/usr/bin/env python3
"""Evaluate five frozen original-BF16-sourced native BEGLIN Qwen CPU policies.

Runs after all source-only MILP policy SHA and eight public eval tokens are
frozen. Executes serially to avoid CPU contention confounds. Tests same
native prompt(s) with dump logits, 8-window PPL and 1-token decoder bench.
This does NOT perform the previously blocked per-tensor logit interventions
and does NOT implement whole-model asynchronous QKV execution.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,subprocess,time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_native_8public_eval_20261022"
SOURCE_MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
OUT=LAB/"native_8public_QV2_policy_evaluation_receipt.json"
MODES=("async_DAG","serial_only","old_random","q4","f32")
EXPECTED={
"async_DAG":(0,28,168),
"serial_only":(0,28,168),
"old_random":(0,28,168),
"q4":(0,0,196),
"f32":(196,0,0)
}
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def private(p:pathlib.Path)->bytes:
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private model-derived file missing/insecure: "+str(p))
    return p.read_bytes()
def persist(path:pathlib.Path,data:bytes):
    if path.exists():raise FileExistsError("refuse overwrite source or model evaluation")
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)

def main():
    if OUT.exists():raise FileExistsError("prior prospective public evaluation already recorded")
    os.umask(0o077)
    pre=json.loads(private(LAB/"public_native_eval_fixture_manifest.json"))
    if pre["status"]!="PREPARED_AFTER_SOURCE_POLICY_FROZEN_BEFORE_NATIVE_EVAL" or pre["eval_public_prompt_count"]!=8:
        raise RuntimeError("invalid source-only policy and held evaluation prelock")
    if set(pre["model_workloads"])!=set(MODES):raise RuntimeError("wrong evaluation model cohort")
    if not (SOURCE_MODEL/"model.safetensors").is_file():raise RuntimeError("pinned native BF16 source model absent")
    expected_bytes=pre["input_window_shas"]
    results={}
    for name in MODES:
        manifest=pre["model_workloads"][name]
        base=LAB/name
        for i in range(8):
            if sha(private(base/f"eval/ppl_work/w{i}.i32"))!=expected_bytes[i]:
                raise RuntimeError("model-specific public held window drift")
        path=pathlib.Path(manifest["binary"])
        if not path.is_file() or sha(path.read_bytes())!=manifest["binary_sha256"]:
            raise RuntimeError("native BEGLIN model kernel binary changed")
        env=os.environ.copy()
        for conflict in ("QWEN_GGUF","QWEN_INT4_BIN","QWEN_WT_PROMOTION_FILE",
                         "QWEN_DENSE_QNG64_OVERRIDE","BEGLIN_Q_LAB_N8_ROLES_FILE"):
            if conflict in env:raise RuntimeError("unexpected inherited precision override "+conflict)
        env.update({
         "VECLIB_MAXIMUM_THREADS":"1",
         "QWEN_SAFETENSORS":str(SOURCE_MODEL/"model.safetensors"),
         "QWEN_HF_CONFIG":str(SOURCE_MODEL/"config.json"),
         "QWEN_SAFETENSORS_MAXSEQ":"256",
         "QWEN_BASE":str(base.resolve()),
         "QWEN_PROMPT":str((base/"prompt_ids.i32").resolve()),
         "BEGLIN_Q_LAB_PRECISION":manifest["mode"]})
        if manifest["precision_policy_file"] is not None:
            policy=pathlib.Path(manifest["precision_policy_file"])
            if sha(policy.read_bytes())!=manifest["precision_policy_SHA256"]:
                raise RuntimeError("frozen role mask changed before native run")
            env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(policy)
        result={}
        for mode in ("dump","ppl","bench"):
            command=[str(path),mode]
            if mode=="ppl":command=["/usr/bin/time","-l",str(path),"ppl","8"]
            if mode=="bench":command.append("1")
            start=time.monotonic()
            done=subprocess.run(command,env=env,cwd=str(ROOT),capture_output=True,text=True,timeout=240)
            log=(done.stdout+"\n"+done.stderr).encode("utf-8")
            persist(base/f"QV2_{mode}_native_run_log.txt",log)
            if done.returncode!=0:raise RuntimeError(name+" "+mode+" exit "+str(done.returncode)+done.stderr[-2000:])
            observed=done.stdout+"\n"+done.stderr
            roles=re.search(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)",observed)
            if roles is None or tuple(map(int,roles.groups()))!=EXPECTED[name]:
                raise RuntimeError("wrong native Qwen tensor counts "+name+" "+mode)
            row={"exit_code":0,"wall_seconds":round(time.monotonic()-start,2),
                 "native_log_SHA256":sha(log),"source_28layer_forward":True,
                 "projector_counts":list(map(int,roles.groups()))}
            if mode=="dump":
                f=base/"results/c_logits_last.f32"
                data=private(f)
                if len(data)!=151936*4:raise RuntimeError("unexpected vocab width")
                logits=np.frombuffer(data,dtype="<f4")
                if not np.isfinite(logits).all():raise RuntimeError("nonfinite model logits")
                row.update(logits_SHA256=sha(data),argmax_token_id=int(np.argmax(logits)))
            elif mode=="ppl":
                m=re.search(r"\[ppl\]\s+fp32\s+ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",observed)
                rss=re.search(r"(\d+)\s+maximum resident set size",observed)
                windows=re.findall(r"\[ppl\] window (\d+) done \((\d+) toks\)",observed)
                if m is None or rss is None or len(windows)!=8:
                    raise RuntimeError("native full model eight-PPL or max RSS missing")
                ppl=float(m.group(1))
                if not math.isfinite(ppl) or ppl<=0:raise RuntimeError("invalid ppl")
                row.update(ppl=ppl,nll_nats_per_token=math.log(ppl),
                    total_evaluated_tokens=int(m.group(2)),RSS_max_bytes=int(rss.group(1)),
                    window_token_counts=[int(v) for i,v in windows])
            else:
                m=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",observed)
                if m is None:raise RuntimeError("bench native token latency unavailable")
                row.update(tps_single_token_best_of_three=float(m.group(1)),
                    ms_single_token_best_of_three=float(m.group(2)))
            result[mode]=row
            print("BEGLIN_QV2_HELD_PUBLIC_NATIVE",name,mode,json.dumps(row,sort_keys=True),flush=True)
        results[name]=result
    teacher=np.frombuffer(private(LAB/"f32/results/c_logits_last.f32"),dtype="<f4").astype(float)
    p=np.exp(teacher-teacher.max());p/=p.sum()
    for name in MODES:
        v=np.frombuffer(private(LAB/name/"results/c_logits_last.f32"),dtype="<f4").astype(float)
        q=np.exp(v-v.max());q/=q.sum()
        result=results[name]["dump"]
        result["KL_F32teacher_to_candidate"]=float(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300)))))
        result["relative_logit_L2"]=float(np.linalg.norm(v-teacher)/np.linalg.norm(teacher))
        result["top5_overlap_with_F32"]=len(set(np.argsort(v)[-5:])&set(np.argsort(teacher)[-5:]))
    manifest=dict(schema="BEGLIN-QV2-8-public-native-sequential-model-evaluation/1",
      status="ALL_NATIVE_RUNS_SUCCEEDED_QUALITY_NOT_PROMOTED",
      inputs_pinned_public_prelock_sha256=pre["prereg_sha256"],
      source_only_MILP_SHA256=pre["source_optimizer_manifest_sha256"],
      public_eval_prompt_count=8,
      frozen_policy_selected_before_first_eval=True,
      model="Qwen2.5-1.5B-Instruct native BEGLIN CPU",
      device="Apple Silicon EOE Mac13,1",
      model_HF_source_original_BF16_revision="4ae77cb209f06199b8df1c94e21ff341332a3a89",
      mode_results=results,
      production_and_main_unchanged=True,
      model_parallel_full_executor_was_not_run=True,
      measurement_limitations=["Same 8 public synthetic prompts, no external IID/OOD task labels and no guaranteed semantic source exclusion","Bench fastest of 3 after warmup, not sustained TPS","Entire actual decoder is serial; async policy's parallel-graph estimate not measured end-to-end","BF16 source widened F32 teacher is not genuine BF16 compute baseline","Output error proxy is diagonal activation cov, no Fisher/Hessian or causal intervention","Unrecognized old V6 24 record IDs remain sealed"])
    persist(OUT,(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_HELD_PUBLIC_ALL_COMPLETE",json.dumps({
       "receipt_SHA256":sha(private(OUT)),
       "nll":{mode:results[mode]["ppl"]["nll_nats_per_token"] for mode in MODES},
       "tps":{mode:results[mode]["bench"]["tps_single_token_best_of_three"] for mode in MODES},
       "KL":{mode:results[mode]["dump"]["KL_F32teacher_to_candidate"] for mode in MODES}
    },sort_keys=True),flush=True)
if __name__=="__main__":main()
