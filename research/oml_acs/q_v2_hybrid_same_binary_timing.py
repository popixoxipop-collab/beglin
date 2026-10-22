#!/usr/bin/env python3
"""Same-BINARY native BEGLIN Qv2 CPU serial/QKV-only/both async timing controls.

No source/model/precision changes. Runs THREE fresh isolated benchmark folders
sequentially to avoid cross-benchmark CPU contention and preserves full logs.
These are short best-of-three single-token *development* timings, not final
reproducible sustained throughput or an automatic release decision.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,re,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_hybrid_full_cpu_async_20261022"
PREVIOUS=ROOT/"lab_local/q_v2_actual_full_cpu_async_20261022"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
BIN=LAB/"beglin_qwen_stage_selective_native"
BIN_SHA="4cfc89d2ee4137745d7830a32b799beab20fe681d7c3c87581ce8cd67fffe718"
MODES=(("serial_control","0","both"),("qkv_repeat","1","qkv"),("both_control","1","both"))
def sha(b):return hashlib.sha256(b).hexdigest()
def write(path,data):
    with path.open("xb") as fp:fp.write(data);fp.flush();os.fsync(fp.fileno())
    path.chmod(0o600)
def secure(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_mode&0o777!=0o600:
        raise RuntimeError("source not private")
    return path.read_bytes()
def main():
    os.umask(0o077)
    old=LAB/"same_binary_three_trial_results.json"
    if old.exists():raise FileExistsError("prior comparison already completed, do not overwrite")
    if sha(BIN.read_bytes())!=BIN_SHA or not (SNAP/"model.safetensors").is_file():
        raise RuntimeError("not original pinned CPU BEGLIN source model and executable")
    prior=PREVIOUS/"serial"
    paths=["prompt_ids.i32"]+[f"eval/ppl_work/w{i}.i32" for i in range(8)]
    results=[]
    for name,flag,stage in MODES:
        folder=LAB/name
        if folder.exists():raise FileExistsError("fresh timings only")
        folder.mkdir(mode=0o700)
        for sub in ("results","eval","eval/ppl_work"):
            (folder/sub).mkdir(exist_ok=True,mode=0o700)
        for rel in paths:write(folder/rel,secure(prior/rel))
        env=os.environ.copy()
        env.update({
          "VECLIB_MAXIMUM_THREADS":"1",
          "QWEN_SAFETENSORS":str((SNAP/"model.safetensors").resolve()),
          "QWEN_HF_CONFIG":str((SNAP/"config.json").resolve()),
          "QWEN_SAFETENSORS_MAXSEQ":"256",
          "QWEN_BASE":str(folder.resolve()),
          "QWEN_PROMPT":str((folder/"prompt_ids.i32").resolve()),
          "BEGLIN_Q_LAB_PRECISION":"mixed",
          "BEGLIN_Q_LAB_N8_ROLES_FILE":str((LAB/"policy.txt").resolve()),
          "BEGLIN_QV2_FULL_ASYNC":flag,
          "BEGLIN_QV2_ASYNC_STAGE_SPEC":stage,
          "QWEN_FUSED_DISPATCH":"0",
          "QWEN_PROF":"0"
        })
        t=time.perf_counter()
        out=subprocess.run([str(BIN.resolve()),"bench","1"],env=env,capture_output=True,
                           text=True,timeout=310,cwd=str(ROOT))
        txt=out.stdout+"\n"+out.stderr
        log=LAB/f"{name}_same_binary_native_log.txt"
        write(log,txt.encode())
        if out.returncode:raise RuntimeError("real same-binary native bench failed "+out.stderr[-1500:])
        metric=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",txt)
        if not metric:raise RuntimeError("native throughput not measured")
        if "projection_role_tensors: f32=0 n8=28 q4=168" not in txt:
            raise RuntimeError("fixed original native precision changed")
        proof=re.search(r"\[BEGLIN-QV2-FULL-CPU-ASYNC\] qkv_tokens=(\d+) ffn_tokens=(\d+) K_q4=(\d+) K_n8=(\d+) V_q4=(\d+) V_n8=(\d+) U_q4=(\d+) U_n8=(\d+) threads=3 per_worker_Q4pool=6 done=1",txt)
        if flag=="0" and proof is not None:
            raise RuntimeError("serial control accidentally launched async threads")
        if flag=="1":
            if proof is None:raise RuntimeError("async full model did not execute background")
            q,ff=int(proof.group(1)),int(proof.group(2))
            if not q>0 or (stage=="qkv" and ff!=0) or (stage=="both" and (q!=ff or ff<=0)):
                raise RuntimeError("async stage proof mismatch")
        receipt=dict(name=name,stage=stage,flags=flag,binary_sha256=BIN_SHA,
             observed_native_single_token_best_of_three_ms=float(metric.group(2)),
             observed_native_tps=float(metric.group(1)),native_log_sha256=sha(txt.encode()),
             frozen_public_eval_inputs_sha256=sha(secure(folder/"prompt_ids.i32")),
             wall_including_loading_s=round(time.perf_counter()-t,3),
             no_original_qa_or_production_modified=True)
        results.append(receipt)
        print("BEGLIN_QV2_SAME_BINARY_TIMING",json.dumps(receipt,sort_keys=True),flush=True)
    out=dict(schema="BEGLIN-Qv2-actual-CPU-async-same-binary-repeat/1",
       status="PASS_CONTROLLED_SEQUENTIAL_SHORT_NATIVE_TIMING",
       candidates=results,
       repeated_original_all_n8_28_and_Q4_168=True,
       native_standalone_original_BF16_model=True,
       all_same_executable_sha256=BIN_SHA,
       research_only=True,source_evaluation_quality_failure_unmodified=True,
       latency_sample_limitations="one native best-of-three token bench per mode; not confidence interval, sustained performance or energy")
    write(old,(json.dumps(out,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_SAME_BINARY_THREE_TIMINGS_COMPLETE",json.dumps({"sha256":sha(secure(old)),
      "cases":{r["name"]:r["observed_native_single_token_best_of_three_ms"] for r in results}},sort_keys=True),flush=True)
if __name__=="__main__":main()
