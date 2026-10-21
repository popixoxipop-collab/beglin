#!/usr/bin/env python3
"""Re-run two frozen BEGLIN 28-n8/168-q4 policies with in-place n8 repack.

Research-only; source BF16 Qwen native CPU. Independent scalar and on-the-fly
NEON results already exist; this run compares ONLY kernel representation.
No per-tensor sensitivity, dataset labels, release or production operations.
"""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import re,subprocess,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"lab_local"
WORK=BASE/"qng64_n8_repacked_full_20261022"
ORIG=BASE/"q_full_model_mixed_20261021"
FLY=BASE/"qng64_n8_neon_mixed_20261022"
SNAP=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
BINARY=WORK/"beglin_mixed_repacked"
RESULT=WORK/"frozen_mixed_kernel_repack_results.json"
EXPECTED_BINARY_SHA="8c16013a343b924b8117f66edcbfd7fb11a86f991127b9cd97fb6ec5552b6664"
POLICY_SHA={
"random":"72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5",
"proxy":"f65f50421cd9e9bac32bc780805942d68c3d503075ce962ce25315fdb5914cb9",
}
FLY_BENCH_MS={"random":143.85,"proxy":144.43}
SCALAR_BENCH_MS={"random":810.72,"proxy":810.17}
FLY_PPL={"random":3411.2283,"proxy":4747.6983}
def sha(x:bytes)->str:return hashlib.sha256(x).hexdigest()
def private(p:Path)->bytes:
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("missing or insecure private source: "+str(p))
    return p.read_bytes()
def persist(path:Path,blob:bytes):
    if path.exists():raise FileExistsError("will not overwrite previous experiment "+str(path))
    with path.open("xb") as f:
        f.write(blob);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def run_mode(policy:str,task:str)->dict:
    folder=WORK/f"mixed_{policy}"
    snap=SNAP
    env=os.environ.copy()
    keys={
      "VECLIB_MAXIMUM_THREADS":"1",
      "QWEN_SAFETENSORS":str(snap/"model.safetensors"),
      "QWEN_HF_CONFIG":str(snap/"config.json"),
      "QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(folder.resolve()),
      "QWEN_PROMPT":str((folder/"prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":"mixed",
      "BEGLIN_Q_LAB_N8_ROLES_FILE":str((WORK/f"policy_{policy}.txt").resolve()),
    }
    for k in ("QWEN_GGUF","QWEN_DENSE_QNG64_OVERRIDE","QWEN_WT_PROMOTION_FILE"):
        if k in env:raise RuntimeError("unexpected Qwen precision override in environment: "+k)
    env.update(keys)
    if task not in ("dump","bench","ppl"):raise ValueError("unknown test phase")
    args=[str(BINARY.resolve()),task]
    if task=="bench":args+=["1"]
    if task=="ppl":args=["/usr/bin/time","-l"]+args+["2"]
    if task=="dump" and (folder/"results/c_logits_last.f32").exists():
        raise FileExistsError("don't overwrite a prior full model logit")
    t=time.perf_counter()
    completed=subprocess.run(args,env=env,cwd=str(ROOT),capture_output=True,text=True,timeout=270)
    wall=time.perf_counter()-t
    log=(completed.stdout+"\n"+completed.stderr).encode("utf-8")
    persist(WORK/f"{policy}_{task}_native_log.txt",log)
    if completed.returncode!=0:
        raise RuntimeError(policy+" "+task+" native exit "+str(completed.returncode)+completed.stderr[-800:])
    if "[Q-NATIVE-RESEARCH] projection_role_tensors: f32=0 n8=28 q4=168" not in log.decode():
        raise RuntimeError("mixed model role precision not 28n8/168q4")
    counters=re.search(r"\[BEGLIN-N8-REPACKED-LAB\] fast_gemv=(\d+) fast_batch_rows=(\d+) other_bit_gemv=(\d+) other_bit_batches=(\d+)",log.decode())
    if not counters or int(counters.group(1))<=0 or any(int(v)!=0 for v in counters.groups()[1:]):
        raise RuntimeError("repacked decoder not actually used / unexpectedly fell back")
    result=dict(policy=policy,task=task,exit=0,wall_seconds=round(wall,3),
                log_sha256=sha(log),n8_fast_gemv_calls=int(counters.group(1)),
                fallback_calls=0)
    if task=="dump":
        file=folder/"results/c_logits_last.f32"
        if not file.is_file() or file.stat().st_size!=151936*4:
            raise RuntimeError("final logit array absent/bad")
        if file.stat().st_mode&0o777!=0o600:
            raise RuntimeError("final full model logits not private")
        z=np.frombuffer(private(file),dtype="<f4").astype(np.float64)
        if not np.isfinite(z).all():raise RuntimeError("nonfinite full logits")
        reference=np.frombuffer(private(FLY/f"mixed_{policy}/results/c_logits_last.f32"),dtype="<f4").astype(np.float64)
        diff=z-reference
        mx=float(np.max(np.abs(diff)));rel=float(np.linalg.norm(diff)/np.linalg.norm(reference))
        if mx>=.001 or rel>=1e-4:raise RuntimeError("mixed repacked vs on-the-fly mismatched")
        result.update(logits_sha256=sha(private(file)),top1_id=int(z.argmax()),
          old_neon_SHA256=sha(private(FLY/f"mixed_{policy}/results/c_logits_last.f32")),
          parity_max_abs=mx,parity_rel_l2=rel)
    if task=="bench":
        bench=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",log.decode())
        if not bench:raise RuntimeError("no real one-token bench metrics")
        ms=float(bench.group(2))
        result.update(best_of_three_one_token_ms=ms,engine_tps_reported=float(bench.group(1)),
             speedup_over_fly_NEON=FLY_BENCH_MS[policy]/ms,
             speedup_over_original_scalar=SCALAR_BENCH_MS[policy]/ms)
    if task=="ppl":
        v=re.search(r"\[ppl\].*ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",log.decode())
        rss=re.search(r"(\d+)\s+maximum resident set size",log.decode())
        if not v or not rss:raise RuntimeError("PPL or RSS missing from observed time -l")
        ppl=float(v.group(1));tokens=int(v.group(2))
        if tokens!=21:raise RuntimeError("incorrect same synthetic PPL count")
        if abs(ppl-FLY_PPL[policy])/FLY_PPL[policy]>=1e-5:
            raise RuntimeError("native n8 repacked changed PPL")
        result.update(ppl=ppl,ppl_tokens=tokens,relative_ppl_difference_vs_fly_neon=abs(ppl-FLY_PPL[policy])/FLY_PPL[policy],
            maximum_resident_bytes=int(rss.group(1)))
    print("REPACKED_FULL_MIXED_PROGRESS",json.dumps(result,sort_keys=True),flush=True)
    return result
def main():
    os.umask(0o077)
    if RESULT.exists():raise FileExistsError("cannot overwrite previous mixed evaluation")
    if sha(BINARY.read_bytes())!=EXPECTED_BINARY_SHA:
        raise RuntimeError("experiment full model binary drift")
    if not SNAP.is_dir() or not (SNAP/"model.safetensors").is_file():
        raise RuntimeError("pinned cached original BF16 Qwen missing")
    for name,digest in POLICY_SHA.items():
        if sha(private(WORK/f"policy_{name}.txt"))!=digest:
            raise RuntimeError("mixed role-specific unchanged tensor policy drift")
        for inputfile in ("prompt_ids.i32","eval/ppl_work/w0.i32","eval/ppl_work/w1.i32"):
            if private(WORK/f"mixed_{name}/{inputfile}") != private(ORIG/f"mixed_{name}/{inputfile}"):
                raise RuntimeError("public synthetic prompt bytes changed")
    results={}
    for name in ("random","proxy"):
        results[name]=[run_mode(name,phase) for phase in ("dump","bench","ppl")]
    result=dict(schema="beglin-Q-repacked-native-full-mixed/1",
        status="PASS_ISOLATED_28N8_168Q4_TWO_POLICIES",
        candidates=results,
        unchanged_policy_SHA256=POLICY_SHA,source_BF16_rev=SNAP.name,
        production_touched=False, original_tensor_source_untouched=True,
        old_V6_final_holdout_untouched=True,no_task_AUROC=True,
        quality_not_claimed_improved=True,
        caveats=["Only 21 synthetic PPL tokens, 1 synthetic logits prompt; no IID/OOD final",
         "Native bench fastest of 3 single-token trials, not sustained throughput",
         "In-memory n8 byte representation differs from on-disk bitplanes, tag required",
         "Model quantization bits or allocation policy unchanged; decoder optimization only"])
    persist(RESULT,(json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
    print("REPACKED_MIXED_COMPLETE",json.dumps({
        "receipt_SHA256":sha(private(RESULT)),
        "mixed_random_ms":results["random"][1]["best_of_three_one_token_ms"],
        "mixed_proxy_ms":results["proxy"][1]["best_of_three_one_token_ms"],
        "both_parity":True,"no_production_changes":True
    },sort_keys=True),flush=True)
if __name__=="__main__":
    main()
