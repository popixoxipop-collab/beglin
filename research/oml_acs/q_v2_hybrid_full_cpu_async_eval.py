#!/usr/bin/env python3
"""Actual BEGLIN Qv2 full decoder selective async QKV-only or FFN-only tests.

Exactly same source-BF16-derived Q4/n8 tensors and public fixed 131 eval
positions; only stage parallelism is changed. Research-only diagnostic;
posthoc runtime throughput choice is NOT a valid model-quality selection.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,re,subprocess,time,math
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_hybrid_full_cpu_async_20261022"
PRIOR=ROOT/"lab_local/q_v2_actual_full_cpu_async_20261022"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
EXPECTED_BIN_SHA="4cfc89d2ee4137745d7830a32b799beab20fe681d7c3c87581ce8cd67fffe718"
ORIGINAL_PPL=807.363
def sha(b):return hashlib.sha256(b).hexdigest()
def private(path):
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private source unsafe "+str(path))
    return path.read_bytes()
def persist(path,bytestr):
    with path.open("xb") as f:
        f.write(bytestr);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def test(mode,phase):
    if mode not in ("qkv_only","ffn_only") or phase not in ("dump","bench","ppl"):
        raise ValueError("unknown experiment condition")
    os.umask(0o077)
    binfile=LAB/"beglin_qwen_stage_selective_native"
    build=json.loads(private(LAB/"build_receipt.json"))
    if sha(binfile.read_bytes())!=EXPECTED_BIN_SHA or build["native_hybrid_bin_sha256"]!=EXPECTED_BIN_SHA:
        raise RuntimeError("native actual async lab binary drift")
    if private(LAB/"policy.txt")!=private(PRIOR/"policy.txt"):
        raise RuntimeError("original 28/196 frozen role policy changed")
    new=LAB/mode
    for src in ["prompt_ids.i32"]+[f"eval/ppl_work/w{i}.i32" for i in range(8)]:
        if private(new/src)!=private(PRIOR/"serial"/src):
            raise RuntimeError("eval source prompt mismatch "+src)
    out=LAB/f"{mode}_{phase}_native_log.txt"
    rec=LAB/f"{mode}_{phase}_execution_receipt.json"
    if out.exists() or rec.exists():raise FileExistsError("cannot overwrite prior research actual stage outputs")
    if phase=="dump" and (new/"results/c_logits_last.f32").exists():
        raise FileExistsError("previous logits should never be overwritten")
    env=dict(os.environ)
    env.update({
        "VECLIB_MAXIMUM_THREADS":"1",
        "QWEN_SAFETENSORS":str((SNAP/"model.safetensors").resolve()),
        "QWEN_HF_CONFIG":str((SNAP/"config.json").resolve()),
        "QWEN_SAFETENSORS_MAXSEQ":"256",
        "QWEN_BASE":str(new.resolve()),
        "QWEN_PROMPT":str((new/"prompt_ids.i32").resolve()),
        "BEGLIN_Q_LAB_PRECISION":"mixed",
        "BEGLIN_Q_LAB_N8_ROLES_FILE":str((LAB/"policy.txt").resolve()),
        "BEGLIN_QV2_FULL_ASYNC":"1",
        "BEGLIN_QV2_ASYNC_STAGE_SPEC":"qkv" if mode=="qkv_only" else "ffn",
        "QWEN_FUSED_DISPATCH":"0",
        "QWEN_PROF":"0"
    })
    for name in ("QWEN_WT_PROMOTION_FILE","QWEN_GGUF","QWEN_DENSE_QNG64_OVERRIDE"):
        if name in os.environ:
            raise RuntimeError("unexpected source precision env override")
    args=[str(binfile.resolve()),phase]+(["1"] if phase=="bench" else ["8"] if phase=="ppl" else [])
    if phase=="ppl":args=["/usr/bin/time","-l"]+args
    t=time.perf_counter()
    p=subprocess.run(args,env=env,cwd=str(ROOT),text=True,capture_output=True,timeout=300)
    log=p.stdout+"\n"+p.stderr
    persist(out,log.encode())
    if p.returncode:raise RuntimeError("Native stage invocation failed "+p.stderr[-2100:])
    verify=re.search(r"projection_role_tensors: f32=0 n8=28 q4=168",log)
    proof=re.search(r"\[BEGLIN-QV2-FULL-CPU-ASYNC\] qkv_tokens=(\d+) ffn_tokens=(\d+) K_q4=(\d+) K_n8=(\d+) V_q4=(\d+) V_n8=(\d+) U_q4=(\d+) U_n8=(\d+) threads=3 per_worker_Q4pool=6 done=1",log)
    if not verify or not proof:raise RuntimeError("not real active async 28 n8/168 q4 full model")
    q,ffn=(int(proof.group(1)),int(proof.group(2)))
    if mode=="qkv_only" and not (q>0 and ffn==0):raise RuntimeError("hybrid QKV only dependency proof wrong")
    if mode=="ffn_only" and not (ffn>0 and q==0):raise RuntimeError("hybrid FFN only dependency proof wrong")
    result=dict(schema="BEGLIN-Qv2-actual-CPU-async-stage-selective/1",
        status="PASS_NATIVE_EXPERIMENTAL_SINGLE_STAGE_CONCURRENCY",
        condition=mode,phase=phase,source_BF16_rev=SNAP.name,
        original_native_binary_sha256=EXPECTED_BIN_SHA,
        original_28_tensor_policy_SHA256=sha(private(LAB/"policy.txt")),
        same_eight_public_eval_prompts_131_position=True,
        QKV_stage_async_token_pairs=q,FFN_stage_async_token_pairs=ffn,
        native_process_log_sha256=sha(log.encode()),
        wall_time_s_including_initial_model_load=round(time.perf_counter()-t,3),
        original_main_or_live_release_modified=False,
        selected_model_precision_allocation_modified=False)
    if phase=="dump":
        payload=private(new/"results/c_logits_last.f32")
        reference=private(PRIOR/"serial/results/c_logits_last.f32")
        if len(payload)!=151936*4:
            raise RuntimeError("unexpected full Qwen vocab")
        values=np.frombuffer(payload,dtype="<f4").astype(float)
        baseline=np.frombuffer(reference,dtype="<f4").astype(float)
        if not np.isfinite(values).all():raise RuntimeError("nonfinite native model logits")
        d=values-baseline
        mx=float(np.max(np.abs(d)));rl=float(np.linalg.norm(d)/np.linalg.norm(baseline))
        if mx>=.001 or rl>=1e-4 or int(np.argmax(values))!=int(np.argmax(baseline)):
            raise RuntimeError("not same full output under independent projection scheduling")
        result.update(logits_sha256=sha(payload),byte_equal_prior_serial=payload==reference,
                      max_abs_logits=mx,relative_L2_logits=rl,top1=int(values.argmax()))
    if phase=="bench":
        match=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",log)
        if not match:raise RuntimeError("no real token latency proof")
        result.update(best_of_three_one_decode_ms=float(match.group(2)),tok_per_s=float(match.group(1)))
    if phase=="ppl":
        match=re.search(r"\[ppl\].*ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",log)
        rss=re.search(r"(\d+)\s+maximum resident set size",log)
        if not match or not rss or int(match.group(2))!=131:
            raise RuntimeError("original 131-position public PPL not observed")
        ppl=float(match.group(1))
        if abs(ppl-ORIGINAL_PPL)/ORIGINAL_PPL>1e-4:
            raise RuntimeError("partial CPU stage concurrency changed Qwen PPL")
        result.update(PPL=ppl,relative_PPL_vs_existing_source=abs(ppl-ORIGINAL_PPL)/ORIGINAL_PPL,peak_RSS_bytes=int(rss.group(1)))
    persist(rec,(json.dumps(result,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_QV2_HYBRID_STAGE_NATIVE_PASS",json.dumps({
        "mode":mode,"phase":phase,"log_sha256":result["native_process_log_sha256"],
        "ms":result.get("best_of_three_one_decode_ms"),
        "ppl":result.get("PPL"),"qkv":q,"ffn":ffn,
        "logit_abs":result.get("max_abs_logits"),
        "receipt_sha256":sha(private(rec))},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=["qkv_only","ffn_only"])
    p.add_argument("--phase",required=True,choices=["dump","bench","ppl"])
    v=p.parse_args()
    test(v.mode,v.phase)
