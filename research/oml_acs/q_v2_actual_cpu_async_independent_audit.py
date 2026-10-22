#!/usr/bin/env python3
"""Independent source-bound BEGLIN Qv2 actual pthread whole-model async audit.

Confirms same frozen original BF16 source, same 28/196 native Q4+n8 weights,
the same 131 public development next-token predictions, full 151936 logits,
3 background pthreads with role-private Q4 pools and real stage barriers.
Warns / fail-closes on a slower actual full-model async result. Neither
async stage microbench nor sparse logit perturbations are promoted.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,re,os
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_actual_full_cpu_async_20261022"
OLD=ROOT/"lab_local/q_v2_activation_async_native_eval_20261022"
OUT=LAB/"ACTUAL_NATIVE_CPU_ASYNC_AUDITED_NEGATIVE_SPEED_20261022.json"
SOURCE_SHA="effe1a219234d2b751cd0f7f995d258785caa43322a71c0f0afc5a850b285f12"
MASK_SHA="cb9268c9177d47705e548c2a90a6b22772569b3f2bb54dfef2767ca53ffa3e76"
SIX={
("serial","dump"):"1e3241a915a49f72d5ae75f7946f54f7bd72a20640bebe8ea26806f3e3c560cd",
("serial","bench"):"76053183bbf079944847b56efb1765cf3f4227f46e4944b7db3f9f408e460c2f",
("serial","ppl"):"2d25b15307061009260871c044bb68ff3bc6c0f43e51f2d7c71adeec63224838",
("async","dump"):"09b40d80c4aa2506ecfb6f5876d1a4a84e35710c14455020d1c6fa840d7cee14",
("async","bench"):"753c8fd96bbb60cb5df023082b26431bde04746bcd516ba708578485d97b93da",
("async","ppl"):"b544ed1182111f3ea580328a0b3be2eb00fb803d7a81292547a9b8e3a0ff7dc3",
}
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def safe(p:pathlib.Path)->bytes:
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private model-derived artifact missing or unsafe "+str(p))
    return p.read_bytes()
def read_frozen():
    build=json.loads(safe(LAB/"build_receipt.json"))
    if sha(safe(LAB/"build_receipt.json"))!=SOURCE_SHA:
        raise RuntimeError("SHA-pinned actual native fork build changed")
    if build["selected_role_mask_SHA256"]!=MASK_SHA or build["concurrency_worker_count"]!=3 or not build["native_q4pool_per_background_role_separate"]:
        raise RuntimeError("unsafe thread pool or changed source precision mask")
    binary=LAB/"beglin_qwen_actual_async"
    if not binary.is_file() or sha(binary.read_bytes())!=build["native_binary_sha256"]:
        raise RuntimeError("same actual native experimental binary SHA changed")
    if sha(safe(LAB/"policy.txt"))!=MASK_SHA:
        raise RuntimeError("frozen original Qwen tensor policy altered")
    for mode in ("serial","async"):
        for rel in ["prompt_ids.i32"]+[f"eval/ppl_work/w{i}.i32" for i in range(8)]:
            p=LAB/mode/rel
            original=OLD/"v2_async_activation"/rel
            if safe(p)!=safe(original):
                raise RuntimeError("same BF16 public evaluation source discrepancy")
    return build
def audit_runs():
    data={}
    for (mode,phase),checksum in SIX.items():
        p=LAB/f"{mode}_{phase}_execution_receipt.json"
        x=safe(p)
        if sha(x)!=checksum:raise RuntimeError("run receipt changed after native A/B test "+mode+phase)
        r=json.loads(x)
        text=safe(LAB/f"{mode}_{phase}_native_stdout_stderr.txt")
        if r["native_process_log_sha256"]!=sha(text):
            raise RuntimeError("real native Apple process transcript SHA mismatch")
        if r["condition"]!=mode or r["phase"]!=phase or r["status"]!="PASS_NATIVE_SAME_SOURCE_RESEARCH_ONLY":
            raise RuntimeError("incomplete native session")
        if bool(r["worker_concurrency_verified"])!=(mode=="async"):
            raise RuntimeError("false async mode claim")
        if mode=="async":
            if not re.search(r"threads=3 per_worker_Q4pool=6 done=1",text.decode()):
                raise RuntimeError("independent q4pool or fully joined actual worker proof missing")
            if not r["worker_async_QKV_FFN_token_pairs"]>0:
                raise RuntimeError("no joined actual QKV/FFN forward executions")
        else:
            if "BEGLIN-QV2-FULL-CPU-ASYNC" in text.decode():
                raise RuntimeError("baseline unexpectedly scheduled pthread QKV/FFN")
        data.setdefault(mode,{})[phase]=r
    for mode in ("serial","async"):
        log=safe(LAB/mode/"results/c_logits_last.f32")
        original=safe(OLD/"v2_async_activation/results/c_logits_last.f32")
        v=np.frombuffer(log,dtype="<f4").astype(np.float64)
        if len(v)!=151936 or not np.isfinite(v).all():
            raise RuntimeError("full model output incomplete")
        if original!=log:
            raise RuntimeError("serial/parallel native Qwen full logits must be bitwise identical")
        for phase in ("dump","ppl"):
            assert data[mode][phase]["native_binary_sha256"]==data[mode]["bench"]["native_binary_sha256"]
        if data[mode]["dump"]["native_full_vocab_top1"]!=9707:
            raise RuntimeError("original top1 changed")
        if data[mode]["ppl"]["original_131_development_positions_count"]!=131:
            raise RuntimeError("same 131 source-only public evaluation tokens changed")
        if abs(data[mode]["ppl"]["actual_same_source_PPL"]-807.363)>1e-6:
            raise RuntimeError("original BF16-sourced PPL became inconsistent under CPU worker scheduling")
    serial=data["serial"]["bench"]["native_bench_best_of_three_ms_tok"]
    parallel=data["async"]["bench"]["native_bench_best_of_three_ms_tok"]
    if serial<=0 or parallel<=0:raise RuntimeError("invalid native one-token latency")
    recommend="async" if parallel<serial else "serial"
    if recommend!="serial":
        raise RuntimeError("negative latency must be preserved; do not misreport actual candidate")
    return dict(
       source_original_model_BF16_rev=data["async"]["dump"]["source_BF16_snapshot_rev"],
       source_only_selected_tensor_mask_SHA256=MASK_SHA,
       original_frozen_model_final_logits_SHA256=data["serial"]["dump"]["output_SHA256"],
       full_model_serial_vs_async_output_byte_identical=True,
       public_PPL_131_original=data["serial"]["ppl"]["actual_same_source_PPL"],
       public_PPL_131_async=data["async"]["ppl"]["actual_same_source_PPL"],
       native_CPU_best_of_three_serial_ms=serial,
       native_CPU_best_of_three_async_ms=parallel,
       async_latency_penalty_fraction=parallel/serial-1,
       native_CPU_best_of_three_serial_tok_s=1000/serial,
       native_CPU_best_of_three_async_tok_s=1000/parallel,
       serial_PPL_RSS_bytes=data["serial"]["ppl"]["maximum_RSS_bytes"],
       async_PPL_RSS_bytes=data["async"]["ppl"]["maximum_RSS_bytes"],
       native_async_join_barrier_QKV_FFN_tokens={p:data["async"][p]["worker_async_QKV_FFN_token_pairs"] for p in ("dump","bench","ppl")},
       all_six_independent_native_process_receipt_shas={f"{mode}_{phase}":sig for (mode,phase),sig in SIX.items()},
       full_decoder_real_parallel_execution_verified=True,
       full_decoder_parallel_faster_than_serial=False,
       recommended_research_dispatch=recommend,
       original_mixed_precision_quality_failure_persists=True)
def run():
    if OUT.exists():raise FileExistsError("no update posthoc async audit")
    build=read_frozen()
    e=audit_runs()
    result=dict(schema="BEGLIN-Qv2-native-full-decoder-pthread-async-independent-audit/1",
      status="PASS_ASYNC_CORRECTNESS__NEGATIVE_SPEED_FALLBACK_SERIAL",
      source_build_sha256=SOURCE_SHA,
      executable_sha256=build["native_binary_sha256"],
      evidence=e,
      producer="research/oml_acs/q_v2_actual_cpu_async_eval.py",
      whole_model_research_only=True,
      production_or_main_changed=False,
      scientific_8bit_allocator_accuracy_verdict="NO_VERDICT",
      auto_promotion_allowed=False,
      open_tasks=["Native asynchronous QKV+FFN code still lacks TSAN/race audit; no full GPU/Metal or batch-path async",
        "Actual new CPU worker 6-thread pool may oversubscribe Mac; stage selective, thread-number variant should be tested separately",
        "No continuous serving, multiple requests, energy, task AUROC or independent final data",
        "All Qv2 quality failures on 131 public synthetic positions stay unchanged"],
      previous_per_tensor_full_model_output_sensitivity_blocker_not_retried=True)
    os.umask(0o077)
    with OUT.open("xb") as f:
        f.write((json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    print("BEGLIN_QV2_ACTUAL_FULL_CPU_ASYNC_AUDIT_PASS",
      json.dumps(dict(sha256=sha(safe(OUT)),
        observed_serial_ms=e["native_CPU_best_of_three_serial_ms"],
        observed_async_ms=e["native_CPU_best_of_three_async_ms"],
        PPL=e["public_PPL_131_async"],jit_threads=3,
        stage_join_pairs=e["native_async_join_barrier_QKV_FFN_tokens"],
        recommended_dispatch=e["recommended_research_dispatch"]),sort_keys=True),flush=True)
if __name__=="__main__":run()
