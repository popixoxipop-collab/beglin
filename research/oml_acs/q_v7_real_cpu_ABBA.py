#!/usr/bin/env python3
"""TinyLlama source-native Q4 vs 22n8 CPU ABBA repeated token p50/p95/RSS.

Uses exact same source-training prompt, model, 2 isolated instrumented BEGLIN
C native binaries, mixed source-frozen mask, alternating two ABBA blocks.
Each process yields three 64-token decode traces after 8 warmup tokens,
measures per-token forward + full-vocab logits (but not argmax). RSS via
macOS time -l. High host load forces INCONCLUSIVE throughput verdict.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,statistics,subprocess,time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv7_tinyllama_20261022"
SOURCE=LAB/"source_arc64_tinyllama_frozen.json"
MODEL=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv7-tinyllama-model-20261022")
BINS=LAB/"native_latency_traces_instrumented"
POLICY=ROOT/"configs/research/BEGLIN_QV7_TINYLLAMA_22N8_FISHER_FROZEN_POLICY_20261022.txt"
PRE=ROOT/"configs/research/BEGLIN_QV7_NATIVE_ABBA_PRELOCK_20261022.json"
RUN=LAB/"tinyllama_cpu_2ABBA_64tokens_3rep"
PUBLIC=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_CPU_ABBA_TRACE_PUBLIC_20261022.json"
PRE_SHA="5603f4044383ad31928291caabd0a90df6d6a9a1ba580fba5015e3f40eced175"
MODES=("q4","fisher_22n8")
TRACE=re.compile(r"\[qv7_trace\] rep=(\d+) tokens=(\d+) mean_ms=([0-9.eE+-]+) p50_ms=([0-9.eE+-]+) p95_ms=([0-9.eE+-]+)")
BASE=re.compile(r"tok_s\s+([0-9.]+)\s+ms_tok\s+([0-9.]+)")
RSS=re.compile(r"(\d+)\s+maximum resident set size")
ROLES=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def sha(b):return hashlib.sha256(b).hexdigest()
def private(p,b):
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def sources():
    if RUN.exists() or PUBLIC.exists():raise FileExistsError("no same-test ABBA overwrite")
    if sha(PRE.read_bytes())!=PRE_SHA:
        raise RuntimeError("real native CPU timing preregistration tampered")
    config=json.loads(PRE.read_text())
    if config["blocks"]!=2 or config["decode_tokens_per_native_trial"]!=64 or \
       config["order_per_block"]!=["q4","fisher_22n8","fisher_22n8","q4"]:
        raise RuntimeError("ABBA schedule not preregistered")
    if sha(POLICY.read_bytes())!=config["Fisher_mask_sha256"]:
        raise RuntimeError("cannot benchmark any post-eval changed source-only mask")
    return config
def run():
    cfg=sources()
    original=json.loads(SOURCE.read_text())["source64"][0]
    if len(original["prefix_ids"])>180:
        raise RuntimeError("benchmark prompt too long for 64 sampled decode tokens")
    import struct
    arr=struct.pack("<"+"i"*len(original["prefix_ids"]),*original["prefix_ids"])
    os.umask(0o077)
    RUN.mkdir(parents=True,mode=0o700)
    cases=[]
    for block in range(2):
        for ix,mode in enumerate(cfg["order_per_block"]):
            position=len(cases);tag=f"block{block+1}_trial{ix+1}_{mode}"
            exe=BINS/f"beglin_qv7_trace_{mode}"
            expected=cfg[f"native_C_instrumented_{'q4' if mode=='q4' else 'fisher_mixed'}_SHA256"]
            if sha(exe.read_bytes())!=expected:raise RuntimeError("original research native timing executable drift "+mode)
            d=RUN/tag
            d.mkdir(mode=0o700)
            private(d/"prompt_ids.i32",arr)
            (d/"results").mkdir(mode=0o700)
            env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV5_","BEGLIN_QV7_"))}
            env.update({"VECLIB_MAXIMUM_THREADS":"1",
              "QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
              "QWEN_HF_CONFIG":str(MODEL/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
              "QWEN_BASE":str(d.resolve()),"QWEN_PROMPT":str((d/"prompt_ids.i32").resolve()),
              "BEGLIN_Q_LAB_PRECISION":"q4" if mode=="q4" else "mixed",
              "BEGLIN_QV2_NATIVE_ASYNC":"0"})
            if mode!="q4":env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(POLICY.resolve())
            load_before=os.getloadavg()
            t0=time.monotonic()
            p=subprocess.run(["/usr/bin/time","-l",str(exe),"bench","64"],env=env,cwd=str(ROOT),
                capture_output=True,text=True,timeout=600)
            wall=time.monotonic()-t0
            load_after=os.getloadavg()
            raw=(p.stdout+"\n"+p.stderr).encode()
            private(d/"timed_native_source_log.txt",raw)
            if p.returncode!=0:raise RuntimeError("real instrumented C benchmark failed "+tag+" "+p.stderr[-500:])
            tx=TRACE.findall(raw.decode())
            benchmark=BASE.findall(raw.decode())
            resident=RSS.findall(raw.decode())
            counts=ROLES.findall(raw.decode())
            expected_tier=(0,0,154) if mode=="q4" else (0,22,132)
            if len(tx)!=3 or len(benchmark)!=1 or len(resident)!=1 or len(counts)!=1 or \
               tuple(map(int,counts[0]))!=expected_tier:
                raise RuntimeError("native CPU measured F32/vocab/precision/3 trials missing "+tag)
            traces=[]
            for rep,n,mean,p50,p95 in tx:
                if int(rep)!=len(traces) or int(n)!=64 or not all(math.isfinite(float(v)) and float(v)>0 for v in (mean,p50,p95)):
                    raise RuntimeError("invalid repeated 64-token percentiles")
                traces.append({"mean_token_forward_logit_ms":float(mean),"p50_ms":float(p50),"p95_ms":float(p95)})
            row={"sequence":position,"block":block+1,"mode":mode,
                "prompt_token_sha256":sha(arr),"source_only_mask_SHA256":None if mode=="q4" else sha(POLICY.read_bytes()),
                "actual_BEG_native_binary_sha256":sha(exe.read_bytes()),
                "native_output_log_sha256":sha(raw),
                "duration_wall_seconds":wall,"CPU_load_average_before":list(load_before),
                "CPU_load_average_after":list(load_after),
                "max_process_RSS_bytes":int(resident[0]),
                "instrumented_3_trials_64_warm_decode_tokens_each":traces,
                "reported_native_best_of3_tps":float(benchmark[0][0]),
                "reported_native_best_of3_ms_tok":float(benchmark[0][1]),
                "performance_gate_cpu_load_high":load_before[0]>cfg["admission_warning_load_average1_threshold"] or load_after[0]>cfg["admission_warning_load_average1_threshold"]}
            cases.append(row)
            print("QV7_NATIVE_ABBA_TIMING",json.dumps({"mode":mode,"block":block+1,
              "trial":ix+1,"p50":np.median([a["p50_ms"] for a in traces]),
              "p95":np.median([a["p95_ms"] for a in traces]),
              "best_of3_tps":row["reported_native_best_of3_tps"],
              "RSS_MB":round(row["max_process_RSS_bytes"]/1e6,2),
              "host_load":round(load_before[0],2)},sort_keys=True),flush=True)
    records={mode:[r for r in cases if r["mode"]==mode] for mode in MODES}
    if len(records["q4"])!=4 or len(records["fisher_22n8"])!=4:
        raise RuntimeError("must actually run 4 Q4 and 4 mixed native processes")
    perf={}
    for mode,rows in records.items():
        trace=[trial for row in rows for trial in row["instrumented_3_trials_64_warm_decode_tokens_each"]]
        perf[mode]={"process_trials":len(rows),"actual_instrumented_tokens":len(trace)*64,
          "trial_mean_ms_median":float(np.median([x["mean_token_forward_logit_ms"] for x in trace])),
          "trial_p50_ms_median":float(np.median([x["p50_ms"] for x in trace])),
          "trial_p95_ms_median":float(np.median([x["p95_ms"] for x in trace])),
          "best_of3_tps_median":float(np.median([x["reported_native_best_of3_tps"] for x in rows])),
          "native_max_RSS_bytes_median":int(np.median([x["max_process_RSS_bytes"] for x in rows]))}
    ratios=[]
    for block in (1,2):
        a=[r["reported_native_best_of3_tps"] for r in records["q4"] if r["block"]==block]
        b=[r["reported_native_best_of3_tps"] for r in records["fisher_22n8"] if r["block"]==block]
        ratios.append(float(np.mean(b)/np.mean(a)-1))
    contaminated=any(x["performance_gate_cpu_load_high"] for x in cases)
    summary={"schema":"BEGLIN-Qv7-real-TinyLlama-native-2ABBA-64tokens-3rep-CPU-RSS/1",
      "status":"REAL_8_NATIVE_PROCESS_24_WARM_DECODE_TRACES_COMPLETE",
      "model_revision":cfg["model_revision"],
      "preregistration_sha256":PRE_SHA,
      "source_only_Fisher_n8_mask_sha256":sha(POLICY.read_bytes()),
      "actual_native_timed_processes":8,"actual_timed_token_count":24*64,
      "warmups_per_process":8,
      "not_full_persistent_multi_request_serving":True,
      "no_CPU_or_GPU_affinity_control":True,
      "host_high_load_any_process":contaminated,
      "performance_admission_verdict":"INCONCLUSIVE_LOAD_CONTENDED" if contaminated else "EXPLORATORY_CPU_ABBA_ONLY",
      "within_block_mixed_vs_q4_reported_bestof3_tps_fraction_diff":ratios,
      "summary":perf,
      "source_policy_latencies_predicted_separately_not_verified_as_actual":True,
      "native_per_token_forward_logit_latency_excludes_argmax":True,
      "operational_runtime_changed":False}
    evidence={"summary":summary,"process_runs":cases}
    b=(json.dumps(evidence,sort_keys=True,indent=2)+"\n").encode()
    private(RUN/"original_full_native_ABBA_receipt.json",b)
    public=dict(summary)
    public["original_private_process_logs_summary_SHA256"]=sha(b)
    public_bytes=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(public_bytes);f.flush();os.fsync(f.fileno())
    print("QV7_REAL_LLAMA_CPU_ABBA_SOURCE_N8_Q4_TRACE_FINISHED",
      json.dumps({"admission":summary["performance_admission_verdict"],
        "summary":perf,"tps_between_policy_fraction":ratios,"public_sha":sha(public_bytes)},sort_keys=True),flush=True)
if __name__=="__main__":run()
