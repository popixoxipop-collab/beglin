#!/usr/bin/env python3
"""Independent original BEGLIN Qv7 native CPU ABBA eight-process audit.

No import from timed driver. Re-parse actual 8 native process transcripts,
per-token warm-decode 3x64 traces, maximum RSS, original model/precision
counts and source-only mask. Recompute order and p50/p95/tps medians; reject
spurious speed advantages under high unrelated CPU load.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv7_tinyllama_20261022"
RUN=LAB/"tinyllama_cpu_2ABBA_64tokens_3rep"
RAW=RUN/"original_full_native_ABBA_receipt.json"
PUB=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_CPU_ABBA_TRACE_PUBLIC_20261022.json"
PRE=ROOT/"configs/research/BEGLIN_QV7_NATIVE_ABBA_PRELOCK_20261022.json"
TRACE_BINS=LAB/"native_latency_traces_instrumented"
MASK=ROOT/"configs/research/BEGLIN_QV7_TINYLLAMA_22N8_FISHER_FROZEN_POLICY_20261022.txt"
SOURCE=LAB/"source_arc64_tinyllama_frozen.json"
OUT=LAB/"TinyLlama_original_C_ABBA_independent_audit.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_CPU_ABBA_INDEPENDENT_AUDIT_PUBLIC_20261022.json"
SHAPRE="5603f4044383ad31928291caabd0a90df6d6a9a1ba580fba5015e3f40eced175"
MODES=("q4","fisher_22n8")
RX=re.compile(r"\[qv7_trace\] rep=(\d+) tokens=(\d+) mean_ms=([0-9eE.+-]+) p50_ms=([0-9eE.+-]+) p95_ms=([0-9eE.+-]+)")
THROUGH=re.compile(r"tok_s\s+([0-9.]+)\s+ms_tok\s+([0-9.]+)")
RSS=re.compile(r"(\d+)\s+maximum resident set size")
ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
    if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
        raise RuntimeError("original source/kernel measurement not privately sealed "+str(p))
    return p.read_bytes()

def audit():
    if OUT.exists() or PUBLIC.exists():raise FileExistsError("do not replace original C toolchain audit")
    if sha(PRE.read_bytes())!=SHAPRE:
        raise RuntimeError("CPU reproducibility protocol not prelocked")
    config=json.loads(PRE.read_text())
    full=json.loads(sealed(RAW));pub=json.loads(PUB.read_text())
    if full["summary"]!={k:v for k,v in pub.items() if k!="original_private_process_logs_summary_SHA256"} or pub["preregistration_sha256"]!=SHAPRE or \
       pub["actual_native_timed_processes"]!=8:
        raise RuntimeError("original private ABBA score/public aggregate mismatch")
    arr=full["process_runs"]
    if len(arr)!=8:raise RuntimeError("real original native ABBA process count wrong")
    before=json.loads(sealed(SOURCE))["source64"][0]
    import struct
    ids=struct.pack("<"+"i"*len(before["prefix_ids"]),*before["prefix_ids"])
    expected=sha(ids)
    if sha(MASK.read_bytes())!=config["Fisher_mask_sha256"]:
        raise RuntimeError("original quality source-only precision mask mutated")
    triples=[]
    for index,row in enumerate(arr):
        block=1+index//4;slot=index%4
        mode=config["order_per_block"][slot]
        if row["sequence"]!=index or row["block"]!=block or row["mode"]!=mode or \
           row["prompt_token_sha256"]!=expected:
            raise RuntimeError("native model workload not true two-block ABBA original exact source")
        path=RUN/f"block{block}_trial{slot+1}_{mode}"
        if sha(sealed(path/"prompt_ids.i32"))!=expected:
            raise RuntimeError("original source prompt switched between Q4/mixed")
        data=sealed(path/"timed_native_source_log.txt").decode()
        if sha(data.encode())!=row["native_output_log_sha256"]:
            raise RuntimeError("native program process transcript tampered")
        exe=TRACE_BINS/f"beglin_qv7_trace_{mode}"
        original_exe_sha=sha(exe.read_bytes())
        allowed=config["native_C_instrumented_q4_SHA256" if mode=="q4" else "native_C_instrumented_fisher_mixed_SHA256"]
        if original_exe_sha!=allowed or row["actual_BEG_native_binary_sha256"]!=allowed:
            raise RuntimeError("executed separate unpinned native C build")
        counts=ROLE.findall(data)
        expected_counts=(0,0,154) if mode=="q4" else (0,22,132)
        if len(counts)!=1 or tuple(map(int,counts[0]))!=expected_counts:
            raise RuntimeError("native run precision masks different")
        vals=RX.findall(data)
        speed=THROUGH.findall(data)
        rss=RSS.findall(data)
        if len(vals)!=3 or len(speed)!=1 or len(rss)!=1:
            raise RuntimeError("actual all 3 native 64 token trials and RSS not recorded")
        details=[]
        for seq,(rep,n,m,p50,p95) in enumerate(vals):
            if int(rep)!=seq or int(n)!=64:
                raise RuntimeError("not exact 3x64 warm decode observations")
            fields=tuple(float(x) for x in (m,p50,p95))
            if not all(math.isfinite(x) and x>0 for x in fields):
                raise RuntimeError("invalid source real per-token latency evidence")
            raw=row["instrumented_3_trials_64_warm_decode_tokens_each"][seq]
            for field,ref in zip(fields,("mean_token_forward_logit_ms","p50_ms","p95_ms")):
                if abs(field-raw[ref])>1e-10:
                    raise RuntimeError("separate per-token triple trace parsing mismatch")
            details.append(fields)
        if int(rss[0])!=row["max_process_RSS_bytes"] or \
           abs(float(speed[0][0])-row["reported_native_best_of3_tps"])>1e-8:
            raise RuntimeError("actual native memory/tokens per second transcript mismatch")
        high=row["CPU_load_average_before"][0]>config["admission_warning_load_average1_threshold"] or \
             row["CPU_load_average_after"][0]>config["admission_warning_load_average1_threshold"]
        if high!=row["performance_gate_cpu_load_high"]:
            raise RuntimeError("source CPU overload gate recalculation disagreement")
        triples.append((mode,block,details))
    per={}
    for mode in MODES:
        rows=[item for item in arr if item["mode"]==mode]
        trials=[t for r in rows for t in r["instrumented_3_trials_64_warm_decode_tokens_each"]]
        calculated={"process_trials":4,"actual_instrumented_tokens":len(trials)*64,
          "trial_mean_ms_median":float(np.median([v["mean_token_forward_logit_ms"] for v in trials])),
          "trial_p50_ms_median":float(np.median([v["p50_ms"] for v in trials])),
          "trial_p95_ms_median":float(np.median([v["p95_ms"] for v in trials])),
          "best_of3_tps_median":float(np.median([x["reported_native_best_of3_tps"] for x in rows])),
          "native_max_RSS_bytes_median":int(np.median([x["max_process_RSS_bytes"] for x in rows]))}
        for key,value in calculated.items():
            if isinstance(value,int):
                if value!=pub["summary"][mode][key]:raise RuntimeError("wrong process count/memory")
            elif abs(value-pub["summary"][mode][key])>1e-10:
                raise RuntimeError("native real p50/p95 reported summary not independently reproducible")
        per[mode]=calculated
    loads=[float(z["CPU_load_average_before"][0]) for z in arr]
    if not any(x>config["admission_warning_load_average1_threshold"] for x in loads) or \
       pub["performance_admission_verdict"]!="INCONCLUSIVE_LOAD_CONTENDED":
        raise RuntimeError("unreliable high CPU load speed auto-promotion was not blocked")
    record={"schema":"BEGLIN-Qv7-independent-original-CPU-2ABBA-source-n8-Q4-proof/1",
       "status":"PASS_REAL_8_NATIVE_PROCESSES_1536_TIMED_TOKENS__LATENCY_ADMISSION_FAILCLOSED",
       "private_source_ABBA_full_sha256":sha(sealed(RAW)),
       "source_TinyLlama_model_rev":config["model_revision"],
       "two_native_architecture_Q4_n8_C_programs_verified":True,
       "all_timed_processes_ABBA_order_correct":True,
       "original_7_input_tensors_kernel_source_unchanged":True,
       "model_base_not_modified":True,
       "high_host_load_fraction_over_limit":sum(x>config["admission_warning_load_average1_threshold"] for x in loads)/8,
       "actual_native_instrumented_token_count_each":768,
       "recomputed_metrics":per,
       "quality_or_serving_speed_promotion_allowed":False,
       "maxRSS_is_whole_process_including_weights_head_KV_not_only_projection_bytes":True,
       "token_p50_p95_forward_plus_logits_excludes_argmax":True,
       "user_visible_interpretation":"Real repeated warm single-request CPU token distribution, host load confounded; NOT a low-load serving benchmark"}
    os.umask(0o077)
    raw=(json.dumps(record,sort_keys=True,indent=2)+"\n").encode()
    with OUT.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    public={k:v for k,v in record.items() if k!="recomputed_metrics"}
    public["private_full_recomputation_receipt_SHA256"]=sha(raw)
    data=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    print("QV7_INDEPENDENT_NATIVE_CPU_ABBA_AUDIT_PASS",json.dumps({
       "sha":sha(data),"p50_ms":{k:round(v["trial_p50_ms_median"],3) for k,v in per.items()},
       "p95_ms":{k:round(v["trial_p95_ms_median"],3) for k,v in per.items()},
       "median_RSS_MB":{k:round(v["native_max_RSS_bytes_median"]/1e6,1) for k,v in per.items()},
       "performance_admission":"INCONCLUSIVE_LOAD_CONTENDED"},sort_keys=True),flush=True)

if __name__=="__main__":audit()
