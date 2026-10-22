#!/usr/bin/env python3
"""Real BEGLIN full-Qwen source-frozen stage-selective pthread scheduler A/B.

Runs ONE SHA-pinned binary, SAME tensor precision mask, SAME public 8-window
input for seven mode variants and original serial. Each mode gets its own
private workspace; two time-balanced forward/reverse timing sweeps.
Selects a fastest mode only as *post-hoc engineering* then independently
verifies bytewise first-logit and PPL parity against serial.

Do NOT claim scientific out-of-sample gains. Never enables production async.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,re,statistics,subprocess,time

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_selective_native_async_20261022"
BASE=LAB/"eval"
EXE=LAB/"beglin_qv2_selective_stage"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PREREG=ROOT/"configs/research/BEGLIN_QV2_SELECTIVE_NATIVE_STAGE_ABBA_PRELOCK_20261022.json"
PRELOCK_SHA="140d65b63aa3cd5dfdfae4b0edd7bb13de7157a7600e3dfd485965de208621d6"
BINARY_SHA="64fff02ea6072469bbf8e033ea0769c87082aba868d84096427a4c58a9a3b39a"
POLICY_SHA="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
MODES=("serial","qkv6","gu6","both6","qkv1","gu1","both1","frozen")
OUTPUT=LAB/"selective_stage_native_exploratory_ABBA_receipt.json"
EXPECTED_PUBLIC_PPL=827.4999

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def create(path:pathlib.Path,data:bytes):
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def secure(path:pathlib.Path):
    if not path.is_file() or path.is_symlink() or path.stat().st_mode&0o777!=0o600:
        raise RuntimeError("source public fixture absent or insecure: "+str(path))
    return path.read_bytes()
def prepare():
    if OUTPUT.exists():raise FileExistsError("cannot retune previously tested stage scheduling result")
    if sha(PREREG.read_bytes())!=PRELOCK_SHA or sha(EXE.read_bytes())!=BINARY_SHA:
        raise RuntimeError("predeclared native candidate binary/protocol changed")
    if sha(secure(LAB/"frozen_28_n8_policy.txt"))!=POLICY_SHA:
        raise RuntimeError("changed source-only BF16 activation precision mask")
    if not (SNAP/"model.safetensors").is_file():
        raise RuntimeError("frozen original Qwen checkpoint missing")
    for mode in MODES:
        folder=LAB/mode
        if folder.exists():
            raise FileExistsError("prior stage mode directory "+str(folder))
        folder.mkdir(mode=0o700)
        for name in ("results","eval","eval/ppl_work"):
            (folder/name).mkdir(mode=0o700,exist_ok=True)
        for item in ("prompt_ids.i32",)+tuple(f"eval/ppl_work/w{k}.i32" for k in range(8)):
            create(folder/item,secure(BASE/item))
def env_for(mode):
    env=os.environ.copy()
    for name in ("QWEN_GGUF","QWEN_INT4_BIN","QWEN_DENSE_QNG64_OVERRIDE","QWEN_WT_PROMOTION_FILE"):
        if name in env:raise RuntimeError("uncontrolled precision override in environment")
    folder=(LAB/mode).resolve()
    env.update({"VECLIB_MAXIMUM_THREADS":"1",
        "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
        "QWEN_HF_CONFIG":str(SNAP/"config.json"),
        "QWEN_SAFETENSORS_MAXSEQ":"256",
        "QWEN_BASE":str(folder),
        "QWEN_PROMPT":str(folder/"prompt_ids.i32"),
        "BEGLIN_Q_LAB_PRECISION":"mixed",
        "BEGLIN_Q_LAB_N8_ROLES_FILE":str((LAB/"frozen_28_n8_policy.txt").resolve()),
        "BEGLIN_QV2_NATIVE_ASYNC":"0" if mode=="serial" else "1",
        "BEGLIN_QV2_NATIVE_STAGE_POLICY":"frozen" if mode=="serial" else mode})
    return env
def run_once(mode,phase,serial):
    if mode not in MODES or phase not in ("bench","dump","ppl"):raise ValueError("unlisted mode or phase")
    cmd=[str(EXE.resolve()),phase]
    if phase=="bench":cmd.append("1")
    if phase=="ppl":cmd=["/usr/bin/time","-l"]+cmd+["8"]
    start=time.monotonic()
    v=subprocess.run(cmd,env=env_for(mode),cwd=str(ROOT),capture_output=True,text=True,timeout=180)
    elapsed=time.monotonic()-start
    content=(v.stdout+"\n"+v.stderr)
    dest=LAB/f"{phase}_{mode}_trial{serial}.log"
    create(dest,content.encode())
    if v.returncode:raise RuntimeError(f"{mode} {phase} exited {v.returncode}: "+content[-1200:])
    if "[Q-NATIVE-RESEARCH] projection_role_tensors: f32=0 n8=28 q4=168" not in content:
        raise RuntimeError("precision policy budget changed in native model")
    matches=re.findall(r"\[BEGLIN-QV2-NATIVE-ASYNC\]\s*qkv_joined=(\d+)\s*gateup_joined=(\d+)\s*serial_stage=(\d+)",content)
    qkv,gu,serialcnt=(int(x) for x in matches[-1]) if matches else (0,0,0)
    if mode=="serial":
        if matches:raise RuntimeError("original pure serial unexpectedly entered experimental stage dispatch")
    else:
        # n8 code repack GEMV still native; Q/K/V and GU join counters
        expected_qkv=mode in ("qkv6","both6","qkv1","both1","frozen")
        expected_gu=mode in ("gu6","both6","gu1","both1","frozen")
        if not matches:raise RuntimeError("enabled selective native QKV/GU should return counters")
        if (qkv>0)!=expected_qkv or (gu>0)!=expected_gu:
            raise RuntimeError(f"wrong actual joined stage counts {mode} qkv={qkv} gu={gu}")
    stats=dict(mode=mode,phase=phase,index=serial,wall_seconds=round(elapsed,3),
      log_sha256=sha(content.encode()),process_exit_code=0,
      qkv_joined=qkv,gateup_joined=gu,serial_stage=serialcnt)
    if phase=="bench":
        m=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",content)
        if not m:raise RuntimeError("actual full-model one-token benchmark missing")
        stats.update(native_report_tps=float(m.group(1)),single_decode_best_of_3_ms=float(m.group(2)))
    elif phase=="ppl":
        m=re.search(r"\[ppl\].*ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",content)
        rss=re.search(r"(\d+)\s+maximum resident set size",content)
        if not m or not rss:raise RuntimeError("native PPL/RSS malformed")
        value=float(m.group(1))
        if int(m.group(2))!=131 or abs(value-EXPECTED_PUBLIC_PPL)>0.01:
            raise RuntimeError("source-only precision policy changed NLL under stage scheduling")
        stats.update(native_public_ppl=value,scored_public_tokens=131,peak_RSS_bytes=int(rss.group(1)))
    elif phase=="dump":
        file=LAB/mode/"results/c_logits_last.f32"
        buf=secure(file)
        if len(buf)!=151936*4:raise RuntimeError("native full-vocab outputs missing")
        stats.update(logits_sha256=sha(buf),num_vocab_values=151936)
    print("BEGLIN_STAGE_NATIVE_ONE",json.dumps(stats,sort_keys=True),flush=True)
    return stats
def main():
    os.umask(0o077)
    prepare()
    # Reverse second sweep minimizes correlation of time drift with mode order.
    schedule=list(MODES)+list(reversed(MODES))
    bench=[]
    for order,mode in enumerate(schedule,1):
        bench.append(run_once(mode,"bench",order))
    values={mode:[q["single_decode_best_of_3_ms"] for q in bench if q["mode"]==mode] for mode in MODES}
    if any(len(x)!=2 for x in values.values()):raise RuntimeError("missing controlled paired runs")
    summary={mode:dict(samples_ms=values[mode],mean_ms=statistics.mean(values[mode]),
        ratio_serial_div_mode=statistics.mean(values["serial"])/statistics.mean(values[mode]))
        for mode in MODES}
    fastest=min((n for n in MODES if n!="serial"),key=lambda name:(summary[name]["mean_ms"],name))
    top_ratio=summary[fastest]["ratio_serial_div_mode"]
    # Conservative engineering admission must be satisfied ON BOTH sweeps,
    # not just a noisy best-of-3 single sample; this is NOT a statistical test.
    serials=values["serial"]
    trial1={q["mode"]:q["single_decode_best_of_3_ms"] for q in bench[:len(MODES)]}
    trial2={q["mode"]:q["single_decode_best_of_3_ms"] for q in bench[len(MODES):]}
    repeated_gain=(trial1["serial"]>=1.05*trial1[fastest] and
                    trial2["serial"]>=1.05*trial2[fastest])
    eligible=(top_ratio>=1.05 and repeated_gain)
    # Validation is post-hoc exploratory, NOT independent quality selection.
    validations=[run_once(mode,"dump",1) for mode in ("serial",fastest)]
    if validations[0]["logits_sha256"]!=validations[1]["logits_sha256"]:
        raise RuntimeError("selected native concurrency mode changed original full logits")
    ppl=[run_once(mode,"ppl",1) for mode in ("serial",fastest)]
    if ppl[0]["native_public_ppl"]!=ppl[1]["native_public_ppl"]:
        raise RuntimeError("selected native concurrency mode changed full PPL")
    result={
       "schema":"BEGLIN-Qv2-selective-QKV-GU-pthread-engineering-probe/1",
       "status":"PASS_ALL_STAGED_CALLS_AND_PARITY_NO_PRODUCTION_PROMOTION",
       "source_mode_prelock_SHA256":PRELOCK_SHA,
       "actual_pinned_research_binary_SHA256":BINARY_SHA,
       "frozen_source_only_quant_bitmask_SHA256":POLICY_SHA,
       "same_eight_public_dev_windows_token_positions":131,
       "modes":list(MODES),
       "real_native_two_reverse_sweeps":bench,
       "per_mode":summary,
       "posthoc_fastest_stage_mask":fastest,
       "conservative_min_5pct_both_sweeps_speedup":eligible,
       "posthoc_full_logit_byte_parity":True,
       "posthoc_131_position_NLL_PPL_parity":True,
       "posthoc_dump":validations,
       "posthoc_PPL":ppl,
       "source_only_decision_after_used_eval_timing_not_prospective":True,
       "candidate_cannot_auto_promote_to_production":True,
       "scientific_generalization":"NO_VERDICT",
       "source_BF16_model_untouched":True,
       "main_release_untouched":True,
       "comments":["fastest of three native one-token measurements per run; two sweeps is not robust sustained benchmark",
          "posthoc engineering winner must be retested on fresh performance data before claiming a reliable speed gain",
          "This changes dispatch only, not precision masks, and demonstrates conditional stage scheduling is valid, not per-token bit relocation",
          "No old V6 24-row holdout, no previously blocked full-model per-tensor interventions or real task AUROC"]
    }
    out=LAB/"selective_stage_native_exploratory_ABBA_receipt.json"
    create(out,(json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_SELECTIVE_NATIVE_CPU_STAGE_EXPERIMENT_COMPLETE",
      json.dumps({"receipt_sha256":sha(secure(out)),
       "fastest":fastest,"ratio":top_ratio,
       "eligible_5pct":eligible,
       "means_ms":{name:round(q["mean_ms"],5) for name,q in summary.items()},
       "quality_parity":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
