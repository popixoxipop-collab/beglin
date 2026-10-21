#!/usr/bin/env python3
"""BEGLIN Qv2 SAME native binary async ON/OFF prospective dev evaluation.

Independent source-frozen precision policy and eight public token windows.
Measured within one isolated EOE Apple Silicon CPU process at a time; no
other native model runs concurrently. Benchmark order OFF/ON/ON/OFF to
reduce monotonic thermal/order confounding; PPL ON/OFF must agree.
Only QKV and gate/up independent sibling operators execute concurrently
when enabled; stage join before attention/SwiGLU, no cross-layer concurrency.

NEVER writes main, production, checkpoint or source policy. Future loading
needs explicit PPL and attention parallel gate, no automatic promotions.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,subprocess,time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_actual_native_async_retry1_20261022"
EVAL=ROOT/"lab_local/q_v2_8public_evaluation_20261022/v2_activation_async_MILP"
BASE=ROOT/"lab_local/q_v2_8public_evaluation_20261022"
CHECKPOINT=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
BIN=LAB/"beglin_Qv2_native_async"
MASK=LAB/"frozen_selected_28_roles.txt"
SCORE=BASE/"all_ppl_public_eval_receipt.json"
RECEIPT=LAB/"source_frozen_actual_async_on_off_receipt_retry1.json"
EXPECTED_BIN="fb947372495962ed67bb00f91199030b2e41d50ab283f4a5bd81ab2e51cdc054"
EXPECTED_MASK="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
EXPECTED_EVAL_PPL_SHA="043ee279a2f5a7e0880286aa0c1b5e5cae7692dbf4775bf53c49a27b803383a9"
EXPECTED_FIRST_OUTPUT_SHA="b46dc07b9b5ff32271a807d18c49444d8f0c33d0fb5533b17d83006de5d33d1e"

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def priv(p:pathlib.Path)->bytes:
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private evidence inaccessible/unsafe "+str(p))
    return p.read_bytes()
def write_private(p:pathlib.Path,data:bytes):
    if p.exists():raise FileExistsError("will not overwrite earlier run "+str(p))
    with p.open("xb") as fd:
        fd.write(data);fd.flush();os.fsync(fd.fileno())
    p.chmod(0o600)
def ensure_preflight():
    if RECEIPT.exists():raise FileExistsError("prior actual source-frozen ON/OFF receipt exists")
    if not CHECKPOINT.is_dir():raise RuntimeError("pinned original model missing")
    if sha(BIN.read_bytes())!=EXPECTED_BIN or sha(priv(MASK))!=EXPECTED_MASK:
        raise RuntimeError("full BEGLIN Qv2 native binary or fixed precision mask changed")
    if sha(priv(SCORE))!=EXPECTED_EVAL_PPL_SHA:
        raise RuntimeError("prior eight-window evaluation provenance drift")
    j=json.loads(priv(SCORE))
    expected=float(j["models"]["v2_activation_async_MILP"]["perplexity"])
    if abs(expected-827.4999)>1e-5:raise RuntimeError("prior public baseline unexpectedly differs")
    old=EVAL/"results/c_logits_last.f32"
    actual=LAB/"isolated/results/c_logits_last.f32"
    if sha(priv(old))!=EXPECTED_FIRST_OUTPUT_SHA or priv(old)!=priv(actual):
        raise RuntimeError("first prompt complete-vocab actual async vs old sequential not bitwise equal")
    for rel in ["prompt_ids.i32"]+[f"eval/ppl_work/w{k}.i32" for k in range(8)]:
        if priv(EVAL/rel)!=priv(LAB/"isolated"/rel):
            raise RuntimeError("source-evaluation token window drift: "+rel)
    return expected
def run(mode:str,phase:str,index:int)->dict:
    env=os.environ.copy()
    for x in ("QWEN_WT_PROMOTION_FILE","QWEN_INT4_BIN","QWEN_GGUF","QWEN_DENSE_QNG64_OVERRIDE"):
        if env.get(x):raise RuntimeError("unexpected external quantization override")
    env.update({
      "VECLIB_MAXIMUM_THREADS":"1",
      "QWEN_SAFETENSORS":str((CHECKPOINT/"model.safetensors").resolve()),
      "QWEN_HF_CONFIG":str((CHECKPOINT/"config.json").resolve()),
      "QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str((LAB/"isolated").resolve()),
      "QWEN_PROMPT":str((LAB/"isolated/prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":"mixed",
      "BEGLIN_Q_LAB_N8_ROLES_FILE":str(MASK.resolve()),
      "BEGLIN_QV2_NATIVE_ASYNC":"1" if mode=="on" else "0",
    })
    native=[str(BIN.resolve()),phase]
    if phase=="bench":native.append("1")
    elif phase=="ppl":native.append("8")
    else:raise ValueError("only native bench/ppl supported")
    wrapped=["/usr/bin/time","-l"]+native if phase=="ppl" else native
    t=time.monotonic()
    result=subprocess.run(wrapped,env=env,cwd=str(ROOT),capture_output=True,text=True,timeout=240)
    seconds=time.monotonic()-t
    b=(result.stdout+"\n"+result.stderr).encode()
    log=LAB/f"actual_async_retry1_{mode}_{phase}_{index}.txt"
    write_private(log,b)
    if result.returncode!=0:raise RuntimeError("native Qwen "+mode+"/"+phase+" error "+result.stderr[-1100:])
    text=b.decode()
    if "[Q-NATIVE-RESEARCH] projection_role_tensors: f32=0 n8=28 q4=168" not in text:
        raise RuntimeError("wrong actual 28n8/168q4 precision policy")
    if "[engine] safetensors" not in text or "[BEGLIN-N8-REPACKED-LAB]" not in text or "source_unchanged=1" not in text:
        raise RuntimeError("native source provenance/print guard absent")
    pattern=re.search(r"\[BEGLIN-QV2-NATIVE-ASYNC\]\s+qkv_joined=(\d+)\s+gateup_joined=(\d+)\s+serial_stage=(\d+).*production_changed=0",text)
    if mode=="on":
        if pattern is None:raise RuntimeError("async ON missing bounded-stage join counters")
        joins=[int(s) for s in pattern.groups()]
        if joins[0]<=0 or joins[1]<=0:raise RuntimeError("async ON didn't run both independent stages")
    else:
        if pattern is not None:raise RuntimeError("async OFF still entered async QKV/GU path")
        joins=[0,0,0]
    record=dict(mode=mode,phase=phase,attempt=index,wall_seconds=round(seconds,3),
         native_log_sha256=sha(b),qkv_joined=joins[0],gateup_joined=joins[1],
         serial_stage=joins[2],exit_code=result.returncode)
    if phase=="bench":
        m=re.search(r"tok_s\s+([0-9.]+)\s+ms_tok\s+([0-9.]+)",text)
        if not m:raise RuntimeError("native one-token elapsed benchmark missing")
        record.update(tokens_per_second=float(m.group(1)),ms_per_token=float(m.group(2)))
    if phase=="ppl":
        m=re.search(r"\[ppl\].*ppl\s+([0-9.]+)\s+over\s+(\d+)\s+tokens",text)
        rss=re.search(r"(\d+)\s+maximum resident set size",text)
        if not m or not rss:raise RuntimeError("native PPL or maximum RSS missing")
        record.update(perplexity=float(m.group(1)),scored_tokens=int(m.group(2)),
                      NLL_nats_per_token=math.log(float(m.group(1))),
                      RSS_peak_bytes=int(rss.group(1)))
        if record["scored_tokens"]!=131:raise RuntimeError("incorrect eight public evaluation token positions")
    print("NATIVE_QV2_ASYNC_ONOFF",json.dumps(record,sort_keys=True),flush=True)
    return record
def main():
    os.umask(0o077)
    baseline_ppl=ensure_preflight()
    runs=[]
    # paired ABBA layout combats temperature/order and includes all active
    # Q4 role pools correctly by using one physical whole-model process/run.
    for i,mode in enumerate(("off","on","on","off"),1):
        runs.append(run(mode,"bench",i))
    for i,mode in enumerate(("on","off"),1):
        runs.append(run(mode,"ppl",i))
    bymode={name:sorted([r["ms_per_token"] for r in runs if r["phase"]=="bench" and r["mode"]==name]) for name in ("on","off")}
    med={k:sum(v)/len(v) for k,v in bymode.items()}
    ppl={name:next(r for r in runs if r["phase"]=="ppl" and r["mode"]==name) for name in ("on","off")}
    if abs(ppl["on"]["perplexity"]-ppl["off"]["perplexity"])/ppl["off"]["perplexity"]>1e-4:
        raise RuntimeError("actual native async changes NLL beyond 1e-4")
    if abs(ppl["off"]["perplexity"]-baseline_ppl)/baseline_ppl>1e-4:
        raise RuntimeError("source frozen sequential native PPL drift")
    summary=dict(schema="BEGLIN-Qv2-actual-native-QKV-GU-ONOFF-measured/1",
        status="ACTUAL_FULL_NATIVE_ASYNC_ON_OFF_VERIFIED",
        original_8prompt_policy_sha=EXPECTED_MASK,
        complete_first_prompt_logits_identical_on_off=True,
        same_native_binary_SHA256=EXPECTED_BIN,
        public_eight_eval_positions=131,
        best_of_three_single_token_ms_by_mode_ABBA=med,
        measured_native_parallel_speed_ratio=med["off"]/med["on"],
        exact_same_public_eval_PPL_by_mode={k:ppl[k]["perplexity"] for k in ppl},
        PPL_peak_RSS={k:ppl[k]["RSS_peak_bytes"] for k in ppl},
        test_order=("off","on","on","off","on_ppl","off_ppl"),
        actual_native_parallel_QKV_gateup=True,
        actual_main_or_release_untouched=True,
        source_original_BF16_untouched=True,
        scientific_quality_generalization="NOT_ESTABLISHED",
        limitations=["All experiments one public first-token prompt and 131 public next-token positions",
            "Bench internally fastest of three single-token iterations, 4 ABBA repetitions; not sustained serving throughput",
            "Full CPU async stage pools are research-only and have not been stress tested under variable sequence length and concurrent user requests",
            "Mixed v2 activation policy NLL loses to previous frozen seed73 random and all4 on these development prompts",
            "No original V6 used 24 or separate heldout test, no Metal GPU, no production promotion"],
        individual_runs=runs)
    write_private(RECEIPT,(json.dumps(summary,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_ACTUAL_ASYNC_NATIVE_VERIFIED",json.dumps({
         "receipt_sha256":sha(priv(RECEIPT)),
         "real_async_speed_ratio":summary["measured_native_parallel_speed_ratio"],
         "OFF_med_ms":med["off"],"ON_med_ms":med["on"],
         "same_PPL":ppl["on"]["perplexity"]==ppl["off"]["perplexity"]},
         sort_keys=True),flush=True)
if __name__=="__main__":main()
