#!/usr/bin/env python3
"""Independent source-locked BEGLIN Qv2 native CPU async + quant admission.

Read-only from private EOE isolated Qv2 receipts. Verifies public fixed
8-prompt/131-position same-input source Qwen evaluation and genuinely
executed full-model native QKV/FFN pthread serial vs async. Distinguishes
measured stage-level microbench from actual sequential/full parallel.

Reject an observed activation-MILP policy if dominated on PPL, latency,
resident memory and physical projection weight bytes by a feasible control.
Reject automatic asynchronous execution unless supported by paired stable
full-model improvement. No production release or main modifications.
"""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local"
PUBLIC=LAB/"q_v2_8public_evaluation_20261022"
PARALLEL=LAB/"q_v2_actual_native_async_retry1_20261022"
FALLBACK=LAB/"q_v2_selective_native_async_20261022"
OUT=LAB/"q_v2_independent_admission_20261022"
PINS={
 "input":"352d2f1b2161ce47233dd25ffe377971b54f321ef3e6ea833c379619d4daa23b",
 "ppl":"043ee279a2f5a7e0880286aa0c1b5e5cae7692dbf4775bf53c49a27b803383a9",
 "bench":"22098c95f37a524f491c09a8775a854ab4ea6eca83ed948895e370e25653e943",
 "pair":"26096060a809b380cbf5bef749b87f09dcec1a253c3ecc99d6ef20652daec692",
 "abba":"8f03f93670400c3e2b34f83bd818fc65e1dd0bd915556dafd4f73b0201bab00",
 "guard":"6e9ed2794964aa4c3efba307744db61388730ada2aa18c81f4f65838c5da5341",
 "selective":"2217e93a9a140f11299073b5b99e19b9fa33c7fdb79db6cd02781f3afb6024c5"
}
NAMES={
 "input":PUBLIC/"input_lock_receipt.json",
 "ppl":PUBLIC/"all_ppl_public_eval_receipt.json",
 "bench":PUBLIC/"all_bench_public_eval_receipt.json",
 "pair":PARALLEL/"paired_native_async_serial_8public.json",
 "abba":PARALLEL/"source_frozen_actual_async_on_off_receipt_retry1.json",
 "guard":PARALLEL/"actual_async_guard_receipt.json",
 "selective":FALLBACK/"selective_native_async_failclosed_admission_receipt.json"
}
MASK="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
STEPS=("original_BF16_widened_F32","all_Q4G64","frozen_seed73_random",
       "former_weight_kernel_aware","v2_activation_async_MILP")
Q4_BYTES=736985088
V2_BYTES=776896512
ASYNC_RELATIVE_BENEFIT_REQUIRED=1.05 # conservative *post-hoc engineering guard*, NOT prereg scientific threshold

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()

def read():
    evidence={}
    for name,path in NAMES.items():
        if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
            raise RuntimeError("missing/private provenance "+name)
        blob=path.read_bytes()
        if sha(blob)!=PINS[name]:
            raise RuntimeError("Qv2 native evidence SHA drift "+name)
        evidence[name]=json.loads(blob)
    return evidence

def dominates(challenger:dict,candidate:dict)->bool:
    keys=("ppl","ms","rss","bytes")
    if any(not math.isfinite(float(challenger[k])) or
           not math.isfinite(float(candidate[k])) or
           challenger[k]<=0 or candidate[k]<=0 for k in keys):
        raise ValueError("invalid timing, memory or NLL comparator")
    return (all(challenger[k]<=candidate[k] for k in keys) and
            any(challenger[k]<candidate[k] for k in keys))

def audit():
    z=read()
    frozen=z["input"]
    if frozen["status"]!="INPUTS_SHA_FROZEN_NO_MODEL_OUTPUT_READ" or not frozen["all_five_conditions_have_byte_identical_inputs"]:
        raise RuntimeError("input source no longer prelocked before model output")
    if frozen["source_only_policy_sha256"]!=MASK or frozen["windows"]!=8:
        raise RuntimeError("native Qwen fixed eight-window validation provenance drift")
    if sum(frozen["token_lengths"])-len(frozen["token_lengths"])!=131:
        raise RuntimeError("native sample token count mismatch")
    ppl=z["ppl"]["models"]
    bench=z["bench"]["models"]
    if set(ppl)!=set(STEPS) or set(bench)!=set(STEPS):
        raise RuntimeError("native five-case source comparison incomplete")
    if z["ppl"]["status"]!="PASS_ALL_FIVE_PUBLIC_EVALUATIONS" or z["bench"]["status"]!="PASS_ALL_FIVE_PUBLIC_EVALUATIONS":
        raise RuntimeError("not confirmed native Qwen")
    src_rev=frozen["original_BF16_rev"]
    for name in STEPS:
        for record in (ppl[name],bench[name]):
            if record["exit"]!=0 or record["source_revision"]!=src_rev:
                raise RuntimeError("different weight or native process failure "+name)
        if ppl[name]["scored_tokens"]!=131:
            raise RuntimeError("not the same eight synthetic windows "+name)
        if ppl[name]["perplexity"]<=0 or bench[name]["single_decode_best_of_three_ms"]<=0:
            raise RuntimeError("invalid native model measurement")
    if ppl["v2_activation_async_MILP"]["source_only_policy_sha256"]!=MASK:
        raise RuntimeError("Qv2 model was not source-frozen")
    def rec(name,bytes_) -> dict:
        return dict(ppl=float(ppl[name]["perplexity"]),
                    ms=float(bench[name]["single_decode_best_of_three_ms"]),
                    rss=int(ppl[name]["max_rss_bytes"]),
                    bytes=bytes_)
    Q4=rec("all_Q4G64",Q4_BYTES)
    V2=rec("v2_activation_async_MILP",V2_BYTES)
    # Q4 beats selected V2 precision policy in every metric on these 131
    # development positions; reject irrespective of other Pareto candidates.
    if not dominates(Q4,V2):
        raise RuntimeError("source data is no longer known poor candidate; do not preempt judgment")
    if z["pair"]["status"]!="PASS_NATIVE_ASYNC_FULL_MODEL_OUTPUT_AND_PPL_PARITY":
        raise RuntimeError("native whole model async pairing not verified")
    p=z["pair"]
    if p["policy_sha256"]!=MASK or not p["real_pipeline_qkv_gateup_async"] or not p["native_ASYNC_threads_join_before_attention_and_swiglu"]:
        raise RuntimeError("not actual native Qwen pthread DAG")
    if p["before_after"]["ppl"]["async"]["PPL"]!=p["before_after"]["ppl"]["serial"]["PPL"]:
        raise RuntimeError("full native PPL drift during async")
    if p["before_after"]["ppl"]["async"]["PPL"]!=V2["ppl"]:
        raise RuntimeError("async model policy different from same public baseline")
    if p["before_after"]["bench"]["async"]["genuine_pthread_gateup_join_count"]<1 or p["before_after"]["bench"]["async"]["genuine_pthread_qkv_join_count"]<1:
        raise RuntimeError("QKV and gateup native async execution not observed")
    if p["relative_ppl_delta"]!=0:
        raise RuntimeError("true native async failed exact public NLL parity")
    abba=z["abba"]
    runs=abba["individual_runs"]
    speed_off=[float(v["ms_per_token"]) for v in runs if v["phase"]=="bench" and v["mode"]=="off"]
    speed_on=[float(v["ms_per_token"]) for v in runs if v["phase"]=="bench" and v["mode"]=="on"]
    if len(speed_off)!=2 or len(speed_on)!=2:
        raise RuntimeError("not true native paired ABBA sample")
    # This is a *post hoc* safety threshold for research candidate eligibility.
    off=sum(speed_off)/len(speed_off)
    on=sum(speed_on)/len(speed_on)
    actual_ratio=off/on
    if abs(actual_ratio-z["guard"]["measured_overall_ratio_OFF_div_ON"])>1e-10:
        raise RuntimeError("existing pairwise native timing already contradicted")
    if abs(float(p["before_after"]["bench"]["async"]["single_decode_best_of_3_ms"])-92.66)>1e-8:
        raise RuntimeError("native full-model async sample changed")
    if abs(float(p["before_after"]["bench"]["serial"]["single_decode_best_of_3_ms"])-92.04)>1e-8:
        raise RuntimeError("native full-model serial sample changed")
    if z["guard"]["async_speed_gate_pass"] or z["guard"]["eligible_for_auto_precision_policy_promotion"]:
        raise RuntimeError("some guard approved a negative candidate")
    if z["selective"]["available_auto_runtime_promotion"] or z["selective"]["default_BEGLIN_QV2_NATIVE_ASYNC"]!="0":
        raise RuntimeError("research native stage fail closed guard changed")
    if not abba["complete_first_prompt_logits_identical_on_off"]:
        raise RuntimeError("native async vs serial logits not identical")
    controls={
      n:dict(PPL=float(ppl[n]["perplexity"]),
         NLL=float(ppl[n]["nll_nats_per_token"]),
         bench_ms=float(bench[n]["single_decode_best_of_three_ms"]),
         native_tok_s=float(bench[n]["native_report_tps"]),
         maxRSS=int(ppl[n]["max_rss_bytes"]),
         source_output_binary_SHA256=ppl[n]["native_binary_sha256"])
      for n in STEPS
    }
    return dict(schema="BEGLIN-Qv2-independent-native-Pareto-async-admission/1",
        status="PASS_SOURCE_AND_NATIVE_EVIDENCE__FAIL_CANDIDATE_SPEED_QUALITY_GATES",
        source_sha_by_receipt=PINS,
        locked_evaluation_window_count=8,
        total_token_positions=131,
        original_source_model_snapshot=src_rev,
        source_activation_quality_policy_sha256=MASK,
        same_input_all_five_native_ppl_runs=True,
        source_only_MILP_precision_candidate_dominated_by_all_Q4=True,
        dominance_comparator_metrics={
            "all_Q4":Q4,"v2_activation_MILP":V2},
        five_real_native_cases=controls,
        native_actual_parallel_QKV_and_gateup_executed=True,
        output_logits_byte_equal_async_vs_serial=True,
        PPL_equal_async_vs_serial=True,
        paired_actual_one_token_native_ms={
            "serial_first_pair":p["before_after"]["bench"]["serial"]["single_decode_best_of_3_ms"],
            "async_first_pair":p["before_after"]["bench"]["async"]["single_decode_best_of_3_ms"],
            "serial_ABBA_mean":off,"async_ABBA_mean":on,
            "serial_div_async_mean_ratio":actual_ratio,
            "samples_per_mode":2},
        posthoc_conservative_async_min_speedup=ASYNC_RELATIVE_BENEFIT_REQUIRED,
        native_async_meets_min_speedup=bool(actual_ratio>=ASYNC_RELATIVE_BENEFIT_REQUIRED),
        research_default_async="disabled",
        optimized_precision_policy_auto_promotion="denied",
        optional_tradeoff="frozen random mixed improves PPL but costs more RAM and is slower than all-Q4; not universally dominant",
        original_BF16_checkpoint_modified=False,
        immutable_tailnet_release_or_repo_main_modified=False,
        full_model_sustained_latency_or_task_AUROC_measured=False,
        independent_final_IID_OOD_test_verified=False,
        existing_V6_used_24_provenance_not_known=True,
        scientific_Q_quality_verdict="NO_VERDICT",
        limitations=[
          "Only eight synthetic prompts/131 positions from publicly frozen research development, not independent final task accuracy.",
          "PPL is full teacher-forced chat-template token distribution, not a task-specific assistant-only NLL.",
          "One-token benchmark uses best-of-three internally; two ABBA samples/mode cannot establish sustained throughput confidence interval.",
          "The 1.05 async threshold is a post-hoc engineering guard rather than preregistered hypothesis.",
          "Whole-model Q4 has fewer logical projection bytes and lower RSS but differs in quantization quality tradeoffs outside the tested inputs.",
          "Actual private per-tensor source-BF16 activations remain private; no prohibited historical V6 24 records opened.",
          "Not safe to enable production/model promotion; no Metal GPU, multi-request race or cold-boot tests."
        ]
    )

def write():
    if OUT.exists():raise FileExistsError("this audit receipt already frozen; refuse replacement")
    os.umask(0o077)
    z=audit()
    OUT.mkdir(parents=True,mode=0o700)
    target=OUT/"qv2_source_locked_native_async_pareto_guard.json"
    with target.open("xb") as f:
        f.write((json.dumps(z,sort_keys=True,indent=2)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    target.chmod(0o600)
    print("BEGLIN_QV2_PARETO_AND_NATIVE_ASYNC_ADMISSION_PASS",json.dumps({
     "receipt_sha256":sha(target.read_bytes()),
     "candidate_dominated":z["source_only_MILP_precision_candidate_dominated_by_all_Q4"],
     "async_on_off_mean_speed_ratio":z["paired_actual_one_token_native_ms"]["serial_div_async_mean_ratio"],
     "async_approved":z["native_async_meets_min_speedup"],
     "promote":z["optimized_precision_policy_auto_promotion"],
     "inputs":z["locked_evaluation_window_count"]},sort_keys=True),flush=True)
if __name__=="__main__":
    write()
