#!/usr/bin/env python3
"""Read-only research Qv2 async admission gate; FAIL CLOSED, no live toggle.

Run only against SHA-pinned actual whole Qwen ON/OFF CPU measurements,
same binary and locked public 8 windows (131 positions). A conservative
5% speed threshold and noninferior PPL to both allQ4 and prior seed73
random control are engineering thresholds, NOT preregistered science.
Regardless of score, this artifact cannot activate production.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,statistics,re

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_actual_native_async_retry1_20261022"
EVAL=ROOT/"lab_local/q_v2_8public_evaluation_20261022"
OUT=LAB/"actual_async_guard_receipt.json"
FILES={
  "ABBA":(LAB/"source_frozen_actual_async_on_off_receipt_retry1.json",
           "8f03f93670400c3e2b34f83bdc1d8fc65e1dd0bd915556dafd4f73b0201bab00"),
  "firstpair":(LAB/"paired_native_async_serial_8public.json",
           "26096060a809b380cbf5bef749b87f09dcec1a253c3ecc99d6ef20652daec692"),
  "public_eval":(EVAL/"all_ppl_public_eval_receipt.json",
           "043ee279a2f5a7e0880286aa0c1b5e5cae7692dbf4775bf53c49a27b803383a9"),
  "existing_audit":(LAB/"BEGLIN_QV2_ACTIVATION_ASYNC_INDEPENDENT_AUDIT.json",
           "2f326545ef39f02988138f6c0c0d50ea23bc7c528fcb68b23c1b96144b6af438")
}
EXPECTED_BINARY="fb947372495962ed67bb00f91199030b2e41d50ab283f4a5bd81ab2e51cdc054"
EXPECTED_POLICY="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
EXPECTED_LOGITS="b46dc07b9b5ff32271a807d18c49444d8f0c33d0fb5533b17d83006de5d33d1e"
REQUIRED_SPEEDUP_RATIO=1.05
MAX_RELATIVE_PPL_DRIFT=1e-4

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private(path:pathlib.Path)->bytes:
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("source or private receipt missing/insecure "+str(path))
    return path.read_bytes()
def inputs():
    rows={}
    for name,(path,checksum) in FILES.items():
        content=private(path)
        if sha(content)!=checksum:raise RuntimeError("pinned original source evidence drift "+name)
        rows[name]=json.loads(content)
    if sha((LAB/"beglin_Qv2_native_async").read_bytes())!=EXPECTED_BINARY:
        raise RuntimeError("separate native Qwen binary was replaced")
    if sha(private(LAB/"frozen_selected_28_roles.txt"))!=EXPECTED_POLICY:
        raise RuntimeError("frozen 28 tensor allocation mask changed")
    cur=private(LAB/"isolated/results/c_logits_last.f32")
    prior=private(EVAL/"v2_activation_async_MILP/results/c_logits_last.f32")
    if sha(cur)!=EXPECTED_LOGITS or cur!=prior:
        raise RuntimeError("full original 151936 logits not exactly equal")
    return rows

def audit():
    rows=inputs()
    a=rows["ABBA"]
    pair=rows["firstpair"]
    prior=rows["public_eval"]["models"]
    if a["status"]!="ACTUAL_FULL_NATIVE_ASYNC_ON_OFF_VERIFIED":
        raise RuntimeError("not a complete actual native on/off ABBA run")
    if pair["status"]!="PASS_NATIVE_ASYNC_FULL_MODEL_OUTPUT_AND_PPL_PARITY":
        raise RuntimeError("independent prior real pthread Qwen trial failed")
    if not rows["existing_audit"]["actual_native_pthread_execution"]["output_logits_identical_to_former_serial"]:
        raise RuntimeError("source verified native logit equal check missing")
    sessions=a["individual_runs"]
    if len(sessions)!=6 or [x["mode"] for x in sessions[:4]]!=["off","on","on","off"]:
        raise RuntimeError("expected two on and two off ABBA stages")
    off=[float(v["ms_per_token"]) for v in sessions[:4] if v["mode"]=="off"]
    on=[float(v["ms_per_token"]) for v in sessions[:4] if v["mode"]=="on"]
    off.append(float(pair["before_after"]["bench"]["serial"]["single_decode_best_of_3_ms"]))
    on.append(float(pair["before_after"]["bench"]["async"]["single_decode_best_of_3_ms"]))
    if len(off)!=3 or len(on)!=3:raise RuntimeError("insufficient native timing points")
    ratio=statistics.mean(off)/statistics.mean(on)
    PPL=[q for q in sessions if q["phase"]=="ppl"]
    if len(PPL)!=2 or {q["mode"] for q in PPL}!={"on","off"}:
        raise RuntimeError("not same 8 public source windows for native paired NLL")
    ppls={q["mode"]:float(q["perplexity"]) for q in PPL}
    if any(q["scored_tokens"]!=131 for q in PPL):
        raise RuntimeError("token count disagrees with preregistered source evaluation")
    if abs(ppls["on"]-ppls["off"])/ppls["off"]>MAX_RELATIVE_PPL_DRIFT:
        raise RuntimeError("native async changes predicted source NLL materially")
    if ppls["off"]!=prior["v2_activation_async_MILP"]["perplexity"]:
        raise RuntimeError("source policy or model baseline NLL drift")
    if pair["before_after"]["ppl"]["async"]["PPL"]!=ppls["on"] or pair["before_after"]["ppl"]["serial"]["PPL"]!=ppls["off"]:
        raise RuntimeError("the separate actual native pthread trial disagrees on fixed PPL")
    for q in sessions:
        file=LAB/f"actual_async_retry1_{q['mode']}_{q['phase']}_{q['attempt']}.txt"
        if sha(private(file))!=q["native_log_sha256"]:
            raise RuntimeError("actual hardware ON/OFF log changed")
        if q["mode"]=="on" and min(q["qkv_joined"],q["gateup_joined"])<=0:
            raise RuntimeError("async stage did not execute in actual model")
        if q["mode"]=="off" and max(q["qkv_joined"],q["gateup_joined"])>0:
            raise RuntimeError("serial baseline accidentally entered async stage")
    baseline=min(float(prior["all_Q4G64"]["perplexity"]),float(prior["frozen_seed73_random"]["perplexity"]))
    async_gain_pass=ratio>=REQUIRED_SPEEDUP_RATIO
    quality_noninferiority=ppls["on"]<=baseline
    return {
      "schema":"BEGLIN-Qv2-source-locked-actual-pthread-eligibility/1",
      "status":"PASS_INDEPENDENT_AUDIT__REJECT_ASYNC_PROMOTION_AND_QV2_BIT_POLICY",
      "observed_full_model_native_Qwen_CPU_async":True,
      "native_binary_sha256":EXPECTED_BINARY,
      "selected_role_mask_sha256":EXPECTED_POLICY,
      "complete_first_prompt_full_logits_byte_equal":True,
      "fixed_public_eval_token_positions":131,
      "timed_native_OFF_single_token_ms":off,
      "timed_native_ON_single_token_ms":on,
      "means_ms":{"off":statistics.mean(off),"on":statistics.mean(on)},
      "measured_overall_ratio_OFF_div_ON":ratio,
      "conservative_engineering_async_min_speedup_ratio":REQUIRED_SPEEDUP_RATIO,
      "async_speed_gate_pass":async_gain_pass,
      "fixed_public_PPL_ON_OFF":ppls,
      "same_131_token_PPL":ppls["on"]==ppls["off"],
      "native_development_seed73_random_PPL":float(prior["frozen_seed73_random"]["perplexity"]),
      "native_development_all_Q4_PPL":float(prior["all_Q4G64"]["perplexity"]),
      "quantized_source_quality_gate_pass":quality_noninferiority,
      "recommended_default_research_env_BEGLIN_QV2_NATIVE_ASYNC":"0",
      "approval_to_modify_production_or_main":False,
      "eligible_for_auto_precision_policy_promotion":False,
      "quality_hypothesis_verdict":"NO_VERDICT",
      "original_model_checkpoint_untouched":True,
      "original_V6_24_used_rows_sealed":True,
      "calibration_source_private_196_activations_not_shared":True,
      "source_sha256_by_receipt":{k:v[1] for k,v in FILES.items()},
      "limitations":[
        "Five-percent speed threshold is a conservative post-hoc engineering risk gate, not a preregistered statistical boundary.",
        "Only three single-token timing samples/mode, each best of 3 within native benchmark; not throughput or thermal stress.",
        "Qv2 source activation diagonal precision mask loses public NLL vs both prior seed73 random and all4.",
        "Native experimental stage executor has not been subjected to concurrent multi-request race/latency testing or Metal GPU deployment.",
        "Unable to rule out past V6 24 used-row provenance; independent real task AUROC not measured."
      ]
    }

def run():
    if OUT.exists():raise FileExistsError("will not overwrite prior release gate")
    report=audit()
    os.umask(0o077)
    with OUT.open("xb") as f:
        f.write((json.dumps(report,indent=2,sort_keys=True)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    print("BEGLIN_QV2_NATIVE_ASYNC_RELEASE_GATE_PASS",
      json.dumps({"receipt_SHA256":sha(private(OUT)),
        "speed_ratio":report["measured_overall_ratio_OFF_div_ON"],
        "speed_gate":report["async_speed_gate_pass"],
        "precision_quality_gate":report["quantized_source_quality_gate_pass"],
        "production_touched":False},sort_keys=True),flush=True)
if __name__=="__main__":run()
