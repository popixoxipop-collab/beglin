#!/usr/bin/env python3
"""Independent engineering-only BEGLIN Qv2 selective pthread scheduler audit.

Rechecks frozen SHA provenance, same-model public input bytes, native stage
join counters, eight modes in forward/reverse order and selected posthoc
full logits/PPL equality. Does NOT rerun a model, calculate tensor causal
sensitivity, certify speed generalization, or alter production defaults.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,statistics

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_selective_native_async_20261022"
PREREG=ROOT/"configs/research/BEGLIN_QV2_SELECTIVE_NATIVE_STAGE_ABBA_PRELOCK_20261022.json"
SRC=LAB/"qwen_qv2_selective_stage.c"
BIN=LAB/"beglin_qv2_selective_stage"
BUILD=LAB/"build_receipt.json"
RECEIPT=LAB/"selective_stage_native_exploratory_ABBA_receipt.json"
OUT=LAB/"selective_stage_native_independent_audit.json"
ORIGINAL_BF16_REV="4ae77cb209f06199b8df1c94e21ff341332a3a89"
ORIGINAL_MASK_SHA="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
EXPECTED_CONFIG_SHA="140d65b63aa3cd5dfdfae4b0edd7bb13de7157a7600e3dfd485965de208621d6"
MODE_ORDER=("serial","qkv6","gu6","both6","qkv1","gu1","both1","frozen")
BIN_SHA="64fff02ea6072469bbf8e033ea0769c87082aba868d84096427a4c58a9a3b39a"

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def secure(p:pathlib.Path)->bytes:
    if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
        raise RuntimeError("private artifact missing/insecure: "+str(p))
    return p.read_bytes()
def inspect():
    if not RECEIPT.is_file():raise FileNotFoundError("native stage experiment has not completed")
    if sha(PREREG.read_bytes())!=EXPECTED_CONFIG_SHA or sha(secure(LAB/"frozen_28_n8_policy.txt"))!=ORIGINAL_MASK_SHA:
        raise RuntimeError("prelocked source precision/stages changed")
    build=json.loads(secure(BUILD))
    if build["native_both_stage_joins_preserved"] is not True or build["source_BF16_cache_revision"]!=ORIGINAL_BF16_REV:
        raise RuntimeError("not same source/role model")
    if sha(SRC.read_bytes())!=build["isolated_new_C_sha256"]:
        raise RuntimeError("selected C source mutated after build")
    if sha(BIN.read_bytes())!=build["new_native_binary_sha256"] or sha(BIN.read_bytes())!=BIN_SHA:
        raise RuntimeError("native binary source sha changed")
    data=json.loads(secure(RECEIPT))
    if data["status"]!="PASS_ALL_STAGED_CALLS_AND_PARITY_NO_PRODUCTION_PROMOTION":
        raise RuntimeError("stage native status is not success")
    if data["source_mode_prelock_SHA256"]!=EXPECTED_CONFIG_SHA or data["frozen_source_only_quant_bitmask_SHA256"]!=ORIGINAL_MASK_SHA:
        raise RuntimeError("candidate tuned after source precision selection")
    if tuple(data["modes"])!=MODE_ORDER:
        raise RuntimeError("missing eight locked candidate modes")
    bench=data["real_native_two_reverse_sweeps"]
    if len(bench)!=16:
        raise RuntimeError("two independent balanced sweeps required")
    if [r["mode"] for r in bench[:8]]!=list(MODE_ORDER) or [r["mode"] for r in bench[8:]]!=list(reversed(MODE_ORDER)):
        raise RuntimeError("native benchmark run order differs from locked two sweeps")
    per_mode={}
    for mode in MODE_ORDER:
        rows=[r for r in bench if r["mode"]==mode]
        if len(rows)!=2 or any(r["phase"]!="bench" or r["process_exit_code"]!=0 for r in rows):
            raise RuntimeError("missing/failed same-binary bench in "+mode)
        for r in rows:
            log=secure(LAB/f"bench_{mode}_trial{r['index']}.log")
            if sha(log)!=r["log_sha256"]:
                raise RuntimeError("native stdout receipt does not match logged timing/counters")
            if (r["qkv_joined"]>0)!= (mode in ("qkv6","both6","qkv1","both1","frozen")):
                raise RuntimeError("native QKV worker dispatch evidence drift")
            if (r["gateup_joined"]>0)!=(mode in ("gu6","both6","gu1","both1","frozen")):
                raise RuntimeError("native GU worker dispatch evidence drift")
            if not math.isfinite(r["single_decode_best_of_3_ms"]) or r["single_decode_best_of_3_ms"]<=0:
                raise RuntimeError("bad native timing")
        expected=statistics.mean(r["single_decode_best_of_3_ms"] for r in rows)
        if abs(data["per_mode"][mode]["mean_ms"]-expected)>1e-6:
            raise RuntimeError("not source-observed mean native timing")
        per_mode[mode]=dict(samples_ms=[r["single_decode_best_of_3_ms"] for r in rows],
             mean_ms=expected,
             ratio_original_serial_div_mode=data["per_mode"][mode]["ratio_serial_div_mode"])
    quickest=min((n for n in MODE_ORDER if n!="serial"),key=lambda n:(per_mode[n]["mean_ms"],n))
    if quickest!=data["posthoc_fastest_stage_mask"]:
        raise RuntimeError("posthoc stage winner tampered")
    ratio=per_mode["serial"]["mean_ms"]/per_mode[quickest]["mean_ms"]
    if abs(ratio-per_mode[quickest]["ratio_original_serial_div_mode"])>1e-8:
        raise RuntimeError("misstated measured speed gain")
    if ratio<1.05:
        eligible=False
    else:
        ids1={r["mode"]:r["single_decode_best_of_3_ms"] for r in bench[:8]}
        ids2={r["mode"]:r["single_decode_best_of_3_ms"] for r in bench[8:]}
        eligible=(ids1["serial"]>=1.05*ids1[quickest] and ids2["serial"]>=1.05*ids2[quickest])
    if bool(data["conservative_min_5pct_both_sweeps_speedup"])!=eligible:
        raise RuntimeError("do not fake conservative speed eligibility")
    compare=data["posthoc_dump"]
    if [r["mode"] for r in compare]!=["serial",quickest]:
        raise RuntimeError("the original serial output not first in parity comparison")
    for r in compare:
        path=LAB/r["mode"]/"results/c_logits_last.f32"
        raw=secure(path)
        if len(raw)!=151936*4 or sha(raw)!=r["logits_sha256"]:
            raise RuntimeError("model emitted wrong raw logit tensor")
    if compare[0]["logits_sha256"]!=compare[1]["logits_sha256"] or not data["posthoc_full_logit_byte_parity"]:
        raise RuntimeError("full model output changed under selective concurrency")
    pp=data["posthoc_PPL"]
    if [q["mode"] for q in pp]!=["serial",quickest] or pp[0]["native_public_ppl"]!=pp[1]["native_public_ppl"] or not data["posthoc_131_position_NLL_PPL_parity"]:
        raise RuntimeError("PPL discrepancy not admitted")
    if pp[0]["scored_public_tokens"]!=131 or pp[1]["scored_public_tokens"]!=131:
        raise RuntimeError("wrong public 8-window PPL token positions")
    if data["candidate_cannot_auto_promote_to_production"] is not True or data["source_only_decision_after_used_eval_timing_not_prospective"] is not True:
        raise RuntimeError("scientific honesty / deployment guard failure")
    # Existing users' prior V6 and future unseen task tests remain unopened.
    out={
      "schema":"BEGLIN-Qv2-selective-native-QKV-GU-independent-audit/1",
      "status":"PASS_LAB_BENCH_QUANT_PPL_PARITY_NOT_PRODUCTION_CERTIFICATION",
      "prelocked_source_protocol_sha256":EXPECTED_CONFIG_SHA,
      "original_BF16_snapshot_revision":ORIGINAL_BF16_REV,
      "selected_196_tensor_bitmask_SHA256":ORIGINAL_MASK_SHA,
      "C_research_source_sha256":build["isolated_new_C_sha256"],
      "single_native_binary_sha256":BIN_SHA,
      "private_original_receipt_sha256":sha(secure(RECEIPT)),
      "stage_mode_2_sweep_native_observations":per_mode,
      "fastest_posthoc_mode":quickest,
      "fastest_speed_ratio":ratio,
      "safe_engineering_admission_both_sweeps_5pct":eligible,
      "recommended_operating_stage_policy":"serial" if not eligible else "EXPLORATORY_ONLY_NEEDS_FRESH_REPLICATES",
      "serial_candidate_logits_151936_byte_identical":True,
      "serial_candidate_same_131_position_PPL":pp[0]["native_public_ppl"],
      "PPL_relative_diff":0.0,
      "calibration_and_eval_non_private_source":True,
      "all_output_files_private_0600":True,
      "actual_main_or_release_touched":False,
      "old_V6_24_consumed_ids_read":False,
      "real_task_AUROC_heldout_tested":False,
      "not_latency_per_token_dynamic_adaptation":True,
      "caveats":["Mode fastest derived on same 8 public eval prompts, post-hoc engineering only.",
          "Single token fastest-of-three timing is not sustained throughput; 2 sweeps not statistically robust.",
          "Q4 6t per-role pools include possible CPU oversubscription even if stage kernel alone faster.",
          "Same quant bit policy and source-BF16 model, NO new dynamic precision allocation.",
          "Do not turn on production real async without independent performance and multi-request stress gates."]
    }
    return out
def main():
    if OUT.exists():raise FileExistsError("cannot overwrite prior completed independent audit")
    o=inspect()
    with OUT.open("xb") as f:
        f.write((json.dumps(o,sort_keys=True,indent=2)+"\n").encode())
        f.flush()
    OUT.chmod(0o600)
    print("BEGLIN_QV2_SELECTIVE_STAGE_INDEPENDENT_AUDIT_PASS",
          json.dumps({"private_receipt_sha256":o["private_original_receipt_sha256"],
                      "fastest_mode":o["fastest_posthoc_mode"],
                      "serial_ms":o["stage_mode_2_sweep_native_observations"]["serial"]["mean_ms"],
                      "winner_mean_ms":o["stage_mode_2_sweep_native_observations"][o["fastest_posthoc_mode"]]["mean_ms"],
                      "safe_speed_gate":o["safe_engineering_admission_both_sweeps_5pct"],
                      "audit_sha256":sha(secure(OUT))},sort_keys=True),flush=True)
if __name__=="__main__":main()
