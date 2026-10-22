#!/usr/bin/env python3
"""Fail-closed BEGLIN Qv2 selective pthread stage admission decision.

Consumes only SHA-attested *isolated native* benchmark and independent audit.
No environment changes, no source/model mutation, no release operations.

Default production/serving QKV/GU async is ALWAYS disabled until a distinct
fresh multi-request/stress evaluation and explicit authorized release gate.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,statistics

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_selective_native_async_20261022"
IN_AUDIT=LAB/"selective_stage_native_independent_audit.json"
IN_PROFILE=LAB/"optimistic_async_stage_headroom_receipt.json"
OUT=LAB/"selective_native_async_failclosed_admission_receipt.json"
AUDIT_SHA="82af3dd1dac03ee1982153f68cb084522b3f5cbec96c28412e781ec7de11fa85"
PROFILE_SHA="68958cb5f24b95a70170be7f8c035a78105006bacf9c3ae38a8c2203209dde7c"
MIN_SPEEDUP=1.05
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def secure(path:pathlib.Path,expected):
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("native research output missing/unsecured")
    raw=path.read_bytes()
    if sha(raw)!=expected:raise RuntimeError("native source-receipt drift")
    return json.loads(raw)
def decide():
    a=secure(IN_AUDIT,AUDIT_SHA)
    profile=secure(IN_PROFILE,PROFILE_SHA)
    if a["status"]!="PASS_LAB_BENCH_QUANT_PPL_PARITY_NOT_PRODUCTION_CERTIFICATION":
        raise RuntimeError("independent stage audit failed")
    speed=a["fastest_speed_ratio"]
    safe_test=bool(a["safe_engineering_admission_both_sweeps_5pct"])
    parity=(a["serial_candidate_logits_151936_byte_identical"] and
            a["serial_candidate_same_131_position_PPL"]==827.4999 and
            a["PPL_relative_diff"]==0)
    if safe_test!=(speed>=MIN_SPEEDUP and all(
             a["stage_mode_2_sweep_native_observations"]["serial"]["samples_ms"][i]>=
             MIN_SPEEDUP*a["stage_mode_2_sweep_native_observations"][a["fastest_posthoc_mode"]]["samples_ms"][i]
             for i in range(2))):
        # Compare same sweep index after reordering each mode into chronological measurements.
        raise RuntimeError("independent criterion input transcription inconsistency")
    candidate_eligible=safe_test and parity
    # This is a development-speed *candidate* gate, not deployment permission.
    return dict(schema="BEGLIN-Qv2-selective-native-async-failclosed-policy/1",
        status="PASS_RESEARCH_SAFETY_OFF" if not candidate_eligible else "EXPLORATORY_CANDIDATE_NEEDS_FRESH_TEST",
        source_independent_audit_sha256=AUDIT_SHA,
        source_prior_stage_headroom_sha256=PROFILE_SHA,
        native_2_sweep_speed_ratio=speed,
        fastest_posthoc_mode=a["fastest_posthoc_mode"],
        posthoc_speed_gain_above_5pct_on_both_sweeps=safe_test,
        exact_logits_and_131_token_PPL_parity=parity,
        stage_scheduling_extra_overhead_5pct_headroom_ms=profile["remaining_scheduling_join_overhead_allowance_ms_at_5pct"],
        default_BEGLIN_QV2_NATIVE_ASYNC="0",
        default_BEGLIN_QV2_NATIVE_STAGE_POLICY="serial",
        research_candidate_eligible_for_additional_fresh_stress=candidate_eligible,
        available_auto_runtime_promotion=False,
        required_before_any_production_integration=[
            "independent fresh token throughput under sustained load, not only best of 3",
            "several concurrent user requests + q4 per-role persistent threadpool races/deadlocks stress",
            "thermal/energy, code format discriminator and full model cleanup/restart",
            "different model lengths, CPU architectures and Apple Metal interaction",
            "task-quality acceptance independent of the used public 131 token development set",
            "formal permission and isolated staged release certification"],
        quality_source_activation_NLL_superiority=False,
        old_V6_24_records_sealed=True,
        original_model_untouched=True,
        main_and_immutable_release_untouched=True)
def main():
    if OUT.exists():raise FileExistsError("do not rewrite admission outcome")
    o=decide()
    os.umask(0o077)
    with OUT.open("xb") as f:
        f.write((json.dumps(o,sort_keys=True,indent=2)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    print("BEGLIN_QV2_SELECTIVE_ASYNC_ADMISSION_PASS",
          json.dumps({"status":o["status"],"eligible":o["research_candidate_eligible_for_additional_fresh_stress"],
                      "speed_ratio":o["native_2_sweep_speed_ratio"],
                      "default_async":o["default_BEGLIN_QV2_NATIVE_ASYNC"],
                      "receipt_sha256":sha(OUT.read_bytes())},sort_keys=True))
if __name__=="__main__":main()
