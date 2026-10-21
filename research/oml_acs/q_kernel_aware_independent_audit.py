#!/usr/bin/env python3
"""Independent source-only exact allocator and native Qwen execution audit.

Every component is SHA-bound. New policy was fixed before full-model logits,
PPL or token speed; previous original Q4/role-balanced comparisons are used
only as *post-selection* development metrics. Final Q science = NO_VERDICT.
"""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_kernel_aware_native_20261022"
ALLOC=ROOT/"lab_local/q_kernel_aware_allocator_20261022"
PRIOR=ROOT/"lab_local"
OUTPUT=LAB/"KERNEL_AWARE_196_NATIVE_INDEPENDENT_AUDIT.json"
SELECTION_SHA="5841c6505cb4e5805b228e7af5609ecc11f38817fc7bccd34e5272f93f400df1"
BUILD_SHA="4bbd0fa406e36340c42e6ef7c3e8727b5bbf0d76ad3e6dbd590caea246fb9dc4"
LOGIT_SHA="0fe08ecb8ce0ce0dd78a9c079d103062fce2bfb275be53fa10b343985d8da82c"
BASE_PPL={"f32":1988.5681,"all_q4":4649.3209,"prior_balanced_proxy":4747.6983,"prior_balanced_random":3411.2283}
ACTUAL_PPL=4568.8686
ACTUAL_RSS=4859985920
ACTUAL_BENCH_MS=82.96
PROCESS_IDS={"dump":"proc_19ad2226928c9fff1ccdc5d3737abb07",
             "bench":"proc_d8f9c880c7054e1fb458a21664266228",
             "ppl":"proc_f6c388f6fef452eb6921f3159911ef1d"}
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_kernel_aware_allocator as allocator

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def secure(p:Path)->bytes:
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private source missing/insecure: "+str(p))
    return p.read_bytes()
def read_json(p:Path,expected:str):
    raw=secure(p)
    if sha(raw)!=expected:raise RuntimeError("frozen source identity SHA changed")
    return json.loads(raw)

def source_only_optimum():
    raw=read_json(ALLOC/"kernel_aware_source_only_selection.json",SELECTION_SHA)
    scan=allocator.read_secure(allocator.QSCAN,allocator.EXPECTED_SCAN)
    role_profile=allocator.read_secure(allocator.PROFILE,allocator.EXPECTED_PROFILE)
    data=allocator.mk_roles(scan,role_profile)
    q4_bytes,gamma,base_score=allocator.build_optimizer(data)
    fresh,stats,_=allocator.exhaustive_groups(data,q4_bytes,gamma)
    policy=raw["policies"]["kernel_aware"]
    for key in ("kernel_aware","memory_only"):
        for field in ("counts","packed_projection_memory_bytes","source_proxy_gain","native_approx_total_decode_ms"):
            expected=raw["policies"][key][field]
            actual=fresh[key]["counts"] if field=="counts" else fresh[key][field]
            if field=="counts":
                if actual!=expected:raise RuntimeError("source-only exact solver count proof differs")
            elif abs(actual-expected)>1e-8:raise RuntimeError("source-only exact policy metric drift")
    text=secure(ALLOC/"kernel_aware_n8_roles.txt")
    if sha(text)!=policy["policy_sha256"]:
        raise RuntimeError("frozen selected tensor IDs replaced")
    lines=[ln.split() for ln in text.decode().splitlines()]
    if len(lines)!=28 or len(set(map(tuple,lines)))!=28:
        raise RuntimeError("policy not exactly 28 role layers")
    if [sum(r==role for r,layer in lines) for role in ROLES]!=policy["counts"]:
        raise RuntimeError("source-only optimizer role order drift")
    if policy["packed_projection_memory_bytes"]>raw["memory_ceiling_bytes"]:
        raise RuntimeError("selected policy memory constraint violated")
    if policy["native_approx_total_decode_ms"]>raw["kernel_aware_predicted_latency_ceiling_ms"]:
        raise RuntimeError("estimated latency constraint violated")
    if not raw["production_main_untouched"]:raise RuntimeError("unsafe source policy")
    # Counterfactual tighter constraints: source-data only, NO candidate NLL tuning.
    saved=allocator.KERNEL_AWARE_MAX_MS
    try:
        allocator.KERNEL_AWARE_MAX_MS=80.0
        strict,strict_stats,_=allocator.exhaustive_groups(data,q4_bytes,gamma)
    finally:
        allocator.KERNEL_AWARE_MAX_MS=saved
    if strict["kernel_aware"]["counts"]==policy["counts"]:
        raise RuntimeError("tightening kernel latency must materially change source-only assignment in this case")
    return dict(
      scan_sha256=allocator.EXPECTED_SCAN,
      seven_role_native_cost_sha256=allocator.EXPECTED_PROFILE,
      policy_manifest_sha256=SELECTION_SHA,
      exact_source_objective_optimality_recomputed=True,
      original_BF16_weights_pinned=True,
      actual_kernel_profile_used=True,
      model_source_prompts_not_used_for_selection=True,
      memory_limit_bytes=raw["memory_ceiling_bytes"],
      kernel_latency_ceiling_ms=raw["kernel_aware_predicted_latency_ceiling_ms"],
      role_counts=policy["counts"],
      selected_policy_sha256=policy["policy_sha256"],
      selected_projection_bytes=policy["packed_projection_memory_bytes"],
      selected_predicted_single_token_ms=policy["native_approx_total_decode_ms"],
      base_q4_projection_bytes=q4_bytes,
      proxy_residual_error=base_score-policy["source_proxy_gain"],
      reference_role_balanced_projection_bytes=830570496,
      groups_exact_memory_feasible_compositions=stats["memory_feasible"],
      groups_exact_latency_feasible_compositions=stats["kernel_time_feasible"],
      kernel_latency_sensitivity_80ms_role_counts=strict["kernel_aware"]["counts"],
      kernel_latency_sensitivity_80ms_predicted_ms=strict["kernel_aware"]["native_approx_total_decode_ms"],
      kernel_latency_sensitivity_80ms_memory_bytes=strict["kernel_aware"]["packed_projection_memory_bytes"],
      kernel_time_constraint_100ms_changed_memory_only_optimum=(policy["counts"]!=raw["policies"]["memory_only"]["counts"]),
      source_only_baseline_gammas=g    if False else float(gamma),
    )

def native_model():
    build=read_json(LAB/"build_receipt.json",BUILD_SHA)
    if not build["full_BEG_original_source_readonly"] or build["production_modified"] or build["main_modified"]:
        raise RuntimeError("native changed immutable authority")
    exe=LAB/"beglin_kernel_aware_native_Qwen"
    if not exe.is_file() or sha(exe.read_bytes())!=build["native_binary_sha256"]:
        raise RuntimeError("real BEGLIN native compiled binary SHA drift")
    selection=read_json(ALLOC/"kernel_aware_source_only_selection.json",SELECTION_SHA)
    if build["preselected_allocator_sha256"]!=SELECTION_SHA or build["preselected_28_tensor_policy_sha256"]!=selection["policies"]["kernel_aware"]["policy_sha256"]:
        raise RuntimeError("new native precision policy not fixed to prior source-only optimizer")
    for name in ("prompt_ids.i32","eval/ppl_work/w0.i32","eval/ppl_work/w1.i32"):
        original=PRIOR/"q_full_model_20261021_v2/q4"/name
        new=LAB/"chosen"/name
        if secure(original)!=secure(new):raise RuntimeError("native input/order contract changed")
    sources={
       "f32":PRIOR/"q_full_model_20261021_v2/f32/results/c_logits_last.f32",
       "all_q4":PRIOR/"q_full_model_20261021_v2/q4/results/c_logits_last.f32",
       "prior_balanced_proxy":PRIOR/"q_full_model_mixed_20261021/mixed_proxy/results/c_logits_last.f32",
       "prior_balanced_random":PRIOR/"q_full_model_mixed_20261021/mixed_random/results/c_logits_last.f32",
       "new_kernel_aware":LAB/"chosen/results/c_logits_last.f32",
    }
    vals={};hashes={}
    for name,path in sources.items():
        data=secure(path)
        if len(data)!=151936*4:raise RuntimeError("real native vocab size drift")
        v=np.frombuffer(data,dtype="<f4").astype(np.float64)
        if not np.isfinite(v).all():raise RuntimeError("nonfinite native logits")
        vals[name]=v;hashes[name]=sha(data)
    if hashes["new_kernel_aware"]!=LOGIT_SHA:raise RuntimeError("native candidate logits changed")
    f32=vals["f32"]
    p=np.exp(f32-f32.max());p/=p.sum()
    models={}
    for mode,logits in vals.items():
        q=np.exp(logits-logits.max());q/=q.sum()
        delta=logits-f32
        models[mode]=dict(
          logits_sha256=hashes[mode],
          KL_widened_F32_to_candidate_nats=float(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))))),
          relative_logit_L2=float(np.linalg.norm(delta)/np.linalg.norm(f32)),
          top1_id=int(logits.argmax()),
          same_top1_as_F32=bool(logits.argmax()==f32.argmax()),
          top5_overlap_with_F32=len(set(np.argsort(logits)[-5:])&set(np.argsort(f32)[-5:])),
          PPL_public_21_token_windows=ACTUAL_PPL if mode=="new_kernel_aware" else BASE_PPL[mode],
          NLL_public_21_token_windows=math.log(ACTUAL_PPL if mode=="new_kernel_aware" else BASE_PPL[mode])
        )
    pred=build["predicted_proxy_decode_ms"]
    if abs(pred-selection["policies"]["kernel_aware"]["native_approx_total_decode_ms"])>1e-8:
        raise RuntimeError("model predicted cost altered after policy freeze")
    if models["new_kernel_aware"]["KL_widened_F32_to_candidate_nats"]<=models["all_q4"]["KL_widened_F32_to_candidate_nats"]:
        raise RuntimeError("do not hide negative first-token KL outcome")
    if models["new_kernel_aware"]["NLL_public_21_token_windows"]<=models["prior_balanced_random"]["NLL_public_21_token_windows"]:
        raise RuntimeError("do not hide negative comparison to frozen random control")
    return dict(
       native_original_source_Qwen_and_checkpoint_unchanged=True,
       full_28_decoder_layers_executed=True,
       packed_n4_projection_count=168,
       packed_n8_repacked_projection_count=28,
       native_binary_sha256=build["native_binary_sha256"],
       first_full_vocab_logits_all_151936_finite=True,
       logits_and_PPL_by_condition=models,
       frozen_actual_best_of_3_single_token_ms=ACTUAL_BENCH_MS,
       frozen_actual_best_of_3_tokens_per_s=1000/ACTUAL_BENCH_MS,
       original_role_balanced_best_of_3_ms=116.18,
       all_q4_best_of_3_ms=75.26,
       predicted_ms=pred,
       predicted_absolute_latency_error_ms=ACTUAL_BENCH_MS-pred,
       PPL_process_observed_maximum_RSS_bytes=ACTUAL_RSS,
       case_native_process_ids=PROCESS_IDS,
       modes_have_same_public_synthetic_21_position_PPL_inputs=True,
       other_tier_results_from_prior_known_development_only=True,
       source_21_token_PPL_is_not_independent_new_holdout=True,
       tiny_eval_but_quality_proxy_failure_must_remain_public=True)

def run():
    if OUTPUT.exists():raise FileExistsError("do not overwrite source-frozen engineering evidence")
    src=source_only_optimum()
    out=native_model()
    obj=dict(
       schema="BEGLIN-kernel-aware-precision-allocator-native-verified/1",
       status="PASS_ENGINEERING_ALLOCATION__NO_QUALITY_ADVANTAGE_ESTABLISHED",
       date="2026-10-22",
       source_only_exact_integer_policy=src,
       native_BEGLIN_Qwen_28_layer_model_measurement=out,
       native_role_cost_is_per_role_L0_representative_not_full_context=True,
       condition_not_selected_on_native_logits_or_PPL=True,
       higher_output_quality_not_demonstrated=True,
       old_V6_24_used_holdout_exclusion_not_attested=True,
       no_live_release_or_main_change=True,
       no_blocked_output_sensitivity_retry=True,
       scientific_Q_final_verdict="NO_VERDICT",
       limitations=[
         "The selected policy is also the memory-only source-weight-error optimum; 100ms kernel latency constraint was nonbinding.",
         "With 80ms predicted latency, the optimizer chooses different roles based on native kernel costs; no quality scoring for that alternative.",
         "Source-relative Frobenius error is not actual model behavior sensitivity and already failed against role-matched random controls.",
         "Native Q4 6-thread and n8 repacked samples are L0 representative for each role; later layer activations can differ.",
         "Native full-model decode latency is one-token fastest of three after warmup, not sustained TPS; predicted cost is additive approximation.",
         "New policy worsens first-token KL vs all Q4 and 21-token NLL vs role-matched random. Do not suppress failures.",
         "F32 is BF16 checkpoint widened to F32, not real BF16 runtime. No Metal GPU, task AUROC, real heldout PPL.",
         "New C experimental dump initially created a 0644 logit file inside 0700 lab; file corrected to 0600; future lab loaders require restrictive umask.",
         "Repacked n8 is a distinct in-memory encoding, not bitplane-compatible until tagged as a separate kernel layout.",
         "Original V6 24 consumed IDs and third-party semantic contamination not attested. No production/model deployment."
       ]
    )
    os.umask(0o077)
    with OUTPUT.open("xb") as fp:
        fp.write((json.dumps(obj,sort_keys=True,indent=2)+"\n").encode())
        fp.flush();os.fsync(fp.fileno())
    OUTPUT.chmod(0o600)
    print("BEGLIN_KERNEL_AWARE_INDEPENDENT_NATIVE_AUDIT_PASS",json.dumps({
      "sha256":sha(secure(OUTPUT)),
      "source_only_counts":src["role_counts"],
      "observed_single_token_ms":out["frozen_actual_best_of_3_single_token_ms"],
      "predicted_ms":out["predicted_ms"],
      "observed_PPL":out["logits_and_PPL_by_condition"]["new_kernel_aware"]["PPL_public_21_token_windows"],
      "observed_KL":out["logits_and_PPL_by_condition"]["new_kernel_aware"]["KL_widened_F32_to_candidate_nats"],
      "quality_advantage":False},sort_keys=True),flush=True)
if __name__=="__main__":run()
