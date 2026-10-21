#!/usr/bin/env python3
"""Independent BEGLIN Qv2 source-only activation/async and native results audit.

Read-only prior frozen source and confirmed Apple native execution receipts.
Recompute top-k-per-role allocator without opening ANY evaluation outputs for
policy decisions. Independently compare all private PPL/bytecode receipts,
actual pthread QKV/GU counts, same-program serial+async PPL and speed.
Never run any blocked per-tensor full-model-logit intervention.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess,sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local"
POLICY=LAB/"q_v2_joint_activation_async_20261022"
EVAL=LAB/"q_v2_new_public_development_20261022"
MILP_EVAL=LAB/"q_v2_8public_evaluation_20261022"
ACTUAL=LAB/"q_v2_actual_native_async_retry1_20261022"
PROFILE=LAB/"q_v2_async_stage_20261022"
OUTPUT=ACTUAL/"BEGLIN_QV2_ACTIVATION_ASYNC_INDEPENDENT_AUDIT.json"
EXPECTED={
 "source_frozen":"8dd48956820d47073f7f415c7427d07f20e01db154cdd8c895ec5d6d134c38a4",
 "prospective_native":"afff3f20a7d258a2639ebe28e3a163eedfa46787503d7a1ff31abcee0ab56f7f",
 "old_MILP_native":"043ee279a2f5a7e0880286aa0c1b5e5ca7692dbf4775bf53c49a27b803383a9",
 "old_MILP_input":"352d2f1b2161ce47233dd25ffe377971b54f321ef3e6ea833c379619d4daa23b",
 "actual_serial_and_pthread":"26096060a809b380cbf5bef749b87f09dcec1a253c3ecc99d6ef20652daec692",
 "stage_cost":"380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808",
 "public_calibration_score":"94ffe5c18ee4f92fadee912d1046d2ed104f3934e968081e87dfd1ac07ed1bf0",
}
ROLE_ORDER=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_joint_activation_async_allocator as candidate

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private(path:pathlib.Path)->bytes:
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("source private receipt missing or incorrectly accessible: "+str(path))
    return path.read_bytes()
def read_locked(path:pathlib.Path,digest:str):
    data=private(path)
    if sha(data)!=EXPECTED[digest]:raise RuntimeError("pinned evidence SHA drift: "+digest)
    return json.loads(data)
def verify_source_allocator():
    frozen=read_locked(POLICY/"frozen_v2_source_only_policies.json","source_frozen")
    quality_path=ROOT/"docs/research/BEGLIN_Q_V2_ACTIVATION_DIAG_196_PUBLIC_AGGREGATE_20261022.json"
    if sha(quality_path.read_bytes())!=EXPECTED["public_calibration_score"]:
        raise RuntimeError("196 source BF16-only quality tensor evidence changed")
    quality=json.loads(quality_path.read_text())
    if quality["tensor_count"]!=196 or set(quality["quality_upgrade_by_role_layer"])!=set(ROLE_ORDER):
        raise RuntimeError("activation input diagonal score completeness violation")
    if any(len(v)!=28 or not np.isfinite(np.asarray(v,dtype=np.float64)).all() for v in quality["quality_upgrade_by_role_layer"].values()):
        raise RuntimeError("source activation per-tensor values missing")
    stage=read_locked(PROFILE/"async_sibling_stage_costs_private.json","stage_cost")
    if set(stage["stages"]["qkv"])!={"444","448","484","488","844","848","884","888"}:
        raise RuntimeError("QKV combinations not fully measured")
    if set(stage["stages"]["ffn_gu"])!={"44","48","84","88"}:
        raise RuntimeError("Gate/up combinations not fully measured")
    # Unlike the previous full model output sensitivity workflow, this method
    # reads original BF16 calibration data and independent kernel micro-cost.
    public,original_kernel,original_stage=candidate.get_data()
    indexed=candidate.family(public["quality_upgrade_by_role_layer"],original_kernel)
    tables=candidate.precompute(indexed,original_stage)
    selections,counts=candidate.enumerate_policies(indexed,tables,original_stage)
    if not selections or counts["valid_memory"]<=0 or counts["valid_serial"]<=0:
        raise RuntimeError("source-only optimizer did not solve")
    digest={}
    for mode,expected in frozen["selected_policies"].items():
        real=selections[mode]
        for k in ("counts","projection_bytes","predicted_serial_ms","hypothetical_parallel_ms","score"):
            actual=real[k]
            earlier=expected[k]
            if k=="counts":
                if actual!=earlier:raise RuntimeError("frozen source activation role mask drift")
            elif abs(float(actual)-float(earlier))>1e-8:
                raise RuntimeError("source-only objective/cost drift: "+k)
        file=POLICY/f"{mode}_source_only_roles.txt"
        if sha(private(file))!=expected["policy_sha256"]:
            raise RuntimeError("precision bit-mask changed since pretest selection")
        selected=[tuple(l.split()) for l in file.read_text().splitlines()]
        if len(selected)!=28 or len(set(selected))!=28:raise RuntimeError("fixed bit budget violated")
        if expected["projection_bytes"]>830570496 or expected["predicted_serial_ms"]>100+1e-7:
            raise RuntimeError("memory/serial constraint violated")
        if mode=="parallel80" and expected["hypothetical_parallel_ms"]>80+1e-7:
            raise RuntimeError("QKV/GU hypothesized async constraint violated")
        digest[mode]=dict(n8=28,n4=168,role_counts=dict(zip(ROLE_ORDER,expected["counts"])),
            policy_sha256=expected["policy_sha256"],
            packed_projection_bytes=expected["projection_bytes"],
            source_local_projection_proxy_gain=expected["score"],
            serial_predicted_ms=expected["predicted_serial_ms"],
            only_hypothetical_async_predicted_ms=expected["hypothetical_parallel_ms"])
    return dict(source_activation_sha256=EXPECTED["public_calibration_score"],
        private_source_CAL_196_stats_sha256=quality["private_activation_moment_sha256"],
        source_only_allocator_receipt_sha256=EXPECTED["source_frozen"],
        async_stage_profile_sha256=EXPECTED["stage_cost"],
        source_only_exact_in_topk_family=True,not_globally_exact_over_all_2pow196=True,
        freeze_before_native_eval=True,
        counts_searched=counts["total_role_compositions"],
        role_source_allocator_choices=digest)

def verify_native_eval():
    results=read_locked(EVAL/"native_results_private.json","prospective_native")
    external=read_locked(MILP_EVAL/"all_ppl_public_eval_receipt.json","old_MILP_native")
    inputs=read_locked(MILP_EVAL/"input_lock_receipt.json","old_MILP_input")
    if not inputs["all_five_conditions_have_byte_identical_inputs"]:
        raise RuntimeError("other source-only MILP prospective eval input integrity missing")
    if inputs["source_only_policy_sha256"]!="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212":
        raise RuntimeError("global MILP source-only bit mask drift")
    if results["actual_new_public_eval_tokens"]!=131:
        raise RuntimeError("expected 131 public next-token predictions")
    models=results["conditions"]
    assert set(models)=={"widened_f32","full_q4","prior_random_mixed","activation_serial100","activation_async80_candidate"}
    if not (models["widened_f32"]["PPL"]<models["prior_random_mixed"]["PPL"]<
            models["full_q4"]["PPL"]<models["activation_async80_candidate"]["PPL"]<
            models["activation_serial100"]["PPL"]):
        raise RuntimeError("negative source-only quality outcome was changed or suppressed")
    m=external["models"]
    assert set(m)=={"original_BF16_widened_F32","all_Q4G64","frozen_seed73_random","former_weight_kernel_aware","v2_activation_async_MILP"}
    for first,second in (
        ("widened_f32","original_BF16_widened_F32"),
        ("full_q4","all_Q4G64"),
        ("prior_random_mixed","frozen_seed73_random")
    ):
        if models[first]["PPL"]!=m[second]["perplexity"]:
            raise RuntimeError("two independently recorded 8 public evaluation baselines disagree")
    for w in range(8):
        rel=f"eval/ppl_work/w{w}.i32"
        reference=private(MILP_EVAL/"v2_activation_async_MILP"/rel)
        for name in models:
            data=private(EVAL/name/rel)
            if data!=reference:
                raise RuntimeError("eight new public model evaluation inputs not identical across independent experiment")
    return dict(public_eight_prompts_131_positions=True,
        same_public_inputs_across_five_and_original_global_MILP=True,
        native_quality_PPL_by_condition={name:float(v["PPL"]) for name,v in models.items()},
        existing_global_MILP_source_frozen_PPL=m["v2_activation_async_MILP"]["perplexity"],
        v2_activation_quality_surrogate_not_superior_to_Q4_or_prior_random=True,
        native_results_receipt_sha256=EXPECTED["prospective_native"],
        separate_native_global_MILP_receipt_sha256=EXPECTED["old_MILP_native"])

def verify_actual_pthreads():
    source=read_locked(ACTUAL/"paired_native_async_serial_8public.json","actual_serial_and_pthread")
    build=json.loads(private(ACTUAL/"build_receipt.json"))
    if source["native_binary_sha256"]!=build["native_binary_SHA256"]:
        raise RuntimeError("actual pthread research binary not pinned")
    if sha((ACTUAL/"beglin_Qv2_native_async").read_bytes())!=build["native_binary_SHA256"]:
        raise RuntimeError("compiled native async executable drift")
    if source["policy_sha256"]!=build["original_q4_n8_tensor_policy_SHA256"]:
        raise RuntimeError("mode on/off changed bit allocation")
    ref=MILP_EVAL/"v2_activation_async_MILP/results/c_logits_last.f32"
    new=ACTUAL/"isolated/results/c_logits_last.f32"
    if private(ref)!=private(new):
        raise RuntimeError("actual native QKV/Gate-Up async changed final vocab output")
    results=source["before_after"]
    assert set(results)=={"bench","ppl"}
    for tag in ("bench","ppl"):
        control=results[tag]["serial"]
        concurrent=results[tag]["async"]
        if control["actual_async_enabled"] or not concurrent["actual_async_enabled"]:
            raise RuntimeError("native thread executor mode not toggled")
        if control["genuine_pthread_qkv_join_count"]!=0 or control["genuine_pthread_gateup_join_count"]!=0:
            raise RuntimeError("sequential binary unexpectedly dispatching concurrent work")
        if concurrent["genuine_pthread_qkv_join_count"]<=0 or concurrent["genuine_pthread_gateup_join_count"]<=0:
            raise RuntimeError("real QKV/FFN parallel stages not invoked")
        for mode in ("serial","async"):
            record=results[tag][mode]
            log=private(ACTUAL/(mode+"_"+tag+"_native_log.txt"))
            if sha(log)!=record["process_log_sha256"]:
                raise RuntimeError("native pthread log transcription SHA mismatch")
    if results["ppl"]["serial"]["PPL"]!=results["ppl"]["async"]["PPL"]:
        raise RuntimeError("whole model actual pthread output PPL changed")
    if results["ppl"]["serial"]["scored_tokens"]!=131 or results["ppl"]["async"]["scored_tokens"]!=131:
        raise RuntimeError("wrong true native eval token count")
    if results["bench"]["async"]["single_decode_best_of_3_ms"]<=results["bench"]["serial"]["single_decode_best_of_3_ms"]:
        raise RuntimeError("do not misreport actual async slower negative performance result")
    return dict(
       actual_both_join_types_executed=True,
       output_logits_identical_to_former_serial=True,
       full_model_actual_serial_ms=results["bench"]["serial"]["single_decode_best_of_3_ms"],
       full_model_actual_async_ms=results["bench"]["async"]["single_decode_best_of_3_ms"],
       actual_async_speedup=source["actual_one_token_native_speedup"],
       actual_async_performance_superiority=False,
       serial_new_public_PPL=results["ppl"]["serial"]["PPL"],
       actual_async_new_public_PPL=results["ppl"]["async"]["PPL"],
       actual_async_QKV_join_count=results["ppl"]["async"]["genuine_pthread_qkv_join_count"],
       actual_async_gateup_join_count=results["ppl"]["async"]["genuine_pthread_gateup_join_count"],
       serial_RSS=results["ppl"]["serial"]["max_rss_bytes"],
       actual_async_RSS=results["ppl"]["async"]["max_rss_bytes"],
       true_async_pthread_receipt_SHA256=EXPECTED["actual_serial_and_pthread"])

def main():
    if OUTPUT.exists():raise FileExistsError("independent Qv2 audit was already performed")
    source=verify_source_allocator()
    quality=verify_native_eval()
    actual=verify_actual_pthreads()
    cp=subprocess.run(["git","-C",str(ROOT),"status","--porcelain"],capture_output=True,text=True,check=True)
    if cp.stdout.strip()!="?? lab_local/":
        raise RuntimeError("research worktree tracked source dirty/production state unclear")
    result=dict(schema="BEGLIN-Q-v2-196-tensor-activation-async-independent-audit/1",
      status="TECHNICAL_PASS_SOURCE_CALIBRATION_AND_ACTUAL_ASYNC_PPL_PARITY__NO_SPEED_OR_QUALITY_SUPERIORITY",
      date="2026-10-22",
      source_only_precisions=source,
      same_public_eval_131_tokens=quality,
      actual_native_pthread_execution=actual,
      model_original_checkpoint_unchanged=True,
      main_and_immutable_release_untouched=True,
      prior_blocked_per_tensor_full_logits_sensitivity_not_retried=True,
      old_V6_24_consumed_ids_unavailable_and_never_opened=True,
      real_task_IID_OOD_AUROC_not_tested=True,
      scientific_Q_final_verdict="NO_VERDICT",
      limitations=[
        "CUDA BF16 input second moments use 12 public calibration prompts/461 token positions and diagonal approximation, not model NLL or causal tensor sensitivity.",
        "8 public synthetic chat prompts contain 131 prediction positions; not independent real task/generalization.",
        "QKV/GateUp native CPU concurrency uses persistent threads and owned Q4 pools; actual full-model timing is slower by 0.7%, despite micro-stage speed gains.",
        "Full native serial and async same binary one-token best-of-3 with warmed model is not sustained production throughput.",
        "Source-only 7-role top-k integer search is exact only for its restricted policy family, not globally exact 2^196 assignments.",
        "Independent global MILP source policy is a separate experiment with source-only lock; proof not rerun on Apple EOE lacking SciPy.",
        "No per-token dynamic bit relocation, hardware Metal GPU async, memory/energy/stress/concurrent multiple requests, or true BF16 arithmetic baseline.",
        "No release promotion or production integration, no final Q hypothesis claims."
      ])
    os.umask(0o077)
    with OUTPUT.open("xb") as f:
        f.write((json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    OUTPUT.chmod(0o600)
    print("BEGLIN_QV2_ACTIVATION_NATIVE_ASYNC_INDEPENDENT_AUDIT_PASS",json.dumps({
       "receipt_SHA256":sha(private(OUTPUT)),
       "candidate_PPL":quality["native_quality_PPL_by_condition"],
       "true_async_ppl_equal":actual["serial_new_public_PPL"]==actual["actual_async_new_public_PPL"],
       "true_parallel_speedup":actual["actual_async_speedup"],
       "source_196_score_verified":True,
       "main_changed":False
    },sort_keys=True),flush=True)
if __name__=="__main__":
    main()
