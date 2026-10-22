#!/usr/bin/env python3
"""Independent read-only re-evaluation of strict BF16-activation × CPU-async Qv2.

Verifies all source-only 448-option selection arithmetic WITHOUT SciPy,
policy tensor IDs vs per-layer DAG bit choices, original stage cost tables,
original role byte costs, bounded native sequential Qwen results, and real
native logit/provenance. Does not retry previously blocked full-model
per-tensor output perturbations, mutate immutable release, or tune bit mask.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re
from collections import Counter
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
FROZEN=ROOT/"configs/research/BEGLIN_Q_V2_STRICT_50MS_SOURCE_POLICY_FROZEN_20261022.json"
SOURCE_ACT=ROOT/"configs/research/BEGLIN_Q_V2_ACTIVATION_DIAGONAL_GAINS_20261022.json"
SOURCE_ASYNC=ROOT/"configs/research/BEGLIN_Q_V2_NATIVE_ASYNC_STAGE_COSTS_20261022.json"
SOURCE_KERNELS=ROOT/"docs/research/BEGLIN_Q_KERNEL_AWARE_ALLOCATOR_NATIVE_EVIDENCE_20261022.json"
RUN=ROOT/"lab_local/q_v2_strict_50ms_native_20261022"
PRIOR=ROOT/"lab_local/q_v2_8public_evaluation_20261022"
SOURCE_NATIVE=ROOT/"lab_local/q_kernel_aware_native_20261022/beglin_kernel_aware_native_Qwen"
EVIDENCE=RUN/"qv2_strict50_source_and_native_independent_audit.json"
ORIGINAL_INPUT_LOCK="352d2f1b2161ce47233dd25ffe377971b54f321ef3e6ea833c379619d4daa23b"
POLICY_SHA="cb9268c9177d47705e548c2a90a6b22772569b3f2bb54dfef2767ca53ffa3e76"
PRIVATE_SOURCE_RECEIPT_SHA="7d19591c22846f8d001256f4c4addacfa7babeebf49eae9b3783128e9c464ea4"
ORIGINAL_NATIVE_BINARY_SHA="172613dc69f79f38c8553ff7087e653ec9c0031ca80aec8efd2f76f948599f14"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
GROUPS=(("qkv",ROLES[:3]),("o",(ROLES[3],)),("ffn_gu",ROLES[4:6]),("down",(ROLES[6],)))
OTHER_8_PPL_RECEIPT_SHA="043ee279a2f5a7e0880286aa0c1b5e5cae7692dbf4775bf53c49a27b803383a9"

def sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()
def private(p:pathlib.Path)->bytes:
    if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
        raise RuntimeError("missing / unsafe model-derived 0600 artifact "+str(p))
    return p.read_bytes()
def load_inputs():
    j=json.loads(FROZEN.read_text())
    src=json.loads(SOURCE_ACT.read_text())
    stages=json.loads(SOURCE_ASYNC.read_text())
    kernel=json.loads(SOURCE_KERNELS.read_text())
    if j["private_receipt_sha256"]!=PRIVATE_SOURCE_RECEIPT_SHA or j["policy_text_sha256"]!=POLICY_SHA:
        raise RuntimeError("source policy frozen SHA changed")
    if j["new_output_or_evaluation_used"] or not j["no_fullmodel_parallel_speed_claim"]:
        raise RuntimeError("source-only optimization vs native async semantics falsified")
    if src["source_SHA"]!="94ffe5c18ee4f92fadee912d1046d2ed104f3934e968081e87dfd1ac07ed1bf0":
        raise RuntimeError("activation BF16 quality source SHA changed")
    if stages["source_SHA"]!="380670d42ccd2f12c4a015600be3d921f672c570a3d3a4a1652c61e217791808":
        raise RuntimeError("actual native sibling async stage source SHA changed")
    if src["bf16_revision"]!="989aa7980e4cf806f80c7fef2b1adb7bc71aa306":
        raise RuntimeError("original teacher BF16 rev changed")
    if sha(SOURCE_NATIVE.read_bytes())!=ORIGINAL_NATIVE_BINARY_SHA:
        raise RuntimeError("native source executable SHA changed")
    if sha(private(RUN/"chosen_n8_roles.txt"))!=POLICY_SHA:
        raise RuntimeError("actual runtime role mask differs from source-only MILP")
    return j,src,stages,kernel

def recompute_source_policy(j,src,stages,kernel):
    costs={row["role"]:row for row in kernel["role_kernel_cost_measured"]}
    if set(costs)!=set(ROLES):
        raise RuntimeError("7 unique original native kernel cost roles missing")
    stage_specs=j["stage_per_layer_modes"]
    if len(stage_specs)!=112:
        raise RuntimeError("not 4 dependency stages x28 decoder layers")
    selected={}
    for row in stage_specs:
        l=int(row["layer"]);stage=row["stage"]
        roles=dict(GROUPS)[stage]
        assert 0<=l<28 and len(row["bits"])==len(roles)
        key=(l,stage)
        if key in selected:raise RuntimeError("multiple stage decisions for one native layer")
        selected[key]=row
    if len(selected)!=112 or set(selected)!=set((l,stage) for l in range(28) for stage,_ in GROUPS):
        raise RuntimeError("missing whole Qwen layer DAG")
    pairs=[]
    async_ms=0.;serial_delta=0.;q4_total=sum(int(costs[r]["q4_with_scales_bytes"])*28 for r in ROLES)
    bytes_delta=0.;gain=0.
    schedule=[]
    for l in range(28):
        for stage,roles in GROUPS:
            row=selected[(l,stage)]
            mask=row["bits"]
            if any(bit not in ("4","8") for bit in mask):
                raise RuntimeError("unsupported bit width in frozen Qv2 stage")
            measured=stages["data"][stage][mask] if stage in ("qkv","ffn_gu") else None
            if measured:
                cost=float(measured["best_ms"]);mode=measured["best_mode"]
            else:
                role=roles[0]
                value=costs[role]
                cost=float(value["n8_repacked_neon_median_ms"] if mask=="8"
                     else min(value["q4_neon_1thread_median_ms"],value["q4_neon_6thread_median_ms"]))
                mode="serial_one_role"
            if abs(row["observed_stage_median_ms"]-cost)>1e-8 or row["dispatcher"]!=mode:
                raise RuntimeError("frozen dependency-stage cost or dispatch mode changed")
            async_ms+=cost
            schedule.append((l,stage,mode,mask))
            for role,b in zip(roles,mask):
                if b=="8":
                    pairs.append((role,l))
                    original=costs[role]
                    serial_delta+=float(original["n8_repacked_neon_median_ms"]-min(original["q4_neon_1thread_median_ms"],original["q4_neon_6thread_median_ms"]))
                    bytes_delta+=int(original["n8_with_scales_bytes"]-original["q4_with_scales_bytes"])
                    gain+=float(src["quality_upgrade_by_role_layer"][role][l])
    pairs.sort(key=lambda x:(ROLES.index(x[0]),x[1]))
    if pairs !=[(str(role),int(idx)) for role,idx in j["selected_n8_role_layer_pairs"]]:
        raise RuntimeError("chosen role layer IDs not same as selected 112 DAG options")
    if len(pairs)!=28 or len(set(pairs))!=28:
        raise RuntimeError("not exactly 28 distinct bit-8 tensors")
    selectedbytes=int(q4_total+bytes_delta)
    native_serial=75.26+1.039406406482483*serial_delta
    if selectedbytes!=j["packed_projection_bytes"] or abs(native_serial-j["predicted_serial_ms"])>1e-8 or abs(async_ms-j["measured_stage_async_DAG_ms"])>1e-8:
        raise RuntimeError("real activation score/async measured stage constraints cannot reproduce frozen policy")
    if selectedbytes>830570496 or native_serial>100. or async_ms>50.:
        raise RuntimeError("frozen quality optimization violates exact source constraints")
    if abs(gain-j["quality_gain"])>1e-8:
        raise RuntimeError("source activation benefit does not agree with actual original 196 scores")
    return dict(role_counts={r:sum(p[0]==r for p in pairs) for r in ROLES},
       n8=28,n4=168,source_optimizer_MILP_reported_gap=j["solver_gap"],
       activation_weighted_source_gain=gain,
       packed_projection_bytes=selectedbytes,
       serial_C_kernel_cost_predicted_ms=native_serial,
       measured_sibling_parallel_stage_sum_ms=async_ms,
       QKV_or_gateup_parallel_dispatch_expected=sum(1 for k in schedule if k[2].startswith("parallel_")),
       stage_has_true_pthread_runtime=False,
       full_model_heldout_inputs_not_part_of_source_selection=True)

def validate_actual_native(j):
    if sha(private(RUN/"source_receipt.json"))!="65948c88316bf50c0ee75b6ed86481bb79eded3adf65fda6dc5a783615054003":
        raise RuntimeError("evaluation input provenance already changed")
    frozen=json.loads(private(RUN/"source_receipt.json"))
    if frozen["source_only_policy_mask_sha256"]!=POLICY_SHA or frozen["windows"]!=8 or not frozen["native_full_model_exec_will_be_sequential"]:
        raise RuntimeError("new async native runtime claim unsupported")
    oldlock=private(PRIOR/"input_lock_receipt.json")
    if sha(oldlock)!=ORIGINAL_INPUT_LOCK:
        raise RuntimeError("original public 8-window evaluator changed")
    inputs=json.loads(oldlock)
    source=PRIOR/"v2_activation_async_MILP"
    for filename in ["prompt_ids.i32"]+[f"eval/ppl_work/w{i}.i32" for i in range(8)]:
        left=private(RUN/"chosen"/filename)
        right=private(source/filename)
        if left!=right:raise RuntimeError("native evaluation input not original 8 public prompts")
    result={}
    originals={"dump":"8d51263509e17a3f04864f23b55eff8ecd79d6777f747acf46b9f58db0de000c",
               "bench":"45f558ca219f57d3551e0767c6ce26b248c2527c06873105357df75c46c5b184",
               "ppl8":"7c20e131d2509d38c0921701058ea586fe0334e00d100f92dc4ac22e88141b49"}
    for stage,h in originals.items():
        rec=json.loads(private(RUN/(stage+"_receipt.json")))
        if sha(private(RUN/(stage+"_receipt.json")))!=h:
            raise RuntimeError("original native measured phase SHA drift "+stage)
        if rec["phase"]!=stage or not rec["expected_serial_process"] or rec["source_only_policy_sha256"]!=POLICY_SHA:
            raise RuntimeError("native model did not execute original policy")
        if rec["n8_repacked_native_GEMV_calls"]<=0 or rec["status"]!="PASS_SEQUENTIAL_REAL_NATIVE_MODEL":
            raise RuntimeError("true packed 8-bit kernel not actually called")
        raw=private(RUN/"chosen/logs"/(stage+".log"))
        if sha(raw)!=rec["native_log_SHA256"]:
            raise RuntimeError("native log SHA mismatch")
        if not re.search(rb"projection_role_tensors: f32=0 n8=28 q4=168",raw):
            raise RuntimeError("native full Qwen precision matrix wrong")
        if not re.search(rb"\[BEGLIN-N8-REPACKED-LAB\] fast_gemv=(\d+)",raw):
            raise RuntimeError("actual n8 CPU kernel usage not auditable")
        result[stage]=rec
    if result["dump"]["first_full_vocab_logits_SHA256"]!=sha(private(RUN/"chosen/results/c_logits_last.f32")):
        raise RuntimeError("full-vocab native logits SHA mismatch")
    if result["ppl8"]["public_development_token_positions"]!=131:
        raise RuntimeError("native 8-public PPL token count changed")
    if not math.isfinite(result["bench"]["best_of_three_one_token_ms"]) or result["bench"]["best_of_three_one_token_ms"]<=0:
        raise RuntimeError("invalid native timing")
    baseline_data=private(PRIOR/"all_ppl_public_eval_receipt.json")
    if sha(baseline_data)!=OTHER_8_PPL_RECEIPT_SHA:
        raise RuntimeError("original all-5 baseline PPL source drift")
    baselines=json.loads(baseline_data)["models"]
    y=np.frombuffer(private(RUN/"chosen/results/c_logits_last.f32"),dtype="<f4").astype(np.float64)
    x=np.frombuffer(private(PRIOR/"original_BF16_widened_F32/results/c_logits_last.f32"),dtype="<f4").astype(np.float64)
    assert y.shape==x.shape==(151936,) and np.isfinite(y).all() and np.isfinite(x).all()
    px=np.exp(x-x.max());px/=px.sum()
    py=np.exp(y-y.max());py/=py.sum()
    kl=float(np.sum(px*(np.log(np.maximum(px,1e-300))-np.log(np.maximum(py,1e-300)))))
    if result["ppl8"]["public_development_PPL"]<=baselines["all_Q4G64"]["perplexity"]:
        raise RuntimeError("do not suppress real activation/async proxy quality regression")
    if result["ppl8"]["public_development_PPL"]<=baselines["frozen_seed73_random"]["perplexity"]:
        raise RuntimeError("do not suppress better random-control quality")
    return dict(
      native_binary_SHA256=ORIGINAL_NATIVE_BINARY_SHA,
      frozen_source_only_policy_SHA256=POLICY_SHA,
      actual_native_execution_parallel=False,
      NLL_public_131_tokens=math.log(result["ppl8"]["public_development_PPL"]),
      PPL_public_131_tokens=result["ppl8"]["public_development_PPL"],
      observed_ppl_process_max_RSS_bytes=result["ppl8"]["maximum_resident_bytes"],
      actual_single_token_best_of_three_ms=result["bench"]["best_of_three_one_token_ms"],
      actual_single_token_tps=result["bench"]["native_tok_per_s_reported"],
      source_predicted_serial_ms=j["predicted_serial_ms"],
      source_predicted_stage_parallel_sum_ms=j["measured_stage_async_DAG_ms"],
      actual_first_prompt_full_vocab_logit_SHA256=result["dump"]["first_full_vocab_logits_SHA256"],
      first_prompt_KL_f32_to_candidate_nats=kl,
      native_first_prompt_top1_id=int(y.argmax()),
      source_original_F32_PPL=baselines["original_BF16_widened_F32"]["perplexity"],
      frozen_role_balanced_random_PPL=baselines["frozen_seed73_random"]["perplexity"],
      all_q4_PPL=baselines["all_Q4G64"]["perplexity"],
      prior_v2_90ms_MILP_PPL=baselines["v2_activation_async_MILP"]["perplexity"],
      prior_v2_90ms_MILP_best_three_one_token_ms=89.54,
      previous_gu6_native_whole_model_async_winner_speed_ratio=1.0024039774900289,
      previous_native_async_speed_admission_gate_pass=False,
      reused_previously_seen_public_development_windows=True,
      main_and_immutable_release_unchanged=True)

def run():
    if EVIDENCE.exists():raise FileExistsError("audit is immutable; refuse overwrite")
    data=load_inputs()
    source=recompute_source_policy(*data)
    actual=validate_actual_native(data[0])
    rec=dict(schema="BEGLIN-Q-v2-strict50-independent-evidence/1",
      status="PASS_SOURCE_OPTIMAL_STAGE_BUDGET_AND_NATIVE_EXECUTION__QUALITY_NOT_SUPERIOR",
      original_BF16_source_model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      frozen_source_only_policy_public_SHA256=sha(FROZEN.read_bytes()),
      source_cost_and_quality_policy_audit=source,
      native_Qwen_28_layer_sequential_evaluation=actual,
      native_continuous_TPS_not_measured=True,
      only_native_cpu_stage_parallel_gain_measured_before_no_full_async_v2_new_policy=True,
      previous_native_full_model_async_measured_no_gain=True,
      independently_new_data_not_claimed=True,
      active_full_model_parallel_default_remains_OFF=True,
      no_production_or_main_change=True,
      scientific_Q_hypothesis="NO_VERDICT")
    os.umask(0o077)
    with EVIDENCE.open("xb") as f:
        f.write((json.dumps(rec,sort_keys=True,indent=2)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    EVIDENCE.chmod(0o600)
    print("BEGLIN_Q_V2_STRICT_50MS_INDEPENDENT_AUDIT_PASS",json.dumps({
      "receipt_SHA256":sha(private(EVIDENCE)),
      "real_native_ms":actual["actual_single_token_best_of_three_ms"],
      "real_native_PPL":actual["PPL_public_131_tokens"],
      "measured_async_stage_sum_ms":source["measured_sibling_parallel_stage_sum_ms"],
      "native_full_model_actual_async_measured_for_new_policy":False,
      "scientific_verdict":"NO_VERDICT"},sort_keys=True),flush=True)
if __name__=="__main__":run()
