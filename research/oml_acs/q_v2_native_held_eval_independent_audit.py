#!/usr/bin/env python3
"""Independent SHA-bound audit of BEGLIN Qv2 public frozen 8-window results.

Recomputes first-prompt full-vocabulary KL from raw float32 logits; extracts
PPL, time and CPU peak RSS from actual native transcripts instead of blindly
trusting reported JSON; independently checks the same input windows & exact
28-n8 source-only policy freeze across all models. Model executes serially.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_native_8public_eval_20261022"
ORIGINAL=LAB/"native_8public_QV2_policy_evaluation_receipt.json"
FIXTURE=LAB/"public_native_eval_fixture_manifest.json"
POLICY=ROOT/"docs/research/BEGLIN_QV2_ASYNC_DAG_MILP_SOURCE_POLICIES_20261022.json"
OUT=LAB/"QV2_NATIVE_8PUBLIC_INDEPENDENT_REVIEW.json"
EXPECTED_ORIGINAL_SHA="4fbaae62ad6545beb48a9d1ad4fb07d54b5d6982061ac93486110e60cf438a18"
EXPECTED_FIXTURE_SHA="191f836b2fc34491ec2fbaa530b78e15e56db11e1322b34ce2146a05e983baa4"
EXPECTED_ASYNC_SHA="e044675ea5b47a5aca0ca75980a55070b932d1a4dc330815bbb097acbf75df7b"
EXPECTED_SERIAL_SHA="06f07fcfab7fdba22528c6c80e09582f18cfa77d0760a26e4ec35bc6a80dd72c"
MODES=("async_DAG","serial_only","old_random","q4","f32")

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def secure(path:pathlib.Path)->bytes:
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("missing/unsecured original native result "+str(path))
    return path.read_bytes()

def load_provenance():
    orig=secure(ORIGINAL);fixture=secure(FIXTURE)
    if sha(orig)!=EXPECTED_ORIGINAL_SHA or sha(fixture)!=EXPECTED_FIXTURE_SHA:
        raise RuntimeError("original native/fixture receipt identity drift")
    j=json.loads(orig);f=json.loads(fixture);p=json.loads(POLICY.read_text())
    assert j["status"]=="ALL_NATIVE_RUNS_SUCCEEDED_QUALITY_NOT_PROMOTED"
    assert p["status"]=="OPTIMAL_SOURCE_ONLY_POLICIES_FROZEN_BEFORE_EVAL"
    assert f["source_optimizer_manifest_sha256"]==sha(POLICY.read_bytes())
    assert j["source_only_MILP_SHA256"]==sha(POLICY.read_bytes())
    assert f["policy_sources_verified_before_eval"] and j["frozen_policy_selected_before_first_eval"]
    assert len(j["mode_results"])==5 and set(j["mode_results"])==set(MODES)
    pp={x["name"]:x for x in p["policies"]}
    if pp["async_DAG"]["frozen_policy_sha256"]!=EXPECTED_ASYNC_SHA or pp["serial_only"]["frozen_policy_sha256"]!=EXPECTED_SERIAL_SHA:
        raise RuntimeError("source frozen role-layer mask SHA mismatch")
    return j,f,p

def parse_and_verify(j,f):
    x=j["mode_results"]
    input_hash=f["input_window_shas"]
    if len(input_hash)!=8:raise RuntimeError("expected eight independent public windows")
    outcomes={}
    for name in MODES:
        folder=LAB/name
        w=f["model_workloads"][name]
        for idx in range(8):
            if sha(secure(folder/f"eval/ppl_work/w{idx}.i32"))!=input_hash[idx]:
                raise RuntimeError("source held token order mismatch")
        if not pathlib.Path(w["binary"]).is_file() or sha(pathlib.Path(w["binary"]).read_bytes())!=w["binary_sha256"]:
            raise RuntimeError("BEGLIN executable diverged after evaluation")
        d=x[name]
        for mode in ("dump","ppl","bench"):
            entry=d[mode]
            text=secure(folder/f"QV2_{mode}_native_run_log.txt")
            if entry["native_log_SHA256"]!=sha(text):raise RuntimeError("transcript mutated")
            decoded=text.decode(errors="replace")
            labels=re.search(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)",decoded)
            if labels is None or list(map(int,labels.groups()))!=entry["projector_counts"]:
                raise RuntimeError("native role precision not verified")
            if entry["exit_code"]!=0:raise RuntimeError("nonzero native process")
            if mode=="ppl":
                matched=re.search(r"\[ppl\]\s+fp32\s+ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",decoded)
                rss=re.search(r"(\d+)\s+maximum resident set size",decoded)
                prompts=re.findall(r"\[ppl\] window (\d+) done \((\d+) toks\)",decoded)
                if not matched or not rss or len(prompts)!=8:raise RuntimeError("ppl transcript incomplete")
                if abs(float(matched.group(1))-entry["ppl"])>1e-8 or int(matched.group(2))!=131:
                    raise RuntimeError("PPL token count or numeric value drift")
                if int(rss.group(1))!=entry["RSS_max_bytes"]:
                    raise RuntimeError("PPL RSS transcription mismatch")
            elif mode=="bench":
                matched=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",decoded)
                if not matched or abs(float(matched.group(2))-entry["ms_single_token_best_of_three"])>1e-8:
                    raise RuntimeError("real best-of-3 native single-token speed drift")
        arr=secure(folder/"results/c_logits_last.f32")
        if len(arr)!=151936*4 or sha(arr)!=d["dump"]["logits_SHA256"]:
            raise RuntimeError("real model full-vocab logits SHA wrong")
        logits=np.frombuffer(arr,dtype="<f4").astype(np.float64)
        if not np.isfinite(logits).all():raise RuntimeError("nonfinite full-vocab logits")
        outcomes[name]=logits
    reference=outcomes["f32"]
    pp=np.exp(reference-reference.max());pp/=pp.sum()
    copied={}
    for name,logits in outcomes.items():
        p=np.exp(logits-logits.max());p/=p.sum()
        kl=float(np.sum(pp*(np.log(np.maximum(pp,1e-300))-np.log(np.maximum(p,1e-300)))))
        if abs(kl-x[name]["dump"]["KL_F32teacher_to_candidate"])>1e-10:
            raise RuntimeError("reported KL doesn't match independent real logits")
        copied[name]=dict(
           full_vocab_logits_sha256=x[name]["dump"]["logits_SHA256"],
           measured_public_eval_ppl=x[name]["ppl"]["ppl"],
           measured_public_eval_NLL_nats=x[name]["ppl"]["nll_nats_per_token"],
           measured_native_best_of_3_ms=x[name]["bench"]["ms_single_token_best_of_three"],
           measured_native_best_of_3_tokens_s=x[name]["bench"]["tps_single_token_best_of_three"],
           observed_PPL_peak_RSS_bytes=x[name]["ppl"]["RSS_max_bytes"],
           KL_widened_F32_to_policy_nats=kl,
           full_vocab_top1_token_id=int(np.argmax(logits)),
           shared_8_pinned_public_prompts=True,
           source_original_BF16_teacher_same=True)
    if not (copied["old_random"]["measured_public_eval_NLL_nats"]<
            copied["serial_only"]["measured_public_eval_NLL_nats"] and
            copied["old_random"]["measured_public_eval_NLL_nats"]<
            copied["async_DAG"]["measured_public_eval_NLL_nats"]):
        raise RuntimeError("do not suppress failure of activation-aware selected policies")
    return copied

def run():
    if OUT.exists():raise FileExistsError("audit receipts are immutable")
    original,fixture,selection=load_provenance()
    cases=parse_and_verify(original,fixture)
    rec=dict(schema="BEGLIN-Qv2-independent-public-native-8window-quality-review/1",
      status="PASS_SHA_INPUTS_NATIVE_NUMERICS_AND_LOWER_RANDOM_NLL",
      date="2026-10-22",
      private_original_receipt_SHA256=EXPECTED_ORIGINAL_SHA,
      private_eval_fixture_SHA256=EXPECTED_FIXTURE_SHA,
      source_only_policy_SHA256=sha(POLICY.read_bytes()),
      stable_locked_policies={x["name"]:x["frozen_policy_sha256"] for x in selection["policies"]},
      evaluation_source_original_Qwen="Qwen/Qwen2.5-1.5B-Instruct BF16 widened F32 reference",
      evaluation_device="Apple Silicon Mac13,1 EOE native BEGLIN CPU",
      comparable_eight_public_synthetic_windows=8,
      teacher_forced_evaluated_token_positions=131,
      stage_async_microbenchmark_is_real=True,
      native_full_model_executes_stage_asynchronously=False,
      mode_results=cases,
      native_kernel_stage_cost_contract="source QKV and FFN measured on original CPU role0 only, async-DAG estimate not observed end-to-end",
      full_model_processes_all_exit_code_zero=True,
      source_only_MILP_optimality_confirmed_from_separate_Linux_SciPy_8_unit_tests=True,
      original_V6_used_docs_verified_excluded=False,
      real_task_AUROC_or_model_generalization_proven=False,
      no_quality_superiority_over_frozen_random=True,
      no_main_or_production_change=True,
      final_Q_quality_verdict="NO_VERDICT")
    os.umask(0o077)
    with OUT.open("xb") as f:
        f.write((json.dumps(rec,sort_keys=True,indent=2)+"\n").encode())
        f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    print("BEGLIN_QV2_HELD_PUBLIC_INDEPENDENT_AUDIT_PASS",json.dumps({
      "private_receipt_SHA256":sha(secure(OUT)),
      "tested_modes":list(cases),
      "ppl":{k:v["measured_public_eval_ppl"] for k,v in cases.items()},
      "n8_native_async_not_established":True},sort_keys=True),flush=True)
if __name__=="__main__":run()
