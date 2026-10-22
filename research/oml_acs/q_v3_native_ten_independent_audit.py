#!/usr/bin/env python3
"""Independent native BEGLIN Qv3 10-new-public prompt / five-policy audit.

Checks the historical source-only v3 full-covariance MILP mask and its
10 frozen token arrays BEFORE considering quality. Then compares original
native process transcripts, first-token 151936-vocab F32/logits and real
ten-window PPL, CPU memory and warmed one-token decode.
Never changes source model, precision masks, or production settings.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v3_native_10dev_20261022"
ORIGINAL=LAB/"v3_native_10public_receipt.json"
FROZEN=LAB/"frozen_10public_manifest.json"
OUTPUT=LAB/"v3_native_10public_independent_audit.json"
PRELOCK=ROOT/"configs/research/BEGLIN_Q_V3_EMPIRICAL_COVARIANCE_PRELOCK_20261022.json"
MILP=ROOT/"docs/research/BEGLIN_Q_V3_FULL_COV_MILP_SOURCE_EVIDENCE_20261022.json"
V3MASK=ROOT/"configs/research/BEGLIN_Q_V3_FROZEN_28N8_POLICY_20261022.txt"
ORIGINAL_INPUTS=ROOT/"lab_local/q_v3_fresh_10dev_20261022"
PRELOCK_SHA="5e8824f74b7d80dac191bf944a48d418f87212ca5ed53835f7ab7edf8eb0c267"
MILP_SHA="f8346e975777a261cd703daafc66d8b68479ddf6508cb6a3850fc707837c5f33"
MASK_SHA="82ae93e06afa7380175afa57535c1f0d1f7fd4320b8d28f981cdc8ee0cf24ef3"
INPUT_SHA="c2b33c5b15a5bcbba02df53d4122a1c1807f4595894c257af40f988c97c5d2cc"
MODES=("f32","q4","role_random","v2_diag","v3_cov")
EXPECTED_COUNTS={"f32":[196,0,0],"q4":[0,0,196],
 "role_random":[0,28,168],"v2_diag":[0,28,168],"v3_cov":[0,28,168]}
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private(p:pathlib.Path)->bytes:
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private original model data missing/unsafe: "+str(p))
    return p.read_bytes()
def verify_source():
    for p,expected in ((PRELOCK,PRELOCK_SHA),(MILP,MILP_SHA),(V3MASK,MASK_SHA)):
        if sha(p.read_bytes())!=expected:
            raise RuntimeError("original prospective v3 source-only policy changed")
    pre=json.loads(PRELOCK.read_text())
    policy=json.loads(MILP.read_text())
    if pre["status"]!="LOCKED_BEFORE_COVARIANCE_CALIBRATION_AND_NEW_PUBLIC_DEV_EVALUATION":
        raise RuntimeError("original prelock not authoritative")
    if policy["status"]!="SOURCE_ONLY_OPTIMAL_MILP_POLICY_FROZEN_BEFORE_NEW_EVAL":
        raise RuntimeError("v3 selected after viewing model output")
    if policy["source_policy_mask_sha256"]!=MASK_SHA:
        raise RuntimeError("impossible branch")
    ids=json.loads(private(ORIGINAL_INPUTS/"pre_eval_input_receipt.json"))
    if sha(private(ORIGINAL_INPUTS/"pre_eval_input_receipt.json"))!=INPUT_SHA:
        raise RuntimeError("first held ten prompt source changed")
    if ids["prompt_count"]!=10 or ids["teacher_forced_prediction_positions_expected_sum_input_lengths_minus_10"]!=187:
        raise RuntimeError("not ten original public evaluation inputs")
    rows=[line.split() for line in V3MASK.read_text().splitlines()]
    if len(rows)!=28 or len(set(map(tuple,rows)))!=28:
        raise RuntimeError("not exact source-frozen v3 mask")
    return ids
def check_actual():
    inputs=verify_source()
    original=json.loads(private(ORIGINAL))
    frozen=json.loads(private(FROZEN))
    if original["status"]!="ALL_NATIVE_EVAL_RUNS_COMPLETED__QUALITY_NEEDS_INDEPENDENT_AUDIT":
        raise RuntimeError("native receipt has incomplete workloads")
    if frozen["status"]!="FROZEN_V3_POLICY_AND_10_PUBLIC_WINDOWS_BEFORE_NATIVE_EVALUATION":
        raise RuntimeError("model eval started before mask/source freeze")
    if original["manifest_sha256"]!=sha(private(FROZEN)) or \
        original["source_frozen_policy_sha256"]!=MASK_SHA:
        raise RuntimeError("native model used different 28-bit Qv3 policy")
    if tuple(original["all_native_mode_results"])!=MODES:
        # JSON sorting erases insertion order; compare as sets instead.
        if set(original["all_native_mode_results"])!=set(MODES):
            raise RuntimeError("native cohort not exactly original five policies")
    if original["window_count"]!=10 or original["prediction_positions"]!=187:
        raise RuntimeError("non-prelocked native inputs/positions")
    hashes=inputs["exact_official_HF_token_ids_sha256"]
    for name in MODES:
        model=frozen["model_workloads"][name]
        if sha(pathlib.Path(model["binary"]).read_bytes())!=model["binary_sha256"]:
            raise RuntimeError("native executable changed since prereg")
        if model["mask_file"] and sha(pathlib.Path(model["mask_file"]).read_bytes())!=model["mask_sha256"]:
            raise RuntimeError("quantizer role mask drift")
        group=original["all_native_mode_results"][name]
        for k in ("dump","ppl","bench"):
            log=private(LAB/name/f"QV3_{k}_native_log.txt").decode("utf-8")
            row=group[k]
            if sha(log.encode())!=row["native_log_sha256"]:
                raise RuntimeError("process transcript altered")
            match=re.search(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)",log)
            if match is None or [int(x) for x in match.groups()]!=EXPECTED_COUNTS[name]:
                raise RuntimeError("quantized C source projection role counts incorrect")
            if k=="ppl":
                e=re.search(r"\[ppl\] fp32 ppl ([\d.]+) over (\d+) tokens",log)
                mem=re.search(r"(\d+)\s+maximum resident set size",log)
                groups=re.findall(r"\[ppl\] window (\d+) done \((\d+) toks\)",log)
                if not e or not mem or len(groups)!=10 or int(e.group(2))!=187:
                    raise RuntimeError("actual PPL process output inconsistent")
                if float(e.group(1))!=row["ppl"] or int(mem.group(1))!=row["peak_RSS_bytes"]:
                    raise RuntimeError("PPL/RSS transcription changed")
            if k=="bench":
                t=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",log)
                if t is None or float(t.group(2))!=row["native_single_token_best_of_3_ms"]:
                    raise RuntimeError("actual warmed CPU benchmark mismatch")
        for i in range(10):
            data=private(LAB/name/f"eval/ppl_work/w{i}.i32")
            if sha(data)!=hashes[i]:
                raise RuntimeError("mode evaluation position/order changed")
        data=private(LAB/name/"results/c_logits_last.f32")
        if len(data)!=151936*4 or sha(data)!=group["dump"]["full_logits_sha256"]:
            raise RuntimeError("missing actual first full-vocab output")
    xs={}
    for name in MODES:
        x=np.frombuffer(private(LAB/name/"results/c_logits_last.f32"),dtype="<f4").astype(np.float64)
        if not np.isfinite(x).all():
            raise RuntimeError("invalid logits")
        xs[name]=x
    teacher=xs["f32"]
    p=np.exp(teacher-teacher.max());p/=p.sum()
    outcomes={}
    for name in MODES:
        row=original["all_native_mode_results"][name]
        x=xs[name]
        q=np.exp(x-x.max());q/=q.sum()
        kl=float(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300)))))
        observed=original["quality_speed_memory"][name]
        if abs(kl-observed["KL_F32_to_policy_nats"])>1e-10:
            raise RuntimeError("independent KL differs from process code")
        nll=math.log(row["ppl"]["ppl"])
        if abs(nll-observed["NLL"])>1e-12:
            raise RuntimeError("teacher-forced NLL differs from PPL")
        outcomes[name]=dict(ppl=row["ppl"]["ppl"],nll_nats=nll,
            KL_F32_to_candidate=kl,peak_RSS_bytes=row["ppl"]["peak_RSS_bytes"],
            best_of_three_ms=row["bench"]["native_single_token_best_of_3_ms"],
            true_full_model_parallel=False)
    return frozen,original,outcomes
def run():
    if OUTPUT.exists():raise FileExistsError("will not overwrite earlier independent quality audit")
    manifest,native,metrics=check_actual()
    q4=metrics["q4"]
    old=metrics["role_random"]
    v3=metrics["v3_cov"]
    report=dict(schema="BEGLIN-Qv3-independent-native-held10-covariance-vs-diag/1",
       status="PASS_INDEPENDENT_NATIVE_AUDIT__RESEARCH_ONLY",
       source_original_BF16_rev="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
       original_milp_source_doc_sha256=MILP_SHA,
       original_10_public_window_sha256=INPUT_SHA,
       original_same_5_native_modes=True,
       native_research_result_source_sha256=sha(private(ORIGINAL)),
       native_eval_manifest_sha256=sha(private(FROZEN)),
       metrics=metrics,
       covariance_quality_better_than_all_q4_PPL=v3["ppl"]<q4["ppl"],
       covariance_quality_better_than_seed73_random_PPL=v3["ppl"]<old["ppl"],
       covariance_nll_gain_vs_diagonal=metrics["v2_diag"]["nll_nats"]-v3["nll_nats"],
       actual_fullmodel_async_on=False,
       source_model_or_production_changed=False,
       new_eval_ten_public_prompts_are_development_not_final=True,
       old_V6_used24_ids_unknown_and_not_opened=True,
       scientific_Q_conclusion="NO_VERDICT",
       limitations=[
         "Frozen v3 empirical covariance source calibration shares the same 12 public prompts as v2 but estimates cross-feature correlations.",
         "New ten public evaluation windows differ from old eight synthetic windows; cross-run NLL/PPL values not directly comparable.",
         "Source full covariance projection error remains local, not downstream final-logit causal sensitivity.",
         "No actual new asynchronous whole-decoder execution; previous v2 real pthread async had no stable speed gain, OFF.",
         "Native warmed one-token fastest of three != sustained serving throughput/p95; source F32 is BF16-widened arithmetic.",
         "Original V6 consumed 24 document IDs remain unknown; do not certify IID/OOD generality or model accuracy superiority."
       ])
    os.umask(0o077)
    with OUTPUT.open("xb") as f:
        f.write((json.dumps(report,indent=2,sort_keys=True)+"\n").encode());f.flush();os.fsync(f.fileno())
    OUTPUT.chmod(0o600)
    print("BEGLIN_QV3_10PUBLIC_NATIVE_INDEPENDENT_PASS",json.dumps({
      "audit_sha256":sha(private(OUTPUT)),
      "PPL":{k:round(v["ppl"],5) for k,v in metrics.items()},
      "v3_better_than_Q4":report["covariance_quality_better_than_all_q4_PPL"],
      "v3_better_than_random":report["covariance_quality_better_than_seed73_random_PPL"],
      "v3_minus_v2_nll":-report["covariance_nll_gain_vs_diagonal"],
      "no_live_promotion":True
    },sort_keys=True),flush=True)
if __name__=="__main__":run()
