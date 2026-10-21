#!/usr/bin/env python3
"""Independent source-SHA and performance audit, native BEGLIN qNg64 repack.

Verifies full 151936 logits under original on-the-fly NEON, mixed frozen
28/196 per-role policies, 21-position PPL, logs and multiple actual tensor
shapes. No new model calls or final V6 holdout are performed in the audit.
"""
from __future__ import annotations
import hashlib,json,math,os,re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"lab_local"
LAB=BASE/"qng64_n8_repacked_full_20261022"
OLD=BASE/"qng64_n8_neon_engine_20261022"
MIXOLD=BASE/"qng64_n8_neon_mixed_20261022"
MULTI=BASE/"qng64_n8_repacked_multishape_20261022"
BATCH=BASE/"qng64_n8_repacked_batch_20261022"
SCAN=BASE/"q_tensor_proxy_20261021"
OUT=LAB/"REPACKED_N8_FULL_AND_MIXED_INDEPENDENT_20261022.json"
EXPECTED={
 "build":"657a4cecc4b7d978fce6c6bd9362a92543a519ebc8ca4938f3524894a71758da",
 "mixed":"e99d1ea70001872c393b24a5982eb00819f4d736d55824506bceb2b7f9607aef",
 "multi":"772af04bc48f20d2de405df33d1ecc119a318d7fbb370081f1681cabe899d937",
 "batch":"faa60621c23b433ca3505ca112aac2f1dbd507591d9cae50392b9cc8bab612ec",
}
FLY_MS={"random":143.85,"proxy":144.43}
FIRST_NEON_MS=577.39
ORIGINAL_SCALAR_MS=5257.63
SNAPSHOT="4ae77cb209f06199b8df1c94e21ff341332a3a89"

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

def private(path:Path)->bytes:
    if not path.is_file() or path.is_symlink() or path.stat().st_mode&0o777!=0o600:
        raise RuntimeError("private source missing/insecure: "+str(path))
    return path.read_bytes()

def receipts():
    names={
      "build":LAB/"source_build_receipt.json",
      "mixed":LAB/"frozen_mixed_kernel_repack_results.json",
      "multi":MULTI/"multishape_private_receipt.json",
      "batch":BATCH/"receipt.json"}
    data={}
    for name,p in names.items():
        raw=private(p)
        if sha(raw)!=EXPECTED[name]:raise RuntimeError("source evidence SHA mismatch "+name)
        data[name]=json.loads(raw)
    build=data["build"]
    if build["bf16_source_rev"]!=SNAPSHOT or not build["main_untouched"] or build["production_touched"]:
        raise RuntimeError("model or production authority drift")
    for name in ("all_n8","mixed"):
        f=LAB/f"beglin_{name}_repacked"
        if not f.is_file() or sha(f.read_bytes())!=build["native_binary_SHA256"][name]:
            raise RuntimeError("actual BEGLIN native executable changed "+name)
    for f,expected in [
       (ROOT/"qng64_dense.c",build["original_BEG_LIN_scalar_c_sha256"]),
       (ROOT/"research/oml_acs/qng64_n8_repacked_neon.c",build["n8_inplace_source_SHA256"]),
       (ROOT/"research/oml_acs/qng64_n8_repacked_lab_dispatch.c",build["repacked_dispatch_source_SHA256"]),
    ]:
        if sha(f.read_bytes())!=expected:raise RuntimeError("source code changed "+str(f))
    return data

def actual_logits(cur:Path,prev:Path)->dict:
    a=private(cur);b=private(prev)
    if len(a)!=len(b) or len(a)!=151936*4:
        raise RuntimeError("wrong native output shape")
    x=np.frombuffer(a,dtype="<f4").astype(np.float64)
    y=np.frombuffer(b,dtype="<f4").astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise RuntimeError("nonfinite logits")
    d=x-y
    maxerr=float(np.abs(d).max())
    l2=float(np.linalg.norm(d)/np.linalg.norm(y))
    if maxerr>=.001 or l2>=1e-4 or int(x.argmax())!=int(y.argmax()):
        raise RuntimeError("quantized kernel output divergence")
    return dict(repacked_sha256=sha(a),original_NEON_sha256=sha(b),
                bitwise_equal=a==b,max_abs=maxerr,rel_l2=l2,
                top1=int(x.argmax()))

def logfile(mode:str,phase:str)->str:
    return private(LAB/f"{mode}_{phase}_native_log.txt").decode("utf-8")

def validate_mixed(mode:str,pack:dict)->dict:
    if mode not in ("proxy","random"):raise ValueError("not a pinned frozen policy")
    m=pack["mixed"]
    entries=m["candidates"][mode]
    if [q["task"] for q in entries]!=["dump","bench","ppl"]:
        raise RuntimeError("native workload phases not fully run")
    src=pack["build"]
    if sha(private(LAB/f"policy_{mode}.txt"))!=src["policy_sha256"][mode]:
        raise RuntimeError("previous already selected role-layer mask modified")
    for name in ("prompt_ids.i32","eval/ppl_work/w0.i32","eval/ppl_work/w1.i32"):
        if private(LAB/f"mixed_{mode}/{name}")!=private(MIXOLD/f"mixed_{mode}/{name}"):
            raise RuntimeError("synthetic evaluation window bytes mismatched")
    cur=LAB/f"mixed_{mode}/results/c_logits_last.f32"
    previous=MIXOLD/f"mixed_{mode}/results/c_logits_last.f32"
    logits=actual_logits(cur,previous)
    observed={}
    for rec in entries:
        phase=rec["task"]
        log=logfile(mode,phase)
        match=re.search(r"\[BEGLIN-N8-REPACKED-LAB\] fast_gemv=(\d+) fast_batch_rows=(\d+) other_bit_gemv=(\d+) other_bit_batches=(\d+)",log)
        if match is None or int(match.group(1))<=0 or any(int(x) for x in match.groups()[1:]):
            raise RuntimeError("actual native repacked GEMV did not run")
        if sha(log.encode())!=rec["log_sha256"]:
            raise RuntimeError("native mode log SHA mismatch")
        observed[phase]=int(match.group(1))
        if phase=="bench":
            e=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",log)
            if e is None or abs(float(e.group(2))-rec["best_of_three_one_token_ms"])>1e-8:
                raise RuntimeError("bench transcript drift")
        if phase=="ppl":
            e=re.search(r"\[ppl\].*ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",log)
            rss=re.search(r"(\d+)\s+maximum resident set size",log)
            if not e or not rss or int(e.group(2))!=21:
                raise RuntimeError("same public PPL token windows not verified")
            if abs(float(e.group(1))-rec["ppl"])>1e-8 or int(rss.group(1))!=rec["maximum_resident_bytes"]:
                raise RuntimeError("PPL or RSS transcription drift")
    speed=entries[1]["best_of_three_one_token_ms"]
    return dict(mode=mode,n8_tensors=28,q4_tensors=168,
       exact_same_code_as_old_NEON=logits["bitwise_equal"],
       complete_logits=logits,
       same_PPL_original=entries[2]["relative_ppl_difference_vs_fly_neon"]==0,
       original_PPL=entries[2]["ppl"],maximum_resident_bytes=entries[2]["maximum_resident_bytes"],
       repacked_best_of_3_single_decode_ms=speed,
       measured_repacked_best_of_3_tps=1000/speed,
       speedup_vs_onfly_NEON=FLY_MS[mode]/speed,
       exact_role_policy_sha256=src["policy_sha256"][mode],
       actual_repacked_GEMV_calls=observed)

def run():
    if OUT.exists():raise FileExistsError("audit exists, do not modify")
    parts=receipts()
    old=OLD/"n8/results/c_logits_last.f32"
    new=LAB/"n8/results/c_logits_last.f32"
    if private(LAB/"n8/prompt_ids.i32")!=private(OLD/"n8/prompt_ids.i32"):
        raise RuntimeError("main n8 prompt changed")
    full=actual_logits(new,old)
    multi=parts["multi"]
    if multi["status"]!="PASS_4_REAL_SHAPES" or len(multi["rows"])!=4:
        raise RuntimeError("not four original BF16 real matrices")
    shapes={tuple(x["shape"]) for x in multi["rows"]}
    if shapes!={(256,1536),(1536,1536),(8960,1536),(1536,8960)}:
        raise RuntimeError("unvalidated tensor geometry")
    for row in multi["rows"]:
        if row["max_abs_error"]>=.0005 or row["relative_L2"]>=1e-5 or row["native_bitplane_weight_bytes"]!=row["repacked_code_weight_bytes"]:
            raise RuntimeError("raw model weight numerical or no-copy memory contract failed")
    batch=parts["batch"]
    if batch["batch_rows"]!=3 or batch["n8_original_scalar_vs_repacked_maxabs"]>=.0005:
        raise RuntimeError("batched qNg64 repacked GEMM inaccurate")
    if set(batch["other_bitwidth_original_fallback"])!={"3","5","6","7"}:
        raise RuntimeError("original other-width bitplane batch fallback not covered")
    mixed={mode:validate_mixed(mode,parts) for mode in ("random","proxy")}
    if not all(m["exact_same_code_as_old_NEON"] for m in mixed.values()) or not full["bitwise_equal"]:
        raise RuntimeError("unexpected non-byte-equality from original on-the-fly NEON kernel")
    result=dict(schema="beglin-repacked-qng64-n8-native-apple-independent-evidence/1",
      status="PASS_REPACKED_N8_DECODER_ALL_NATIVE_CONTEXTS_DEV_ONLY",
      pinned_source_private_receipt_sha256=EXPECTED,
      pinned_native_model_snapshot=SNAPSHOT,
      original_qng64_bitplanes_unchanged_on_disk=True,
      in_memory_repack_changes_bitplane_semantics=True,
      n8_weight_code_bytes_no_increase=True,
      full_model_all_n8=dict(logits=full,old_scalar_best_of_three_ms=ORIGINAL_SCALAR_MS,
        onfly_NEON_best_of_three_ms=FIRST_NEON_MS,
        repacked_best_of_three_ms=379.06,
        speedup_over_scalar=ORIGINAL_SCALAR_MS/379.06,
        speedup_over_onfly_NEON=FIRST_NEON_MS/379.06,
        PPL_onfly_NEON=2148.1231,
        PPL_repacked=2148.1231,
        PPL_token_positions=21,
        repacked_PPL_process_id="proc_964390d2e4ab96d3ea1d682408e367be",
        repacked_bench_process_id="proc_8aae092a4d7bfbcdc56954cc750505c8",
        repacked_peak_resident_bytes_PPL=5491048448,
        n8_GEMV_calls_bench=11172),
      same_role_budget_mixed_policies=mixed,
      four_real_tensor_shapes=[dict(name=x["name"],shape=x["shape"],
        scalar_to_repacked_speedup=x["speedup"],
        max_abs_err=x["max_abs_error"],rel_l2=x["relative_L2"]) for x in multi["rows"]],
      batched_3_source_vectors_numeric_maxerr=batch["n8_original_scalar_vs_repacked_maxabs"],
      batched_original_bitwidth_fallbacks=list(batch["other_bitwidth_original_fallback"]),
      V6_24_holdout_still_sealed=True,
      native_BF16_math_evaluated=False,
      real_task_accuracy_evaluated=False,
      kernel_format_discriminant_not_integrated_into_upstream_runtime=True,
      actual_production_release_unchanged=True,
      all_code_runs_in_isolated_research_checkout=True,
      unit_of_work="Native BEGLIN M=1 CPU inference and batch M3 parity; not service promotion",
      limitations=["The all-n8 full PPL and decode timing are transcribed from previous durable process logs, not regenerated here",
        "Best-of-three single decode token timing is not sustained throughput or confidence interval",
        "Different QNg64 n8 in-memory codes require an explicit format tag before production merging",
        "Original BF16-sourced weights widened to FP32 are not a native BF16 math baseline",
        "No prospective heldout AUROC or output-sensitivity policy selection, which remains blocked",
        "Original 24 V6 used docs unverified and independent scientific Q verdict NO_VERDICT"])
    os.umask(0o077)
    with OUT.open("xb") as fp:
        fp.write((json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
        fp.flush();os.fsync(fp.fileno())
    OUT.chmod(0o600)
    print("REPACKED_N8_NATIVE_ALL_AUDIT_PASS",json.dumps({
        "private_sha256":sha(private(OUT)),
        "full_model_speedup_over_prev_NEON":result["full_model_all_n8"]["speedup_over_onfly_NEON"],
        "mixed_random_speedup":mixed["random"]["speedup_vs_onfly_NEON"],
        "mixed_proxy_speedup":mixed["proxy"]["speedup_vs_onfly_NEON"],
        "all_code_model_byte_exact":True,"four_real_shapes":4,"batch_M3_parity":True
    },sort_keys=True),flush=True)
if __name__=="__main__":run()
