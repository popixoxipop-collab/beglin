#!/usr/bin/env python3
"""Actual full27 teacher-forced NLL *secant* audit for JE01 calibration.

Computes a finite *endpoint secant*, never calls it a parameter gradient or
Fisher information. The candidate perturbation is a simultaneous 24-role
AF4 -> 12AF4/6Q4/6Q5 change. Only previously inspected TRAIN16 gold targets
are used. Bayesian bootstrap models reweight the 16 sentence clusters; this
is exploratory and does not establish independent validation quality.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import random

ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=ROOT/"docs"/"research"
AUDITOR=ROOT/"tools"/"je01_cal240_rolewise_fullgold16_independent_audit.py"
EXPECTED={
  "baseline":("JE01_REAL27_ORIGINAL_AF4_656GOLD_BASELINE_20261122.json",
    "1289088dd3cff691b9a2a8120442e221346b3fd281656fe129ea30443ae9d09e"),
  "off":("JE01_REAL27_CAL240_ROLEWISE_OFF_656GOLD_NLL_20261122.json",
    "22f5ce69b63208e3bd0a6a6bb29e119183110b90517c4e2afb3f7d57095fe661"),
  "on":("JE01_REAL27_CAL240_ROLEWISE_ON_656GOLD_NLL_20261122.json",
    "54afc4f7e7faa86b6c4a9c21286a95cae02856de2d6e3c8dfa309bb1011c0510"),
}
SEED=20261123
DRAWS=20000
spec=importlib.util.spec_from_file_location("je01_real27_independent_audit",AUDITOR)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()


def canon(d):
    return sha(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode())


def pinned_sources(root=DATA):
    files={}
    for k,(fn,h) in EXPECTED.items():
        p=root/fn
        mod.fail(p.is_file() and not p.is_symlink() and sha(p.read_bytes())==h,
                 "JE01_REAL_SOURCE_SHA_DRIFT_"+k)
        files[k]=p
    return files


def quantile(xs,alpha):
    s=sorted(xs)
    pos=(len(s)-1)*alpha
    i=int(pos)
    return (s[i]*(1-(pos-i)) + s[min(i+1,len(s)-1)]*(pos-i))


def bayesian_bootstrap(case_deltas,case_token_counts,draws=DRAWS,seed=SEED):
    """Dirichlet(1,...,1) prior over 16 observed *sentence* support points.

    Does not grant conditional independence to the 656 tokens. Reweight each
    sentence's token sum, divide by reweighted number of target tokens.
    """
    assert len(case_deltas)==len(case_token_counts)==16
    assert all(isinstance(n,int) and n>0 for n in case_token_counts)
    assert all(math.isfinite(x) for x in case_deltas)
    rng=random.Random(seed)
    z=[]
    for _ in range(draws):
        weights=[rng.expovariate(1.0) for _ in range(16)]
        weighted_n=math.fsum(w*n for w,n in zip(weights,case_token_counts))
        z.append(math.fsum(w*d for w,d in zip(weights,case_deltas))/weighted_n)
    return {
        "method":"BAYESIAN_BOOTSTRAP_DIRICHLET_1_OVER_16_OBSERVED_SENTENCE_CLUSTERS",
        "seed":seed,"draws":draws,
        "conditional_95pct_interval_delta_NLL":[quantile(z,0.025),quantile(z,0.975)],
        "posterior_fraction_candidate_mean_NLL_lower":math.fsum(d<0 for d in z)/len(z),
        "conditioning_note":"Previously inspected 16 TRAIN prompts; Bayesian bootstrap conditional on observed support, NOT heldout posterior predictive confidence",
    }


def audit(root=DATA):
    paths=pinned_sources(root)
    signed=mod.audit(paths["baseline"],paths["off"],paths["on"])
    off=mod.signed_json(paths["off"])
    on=mod.signed_json(paths["on"])
    off_case=mod.validate_run(off,"off",mod.OFF_STATUS)
    on_case=mod.validate_run(on,"rolewise",mod.ON_STATUS)
    deltas=[]
    by_sentence=[]
    sums=[]
    lengths=[]
    for i,(o,n) in enumerate(zip(off["per_gold_target_actual_NLL"],on["per_gold_target_actual_NLL"])):
        mod.fail(len(o)==len(n)==on_case[i]["tokens"] and
                 off_case[i]["tokens"]==on_case[i]["tokens"],
                 "JE01_SECANT_TOKEN_ALIGNMENT_DRIFT")
        diffs=[float(a-b) for a,b in zip(n,o)]
        mod.fail(all(math.isfinite(d) for d in diffs) and
                 abs(math.fsum(diffs)-on_case[i]["candidate_minus_original_NLL_sum"])<1e-8,
                 "JE01_SECANT_TOKENS_NO_LONGER_SUM_TO_NATIVE_NLL")
        sums.append(math.fsum(diffs))
        lengths.append(len(diffs))
        deltas+=diffs
        by_sentence.append({
          "sentence_index":i,"gold_tokens":len(diffs),
          "delta_mean_NLL_per_token":math.fsum(diffs)/len(diffs),
          "delta_NLL_sum":math.fsum(diffs),
          "secant_score_rms":math.sqrt(math.fsum(d*d for d in diffs)/len(diffs)),
          "candidate_worse_tokens":sum(d>0 for d in diffs),
          "candidate_better_tokens":sum(d<0 for d in diffs),
        })
    mod.fail(len(deltas)==656 and sum(lengths)==656 and
             abs(math.fsum(deltas)/656-signed["paired_delta_candidate_minus_baseline"])<1e-10,
             "JE01_SECANT_GLOBAL_DELTA_NOT_MATCH_NATIVE")
    mean=math.fsum(deltas)/656
    sq=math.fsum(d*d for d in deltas)/656
    mod.fail(math.isfinite(sq) and sq>=mean*mean-1e-12,"JE01_SECANT_VARIANCE_INVALID")
    rank=sorted(by_sentence,key=lambda v:abs(v["delta_NLL_sum"]),reverse=True)
    result={
      "schema":"beglin-je01-real-original27-656gold-bayesian-bootstrap-and-secant-audit/1",
      "status":"JE01_ACTUAL_REAL27_LOSS_SECANT_NOT_EMPIRICAL_FISHER",
      "source_receipts_sha256":{k:h for k,(_,h) in EXPECTED.items()},
      "off_original_AF_output_sha256":mod.ORIGINAL_OUTPUT_SHA,
      "baseline_NLL":signed["baseline_NLL"],
      "mixed_NLL":signed["candidate_NLL"],
      "mean_delta_NLL":mean,
      "relative_delta_NLL_pct":100*mean/signed["baseline_NLL"],
      "total_token_log_evidence_ratio_candidate_vs_baseline":-math.fsum(deltas),
      "sentence_clusters":16,
      "gold_tokens":656,
      "secant_score_definition":"s_t=logp_mixed(y_t|context)-logp_AF4(y_t|context)=-[NLL_mixed_t-NLL_AF4_t]",
      "secant_score_squared_mean":sq,
      "secant_score_variance_about_mean":max(0.0,sq-mean*mean),
      "secant_score_rms":math.sqrt(sq),
      "token_delta_abs_p95":quantile([abs(d) for d in deltas],0.95),
      "token_delta_abs_max":max(abs(d) for d in deltas),
      "tokens_mixed_NLL_worse":sum(d>0 for d in deltas),
      "tokens_mixed_NLL_better":sum(d<0 for d in deltas),
      "top5_sentence_contribution":rank[:5],
      "per_sentence":by_sentence,
      "bayesian_bootstrap":bayesian_bootstrap(sums,lengths),
      "fisher_diagonal_24_roles":"NOT_MEASURED",
      "logprob_gradient_wrt_mixing_alpha":"NOT_MEASURED",
      "hessian_or_EWC_precision":"NOT_MEASURED",
      "bayesian_bit_allocation_improved_full27_NLL":"NOT_TESTED",
      "independent_validation128":"NOT_ACCESSED",
      "independent_OOD128":"NOT_ACCESSED",
      "empirical_Fisher_disclaimer":"This is a squared whole-policy ENDPOINT SECANT of token log probability, NOT a squared derivative or empirical Fisher, and cannot identify sensitive Expert roles.",
      "read_only":True,"production_modified":False,
    }
    result["receipt_sha256"]=canon(result)
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=pathlib.Path,default=DATA)
    ap.add_argument("--output",type=pathlib.Path,default=None)
    args=ap.parse_args()
    report=audit(args.data_root)
    raw=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if args.output:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("JE01_NO_REWRITE_OF_ALREADY_SEALED_RESULT")
        args.output.write_text(raw)
    print(json.dumps({k:v for k,v in report.items() if k not in ("per_sentence","top5_sentence_contribution")},sort_keys=True))
