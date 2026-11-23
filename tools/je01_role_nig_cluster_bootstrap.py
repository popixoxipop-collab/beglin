#!/usr/bin/env python3
"""JE01 synthetic real-BF16 Expert-role NIG: 8-cluster paired bootstrap audit.

8 Experts, each carries 32 correlated hidden-state evaluations. Resample
EXPERTS (not naive 256 individual samples). The result is diagnostic, not
full-model NLL, and no pre-registered gold NLL thresholds can be reused.
"""
import argparse,hashlib,json,math,random
from pathlib import Path
SOURCE="72769af77d5cfbce683333bd4f1fe1d8aeb728a196e9d477df7ca851fe1c602c"
SCHEMA="beglin-je01-real-bf16-role-NIG-8expert-cluster-bootstrap/1"
N=20000
SEED=20261123
NIG="NIG_proxy_upper_6roles"
CONTROLS=("weight_MSE_proxy_6roles","prelabel_frequency_2experts_x3","fixed_role_permutation_6roles")
def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def percentile(arr,q):
    s=sorted(arr)
    p=(len(s)-1)*q
    i=int(p);f=p-i
    return s[i]*(1-f)+s[min(i+1,len(s)-1)]*f
def audit(source):
    d=json.loads(source.read_text())
    sig=d.pop("receipt_sha256",None)
    if sig!=SOURCE or canonical(d)!=SOURCE or d.get("status")!="REAL_BF16_SOURCE_ROLE_LEVEL_NIG_PROXY_TRAIN32_TEST32_COMPLETE":
        raise ValueError("JE01_EXPERT_ROLE_SOURCE_RECEIPT_SHA_WRONG")
    policies=d["synthetic_test_policy_results"]
    if set(policies)!={NIG,*CONTROLS}:
        raise ValueError("JE01_EXPERT_ROLE_POLICY_SET_CHANGED")
    values={}
    for name,v in policies.items():
        a=v["per_case_relative_MSE"]
        if len(a)!=256 or not all(type(z) in (int,float) and math.isfinite(z) and z>=0 for z in a):
            raise ValueError("JE01_ROLE_CASE_MSE_NONFINITE_OR_COUNT")
        byexpert=[sum(a[k*32:(k+1)*32])/32 for k in range(8)]
        original=sum(byexpert)/8
        if abs(original-v["independent_heldout_synthetic_mean8_relative_FFN_MSE"])>1e-11:
            raise ValueError("JE01_ROLE_MSE_MEAN_FROM_SAMPLES_DRIFT")
        values[name]=byexpert
    rng=random.Random(SEED)
    picks=[tuple(rng.randrange(8) for _ in range(8)) for i in range(N)]
    results={}
    for control in CONTROLS:
        differences=[values[control][i]-values[NIG][i] for i in range(8)]
        observed=sum(differences)/8
        bs=[sum(differences[k] for k in idx)/8 for idx in picks]
        lower=percentile(bs,0.025);upper=percentile(bs,0.975)
        p_one=(1+sum(v<=0 for v in bs))/(N+1)
        results[control]={"NIG_better_positive_mean_proxy_delta":observed,
                          "clustered_95pct_paired_delta_interval":[lower,upper],
                          "one_sided_bootstrap_p_gain_le_zero":p_one,
                          "independent_clustering_unit":"8_EXPERTS_NOT_256_ACTIVATIONS",
                          "CI_lower_positive":lower>0}
    # Valid multiplicity correction on one-sided tail probabilities.
    ordered=sorted(CONTROLS,key=lambda name:results[name]["one_sided_bootstrap_p_gain_le_zero"])
    maximal=0.0
    for rank,name in enumerate(ordered):
        raw=results[name]["one_sided_bootstrap_p_gain_le_zero"]
        maximal=max(maximal,min(1.0,(3-rank)*raw))
        results[name]["Holm_adjusted_one_sided_p"]=maximal
        results[name]["Holm_p_lt_0_05"]=maximal<0.05
    out={"schema":SCHEMA,"status":"REAL_SOURCE_BF16_ROLE_NIG_INDEP_SYNTHETIC_CLUSTER_BOOTSTRAP_COMPLETE",
         "source_real_BF16_synthetic_holdout_receipt_sha256":SOURCE,
         "resample_seed":SEED,"resamples":N,
         "cluster_count":8,"observations_per_cluster":32,
         "source_actual_gold_token_NLL":"NOT_MEASURED",
         "source_actual_27layer_model_forward":"NOT_RUN",
         "three_NIG_vs_locked_control_comparisons":results,
         "any_all_three_Holm_significant":all(v["Holm_p_lt_0_05"] for v in results.values()),
         "not_pre_registered_model_quality_gain":True,
         "production_modified":False}
    out["receipt_sha256"]=canonical(out)
    return out
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        if a.output.exists():raise ValueError("JE01_OUTPUT_ALREADY_EXISTS")
        d=audit(a.source)
        a.output.write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"status":d["status"],"results":d["three_NIG_vs_locked_control_comparisons"],
                          "receipt_sha256":d["receipt_sha256"]},sort_keys=True),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:260]}))
        raise SystemExit(2)
