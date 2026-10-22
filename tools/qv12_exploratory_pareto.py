#!/usr/bin/env python3
"""Exploratory Qv12 7-arm packed-byte/NLL Pareto audit; not a Qv13 allocator."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/research/BEGLIN_QV12_REAL_TWO_N8_NATIVE_INTERACTION_INDEPENDENT_PUBLIC_20261023.json"
COST = ROOT / "docs/research/BEGLIN_QV12_ONE_TWO_N8_PACKED_WEIGHT_COST_PUBLIC_20261023.json"
EXPECTED = {
    "audit": "89771cd5dafed2b8ee299d328125803ee039aaa2efc9195c5847a2bce83f8bac",
    "cost": "29caa29298420cdbcd2bb6b7c1789a99667036d515925ef6c2b93d52d4175073",
}
PRIVATE_SHA = "57e4766da993561ff3264288fdb2d1c446ee89b0b44cf90aa29436d11718d9e9"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path, expected):
    if digest(path) != expected:
        raise ValueError("source SHA mismatch: " + Path(path).name)
    return json.loads(Path(path).read_text())

def ci(values, seed=20261023, draws=10000):
    rng=random.Random(seed)
    n=len(values)
    means=sorted(sum(values[rng.randrange(n)] for j in range(n))/n for k in range(draws))
    return [means[250],means[9749]]

def audit(private_path):
    a=load(AUDIT,EXPECTED["audit"])
    c=load(COST,EXPECTED["cost"])
    s=a["native_summary"]
    prices=c["native_one_two_bit_arm_costs"]
    if set(s)!=set(prices) or len(s)!=7:
        raise ValueError("missing native arm")
    q4=s["q4"]["mean_fullvocab_correct_token_NLL"]
    rows={}
    for arm,z in s.items():
        v=prices[arm]
        if z["n"]!=16 or z["correct"]!=7 or z["n8_count"]!=v["n8_role_count"]:
            raise ValueError("arm size, labels or mask mismatch")
        rows[arm]=dict(correct=7,n8_count=z["n8_count"],nll=z["mean_fullvocab_correct_token_NLL"],
          added_bytes=v["incremental_n8_weight_bytes"],gain_q4=q4-z["mean_fullvocab_correct_token_NLL"])
    def dominates(b,a):
        return (b["added_bytes"]<=a["added_bytes"] and b["nll"]<=a["nll"] and
                (b["added_bytes"]<a["added_bytes"] or b["nll"]<a["nll"]))
    dominated={k:[j for j,v in rows.items() if j!=k and dominates(v,z)] for k,z in rows.items()}
    frontier=sorted([k for k in rows if not dominated[k]],key=lambda k:(rows[k]["added_bytes"],rows[k]["nll"]))
    result=dict(schema="BEGLIN-Qv12-exploratory-packed-byte-pareto/1",
      status="EXPLORATORY_QV12_COHORT_ONLY_NOT_QV13_POLICY_VALIDATION",
      source_sha256=EXPECTED,
      sample_n_articles=16,
      baseline_projection_packed_bytes=c["original_BEGLIN_Q4_projection_baseline_bytes"],
      data=rows,
      pareto_frontier=frontier,
      dominated_by={k:v for k,v in dominated.items() if v},
      limitation="Out-of-sample superiority, original 28-n8 mask, native RSS and throughput NOT measured")

    if private_path:
        private=load(private_path,PRIVATE_SHA)["native_7arm_results"]
        if set(private)!=set(s):
            raise ValueError("private arms mismatch")
        patterns=[]
        for k in s:
            vals=private[k]["per_case_correct_fullvocab_NLL"]
            pats=private[k]["per_case_choice_correct"]
            if len(vals)!=16 or len(pats)!=16 or abs(sum(vals)/16-rows[k]["nll"])>1e-9:
                raise ValueError("private per-article metric mismatch")
            if sum(bool(v) for v in pats)!=7:
                raise ValueError("private accuracy differs")
            patterns.append(tuple(pats))
        if len(set(patterns))!=1:
            raise ValueError("same seven correct not reproducible")
        comparisons=[("v_proj_20__k_proj_0","down_proj_10"),
                     ("v_proj_20__k_proj_0","down_proj_10__k_proj_0"),
                     ("v_proj_20","down_proj_10")]
        pairs=[]
        for better,other in comparisons:
            x=private[better]["per_case_correct_fullvocab_NLL"]
            y=private[other]["per_case_correct_fullvocab_NLL"]
            d=[b-a for a,b in zip(x,y)]
            pairs.append(dict(candidate=better,comparator=other,
              mean_NLL_advantage=sum(d)/len(d), bootstrap95=ci(d),
              articles_candidate_better=sum(v>0 for v in d),
              articles_candidate_worse=sum(v<0 for v in d),
              note="Post-hoc unadjusted article-paired percentile bootstrap, 10000 draws; CI crossing zero is inconclusive"))
        result["private_aggregate_only"]=dict(
          source_sha256=PRIVATE_SHA,
          raw_data_exported=False,
          all_seven_conditions_same_correct_items=True,
          paired_comparisons=pairs)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--private",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    r=audit(args.private)
    value=json.dumps(r,indent=2,sort_keys=True,ensure_ascii=False)+"\n"
    if args.output:
        args.output.write_text(value)
        print("WROTE",args.output,"SHA256",digest(args.output))
    else:
        print(value)
