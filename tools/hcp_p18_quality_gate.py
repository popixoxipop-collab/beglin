#!/usr/bin/env python3
"""Beglin P18 external paired research gate. Never authorizes production.
Only consume independently audited raw BF16/native evidence and a frozen cohort.
The gate does NOT itself authenticate raw GPU arrays or task-output provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import pathlib
import random
from collections import Counter, defaultdict

SCHEMA="beglin-hcp-p18-independently-audited-paired-scores-v1"
COHORT_SCHEMA="beglin-hcp-p18-frozen-external-cohort-v1"
PROTOCOL_SCHEMA="beglin-hcp-p18-preregistered-protocol-v1"
SOURCES=("mmlu","arc_challenge","gsm8k")
COUNT_PER_SOURCE=160
SEED=20261015
REPLICATES=10000
ALPHA=0.05/3
NLL_MARGIN=0.05
ACCURACY_MARGIN=-0.05
LATENCY_RATIO_MARGIN=1.10
P95_LATENCY_RATIO_MARGIN=1.15
BF16_TASK_FLOORS={"mmlu":0.30,"arc_challenge":0.30,"gsm8k":0.10}

class GateInputError(ValueError):
    pass

def reject_if(condition,message):
    if condition:
        raise GateInputError(message)

def is_sha(s):
    return isinstance(s,str) and len(s)==64 and all(c in "0123456789abcdef" for c in s)

def is_git_revision(s):
    """Hugging Face/Git SHA-1 commit is a 40-hex ref; Parquet digests remain SHA-256."""
    return isinstance(s,str) and len(s)==40 and all(c in "0123456789abcdef" for c in s)

def sha_file(p):
    f=pathlib.Path(p)
    reject_if(not f.is_file() or f.is_symlink(),"source must be regular file, not a symlink")
    return hashlib.sha256(f.read_bytes()).hexdigest()

def load_json(p):
    reject_if(pathlib.Path(p).is_symlink(),"no symlink inputs")
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))

def positive(v, name):
    reject_if(type(v) not in (float,int) or not math.isfinite(v) or v<=0,name+" invalid")
    return float(v)

def losses(values,n):
    reject_if(not isinstance(values,list) or len(values)!=n,"missing teacher-forced gold-token NLL")
    reject_if(any(type(x) not in (float,int) or not math.isfinite(x) or not 0<=x<150 for x in values),
              "invalid NLL float")
    return sum(values)/n

def quantile(values,q):
    vals=sorted(values)
    idx=(len(vals)-1)*q
    lo=math.floor(idx); hi=math.ceil(idx); f=idx-lo
    return vals[lo]*(1-f)+vals[hi]*f

def frozen_protocol(p):
    reject_if(p.get("schema")!=PROTOCOL_SCHEMA or p.get("status")!="FROZEN_BEFORE_EVALUATION",
              "protocol is not frozen")
    fixed=p.get("statistics",{})
    reject_if((fixed.get("seed")!=SEED or fixed.get("bootstrap_replicates")!=REPLICATES
        or fixed.get("familywise_alpha")!=0.05
        or fixed.get("bonferroni_endpoints")!=3
        or fixed.get("nll_margin_nats_per_gold_token")!=NLL_MARGIN
        or fixed.get("accuracy_margin_B_minus_A")!=ACCURACY_MARGIN
        or fixed.get("latency_geomean_ratio_ceiling")!=LATENCY_RATIO_MARGIN
        or fixed.get("p95_latency_ratio_ceiling")!=P95_LATENCY_RATIO_MARGIN),
        "preregistered P18 margins changed")
    ds=p.get("datasets")
    reject_if(not isinstance(ds,dict) or set(ds)!=set(SOURCES),"unexpected sources")
    for name in SOURCES:
        x=ds[name]
        reject_if(x.get("n")!=COUNT_PER_SOURCE or x.get("split")!="test"
                  or not is_git_revision(x.get("revision")) or not is_sha(x.get("raw_dataset_sha256"))
                  or not x.get("license"),"dataset revision/hash/license not pinned")
    reject_if(p.get("automatic_live_promotion") is not False
              or p.get("incumbent")!="A" or p.get("challenger")!="B",
              "unsafe P18 serving policy")
    return p

def frozen_cohort(c,protocol):
    reject_if(c.get("schema")!=COHORT_SCHEMA,"wrong cohort schema")
    qs=c.get("items")
    reject_if(not isinstance(qs,list) or len(qs)!=3*COUNT_PER_SOURCE,"requires 480 frozen items")
    ids=set(); counts=Counter(); caps=defaultdict(Counter); shots=defaultdict(Counter)
    for q in qs:
        reject_if(not isinstance(q,dict),"bad item")
        item_id=q.get("item_id"); source=q.get("dataset")
        reject_if(not isinstance(item_id,str) or len(item_id)>120 or item_id in ids,
                  "duplicate or missing item_id")
        ids.add(item_id)
        reject_if(source not in SOURCES,"unexpected dataset")
        counts[source]+=1
        caps[source][q.get("max_new_tokens")]+=1
        shots[source][q.get("shot_count")]+=1
        ds=protocol["datasets"][source]
        reject_if(q.get("dataset_revision")!=ds["revision"]
                  or q.get("split")!="test","dataset revision/split not frozen")
        for k in ("record_sha256","prompt_token_sha256","gold_token_sha256"):
            reject_if(not is_sha(q.get(k)),k+" not pinned")
        n=q.get("gold_token_count")
        limit=q.get("max_new_tokens")
        reject_if(type(n)!=int or not 1<=n<=min(limit,128),"gold answer length outside generation cap")
        reject_if(type(q.get("prompt_token_count"))!=int
                  or not 8<=q["prompt_token_count"]<=4096,"prompt token length invalid")
        reject_if(q.get("decode_steps")!=limit
                  or q.get("shot_count") not in (0,3,5),"bad fixed-work latency configuration")
    reject_if(any(counts[s]!=COUNT_PER_SOURCE for s in SOURCES),"source strata not balanced")
    for s in SOURCES:
        reject_if(any(caps[s][cap]<50 for cap in (8,32,128)),"missing generation length stratum")
        reject_if(any(shots[s][shot]<50 for shot in (0,3,5)),"missing few-shot stratum")
    return qs

def policy_metrics(cell,n,decode_steps):
    reject_if(not isinstance(cell,dict) or not is_sha(cell.get("raw_evidence_sha256")),
              "missing per-policy raw source evidence SHA")
    task_correct=cell.get("independently_verified_task_correct")
    reject_if(type(task_correct)!=bool,"task correctness not independently verified")
    nll=losses(cell.get("gold_token_nll_nats"),n)
    trials=cell.get("latency_trials")
    reject_if(not isinstance(trials,list) or len(trials)!=3,"three native paired trials required")
    lat=[]
    for trial in trials:
        reject_if(not isinstance(trial,dict) or trial.get("decode_steps")!=decode_steps
                  or trial.get("precision_switch_excluded") is not True
                  or trial.get("cache_mode")!="warm" ,"invalid fixed-work latency trial")
        pre=positive(trial.get("prefill_ms"),"prefill_ms")
        dec=positive(trial.get("decode_ms"),"decode_ms")
        total=positive(trial.get("total_ms"),"total_ms")
        reject_if(total+0.001<pre+dec,"total latency inconsistent")
        lat.append(total)
    return nll,int(task_correct),quantile(lat,0.5)

def validate_rows(protocol, cohort, scores, protocol_sha, cohort_sha):
    reject_if(scores.get("schema")!=SCHEMA,"wrong independent audit schema")
    reject_if(scores.get("protocol_sha256")!=protocol_sha
              or scores.get("cohort_sha256")!=cohort_sha,"protocol or cohort SHA drift")
    reject_if(scores.get("independent_raw_full_logits_audit_passed") is not True
              or scores.get("independent_task_output_audit_passed") is not True
              or scores.get("production_touched") is not False
              or scores.get("original_serving_unchanged") is not True
              or scores.get("policy_A_B_identity_verified") is not True
              or scores.get("original_BF16_identity_verified") is not True,
              "real native/teacher audit incomplete")
    for k in ("independent_audit_manifest_sha256","teacher_source_manifest_sha256",
              "native_source_manifest_sha256"):
        reject_if(not is_sha(scores.get(k)), k+" absent")
    observations=scores.get("records")
    reject_if(not isinstance(observations,list)
              or len(observations)!=len(cohort)
              or len({r.get("item_id") for r in observations if isinstance(r,dict)})!=len(cohort),
              "incomplete or duplicate question observations")
    byid={r["item_id"]:r for r in observations}
    rows=[]
    for q in cohort:
        rec=byid.get(q["item_id"])
        reject_if(not rec or rec.get("dataset")!=q["dataset"]
                  or rec.get("record_sha256")!=q["record_sha256"]
                  or rec.get("prompt_token_sha256")!=q["prompt_token_sha256"]
                  or rec.get("gold_token_sha256")!=q["gold_token_sha256"],
                  "paired question source identity mismatch")
        n=q["gold_token_count"]; cap=q["decode_steps"]
        teacher=rec.get("BF16")
        reject_if(not isinstance(teacher,dict)
                  or not is_sha(teacher.get("raw_evidence_sha256"))
                  or type(teacher.get("independently_verified_task_correct"))!=bool,
                  "original BF16 task audit missing")
        t_nll=losses(teacher.get("gold_token_nll_nats"),n)
        an,ac,al=policy_metrics(rec.get("A"),n,cap)
        bn,bc,bl=policy_metrics(rec.get("B"),n,cap)
        rows.append({"item_id":q["item_id"],"dataset":q["dataset"],
                     "BF16_NLL":t_nll,"BF16_task_correct":int(teacher["independently_verified_task_correct"]),
                     "A_NLL":an,"B_NLL":bn,"B_minus_A_NLL":bn-an,
                     "A_task_correct":ac,"B_task_correct":bc,"B_minus_A_accuracy":bc-ac,
                     "A_latency_ms":al,"B_latency_ms":bl,"log_B_over_A_latency":math.log(bl/al)})
    return rows

def bootstrap(rows):
    groups={s:[r for r in rows if r["dataset"]==s] for s in SOURCES}
    rng=random.Random(SEED)
    sampled_nll=[];sampled_acc=[];sampled_lat=[]
    for _ in range(REPLICATES):
        sample=[groups[s][rng.randrange(COUNT_PER_SOURCE)]
                for s in SOURCES for _ in range(COUNT_PER_SOURCE)]
        sampled_nll.append(sum(r["B_minus_A_NLL"] for r in sample)/len(sample))
        sampled_acc.append(sum(r["B_minus_A_accuracy"] for r in sample)/len(sample))
        sampled_lat.append(sum(r["log_B_over_A_latency"] for r in sample)/len(sample))
    return {
      "NLL_upper_one_sided_familywise":quantile(sampled_nll,1-ALPHA),
      "accuracy_lower_one_sided_familywise":quantile(sampled_acc,ALPHA),
      "latency_geomean_ratio_upper_one_sided_familywise":math.exp(quantile(sampled_lat,1-ALPHA))}

def gate(protocol_path,cohort_path,score_path,expected_protocol_sha,expected_cohort_sha):
    reject_if(not is_sha(expected_protocol_sha) or not is_sha(expected_cohort_sha),
              "expected prereg SHA values must be external pinned digests")
    reject_if(sha_file(protocol_path)!=expected_protocol_sha
              or sha_file(cohort_path)!=expected_cohort_sha,"frozen inputs changed")
    protocol=frozen_protocol(load_json(protocol_path))
    cohort=frozen_cohort(load_json(cohort_path),protocol)
    rows=validate_rows(protocol,cohort,load_json(score_path),
                       expected_protocol_sha,expected_cohort_sha)
    b=bootstrap(rows)
    nll=sum(x["B_minus_A_NLL"] for x in rows)/len(rows)
    acc=sum(x["B_minus_A_accuracy"] for x in rows)/len(rows)
    geom=math.exp(sum(x["log_B_over_A_latency"] for x in rows)/len(rows))
    p95=quantile([x["B_latency_ms"] for x in rows],.95)/quantile([x["A_latency_ms"] for x in rows],.95)
    teacher_floors={s:sum(x["BF16_task_correct"] for x in rows if x["dataset"]==s)/COUNT_PER_SOURCE
                    for s in SOURCES}
    controls_ok=all(teacher_floors[s]>=BF16_TASK_FLOORS[s] for s in SOURCES)
    passed=(controls_ok and b["NLL_upper_one_sided_familywise"]<=NLL_MARGIN
            and b["accuracy_lower_one_sided_familywise"]>=ACCURACY_MARGIN
            and b["latency_geomean_ratio_upper_one_sided_familywise"]<=LATENCY_RATIO_MARGIN
            and p95<=P95_LATENCY_RATIO_MARGIN)
    return {
      "schema":"beglin-hcp-p18-research-gate-v1",
      "status":"P18_RESEARCH_GATE_PASS_NOT_PRODUCTION" if passed else "P18_RESEARCH_GATE_BLOCKED",
      "research_gate_pass":passed,
      "question_cluster_units":len(rows),"strata":dict(Counter(x["dataset"] for x in rows)),
      "primary_B_minus_A_question_mean_gold_NLL_nats":nll,
      "secondary_B_minus_A_question_accuracy":acc,
      "secondary_B_over_A_paired_geometric_latency_ratio":geom,
      "latency_p95_B_over_A_ratio":p95,
      "BF16_positive_control_task_accuracy":teacher_floors,
      "BF16_positive_control_floors_pass":controls_ok,
      "bonferroni_familywise_one_sided_CI":b,
      "bootstrap_seed":SEED,"bootstrap_replicates":REPLICATES,
      "expected_protocol_sha256":expected_protocol_sha,
      "expected_cohort_sha256":expected_cohort_sha,
      "raw_logits_reverification_performed_here":False,
      "gate_assumes_separately_attested_raw_audit":True,
      "external_public_benchmark_no_training_contamination_guarantee":True,
      "automatic_live_promotion":False,"production_touched":False
    }

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--protocol",required=True)
    a.add_argument("--cohort",required=True)
    a.add_argument("--scores",required=True)
    a.add_argument("--expected-protocol-sha256",required=True)
    a.add_argument("--expected-cohort-sha256",required=True)
    a.add_argument("--output",required=True)
    x=a.parse_args()
    result=gate(x.protocol,x.cohort,x.scores,x.expected_protocol_sha256,x.expected_cohort_sha256)
    with open(x.output,"x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print(result["status"])
