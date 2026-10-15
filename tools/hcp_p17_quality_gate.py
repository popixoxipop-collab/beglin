#!/usr/bin/env python3
"""P17 frozen paired-question noninferiority gate (research-only).

Consumes independently verified genuine-source per-question score records.
Never mistakes 31 correlated target tokens for 31 independent questions, and
never declares production quality based on 12 internally authored items.
"""
from __future__ import annotations
import hashlib,json,math,pathlib
import numpy as np

FROZEN_SHA="a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521"
SEED=20261015
N_BOOT=10000
MARGIN_NLL=0.05
MARGIN_PREFIX=0.05

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def longest_common_prefix_score(generated,gold):
    if not gold or len(generated)>8 or any(not isinstance(z,int) or not 0<=z<102400 for z in [*generated,*gold]):
        raise ValueError("P17 exact generated/gold token prefix invalid")
    k=0
    for g,t in zip(generated,gold):
        if g!=t:break
        k+=1
    return k/len(gold)
def one_sided_bounds(difference,seed=SEED,reps=N_BOOT):
    d=np.asarray(difference,dtype=np.float64)
    if d.shape!=(12,) or not np.isfinite(d).all() or reps!=10000 or seed!=SEED:
        raise ValueError("P17 bootstrap requires 12 independent question units, preregistered seed/reps")
    rng=np.random.default_rng(seed)
    b=rng.integers(0,len(d),size=(reps,len(d)))
    means=d[b].mean(axis=1)
    return {"mean":float(d.mean()),"lower_one_sided_95":float(np.quantile(means,.05)),
            "upper_one_sided_95":float(np.quantile(means,.95)),
            "seed":seed,"replicates":reps,"cluster_unit":"question_not_token"}

def gate(frozen,data):
    frozen=pathlib.Path(frozen)
    if not frozen.is_file() or frozen.is_symlink() or sha(frozen)!=FROZEN_SHA:
        raise ValueError("P17 source original gold token preregistration SHA mismatch")
    source=json.loads(frozen.read_text())
    if source.get("question_count")!=12 or source.get("gold_token_prefix_count")!=31:
        raise ValueError("P17 wrong preregistered cohort")
    if data.get("schema")!="beglin-hcp-p17-independent-real-teacher-native-question-scores-v1":
        raise ValueError("P17 input scores must be independently cross-checked with real BF16 and A/B arrays")
    if (data.get("frozen_token_sha256")!=FROZEN_SHA or
        data.get("verified_actual_BF16_gold_logit_positions")!=31 or
        data.get("verified_actual_native_gold_logit_positions")!=62 or
        data.get("verified_native_real_greedy_generations")!=24 or
        data.get("independent_actual_logits_source_audit_passed") is not True or
        data.get("production_touched") is not False):
        raise ValueError("P17 real sources not complete and independently validated")
    records=data.get("records")
    if not isinstance(records,list) or len(records)!=12 or len({r.get("item_id") for r in records})!=12:
        raise ValueError("P17 requires all twelve distinct question records")
    index={r["item_id"]:r for r in records}
    per=[]
    for item in source["items"]:
        id_=item["id"]
        if id_ not in index:raise ValueError("P17 missing original frozen question "+id_)
        r=index[id_]
        gold=item["gold_answer_ids"]
        n=len(gold)
        if (r.get("gold_answer_sha256")!=item["gold_answer_sha256"] or
            r.get("gold_token_count")!=n or
            r.get("domain")!=item["domain"]):
            raise ValueError("P17 source answer/token/domain mismatch "+id_)
        mode={}
        for policy in ("A","B"):
            cell=r.get(policy)
            if not isinstance(cell,dict):raise ValueError("P17 absent native real policy "+policy)
            vals=np.asarray(cell.get("gold_token_nll_nats",[]),dtype=np.float64)
            if vals.shape!=(n,) or not np.isfinite(vals).all() or np.any(vals<0) or np.any(vals>=150):
                raise ValueError("P17 missing/invalid true gold-token losses "+id_+policy)
            generated=cell.get("native_greedy_token_ids")
            if not isinstance(generated,list) or not 1<=len(generated)<=8:
                raise ValueError("P17 real native bounded greedy output missing "+id_+policy)
            prefix=longest_common_prefix_score(generated,gold)
            mode[policy]={"gold_nll_mean":float(vals.mean()),"gold_nll_sum":float(vals.sum()),
                          "strict_gold_prefix_accuracy":prefix,"gold_exact_prefix_all_correct":prefix==1,
                          "real_evidence_SHA256":cell.get("real_source_logits_manifest_sha256")}
            if not isinstance(mode[policy]["real_evidence_SHA256"],str) or len(mode[policy]["real_evidence_SHA256"])!=64:
                raise ValueError("P17 native unverified evidence manifest "+id_+policy)
        t=np.asarray(r.get("original_BF16_teacher_gold_nll_nats",[]),dtype=np.float64)
        if t.shape!=(n,) or not np.isfinite(t).all() or np.any(t<0):
            raise ValueError("P17 original teacher loss unavailable or incorrect length")
        per.append({"item_id":id_,"domain":item["domain"],"gold_count":n,
            "original_teacher_mean_gold_NLL_nats":float(t.mean()),
            "A":mode["A"],"B":mode["B"],
            "B_minus_A_mean_gold_NLL":mode["B"]["gold_nll_mean"]-mode["A"]["gold_nll_mean"],
            "B_minus_A_gold_prefix_accuracy":mode["B"]["strict_gold_prefix_accuracy"]-mode["A"]["strict_gold_prefix_accuracy"]})
    if sum(x["gold_count"] for x in per)!=31:raise ValueError("P17 one-shot gold token count drift")
    nll=one_sided_bounds([r["B_minus_A_mean_gold_NLL"] for r in per])
    prefix=one_sided_bounds([r["B_minus_A_gold_prefix_accuracy"] for r in per])
    accuracy=nll["upper_one_sided_95"]<=MARGIN_NLL and prefix["lower_one_sided_95"]>=-MARGIN_PREFIX
    verdict="P17_INTERNAL_RESEARCH_GATE_PASS_NOT_PRODUCTION" if accuracy else "P17_INTERNAL_RESEARCH_GATE_BLOCKED"
    return {"schema":"beglin-hcp-p17-question-cluster-paired-noninferiority-gate-v1",
            "status":verdict,"research_internal_gate_pass":bool(accuracy),
            "question_units":12,"actual_gold_token_positions":31,
            "native_AB_token_logit_positions":62,"native_greedy_calls":24,
            "incumbent":"A","challenger":"B","same_model_original_BF16_teacher":True,
            "B_minus_A_nll_question_bootstrap":nll,
            "B_minus_A_prefix_accuracy_question_bootstrap":prefix,
            "noninferiority_nll_margin_nats_per_token":MARGIN_NLL,
            "noninferiority_prefix_accuracy_margin":MARGIN_PREFIX,
            "question_level_records":per,
            "not_external_independent_benchmark":True,
            "not_independent_of_original_pretraining":True,
            "actual_end_to_end_task_accuracy_certified":False,
            "automatic_live_promotion":False,
            "production_touched":False}
if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser();a.add_argument("--frozen",required=True);a.add_argument("--verified-score-records",required=True);a.add_argument("--output",required=True)
    x=a.parse_args();d=gate(x.frozen,json.loads(pathlib.Path(x.verified_score_records).read_text()))
    with pathlib.Path(x.output).open("x") as f:json.dump(d,f,indent=2,sort_keys=True);f.write("\n")
    print("P17_RESEARCH_ONLY_QUALITY_GATE",json.dumps({k:v for k,v in d.items() if k in ("status","research_internal_gate_pass","B_minus_A_nll_question_bootstrap","B_minus_A_prefix_accuracy_question_bootstrap")},sort_keys=True))
