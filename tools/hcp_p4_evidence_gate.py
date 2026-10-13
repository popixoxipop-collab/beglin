#!/usr/bin/env python3
"""Fail-closed retrospective audit of real 112-token risk + exact native A/B costs.

Never permits an LLM policy cutover; prompt agreement vs original AF is NOT
a supervised quality score. The observed dataset's future labels are protected
by prompt-level split, and all raw inputs/logits stay off GitHub.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,statistics
from collections import defaultdict
import numpy as np
import precision_context as pc

SCHEMA="beglin-hcp-p4-real-context-policy-costquality-audit-v1"
EXPECTED_SPLITS={"train":12,"validation":12,"holdout":12}
A=pc.normalize_policy([{"role":"shared_up_proj","layer":3,"n":6},{"role":"shared_down_proj","layer":26,"n":5}])
B=pc.normalize_policy([{"role":"shared_up_proj","layer":3,"n":5},{"role":"shared_down_proj","layer":26,"n":6}])
FULL_FEATURE="beglin-hcp-p4-long-actual-af-risk-dataset-v1"
FULL_AB="beglin-hcp-p4-matched-native-qng64-ab-v1"

def check_feature(body):
    if body.get("schema")!=FULL_FEATURE or body.get("status")!="ACTUAL_DEEPSEEK_LONGISH_CONTEXT_RISK_TRACES_ONLY":
        raise ValueError("wrong true DeepSeek feature lineage")
    if body.get("production_touched") is not False or body.get("raw_prompt_token_ids_exported") is not False or body.get("full_logits_exported") is not False:
        raise ValueError("original raw export/serving restriction violated")
    if body.get("train_val_holdout")!=EXPECTED_SPLITS or body.get("record_count")!=36:
        raise ValueError("insufficient disjoint prompt contexts")
    records=body.get("runs");sequences=body.get("sequences")
    if not isinstance(records,list) or not isinstance(sequences,list) or len(records)!=36 or len(sequences)!=36:
        raise ValueError("expected exactly 36 original and derived real traces")
    by_id={r["prompt_id"]:r for r in records}
    by_seq={r["prompt_id"]:r for r in sequences}
    if len(by_id)!=36 or set(by_id)!=set(by_seq):
        raise ValueError("duplicate/missing real prompts")
    groups=set();total=0;label_counts={k:0 for k in ("4","16","32")}
    for key,rec in by_id.items():
        seq=by_seq[key]
        length=int(rec["prompt_length"])
        if not(104<=length<=112 and seq["split"]==rec["split"] and seq["domain"]==rec["category"]):
            raise ValueError("context length or grouping changed")
        if not(len(seq["rows"])==length-1 and rec["rank7_observation_count"]==length*26):
            raise ValueError("incomplete 26-layer full model router observations")
        if rec["complete_future_32_labels"]!=length-32:
            raise ValueError("32-step prospective label coverage mismatch")
        groups.add(rec["category"]);total+=length
        for i,row in enumerate(seq["rows"]):
            if int(row.get("position",-1))!=i:
                raise ValueError("future target temporal index invalid")
            for name in ("fast","slow4","slow16"):
                values=np.asarray(row.get(name),dtype=float)
                if values.shape!=(8,) or not np.isfinite(values).all():
                    raise ValueError("invalid causal scalar representation")
            if "token_ids" in row or "full_logits" in row or "h2_vector" in row:
                raise ValueError("raw private information in derived corpus")
            target=row.get("true_future_gap_min")
            if not isinstance(target,dict) or set(target)!={"4","16","32"}:
                raise ValueError("future horizons missing")
            for k in label_counts:
                gap=target[k]
                h=int(k)
                if (i+h < length) != (gap is not None):
                    raise ValueError("leaking incomplete horizon target")
                if gap is not None:
                    if not math.isfinite(float(gap)) or not 0<=float(gap)<=1:
                        raise ValueError("future margin outside real-router bounds")
                    label_counts[k]+=1
    if len(groups)!=12 or body.get("token_position_count")!=total:
        raise ValueError("incorrect dataset diversity or token positions")
    return by_id,by_seq,{"domain_count":len(groups),
        "complete_input_positions":total,"future_target_count":label_counts}

def audit(features,policies,*,bootstrap=1000):
    by_id,by_seq,stats=check_feature(features)
    if policies.get("schema")!=FULL_AB or policies.get("status")!="ACTUAL_MATCHED_NATIVE_POLICY_A_B_AGREEMENT_COST_ONLY":
        raise ValueError("missing real native A/B evidence")
    if policies.get("sample_pairs")!=36 or policies.get("observed_policy_A")!=A or policies.get("observed_policy_B")!=B:
        raise ValueError("exact native qNg64 policies were not both measured")
    if policies.get("native_accuracy_vs_human_reference_measured") is not False or policies.get("production_touched") is not False or policies.get("active_route_sha_before_after_equal") is not True:
        raise ValueError("false task accuracy or live serving mutation")
    seen=set();out=[];by_domain=defaultdict(list)
    for row in policies.get("records",[]):
        key=row.get("prompt_id")
        if key not in by_id or key in seen:raise ValueError("duplicate or unknown policy context")
        seen.add(key);f=by_id[key]
        if row.get("split")!=f["split"] or row.get("domain")!=f["category"] or row.get("prompt_length")!=f["prompt_length"]:
            raise ValueError("paired model input did not match frozen corpus")
        if row.get("matched_policy_order") not in (["A","B"],["B","A"]):
            raise ValueError("unbalanced native policy order")
        if any(type(row.get(k)) is not bool for k in ("a_af_reference_next_token_agreement",
              "b_af_reference_next_token_agreement","a_b_next_token_agreement",
              "a_b_generated_sequence_exact_match")):
            raise ValueError("missing real paired model outputs")
        a=row["policy_A"];b=row["policy_B"]
        for kind,r in (("A",a),("B",b)):
            for metric in ("engine_wall_ms","roundtrip_ms","transition_wall_ms"):
                val=float(r[metric])
                if not(math.isfinite(val) and val>=0):
                    raise ValueError("nonfinite native time cost")
            if float(r["engine_wall_ms"])<=0 or float(r["roundtrip_ms"])<=0 or int(r["rss_bytes"])<=0:
                raise ValueError("invalid actual native worker duration/memory")
        clean={"id":key,"domain":f["category"],"split":f["split"],
            "policyB_minus_A_engine_ms":float(b["engine_wall_ms"])-float(a["engine_wall_ms"]),
            "policyB_minus_A_roundtrip_ms":float(b["roundtrip_ms"])-float(a["roundtrip_ms"]),
            "model_next_agreement_ab":row["a_b_next_token_agreement"],
            "original_af_next_agreement_A":row["a_af_reference_next_token_agreement"],
            "original_af_next_agreement_B":row["b_af_reference_next_token_agreement"],
            "min_router_relative_gap":float(f["real_min_rank6_rank7_gap_min"])}
        out.append(clean)
        by_domain[f["category"]].append(clean)
    if len(out)!=36 or len(by_domain)!=12:
        raise ValueError("missing full real native paired coverage")
    means={k:statistics.mean(x[k] for x in out) for k in ("policyB_minus_A_engine_ms","policyB_minus_A_roundtrip_ms")}
    domain_keys=sorted(by_domain);rng=np.random.default_rng(20261014)
    ci={}
    for measure in means:
        draws=[]
        for _ in range(bootstrap):
            selected=rng.choice(domain_keys,len(domain_keys),replace=True)
            group=[row for cat in selected for row in by_domain[cat]]
            draws.append(float(np.mean([r[measure] for r in group])))
        ci[measure]=list(map(float,np.quantile(draws,[.025,.975])))
    agree={
        "policy_A_vs_AF_reference_first_token":float(np.mean([r["original_af_next_agreement_A"] for r in out])),
        "policy_B_vs_AF_reference_first_token":float(np.mean([r["original_af_next_agreement_B"] for r in out])),
        "policy_A_vs_B_first_token":float(np.mean([r["model_next_agreement_ab"] for r in out])),
        "policy_A_vs_B_entire_generated_sequence":float(policies["a_b_generated_sequence_exact_agreement_rate"])}
    result={"schema":SCHEMA,"status":"COLLECTED_REAL_NATIVE_POLICY_AGREEMENT_COST_RESEARCH_ONLY",
        "source":"36_REAL_104_112_TOKEN_26LAYER_ROUTER_TRACES_AND_NATIVE_GPU_PRECISION_AB",
        "model":"DeepSeek-V2-Lite",
        "prompts":36,"prompt_splits":EXPECTED_SPLITS,
        "actual_routed_layer_count":26,**stats,
        "native_quality_observable":"agreement with original quantized-AF argmax and A/B generated tokens",
        "not_measured":"BF16 ground truth, task success, native logits KL, licensed operating safety or net H-JEPA gains",
        "policy_agreement":agree,
        "paired_policyB_minus_A_mean_ms":means,
        "paired_12_domain_bootstrap_95ci_delta_ms":ci,
        "per_policy_actual_engine_ms":policies["per_policy"],
        "routing_risk_and_precision_error_causal_association_established":False,
        "native_quality_certification":False,
        "native_policy_change_allowed":False,
        "production_touched":False,"automatic_live_promotion":False}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--risk",required=True,type=pathlib.Path)
    p.add_argument("--native",required=True,type=pathlib.Path)
    p.add_argument("--output",required=True,type=pathlib.Path)
    a=p.parse_args()
    risk=a.risk.read_bytes();native=a.native.read_bytes()
    result=audit(json.loads(risk),json.loads(native))
    result["source_risk_sha256"]=hashlib.sha256(risk).hexdigest()
    result["source_native_ab_sha256"]=hashlib.sha256(native).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("P4_REAL_LONG_COST_QUALITY_SAFE_AUDIT",json.dumps({k:v for k,v in result.items()
           if k in ("status","prompts","complete_input_positions","future_target_count",
                    "policy_agreement","paired_policyB_minus_A_mean_ms",
                    "paired_12_domain_bootstrap_95ci_delta_ms",
                    "native_policy_change_allowed")},sort_keys=True))
if __name__=="__main__":main()
