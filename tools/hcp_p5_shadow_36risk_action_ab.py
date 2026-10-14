#!/usr/bin/env python3
"""P5 original-AF 112-token route risk matched to 36 native scratch qNg64 A/B pairs.

Uses SHA-sealed original P4 real full-router risk features and the SAME frozen
preregistered prompt tokenization. No 36x standalone AF baseline rerun; original
AF target logits/next tokens were not archived, so output agreement here is
ONLY between native policy A and B, never called correct/incorrect task output.

GPU C++ and C scratch-only positional cap128 + REBIND_SET required; original
persistent stage and active-route path remain byte-immutable. No live cutover.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,time,statistics
from collections import defaultdict
import numpy as np
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,stats
from hcp_p5_scratch_binary_adapter import file_sha,PINNED_OLD_BINARY_SHA,PINNED_OLD_SOURCE_SHA

SCHEMA="beglin-hcp-p5-36-real-risk-matched-native-policy-pairs-v1"
RISKSHA="c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53"
CONTEXT_MAX=128
NEW_TOKENS=4

def inspect_risk(path):
    path=pathlib.Path(path)
    if file_sha(path)!=RISKSHA:raise ValueError("frozen 3933-token true risk source SHA changed")
    d=json.loads(path.read_text())
    if (d.get("schema")!="beglin-hcp-p4-long-actual-af-risk-dataset-v1" or
        d.get("status")!="ACTUAL_DEEPSEEK_LONGISH_CONTEXT_RISK_TRACES_ONLY" or
        d.get("record_count")!=36 or d.get("token_position_count")!=3933 or
        d.get("train_val_holdout")!={"train":12,"validation":12,"holdout":12} or
        d.get("production_touched") is not False):
        raise ValueError("real 36-prompt P4 risk provenance missing")
    cases=d.get("runs") or []
    keys={x["prompt_id"]:x for x in cases}
    if len(keys)!=36 or len({x["category"] for x in cases})!=12:
        raise ValueError("duplicate or missing immutable risk group")
    return keys

def bootstrap_domain_delta(rows,field,draws=600):
    grouped=defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row["B"][field]-row["A"][field])
    groups=sorted(grouped)
    if len(groups)!=12:raise ValueError("needs 12 independent domain clusters")
    rng=np.random.default_rng(20261014);values=[]
    for _ in range(draws):
        take=rng.choice(groups,size=len(groups),replace=True)
        values.append(float(np.mean([v for domain in take for v in grouped[domain]])))
    return list(map(float,np.quantile(values,[.025,.975])))

def run(*,binary,source,index,model_root,bank,tokenizer,risk,owner,output):
    root=pathlib.Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    owner=pathlib.Path(owner).resolve(strict=True)
    binary=pathlib.Path(binary).resolve(strict=True)
    source=pathlib.Path(source).resolve(strict=True)
    if owner.parent!=root or not owner.name.startswith("hcp-p5-risk36-"):
        raise ValueError("unexpected research root")
    if not binary.is_relative_to(owner) or not source.is_relative_to(owner):
        raise ValueError("shadow research binary/source outside owned root")
    c=source.read_text()
    gpu=source.parent/"mlx_moe.cpp"
    if ("#define MOE_CBATCH_MAXPOS 128" not in c or "REBIND_SET_APPLIED" not in c
        or "#define MLA_L0_MAXPOS 128" not in gpu.read_text()):
        raise ValueError("scratch native cap and ACK source contract absent")
    risk_cases=inspect_risk(risk)
    idx=pathlib.Path(index)
    if file_sha(idx)!="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf":
        raise ValueError("wrong original DeepSeek index")
    corpus=prepare(bank,tokenizer)
    prompts=corpus["prompts"]
    if len(prompts)!=36:raise ValueError("not complete unique corpus")
    for case in prompts:
        p=risk_cases.get(case["id"])
        if not p or case["token_sha256"]!=p["token_sha256"] or case["tokens"]!=p["prompt_length"] or case["split"]!=p["split"]:
            raise ValueError("P4 true-router and current native A/B prompt lineage do not agree")
        if case["tokens"]+NEW_TOKENS>CONTEXT_MAX:
            raise ValueError("native MLX request would exceed 128")
    stage=pathlib.Path(ps.PERSISTENT_BINARY).resolve(strict=True)
    before_stage=file_sha(stage)
    if before_stage!=PINNED_OLD_BINARY_SHA or ps.EXPECTED_PERSISTENT_SOURCE_SHA!=PINNED_OLD_SOURCE_SHA:
        raise ValueError("historical native stage identity changed")
    active=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
    before_live=file_sha(active) if active.is_file() else None
    ps.PERSISTENT_BINARY=binary
    ps.EXPECTED_PERSISTENT_BINARY_SHA=file_sha(binary)
    ps.REQUEST_TIMEOUT_SECONDS=max(190,int(ps.REQUEST_TIMEOUT_SECONDS))
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=owner/"native_worker")
    rows=[];t0=time.monotonic()
    try:
        h=worker.start()
        if not h.get("alive") or h["runtime_policy_hash"]!=pc.policy_hash(A):
            raise ValueError("true native scratch startup policy mismatch")
        pid=int(h["pid"])
        for n,case in enumerate(prompts):
            prompt=case["token_ids"];by={}
            order=("A","B") if n%2==0 else ("B","A")
            for label in order:
                policy=A if label=="A" else B
                actual=worker.submit_with_precision_policy(
                    [(prompt,NEW_TOKENS)],target_policy=policy,
                    admission_id=f"hcp5-risk36-{n:03d}-{label}")
                if actual.get("finite_logits") is not True or int(actual.get("inference_passes",1))!=1:
                    raise ValueError("invalid full-model qNg64 output")
                tokens=actual.get("responses")
                if not isinstance(tokens,list) or len(tokens)!=1 or not tokens[0]:
                    raise ValueError("missing original native generated sequence")
                accepted=worker.health()
                if not accepted.get("alive") or int(accepted["pid"])!=pid or accepted["runtime_policy_hash"]!=pc.policy_hash(policy):
                    raise ValueError("runtime PID or ACK policy hash disagrees with scheduler")
                epoch=actual.get("precision_epoch") or {}
                if epoch.get("target_policy_hash")!=pc.policy_hash(policy):
                    raise ValueError("inconsistent native scheduling policy")
                costs=epoch.get("transition_cost") or {}
                by[label]={"tokens":list(map(int,tokens[0])),
                    "engine_ms":float(actual["engine_wall_ms"]),
                    "roundtrip_ms":float(actual["roundtrip_ms"]),
                    "transition_ms":float(costs.get("transition_wall_ms",0)),
                    "rss_bytes":int(accepted["rss_bytes"])}
                if not(math.isfinite(by[label]["engine_ms"]) and by[label]["engine_ms"]>0
                    and math.isfinite(by[label]["roundtrip_ms"]) and by[label]["roundtrip_ms"]>0
                    and by[label]["rss_bytes"]>0):
                    raise ValueError("invalid actual native precision cost")
            risk_row=risk_cases[case["id"]]
            rows.append({"id":case["id"],"split":case["split"],"domain":case["domain"],
                "position_count":case["tokens"],
                "risk_min_relative_gap":float(risk_row["real_min_rank6_rank7_gap_min"]),
                "risk_p50_relative_gap":float(risk_row["real_min_rank6_rank7_gap_p50"]),
                "first_generated_token_A_B_equal":by["A"]["tokens"][0]==by["B"]["tokens"][0],
                "all_four_A_B_generated_tokens_equal":by["A"]["tokens"]==by["B"]["tokens"],
                "policy_order":list(order),
                "A":{k:v for k,v in by["A"].items() if k!="tokens"},
                "B":{k:v for k,v in by["B"].items() if k!="tokens"}})
            if (n+1)%6==0:
                print("P5_36_ORIGINAL_RISK_NATIVE_ACTION_PROGRESS",json.dumps({
                    "complete_pairs":len(rows),"A_B_first_token_disagreement":
                         sum(not x["first_generated_token_A_B_equal"] for x in rows),
                    "elapsed_s":round(time.monotonic()-t0,2)},sort_keys=True),flush=True)
    finally:
        worker.stop(force=True)
    if file_sha(stage)!=before_stage or (file_sha(active) if active.is_file() else None)!=before_live:
        raise ValueError("production stage or active route mutated")
    if len(rows)!=36:raise ValueError("incomplete original 36 P4 model input comparison")
    result={"schema":SCHEMA,"status":"REAL_36_CONTEXT_NATIVE_QNG64_POLICY_A_B_SHADOW_ONLY",
        "source_original_P4_risk_sha256":RISKSHA,
        "source_model_index_sha256":"d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf",
        "original_stage_unchanged_sha256":before_stage,
        "scratch_binary_sha256":file_sha(binary),
        "original_input_count":len(rows),
        "original_input_positions":sum(p["position_count"] for p in rows),
        "prompt_splits":{"train":12,"validation":12,"holdout":12},
        "true_router_risk_matched_to_exact_native_input_sha":True,
        "per_policy":{
            label:{"policy_hash":pc.policy_hash(A if label=="A" else B),
                "engine_ms":stats([r[label]["engine_ms"] for r in rows]),
                "roundtrip_ms":stats([r[label]["roundtrip_ms"] for r in rows]),
                "transition_ms":stats([r[label]["transition_ms"] for r in rows]),
                "worker_rss_bytes":stats([r[label]["rss_bytes"] for r in rows])}
            for label in ("A","B")},
        "A_B_first_token_exact_agreement_rate":sum(r["first_generated_token_A_B_equal"] for r in rows)/36,
        "A_B_all_4_token_sequence_exact_agreement_rate":sum(r["all_four_A_B_generated_tokens_equal"] for r in rows)/36,
        "mean_B_minus_A_engine_ms":float(np.mean([r["B"]["engine_ms"]-r["A"]["engine_ms"] for r in rows])),
        "domain_block_bootstrap_95ci_B_minus_A_engine_ms":bootstrap_domain_delta(rows,"engine_ms"),
        "not_measured":"native BF16 oracle, human task ground truth, correctness or full production serving",
        "actual_AF_last_token_reference_not_saved_by_P4":True,
        "live_route_and_old_binary_hash_unchanged":True,
        "native_policy_change_production_allowed":False,
        "production_touched":False,"automatic_live_promotion":False,
        "records":rows}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("P5_FULL36_NATIVE_AB_WITH_TRUE_ROUTER_RISK",json.dumps({
        "pairs":36,"positions":result["original_input_positions"],
        "agreement_first":result["A_B_first_token_exact_agreement_rate"],
        "agreement_4tokens":result["A_B_all_4_token_sequence_exact_agreement_rate"],
        "engine_ms":result["per_policy"],
        "B_minus_A":result["mean_B_minus_A_engine_ms"],
        "domain_bootstrap_95ci":result["domain_block_bootstrap_95ci_B_minus_A_engine_ms"],
        "production_touched":False},sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("binary","source","index","model-root","bank","tokenizer","risk","owner","output"):
        p.add_argument("--"+name,required=True,type=pathlib.Path)
    a=p.parse_args()
    run(binary=a.binary,source=a.source,index=a.index,model_root=a.model_root,
        bank=a.bank,tokenizer=a.tokenizer,risk=a.risk,owner=a.owner,output=a.output)
