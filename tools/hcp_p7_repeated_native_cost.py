#!/usr/bin/env python3
"""P7 research-only repeated native A/B with true REBIND_SET and end-to-end cost.

Reuses the same immutable P5 source/checkpoint and *one fresh disposable* GPU
worker. Balanced ABBA/BAAB crossover is repeated across temporal blocks;
all transition and response costs enter the reported measurements.
No production release/policy or active route is changed. Not a quality gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import time
import random

import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A, B, sha, stats

SCHEMA = "beglin-hcp-p7-repeated-native-ab-transition-inclusive-v1"
INDEX_SHA = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
FEATURE_SHA = "c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53"
C128_SHA = "bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
MLX128_SHA = "daded3c0e4bf7cc2a86e0fab8c1f73322b5b0ffea0002d9b2d38857d2e9591ae"
RSS_LIMIT = 14 * 1024 ** 3
SEED = 20261014

def order_for(block:int,domain:int)->tuple[str,...]:
    if block < 0 or not(0 <= domain < 12):
        raise ValueError("invalid preregistered crossover position")
    return ("A","B","B","A") if (block+domain)%2==0 else ("B","A","A","B")

def paired_block_ci(rows:list[dict],field="roundtrip_ms",iterations=10000)->dict:
    if not rows or iterations < 1000:
        raise ValueError("missing complete repeated blocks")
    byblock:dict[int,list[float]]={}
    for row in rows:
        a=row["A"][field];b=row["B"][field]
        if len(a)!=2 or len(b)!=2 or any(not math.isfinite(v) or v<=0 for v in a+b):
            raise ValueError("incomplete or invalid temporal block timing")
        byblock.setdefault(int(row["block"]),[]).append(
            (sum(b)/2)-(sum(a)/2))
    if len(byblock)<2 or len({len(vals) for vals in byblock.values()})!=1:
        raise ValueError("block count/coverage mismatch")
    keys=sorted(byblock);block_means=[statistics.mean(byblock[k]) for k in keys]
    rng=random.Random(SEED)
    bootstrap=[]
    for _ in range(iterations):
        selected=[block_means[rng.randrange(len(block_means))] for __ in keys]
        bootstrap.append(statistics.mean(selected))
    bootstrap.sort()
    lower=bootstrap[int(iterations*.025)]
    upper=bootstrap[min(iterations-1,int(iterations*.975))]
    return {"paired_B_minus_A_mean_ms":statistics.mean(block_means),
            "block_bootstrap_ci95_ms":[lower,upper],
            "seed":SEED,"bootstrap_draws":iterations,
            "temporal_blocks":len(keys),"pairs_per_block":len(byblock[keys[0]])}

def p50(vals):
    return float(statistics.median(vals))

def validate(root,binary,source,index,feature):
    root=pathlib.Path(root).resolve(strict=True)
    runner=pathlib.Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    if root!=runner/("hcp-p7-repeated-"+os.environ["GITHUB_RUN_ID"]):
        raise ValueError("P7 root must be this run's unique scratch")
    binary=pathlib.Path(binary).resolve(strict=True)
    source=pathlib.Path(source).resolve(strict=True)
    gpu=root/"src/mlx_moe.cpp"
    if (binary!=root/"build/qwen_infer_gpu" or source!=root/"src/qwen_infer.c"
        or not binary.is_file() or binary.is_symlink()
        or source.is_symlink() or not gpu.is_file() or gpu.is_symlink()):
        raise ValueError("P7 scratch source/binary escaped owned root")
    if sha(source)!=C128_SHA or sha(gpu)!=MLX128_SHA:
        raise ValueError("P7 uninstrumented P5 C and MLX KV128 source changed")
    c=source.read_text()
    if ("REBIND_SET_APPLIED" not in c or "#define MOE_CBATCH_MAXPOS 128" not in c or
        "moe_cb4b_admit_guard(" not in c or "BEGLIN_HCP_P6_LOGITS_OUTPUT" in c):
        raise ValueError("P7 cannot run modified inference mathematics/logit capture")
    if sha(index)!=INDEX_SHA or sha(feature)!=FEATURE_SHA:
        raise ValueError("checkpoint or frozen 36-prompt AF features not pinned")
    stage=pathlib.Path(ps.PERSISTENT_BINARY)
    original=ps.EXPECTED_PERSISTENT_BINARY_SHA
    if not stage.is_file() or sha(stage)!=original:
        raise ValueError("original stage binary changed before P7")
    return {"root":root,"binary_sha":sha(binary),
            "stage":stage,"stage_sha":original}

def run(*,root,binary,source,bank,tokenizer,index,feature,output,count,blocks):
    if (count,blocks) not in ((2,2),(12,4)):
        raise ValueError("P7 predeclared 2x2 smoke or 12x4 measured pilot only")
    identity=validate(root,binary,source,index,feature)
    root=identity["root"]
    frozen=json.loads(pathlib.Path(feature).read_text())
    if frozen.get("record_count")!=36 or frozen.get("schema")!="beglin-hcp-p4-long-actual-af-risk-dataset-v1":
        raise ValueError("wrong original AF/P4 input lineage")
    expected={r["prompt_id"]:r for r in frozen["runs"]}
    corpus=prepare(bank,tokenizer)
    selected=[corpus["prompts"][3*i+i%3] for i in range(12)][:count]
    if len({x["domain"] for x in selected})!=count:
        raise ValueError("not unique precommitted domains")
    for item in selected:
        ref=expected[item["id"]]
        if (ref["token_sha256"]!=item["token_sha256"] or
            ref["prompt_length"]!=item["tokens"] or not 104<=item["tokens"]<=112 or
            item["tokens"]+4>=128):
            raise ValueError("P7 prompt not the same verified P4/P5 tokenized input")

    live=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
    before=sha(live) if live.is_file() else None
    old_bin=ps.PERSISTENT_BINARY
    old_digest=ps.EXPECTED_PERSISTENT_BINARY_SHA
    ps.PERSISTENT_BINARY=pathlib.Path(binary).resolve(strict=True)
    ps.EXPECTED_PERSISTENT_BINARY_SHA=identity["binary_sha"]
    ps.REQUEST_TIMEOUT_SECONDS=max(int(ps.REQUEST_TIMEOUT_SECONDS),180)
    worker_root=root/("p7_smoke_worker" if count==2 else "p7_measure_worker")
    if worker_root.exists():
        raise ValueError("P7 worker scratch already exists")
    records=[];started=time.monotonic()
    try:
        worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=worker_root)
        old_env=worker._env
        def one_slot_env():
            e=old_env()
            e["QWEN_MOE_CB_SLOTS"]="1"
            e["QWEN_MOE_CB_REQS"]="1"
            return e
        worker._env=one_slot_env
        try:
            initial=worker.start()
            if (not initial.get("alive") or initial.get("runtime_policy_hash")!=pc.policy_hash(A)
                or int(initial["rss_bytes"])>RSS_LIMIT):
                raise ValueError("P7 scratch worker startup PID/policy/RSS mismatch")
            pid=int(initial["pid"])
            for block in range(blocks):
                for i,entry in enumerate(selected):
                    order=order_for(block,i)
                    outcomes={"A":[],"B":[]}
                    for j,kind in enumerate(order):
                        pol=A if kind=="A" else B
                        res=worker.submit_with_precision_policy(
                            [(entry["token_ids"],4)],target_policy=pol,
                            admission_id=f"p7-b{block:02d}-p{i:02d}-{j}-{kind}")
                        if (res.get("finite_logits") is not True or
                            res.get("inference_passes",1)!=1 or
                            not isinstance(res.get("responses"),list) or len(res["responses"])!=1 or
                            not res["responses"][0] or len(res["responses"][0])>4):
                            raise ValueError("P7 incomplete or nonfinite model request")
                        health=worker.health()
                        if (not health.get("alive") or int(health["pid"])!=pid or
                            int(health["rss_bytes"])>RSS_LIMIT or
                            health["runtime_policy_hash"]!=pc.policy_hash(pol)):
                            raise ValueError("P7 epoch policy hash/PID/RSS mismatch")
                        epoch=res.get("precision_epoch") or {}
                        tc=epoch.get("transition_cost") or {}
                        moved=bool(epoch.get("transitioned"))
                        if moved and (len(epoch.get("changed_targets",[]))!=2 or
                            worker.ack.get("txn_id")!=epoch.get("txn_id") or
                            worker.ack.get("status")!="REBIND_SET_APPLIED" or
                            int(epoch["after_epoch"])!=int(epoch["before_epoch"])+1):
                            raise ValueError("P7 native 2-target REBIND_SET txn ACK failed")
                        engine=float(res["engine_wall_ms"])
                        roundtrip=float(res["roundtrip_ms"])
                        transition=float(tc.get("transition_wall_ms",0))
                        if not all(math.isfinite(x) for x in (engine,roundtrip,transition)) or min(engine,roundtrip)<=0 or transition<0:
                            raise ValueError("invalid native P7 timings")
                        outcomes[kind].append({
                            "engine":engine,"roundtrip":roundtrip,"transition":transition,
                            "moved":moved,"generated_count":len(res["responses"][0]),
                            "rss":int(health["rss_bytes"]),
                            "seq":tuple(map(int,res["responses"][0]))})
                    a,b=outcomes["A"],outcomes["B"]
                    records.append({
                        "block":block,"domain":entry["domain"],"split":entry["split"],
                        "prompt_id":entry["id"],"input_tokens":entry["tokens"],
                        "exact_P4_prompt_token_sha_matched":True,
                        "order":list(order),
                        "A":{"engine_ms":[v["engine"] for v in a],
                             "roundtrip_ms":[v["roundtrip"] for v in a],
                             "transition_ms":[v["transition"] for v in a],
                             "transitioned":[v["moved"] for v in a],
                             "generated_count":[v["generated_count"] for v in a]},
                        "B":{"engine_ms":[v["engine"] for v in b],
                             "roundtrip_ms":[v["roundtrip"] for v in b],
                             "transition_ms":[v["transition"] for v in b],
                             "transitioned":[v["moved"] for v in b],
                             "generated_count":[v["generated_count"] for v in b]},
                        "same_policy_repeat_seq_A":a[0]["seq"]==a[1]["seq"],
                        "same_policy_repeat_seq_B":b[0]["seq"]==b[1]["seq"],
                        "A_B_first_token_agree":a[0]["seq"][0]==b[0]["seq"][0],
                        "A_B_generated_length_agree":len(a[0]["seq"])==len(b[0]["seq"])
                    })
                print("P7_REPEATED_TEMPORAL_BLOCK",json.dumps({
                    "block":block,"complete_pairs":len(records),"elapsed_s":round(time.monotonic()-started,2),
                    "all_native_ACK":True,"live_route_touched":False},sort_keys=True),flush=True)
        finally:
            worker.stop(force=True)
    finally:
        ps.PERSISTENT_BINARY=old_bin
        ps.EXPECTED_PERSISTENT_BINARY_SHA=old_digest

    if (sha(identity["stage"])!=identity["stage_sha"]
        or (sha(live) if live.is_file() else None)!=before):
        raise ValueError("P7 original stage or live precision route mutated")
    if len(records)!=count*blocks:
        raise ValueError("P7 incomplete block set")

    summaries={}
    for kind in ("A","B"):
        values=lambda field:[x for row in records for x in row[kind][field]]
        summaries[kind]={
            "engine_ms":stats(values("engine_ms")),
            "roundtrip_ms":stats(values("roundtrip_ms")),
            "transition_ms":stats(values("transition_ms")),
            "epoch_transitions":sum(values("transitioned")),
            "generations":sum(values("generated_count"))}
    paired_roundtrip=paired_block_ci(records,"roundtrip_ms")
    paired_engine=paired_block_ci(records,"engine_ms")
    quality_guard=all(r["same_policy_repeat_seq_A"] and r["same_policy_repeat_seq_B"] for r in records)
    lengths_guard=all(r["A_B_generated_length_agree"] for r in records)
    report={
        "schema":SCHEMA,"status":"ACTUAL_ISOLATED_REPEATED_112TOK_NATIVE_COST_RESEARCH_ONLY",
        "prompt_count_per_block":count,"temporal_blocks":blocks,"matched_pairs":len(records),
        "total_native_admissions":4*len(records),"output_budget":4,
        "real_26_layer_gpu_inference":True,"single_GPU_request_slot":True,
        "source_P4_SHA256":FEATURE_SHA,"source_P5_C128_SHA256":C128_SHA,
        "source_P5_MLX128_SHA256":MLX128_SHA,"scratch_binary_sha256":identity["binary_sha"],
        "original_stage_sha256":identity["stage_sha"],"checkpoint_index_sha256":INDEX_SHA,
        "per_policy":summaries,
        "paired_roundtrip_including_transitions":paired_roundtrip,
        "paired_engine_including_transitions":paired_engine,
        "same_policy_repeated_sequences_all_equal":quality_guard,
        "A_B_same_generation_length_all":lengths_guard,
        "A_B_first_token_agreed_count":sum(row["A_B_first_token_agree"] for row in records),
        "raw_tokens_or_logits_uploaded":False,
        "no_full_context_quality_claim":True,"speed_improvement_proven":False,
        "preregistered_full_twenty_blocks_completed":False,
        "current_study_pilot_not_promotion_grade":True,
        "native_REBIND_SET_ACK_not_mocked":True,
        "old_stage_unchanged":True,"active_route_unchanged":True,
        "production_touched":False,"automatic_live_promotion":False,
        "records":records}
    out=pathlib.Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("P7_REPEATED_NATIVE_AB_RESULT",json.dumps({
        "pairs":len(records),"blocks":blocks,"A_roundtrip_p50":summaries["A"]["roundtrip_ms"]["p50"],
        "B_roundtrip_p50":summaries["B"]["roundtrip_ms"]["p50"],
        "B_minus_A_roundtrip":paired_roundtrip,
        "native_ack":True,"speedup_proven":False},sort_keys=True),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("root","binary","source","bank","tokenizer","index","feature","output"):
        p.add_argument("--"+k,required=True,type=pathlib.Path)
    p.add_argument("--count",type=int,default=2)
    p.add_argument("--blocks",type=int,default=2)
    args=p.parse_args()
    run(root=args.root,binary=args.binary,source=args.source,
        bank=args.bank,tokenizer=args.tokenizer,index=args.index,
        feature=args.feature,output=args.output,count=args.count,blocks=args.blocks)
