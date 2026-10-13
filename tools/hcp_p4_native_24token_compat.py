#!/usr/bin/env python3
"""Exactly 24 input + 4 output tokens, GPU persistent qNg64 A/B compatibility only.

The persistent cbatch code uses MOE_CBATCH_MAXPOS=32 and drops prompt_len>=32.
The verifier's MOE_MAXPOS=160 is not proof of the persistent worker capacity.

This isolates the native policies without changing production routes or weights.
It is NOT a 112-token A/B experiment or an accuracy/JEPA speedup study.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,time
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,sha,stats
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps

SCHEMA="beglin-hcp-p4-24token-native-compat-v1"
MAX_C_BATCH_POS=32
PREFIX=24
GEN=4
INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"

def run(*,bank,tokenizer,index,root,output,sample_count=12):
    if sample_count not in (1,2,3,6,12):raise ValueError("P4 compatibility sample count outside budget")
    if PREFIX+GEN>MAX_C_BATCH_POS-1:
        raise ValueError("P4 prefix would overrun actual online max position")
    idx=pathlib.Path(index).resolve(strict=True)
    if sha(idx)!=INDEX_SHA:raise ValueError("DeepSeek checkpoint source changed")
    # Immutable persistent-stage binary, not copied, rebuilt or patched.
    stage=pathlib.Path(ps.PERSISTENT_BINARY).resolve(strict=True)
    if sha(stage)!=ps.EXPECTED_PERSISTENT_BINARY_SHA:
        raise ValueError("pinned persistent-stage binary differs")
    route=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
    route_before=sha(route) if route.is_file() else None
    corpus=prepare(bank,tokenizer)
    # One preregistered prompt per domain, with cyclic train/val/holdout.
    samples=[corpus["prompts"][3*i+i%3] for i in range(12)][:sample_count]
    if len({row["domain"] for row in samples})!=len(samples):
        raise ValueError("domain diversity lost")
    root=pathlib.Path(root)
    runner=pathlib.Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    owned=runner/("hcp-p4-native-short-"+os.environ["GITHUB_RUN_ID"])
    if (root.exists() or not owned.is_dir() or
        root.parent.resolve()!=owned.resolve() or
        root.name not in ("smoke_worker","measure_worker")):
        raise ValueError("worker workspace must be new and inside exact owned run root")
    # Only isolated worker Python state; no live supervisor config changes.
    ps.REQUEST_TIMEOUT_SECONDS=max(int(ps.REQUEST_TIMEOUT_SECONDS),180)
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root)
    rows=[];started=time.monotonic()
    try:
        alive=worker.start()
        if not alive.get("alive") or alive.get("runtime_policy_hash")!=pc.policy_hash(A):
            raise ValueError("native worker did not start at frozen A policy")
        pid=int(alive["pid"])
        for i,item in enumerate(samples):
            ids=item["token_ids"][:PREFIX]
            if len(ids)!=PREFIX or not all(0<=int(n)<102400 for n in ids):
                raise ValueError("invalid tokenizer prefix identity")
            pair={}
            order=("A","B") if i%2==0 else ("B","A")
            for name in order:
                target=A if name=="A" else B
                result=worker.submit_with_precision_policy([(ids,GEN)],
                    target_policy=target,admission_id=f"p4-short24-{i:02d}-{name}")
                if result.get("finite_logits") is not True or int(result.get("inference_passes",1))!=1:
                    raise ValueError("bad native logits/extra corrective inference pass")
                resp=result.get("responses")
                if not isinstance(resp,list) or len(resp)!=1 or not(1<=len(resp[0])<=GEN):
                    raise ValueError("unexpected native qNg64 response length")
                ans=[int(x) for x in resp[0]]
                ack=worker.health()
                if not ack.get("alive") or ack["runtime_policy_hash"]!=pc.policy_hash(target) or int(ack["pid"])!=pid:
                    raise ValueError("native selected policy ACK or process identity mismatch")
                transition=(result.get("precision_epoch") or {}).get("transition_cost") or {}
                row={"tokens":ans,
                     "engine_ms":float(result["engine_wall_ms"]),
                     "roundtrip_ms":float(result["roundtrip_ms"]),
                     "transition_ms":float(transition.get("transition_wall_ms",0)),
                     "worker_rss_bytes":int(ack["rss_bytes"]),
                     "cache_bytes_added":int(transition.get("cache_bytes_added",0))}
                if (not math.isfinite(row["engine_ms"]) or row["engine_ms"]<=0 or
                    not math.isfinite(row["roundtrip_ms"]) or row["roundtrip_ms"]<=0 or
                    row["worker_rss_bytes"]<=0):
                    raise ValueError("invalid actual measurement")
                pair[name]=row
            # Never emit raw generated tokens or exact input IDs.
            rows.append({
                "domain":item["domain"],"original_source_split":item["split"],
                "input_positions":PREFIX,"generated_token_limit":GEN,
                "policy_order":list(order),
                "same_first_generated_token":pair["A"]["tokens"][0]==pair["B"]["tokens"][0],
                "same_full_generated_sequence":pair["A"]["tokens"]==pair["B"]["tokens"],
                "A":{k:v for k,v in pair["A"].items() if k!="tokens"},
                "B":{k:v for k,v in pair["B"].items() if k!="tokens"}})
            print("P4_NATIVE_SHORT24_PROGRESS",json.dumps({
                "pairs":len(rows),"same_first_token":sum(r["same_first_generated_token"] for r in rows),
                "elapsed_s":round(time.monotonic()-started,1)}),flush=True)
    finally:
        worker.stop(force=True)
    route_after=sha(route) if route.is_file() else None
    if route_before!=route_after:raise ValueError("production active route changed")
    if len(rows)!=sample_count:raise ValueError("native paired data incomplete")
    agg={}
    for name in ("A","B"):
        agg[name]={"policy_hash":pc.policy_hash(A if name=="A" else B),
            "engine_ms":stats([x[name]["engine_ms"] for x in rows]),
            "roundtrip_ms":stats([x[name]["roundtrip_ms"] for x in rows]),
            "transition_ms":stats([x[name]["transition_ms"] for x in rows]),
            "worker_rss_bytes":stats([x[name]["worker_rss_bytes"] for x in rows])}
    out={"schema":SCHEMA,
        "status":"REAL_NATIVE_GPU_SHORT24_COMPATIBILITY_ONLY",
        "context_tokens":PREFIX,"generated_token_budget":GEN,
        "same_device_and_worker":True,
        "sample_pairs":sample_count,
        "source_checkpoint_index_sha256":INDEX_SHA,
        "persistent_binary_sha256":ps.EXPECTED_PERSISTENT_BINARY_SHA,
        "policy_A":A,"policy_B":B,
        "per_policy":agg,"original_AF_reference_for_same_24token_prefix":False,
        "first_token_A_B_agreement_rate":sum(x["same_first_generated_token"] for x in rows)/len(rows),
        "exact_sequence_A_B_agreement_rate":sum(x["same_full_generated_sequence"] for x in rows)/len(rows),
        "first_token_human_accuracy_measured":False,
        "long_112token_native_support":False,
        "native_qng64_quality_certification":False,
        "production_route_hash_unchanged":True,
        "native_policy_changes_only_in_scratch_worker":True,
        "production_touched":False,"auto_live_promotion":False,
        "records":rows}
    output=pathlib.Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    print("P4_NATIVE_SHORT24_COMPATIBILITY_RESULT",json.dumps({
        "pairs":sample_count,"first_token_A_B_agreement":out["first_token_A_B_agreement_rate"],
        "exact_sequences_A_B_agreement":out["exact_sequence_A_B_agreement_rate"],
        "engine_p50_ms":{k:v["engine_ms"]["p50"] for k,v in agg.items()},
        "no_live_policy_change":True},sort_keys=True))
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("bank","tokenizer","index","root","output"):
        p.add_argument("--"+name,required=True,type=pathlib.Path)
    p.add_argument("--sample-count",type=int,default=12)
    a=p.parse_args()
    run(bank=a.bank,tokenizer=a.tokenizer,index=a.index,root=a.root,
        output=a.output,sample_count=a.sample_count)
