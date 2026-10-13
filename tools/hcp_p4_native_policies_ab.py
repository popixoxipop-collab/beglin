#!/usr/bin/env python3
"""Matched, scratch-only native qNg64 DeepSeek A/B quality-agreement and total cost.

Uses the ORIGINAL pinned persistent Beglin binary with an entirely new per-run
worker root; never changes active route, stage binary, policy map, or serving
workers. A and B are researched exact precision maps, not ranked quality
levels (qNg64 precision is NONMONOTONIC).

Measures agreement against the SAME model's original AF reference first-token
argmax, NOT ground-truth task success/BF16 accuracy. Baseline hashes are
retained only in the scratch dataset; exported artifacts contain no token IDs.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,statistics,time
import numpy as np
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps

SCHEMA="beglin-hcp-p4-matched-native-qng64-ab-v1"
A=pc.normalize_policy([
 {"role":"shared_up_proj","layer":3,"n":6},
 {"role":"shared_down_proj","layer":26,"n":5},
])
B=pc.normalize_policy([
 {"role":"shared_up_proj","layer":3,"n":5},
 {"role":"shared_down_proj","layer":26,"n":6},
])
INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def percentile(values,p):
    vals=sorted(float(x) for x in values)
    if not vals:raise ValueError("no observed native inference costs")
    q=(len(vals)-1)*p;i=math.floor(q);j=math.ceil(q)
    return float(vals[i]) if i==j else float(vals[i]+(vals[j]-vals[i])*(q-i))

def stats(values):
    if not values:raise ValueError("no native measurements")
    return {"count":len(values),"p50":percentile(values,.5),
            "p95":percentile(values,.95),"max":max(values)}

def check_manifest(feature_path,refs_path):
    raw=pathlib.Path(feature_path).read_bytes()
    d=json.loads(raw)
    if (d.get("schema")!="beglin-hcp-p4-long-actual-af-risk-dataset-v1" or
        d.get("status")!="ACTUAL_DEEPSEEK_LONGISH_CONTEXT_RISK_TRACES_ONLY"):
        raise ValueError("missing actual P4 36-prompt source")
    if d.get("record_count")!=36 or d.get("train_val_holdout")!={"train":12,"validation":12,"holdout":12}:
        raise ValueError("wrong independent prompt coverage")
    if d.get("native_precision_changed") is not False or d.get("production_touched") is not False:
        raise ValueError("source policy mutation")
    refs=json.loads(pathlib.Path(refs_path).read_text())
    if len(refs)!=36 or len(d["runs"])!=36:raise ValueError("missing scratch-only native references")
    index={r["prompt_id"]:r for r in refs}
    if len(index)!=36:raise ValueError("duplicate prompt internal references")
    rows=[]
    for rec in d["runs"]:
        key=rec["prompt_id"];other=index[key]
        ids=other["prompt_token_ids"]
        token_sha=hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
        if token_sha!=rec["token_sha256"] or token_sha!=other["token_sha256"] or len(ids)!=rec["prompt_length"]:
            raise ValueError("AF reference and qNg64 source prompts mismatched")
        if not(80<=len(ids)<=112):
            raise ValueError("model context not in safe experimental bounds")
        rows.append((rec,other))
    return rows,sha(feature_path)

def matched_ab(feature_path,refs_path,workspace,output,*,limit=36,new_tokens=4):
    if not (1<=limit<=36 and 1<=new_tokens<=8):
        raise ValueError("unapproved research bounds")
    source=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/model.safetensors.index.json")
    if sha(source)!=INDEX_SHA:raise ValueError("source checkpoint index changed")
    current_binary=pathlib.Path(ps.PERSISTENT_BINARY)
    if not current_binary.is_file() or sha(current_binary)!=ps.EXPECTED_PERSISTENT_BINARY_SHA:
        raise ValueError("persistent-stage binary identity mismatch")
    before_default=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
    before_route_sha=sha(before_default) if before_default.is_file() else None
    rows,features_sha=check_manifest(feature_path,refs_path)
    workspace=pathlib.Path(workspace)
    if workspace.exists():raise ValueError("worker scratch must be fresh")
    # The only native writes are new per-run worker queue files and runtime ACK
    # below RUNNER_TEMP; source policies, main repo and live route never changed.
    workspace.parent.mkdir(parents=True,exist_ok=True)
    # 112-token pilot may exceed the legacy 60s request budget. Override only
    # this script's in-memory imported module, not live process configuration.
    ps.REQUEST_TIMEOUT_SECONDS=max(int(ps.REQUEST_TIMEOUT_SECONDS),180)
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=workspace)
    records=[];summaries=[]
    started=time.monotonic()
    try:
        health=worker.start()
        if not health.get("alive") or health.get("runtime_policy_hash")!=pc.policy_hash(A):
            raise ValueError("wrong isolated native worker startup policy")
        pid=int(health["pid"])
        for n,(meta,ref) in enumerate(rows[:limit]):
            ids=ref["prompt_token_ids"];reference=int(ref["reference_last_token"])
            # AB/BA prompt pairing balances local temporal drift. It is not a
            # throughput benchmark with frozen resident cache across requests.
            order=["A","B"] if n%2==0 else ["B","A"]
            by={}
            for kind in order:
                pol=A if kind=="A" else B
                row=worker.submit_with_precision_policy(
                    [(ids,new_tokens)],target_policy=pol,
                    admission_id=f"p4shadow-{n:03d}-{kind}")
                if row.get("finite_logits") is not True or int(row.get("inference_passes",1))!=1:
                    raise ValueError("actual model returned nonfinite logits or adaptive extra pass")
                samples=row.get("responses")
                if not isinstance(samples,list) or len(samples)!=1 or not samples[0]:
                    raise ValueError("native worker yielded no response")
                pred=list(map(int,samples[0]))
                if len(pred)>new_tokens:raise ValueError("unexpected response length")
                epoch=row.get("precision_epoch") or {}
                if str(epoch.get("selected_policy_hash",epoch.get("target_policy_hash",""))) not in (
                    "",pc.policy_hash(pol)):
                    raise ValueError("native precision policy hash mismatch")
                # Canonical post-execution runtime ACK must match target exactly.
                ack=worker.health()
                if ack["runtime_policy_hash"]!=pc.policy_hash(pol) or int(ack["pid"])!=pid:
                    raise ValueError("actual native runtime ACK/PID mismatch")
                tc=epoch.get("transition_cost") or {}
                by[kind]={
                    "generated":pred,
                    "engine_wall_ms":float(row["engine_wall_ms"]),
                    "roundtrip_ms":float(row["roundtrip_ms"]),
                    "transition_wall_ms":float(tc.get("transition_wall_ms",0)),
                    "cache_bytes_added":int(tc.get("cache_bytes_added",0)),
                    "runtime_rss_bytes":int(ack["rss_bytes"]),
                    "runtime_policy_hash":ack["runtime_policy_hash"],
                    "no_model_errors":True,
                }
                if min(by[kind]["engine_wall_ms"],by[kind]["roundtrip_ms"])<=0:
                    raise ValueError("invalid native latency")
            a,b=by["A"],by["B"]
            records.append({
                "prompt_id":meta["prompt_id"],"split":meta["split"],
                "domain":meta["category"],"prompt_length":len(ids),
                "matched_policy_order":order,
                "a_af_reference_next_token_agreement":a["generated"][0]==reference,
                "b_af_reference_next_token_agreement":b["generated"][0]==reference,
                "a_b_next_token_agreement":a["generated"][0]==b["generated"][0],
                "a_b_generated_sequence_exact_match":a["generated"]==b["generated"],
                "relative_gap_p50_at_original_af":meta["real_min_rank6_rank7_gap_p50"],
                "relative_gap_min_at_original_af":meta["real_min_rank6_rank7_gap_min"],
                "policy_A":{"engine_wall_ms":a["engine_wall_ms"],
                            "roundtrip_ms":a["roundtrip_ms"],
                            "transition_wall_ms":a["transition_wall_ms"],
                            "cache_bytes_added":a["cache_bytes_added"],
                            "rss_bytes":a["runtime_rss_bytes"]},
                "policy_B":{"engine_wall_ms":b["engine_wall_ms"],
                            "roundtrip_ms":b["roundtrip_ms"],
                            "transition_wall_ms":b["transition_wall_ms"],
                            "cache_bytes_added":b["cache_bytes_added"],
                            "rss_bytes":b["runtime_rss_bytes"]},
            })
            if (n+1)%6==0:
                print("P4_NATIVE_A_B_PROGRESS",json.dumps({
                    "pairs":n+1,
                    "first_token_policy_disagreements":sum(not r["a_b_next_token_agreement"] for r in records),
                    "seconds":round(time.monotonic()-started,2),
                    "separate_worker_pid_preserved":True}),flush=True)
    finally:
        worker.stop(force=True)
    after_route_sha=sha(before_default) if before_default.is_file() else None
    if before_route_sha!=after_route_sha:raise ValueError("production active route mutated")
    if len(records)!=limit:raise ValueError("incomplete native experimental sample")
    policy_summary={}
    for kind in ("A","B"):
        entry=[r["policy_"+kind] for r in records]
        policy_summary[kind]={
            "precision_policy_hash":pc.policy_hash(A if kind=="A" else B),
            "engine_ms":stats([r["engine_wall_ms"] for r in entry]),
            "roundtrip_ms":stats([r["roundtrip_ms"] for r in entry]),
            "transition_ms":stats([r["transition_wall_ms"] for r in entry]),
            "resident_rss_bytes":stats([r["rss_bytes"] for r in entry]),
            "af_next_token_agreement_rate":sum(r[kind.lower()+"_af_reference_next_token_agreement"] for r in records)/limit}
    out={"schema":SCHEMA,"status":"ACTUAL_MATCHED_NATIVE_POLICY_A_B_AGREEMENT_COST_ONLY",
        "sample_pairs":limit,"original_reference":"original AF C verifier last-prompt-position logits argmax",
        "reference_is_bf16_or_human_ground_truth":False,
        "new_tokens_requested":new_tokens,
        "checkpoint_index_sha256":INDEX_SHA,
        "persistent_binary_sha256":ps.EXPECTED_PERSISTENT_BINARY_SHA,
        "reference_feature_sha256":features_sha,
        "risk_feature_is_certified_bit_safety":False,
        "observed_policy_A":A,"observed_policy_B":B,
        "per_policy":policy_summary,
        "a_b_next_token_agreement_rate":sum(r["a_b_next_token_agreement"] for r in records)/limit,
        "a_b_generated_sequence_exact_agreement_rate":sum(r["a_b_generated_sequence_exact_match"] for r in records)/limit,
        "real_model_gpu_inference":True,"policy_changes_only_in_disposable_worker":True,
        "active_route_sha_before_after_equal":True,
        "native_accuracy_vs_human_reference_measured":False,
        "production_touched":False,"automatic_live_promotion":False,
        "notes":"Observed policy agreement is not model accuracy; qNg64 n is not monotonic in quality.",
        "records":records}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    print("P4_REAL_NATIVE_MATCHED_QNG64_AB_RESULT",json.dumps({
        "count":limit,"policy_summary":policy_summary,
        "first_token_A_B_agreement":out["a_b_next_token_agreement_rate"],
        "4token_A_B_agreement":out["a_b_generated_sequence_exact_agreement_rate"],
        "production_touched":False},sort_keys=True))
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("features","scratch-only-reference","worker-root","output"):
        ap.add_argument("--"+name,required=True,type=pathlib.Path)
    ap.add_argument("--limit",default=36,type=int)
    ap.add_argument("--new-tokens",default=4,type=int)
    a=ap.parse_args()
    matched_ab(a.features,a.scratch_only_reference,a.worker_root,a.output,
               limit=a.limit,new_tokens=a.new_tokens)
