#!/usr/bin/env python3
"""P5 real XOX 104-112 token qNg64 A/B capacity pilot, scratch binary ONLY.

Never modifies pinned deployed binary, live route, original model weights,
CMake checkout or existing supervisor. Two actual native policies are passed
through Beglin's original atomic PrecisionEpochScheduler; a mismatch aborts.
Also uses identical prompt IDs in the scratch standalone AF verifier as an
actual same-input next-token reference. No task correctness claim.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,subprocess,time,shutil
import numpy as np
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,stats
from hcp_p5_scratch_binary_adapter import file_sha,PINNED_OLD_BINARY_SHA,PINNED_OLD_SOURCE_SHA

SCHEMA="beglin-hcp-p5-native-scratch-kv128-real-long-ab-v1"
ORIG_INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
RUNNER_PREFIX="hcp-p5-long128-"
MIN_LEN=104
MAX_LEN=112
NATIVE_CAP=128
NEW_TOKENS=4

def standalone_af_argmax(binary,ids,model_root,root):
    if root.exists():raise ValueError("scratch reference directory must be fresh")
    root.mkdir(parents=True,exist_ok=False)
    encoded=",".join(map(str,ids))
    if not (MIN_LEN<=len(ids)<=MAX_LEN and len(encoded.encode())<900):
        raise ValueError("original verifier prompt input not within reviewed contract")
    env={k:os.environ[k] for k in ("HOME","PATH","USER","LANG","TMPDIR") if k in os.environ}
    env.update({"QWEN_MOE_BASE":str(model_root),
                "QWEN_MOE_PROMPT_IDS":encoded,
                "QWEN_MOE_NEARTIE_CORRECT":"0"})
    t0=time.monotonic()
    try:
        proc=subprocess.run([str(binary)],cwd=root,env=env,
                             stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                             timeout=190)
    except subprocess.TimeoutExpired as e:
        raise ValueError("standalone AF model exceeded fixed P5 timeout") from e
    if proc.returncode:raise ValueError("scratch standalone AF model forward failed")
    result=root/"moe3a_c_logits.bin"
    if not result.is_file() or result.is_symlink():
        raise ValueError("missing full AF forward logits")
    if result.stat().st_size%(4*len(ids))!=0:raise ValueError("AF logits positions do not match input")
    vocab=result.stat().st_size//(4*len(ids))
    if not 100000<=vocab<=110000:raise ValueError("wrong reference checkpoint vocabulary")
    logits=np.memmap(result,dtype="<f4",mode="r",shape=(len(ids),vocab))
    if not np.isfinite(logits[-1]).all():raise ValueError("nonfinite final AF reference")
    target=int(np.argmax(logits[-1]))
    del logits
    elapsed=time.monotonic()-t0
    shutil.rmtree(root)
    return target,float(elapsed)

def run(binary,source,bank,tokenizer,index,model_root,owner,output,*,count=1):
    if count not in (1,3):raise ValueError("P5 long capacity experiment supports one or three")
    if not (MAX_LEN+NEW_TOKENS<NATIVE_CAP):
        raise ValueError("requested output exceeds exact native KV bound")
    runner=pathlib.Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    owner=pathlib.Path(owner).resolve(strict=True)
    if owner.parent!=runner or not owner.name.startswith(RUNNER_PREFIX):
        raise ValueError("worker root must be under isolated P5 long study")
    binary=pathlib.Path(binary).resolve(strict=True)
    source=pathlib.Path(source).resolve(strict=True)
    if (not binary.is_file() or not source.is_file() or
        not binary.is_relative_to(owner) or not source.is_relative_to(owner)):
        raise ValueError("C and native target both must stay under study root")
    source_text=source.read_text()
    if source_text.count("#define MOE_CBATCH_MAXPOS 128")!=1 or "REBIND_SET_APPLIED" not in source_text:
        raise ValueError("not reviewed scratch CBATCH128 and atomic 2-target code")
    if file_sha(index)!=ORIG_INDEX_SHA:raise ValueError("model checkpoint index not pinned")
    old_stage=pathlib.Path(ps.PERSISTENT_BINARY).resolve(strict=True)
    before_stage=file_sha(old_stage)
    if before_stage!=PINNED_OLD_BINARY_SHA or ps.EXPECTED_PERSISTENT_SOURCE_SHA!=PINNED_OLD_SOURCE_SHA:
        raise ValueError("original serving stage not pinned")
    live=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
    before_live=file_sha(live) if live.is_file() else None
    corpus=prepare(bank,tokenizer)
    selected=[corpus["prompts"][3*i+i%3] for i in range(count)]
    if len({x["domain"] for x in selected})!=count:
        raise ValueError("independent topic domains lost")
    # Same-process shadow override, never rewriting ps.PERSISTENT_BINARY itself
    # or the immutable stage or any shared authority.
    ps.PERSISTENT_BINARY=binary
    ps.EXPECTED_PERSISTENT_BINARY_SHA=file_sha(binary)
    ps.REQUEST_TIMEOUT_SECONDS=max(190,ps.REQUEST_TIMEOUT_SECONDS)
    worker=None
    records=[];started=time.monotonic()
    try:
        references=[]
        for case in selected:
            ids=case["token_ids"]
            if not(MIN_LEN<=len(ids)<=MAX_LEN and len(ids)+NEW_TOKENS<NATIVE_CAP):
                raise ValueError("incompatible KV request length")
            ref,elapsed=standalone_af_argmax(binary,ids,pathlib.Path(model_root),
                owner/("original_af_ref_"+case["id"]))
            references.append((case,ids,ref,elapsed))
            print("P5_LONG_KV128_SAME_INPUT_AF_REFERENCE",json.dumps({
                 "processed_prompts":len(references),"source_input_positions":len(ids),
                 "original_af_elapsed_seconds":round(elapsed,2),
                 "no_raw_output_exported":True}),flush=True)
        worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),
                    root=owner/("smoke_worker" if count==1 else "measure_worker"))
        health=worker.start()
        if not health.get("alive") or health.get("runtime_policy_hash")!=pc.policy_hash(A):
            raise ValueError("native stage source no longer begins at A")
        pid=int(health["pid"])
        for n,(case,ids,reference,ref_s) in enumerate(references):
            results={}
            order=("A","B") if n%2==0 else ("B","A")
            for which in order:
                chosen=A if which=="A" else B
                outcome=worker.submit_with_precision_policy(
                    [(ids,NEW_TOKENS)],target_policy=chosen,
                    admission_id=f"hcp5-long128-{n:02d}-{which}")
                if outcome.get("finite_logits") is not True or int(outcome.get("inference_passes",1))!=1:
                    raise ValueError("native P5 qNg64 response not finite or unexpected extra pass")
                answer=outcome.get("responses")
                if not isinstance(answer,list) or len(answer)!=1 or not answer[0]:
                    raise ValueError("empty or malformed real GPU response")
                ack=worker.health()
                if not ack.get("alive") or int(ack["pid"])!=pid or ack["runtime_policy_hash"]!=pc.policy_hash(chosen):
                    raise ValueError("native C terminal ACK does not match exact policy/PID")
                epoch=outcome.get("precision_epoch") or {}
                if epoch.get("target_policy_hash")!=pc.policy_hash(chosen):
                    raise ValueError("precision scheduler target policy mismatch")
                tc=epoch.get("transition_cost") or {}
                results[which]={
                  "out":list(map(int,answer[0])),
                  "engine_ms":float(outcome["engine_wall_ms"]),
                  "roundtrip_ms":float(outcome["roundtrip_ms"]),
                  "transition_ms":float(tc.get("transition_wall_ms",0)),
                  "rss_bytes":int(ack["rss_bytes"]),
                  "cache_bytes_added":int(tc.get("cache_bytes_added",0)),
                }
                if (len(results[which]["out"])>NEW_TOKENS or
                    not math.isfinite(results[which]["engine_ms"]) or
                    results[which]["engine_ms"]<=0 or
                    results[which]["rss_bytes"]<=0):
                    raise ValueError("native long qNg64 unsafe measurement")
            records.append({
                "domain":case["domain"],"split":case["split"],
                "full_context_length":len(ids),"policy_order":list(order),
                "A_same_next_token_as_original_AF":results["A"]["out"][0]==reference,
                "B_same_next_token_as_original_AF":results["B"]["out"][0]==reference,
                "A_B_same_first_token":results["A"]["out"][0]==results["B"]["out"][0],
                "A_B_same_4token_sequence":results["A"]["out"]==results["B"]["out"],
                "original_AF_seconds":ref_s,
                "A":{k:v for k,v in results["A"].items() if k!="out"},
                "B":{k:v for k,v in results["B"].items() if k!="out"}
            })
            print("P5_LONG_KV128_REAL_POLICY_PAIR",json.dumps({
                "pairs":len(records),"context_length":len(ids),
                "next_AF_matches":{"A":records[-1]["A_same_next_token_as_original_AF"],
                                   "B":records[-1]["B_same_next_token_as_original_AF"]},
                "elapsed_s":round(time.monotonic()-started,1),
                "stage_route_untouched":True}),flush=True)
    finally:
        if worker is not None:worker.stop(force=True)
    if file_sha(old_stage)!=before_stage:
        raise ValueError("fixed historic stage binary changed")
    if (file_sha(live) if live.is_file() else None)!=before_live:
        raise ValueError("real active route mutated")
    if len(records)!=count:raise ValueError("native long experiment incomplete")
    stats_by={}
    for which in ("A","B"):
        stats_by[which]={"policy_hash":pc.policy_hash(A if which=="A" else B),
            "engine_ms":stats([r[which]["engine_ms"] for r in records]),
            "roundtrip_ms":stats([r[which]["roundtrip_ms"] for r in records]),
            "transition_ms":stats([r[which]["transition_ms"] for r in records]),
            "resident_rss_bytes":stats([r[which]["rss_bytes"] for r in records]),
            "same_AF_reference_next_token_rate":
                 sum(r[which+"_same_next_token_as_original_AF"] for r in records)/count}
    result={"schema":SCHEMA,"status":"SCRATCH_REAL_104_112_TOKEN_NATIVE_QNG64_KV128_AB_MEASURED",
        "data_points":count,"source":"real quantized AF DeepSeek and native qNg64 shadow",
        "verified_context_range":[min(r["full_context_length"] for r in records),
                                  max(r["full_context_length"] for r in records)],
        "cbatch_maxpos":NATIVE_CAP,"original_verifier_maxpos":160,
        "generated_new_tokens_limit":NEW_TOKENS,
        "old_stage_binary_sha256_unchanged":before_stage,
        "scratch_source_sha256":file_sha(source),
        "scratch_binary_sha256":file_sha(binary),
        "checkpoint_index_sha256":ORIG_INDEX_SHA,
        "per_policy":stats_by,
        "first_token_A_B_agreement":sum(r["A_B_same_first_token"] for r in records)/count,
        "generated_4token_A_B_agreement":sum(r["A_B_same_4token_sequence"] for r in records)/count,
        "original_AF_reference_is_human_accuracy_ground_truth":False,
        "no_original_stage_or_active_route_mutation":True,
        "whole_model_task_quality_verified":False,
        "native_runtime_certified_for_production":False,
        "live_precision_policy_changed":False,
        "production_touched":False,"automatic_live_promotion":False,
        "records":records}
    out=pathlib.Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("P5_SCRATCH_KV128_LONG_AB_RESULT",json.dumps({
        "samples":count,"verified_context":result["verified_context_range"],
        "per_policy":stats_by,"next_AF_and_AB_token_agreement":result["first_token_A_B_agreement"],
        "production_touched":False},sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("binary","source","bank","tokenizer","index","model-root","owner","output"):
        p.add_argument("--"+name,required=True,type=pathlib.Path)
    p.add_argument("--sample-count",default=1,type=int)
    a=p.parse_args()
    run(a.binary,a.source,a.bank,a.tokenizer,a.index,a.model_root,a.owner,a.output,count=a.sample_count)
