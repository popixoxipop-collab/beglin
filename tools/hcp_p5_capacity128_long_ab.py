#!/usr/bin/env python3
"""112-token native DeepSeek A/B ONLY on bounded CB128 scratch compiled binary.

The current production persistent CB32 cannot serve these inputs. This driver
accepts ONLY the audited scratch build whose source increased the whole
MOE_CBATCH_MAXPOS macro from 32 to 128 while retaining existing guards and
REBIND_SET. It may not change or promote the real stage, active route, or
regular workers. 12-domain input order is preregistered, with AB/BA crossover.
Actual qNg64 vs AF next-token quality is NOT proven by this comparison.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,time
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,sha,stats
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p5_scratch_capacity128 import SCHEMA as SOURCE_SCHEMA,ALLOWED_SOURCE_HASH

SCHEMA="beglin-hcp-p5-native-cbatch128-long-ab-v1"
REFERENCE_SHA="c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53"
INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
LIMIT_RSS=14*1024**3
MAX_POS=128

def require_authority(binary,source,root,model_index):
    root=pathlib.Path(root).resolve(strict=True)
    runner=pathlib.Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    expected=runner/("hcp-p5-cbatch128-"+os.environ["GITHUB_RUN_ID"])
    if root!=expected or not root.is_dir():
        raise ValueError("not exact isolated P5 CB128 run root")
    binpath=pathlib.Path(binary).resolve(strict=True)
    cpath=pathlib.Path(source).resolve(strict=True)
    if binpath!=root/"build/qwen_infer_gpu" or cpath!=root/"src/qwen_infer.c":
        raise ValueError("native binary/source escaped owned scratch")
    if not binpath.is_file() or not cpath.is_file() or binpath.is_symlink() or cpath.is_symlink():
        raise ValueError("missing safe scratch source or binary")
    c=cpath.read_text()
    if c.count("#define MOE_CBATCH_MAXPOS 128")!=1 or "#define MOE_MAXPOS 160" not in c:
        raise ValueError("scratch binary's source lacks 128/160 capacity")
    if "REBIND_SET_APPLIED" not in c or "moe_cb4b_admit_guard(" not in c:
        raise ValueError("scratch source lacks native rebind or input guard")
    if sha(model_index)!=INDEX_SHA:
        raise ValueError("DeepSeek checkpoint is not the study's pinned source")
    stage=pathlib.Path(ps.PERSISTENT_BINARY)
    old_sha=ps.EXPECTED_PERSISTENT_BINARY_SHA
    if sha(stage)!=old_sha:
        raise ValueError("old stage changed before isolated test")
    return stage,old_sha,sha(binpath),sha(cpath)

def run(*,binary,source,root,bank,tokenizer,index,risk_evidence,output,count=2,new_tokens=4):
    if count not in (2,12) or new_tokens!=4:raise ValueError("only preregistered 2 or 12 crossover pairs")
    stage,old_sha,bin_sha,c_sha=require_authority(binary,source,root,index)
    if sha(risk_evidence)!=REFERENCE_SHA:
        raise ValueError("P4 frozen actual route risk input SHA mismatch")
    expected=json.loads(pathlib.Path(risk_evidence).read_text())
    if expected.get("schema")!="beglin-hcp-p4-long-actual-af-risk-dataset-v1" or expected.get("record_count")!=36:
        raise ValueError("not original genuine 36-prompt P4 trace corpus")
    frozen={x["prompt_id"]:x for x in expected["runs"]}
    corpus=prepare(bank,tokenizer)
    selected=[corpus["prompts"][3*i+i%3] for i in range(12)][:count]
    if len({r["domain"] for r in selected})!=count:
        raise ValueError("not distinct research domains")
    for entry in selected:
        expected_record=frozen[entry["id"]]
        if (expected_record["token_sha256"]!=entry["token_sha256"] or
            expected_record["prompt_length"]!=entry["tokens"] or
            not 104<=entry["tokens"]<=112 or
            entry["tokens"]+new_tokens>=MAX_POS):
            raise ValueError("one native long prompt does not match original AF capture")
    live=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
    live_before=sha(live) if live.is_file() else None
    worker_root=pathlib.Path(root)/("smoke_worker" if count==2 else "measure_worker")
    if worker_root.exists():raise ValueError("native scratch worker already exists")
    original_bin=ps.PERSISTENT_BINARY
    original_digest=ps.EXPECTED_PERSISTENT_BINARY_SHA
    ps.PERSISTENT_BINARY=pathlib.Path(binary).resolve(strict=True)
    ps.EXPECTED_PERSISTENT_BINARY_SHA=bin_sha
    ps.REQUEST_TIMEOUT_SECONDS=max(int(ps.REQUEST_TIMEOUT_SECONDS),180)
    records=[];started=time.monotonic()
    try:
        worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=worker_root)
        native_env=worker._env
        def one_slot_env():
            e=native_env()
            e["QWEN_MOE_CB_SLOTS"]="1"
            e["QWEN_MOE_CB_REQS"]="1"
            return e
        worker._env=one_slot_env
        try:
            before=worker.start()
            if not before.get("alive") or before["runtime_policy_hash"]!=pc.policy_hash(A):
                raise ValueError("scratch native worker did not start at policy A")
            pid=int(before["pid"])
            if int(before["rss_bytes"])>LIMIT_RSS:
                raise ValueError("scratch worker exceeds preadmission 14 GiB RSS gate")
            for i,entry in enumerate(selected):
                ids=entry["token_ids"]
                answers={}
                order=("A","B") if i%2==0 else ("B","A")
                for name in order:
                    pol=A if name=="A" else B
                    result=worker.submit_with_precision_policy(
                        [(ids,new_tokens)],target_policy=pol,
                        admission_id=f"p5-cbatch128-{i:02d}-{name}")
                    if result.get("finite_logits") is not True or result.get("inference_passes",1)!=1:
                        raise ValueError("native logits not finite or unexpected corrective inference")
                    responses=result.get("responses")
                    if not isinstance(responses,list) or len(responses)!=1 or not responses[0]:
                        raise ValueError("native worker missing long prompt response")
                    seq=[int(v) for v in responses[0]]
                    if len(seq)>new_tokens:
                        raise ValueError("native worker response token count exceeds declared budget")
                    ack=worker.health()
                    if not ack.get("alive") or ack["runtime_policy_hash"]!=pc.policy_hash(pol) or int(ack["pid"])!=pid:
                        raise ValueError("native long prompt exact ACK or PID mismatch")
                    if int(ack["rss_bytes"])>LIMIT_RSS:
                        raise ValueError("native long context exceeded scratch-only 14 GiB RSS gate")
                    epoch=result.get("precision_epoch") or {}
                    transitioned=bool(epoch.get("transitioned"))
                    if transitioned:
                        if (len(epoch.get("changed_targets",[]))!=2 or
                            int(epoch["after_epoch"])!=int(epoch["before_epoch"])+1):
                            raise ValueError("REBIND_SET not atomic exactly-two-target epoch")
                        if (worker.ack.get("txn_id")!=epoch.get("txn_id") or
                            worker.ack.get("status")!="REBIND_SET_APPLIED"):
                            raise ValueError("native long context actual REBIND_SET terminal ACK missing")
                    if not math.isfinite(float(result["engine_wall_ms"])) or float(result["engine_wall_ms"])<=0:
                        raise ValueError("invalid native full model duration")
                    transition=epoch.get("transition_cost") or {}
                    answers[name]={"tokens":seq,"engine_ms":float(result["engine_wall_ms"]),
                        "roundtrip_ms":float(result["roundtrip_ms"]),
                        "transition_ms":float(transition.get("transition_wall_ms",0)),
                        "rss_bytes":int(ack["rss_bytes"]),
                        "num_rebound_targets":len(epoch.get("changed_targets",[])),
                        "epoch_transitioned":transitioned}
                a,b=answers["A"],answers["B"]
                records.append({
                   "prompt_id":entry["id"],"domain":entry["domain"],
                   "original_split":entry["split"],"positions":entry["tokens"],
                   "matched_original_af_prompt_SHA":True,
                   "AB_or_BA":list(order),
                   "same_first_token":a["tokens"][0]==b["tokens"][0],
                   "same_entire_sequence":a["tokens"]==b["tokens"],
                   "A":{k:v for k,v in a.items() if k!="tokens"},
                   "B":{k:v for k,v in b.items() if k!="tokens"},
                })
                print("P5_CAPACITY128_NATIVE_PROMPT",json.dumps({
                   "pairs":len(records),"tokens":entry["tokens"],"matched_original_af":True,
                   "AB_first_agreed_so_far":sum(row["same_first_token"] for row in records),
                   "elapsed_s":round(time.monotonic()-started,2),
                   "production_touched":False},sort_keys=True),flush=True)
        finally:
            worker.stop(force=True)
    finally:
        ps.PERSISTENT_BINARY=original_bin
        ps.EXPECTED_PERSISTENT_BINARY_SHA=original_digest
    if sha(stage)!=old_sha or (sha(live) if live.is_file() else None)!=live_before:
        raise ValueError("old original binary or live route mutated")
    if len(records)!=count:raise ValueError("missing original-AF-matched native long prompts")
    summary={}
    for name in ("A","B"):
        summary[name]={"exact_policy_hash":pc.policy_hash(A if name=="A" else B),
          "engine_ms":stats([x[name]["engine_ms"] for x in records]),
          "roundtrip_ms":stats([x[name]["roundtrip_ms"] for x in records]),
          "transition_ms":stats([x[name]["transition_ms"] for x in records]),
          "rss_bytes":stats([x[name]["rss_bytes"] for x in records])}
    result={"schema":SCHEMA,"status":"REAL_SCRATCH_CB128_LONG_CONTEXT_NATIVE_AB_RESEARCH_ONLY",
        "sample_pairs":count,"input_tokens_min":min(x["positions"] for x in records),
        "input_tokens_max":max(x["positions"] for x in records),
        "generated_token_budget":new_tokens,"routed_layers":26,
        "matched_P4_original_AF_prompt_source_SHA256":REFERENCE_SHA,
        "old_persistent_stage_binary_sha256":old_sha,
        "scratch_C_source_sha256":c_sha,"scratch_binary_sha256":bin_sha,
        "scratched_original_c_batch_cap_from":32,"scratch_cbatch_cap_to":128,
        "scratch_worker_slots":1,
        "policyA_B_first_token_agreement":sum(x["same_first_token"] for x in records)/count,
        "policyA_B_full_sequence_agreement":sum(x["same_entire_sequence"] for x in records)/count,
        "per_policy":summary,"records":records,
        "actual_binary_capacity_112token_tested":True,
        "original_AF_logits_comparison_in_this_run":False,
        "native_intervention_task_quality_measured":False,
        "model_ground_truth_accuracy_measured":False,
        "native_rebind_set_ack_not_mocked":True,
        "live_route_unchanged":True,"old_persistent_stage_unchanged":True,
        "production_touched":False,"automatic_live_promotion":False}
    out=pathlib.Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("P5_CAPACITY128_LONG_NATIVE_AB_RESULT",json.dumps({
        "pairs":count,"first_token_agreement":result["policyA_B_first_token_agreement"],
        "4token_sequence_agreement":result["policyA_B_full_sequence_agreement"],
        "engine_ms_p50":{p:summary[p]["engine_ms"]["p50"] for p in summary},
        "rss_p95":{p:summary[p]["rss_bytes"]["p95"] for p in summary},
        "native_long_support_is_scratch_only":True},sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("binary","source","root","bank","tokenizer","index","risk-evidence","output"):
        p.add_argument("--"+k,required=True,type=pathlib.Path)
    p.add_argument("--count",type=int,default=2)
    a=p.parse_args()
    run(binary=a.binary,source=a.source,root=a.root,bank=a.bank,
        tokenizer=a.tokenizer,index=a.index,risk_evidence=a.risk_evidence,
        output=a.output,count=a.count)
