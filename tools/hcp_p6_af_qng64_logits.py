#!/usr/bin/env python3
"""P6: same-input original AF reference vs scratch qNg64 A/B last-position GPU logits.

This is a precision-fidelity study, NOT supervised accuracy, BF16 teacher
comparison, or a speed benchmark. No raw prompt IDs, logits or generated token
IDs are persisted in upload artifacts. Only temporary local XOX files see them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import time

import numpy as np
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A, B, sha, stats
from hcp_p6_scratch_logit_hook import HOOK, P5_C128_SHA256

SCHEMA = "beglin-hcp-p6-actual-af-qng64-matched-logit-fidelity-v1"
P4_FEATURE_SHA = "c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53"
INDEX_SHA = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
RSS_LIMIT = 14 * (1024 ** 3)
GPU_CAPTURE = "scratch_p6_last_logits.f32"
MOE_BASE_WEIGHTS = "weights_moe/deepseek_moe_af.bin"

def validate(binary, source, root, idx, feature, moe_base):
    root = pathlib.Path(root).resolve(strict=True)
    runner = pathlib.Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    if root != runner / ("hcp-p6-af-qng64-" + os.environ["GITHUB_RUN_ID"]):
        raise ValueError("P6 exact run-owned scratch root required")
    binary = pathlib.Path(binary).resolve(strict=True)
    source = pathlib.Path(source).resolve(strict=True)
    if (binary != root / "build/qwen_infer_gpu" or source != root / "src/qwen_infer.c"
        or binary.is_symlink() or source.is_symlink() or not binary.is_file()):
        raise ValueError("P6 execution is restricted to this run's compiled scratch binary")
    c = source.read_text()
    if (c.count(HOOK) != 1 or "#define MOE_CBATCH_MAXPOS 128" not in c
        or "REBIND_SET_APPLIED" not in c or "moe_cb4b_admit_guard(" not in c):
        raise ValueError("P6 scratch source/hash/capacity/ACK invalid")
    mlx = root / "src/mlx_moe.cpp"
    if (not mlx.is_file() or mlx.is_symlink()
        or "#define MLA_L0_MAXPOS 128" not in mlx.read_text()
        or "#define GQA_L0_MAXPOS 32" not in mlx.read_text()):
        raise ValueError("P6 C and MLX bounds not co-extended")
    if sha(idx) != INDEX_SHA or sha(feature) != P4_FEATURE_SHA:
        raise ValueError("P6 model or 36-prompt AF source SHA mismatch")
    if not (pathlib.Path(moe_base) / MOE_BASE_WEIGHTS).is_file():
        raise ValueError("P6 original AF model not present")
    stage = pathlib.Path(ps.PERSISTENT_BINARY)
    if not stage.is_file() or sha(stage) != ps.EXPECTED_PERSISTENT_BINARY_SHA:
        raise ValueError("P6 immutable original worker binary changed")
    return {"root":root,"binary_sha":sha(binary),"c_sha":sha(source),
            "old_stage_sha":sha(stage),"stage":stage}

def af_reference(binary, moe_base, ids, scratch, *, expected_vocab=None):
    scratch.mkdir(parents=True,exist_ok=False)
    try:
        value = ",".join(map(str,ids))
        if not(104<=len(ids)<=112 and len(value.encode())<900):
            raise ValueError("P6 original AF input exceeded the frozen 112-token bound")
        env = {k:os.environ[k] for k in ("HOME","PATH","USER","TMPDIR","LANG") if k in os.environ}
        env.update({"QWEN_MOE_BASE":str(pathlib.Path(moe_base).resolve(strict=True)),
                    "QWEN_MOE_PROMPT_IDS":value,"QWEN_MOE_NEARTIE_CORRECT":"0"})
        p = subprocess.run([str(binary)],cwd=scratch,env=env,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,timeout=180)
        if p.returncode != 0:
            raise ValueError("original AF full forward failed; raw logs withheld")
        logits = scratch/"moe3a_c_logits.bin"
        if not logits.is_file() or logits.is_symlink():
            raise ValueError("original AF logit file missing")
        if logits.stat().st_size % (len(ids)*4):
            raise ValueError("AF full-forward logits shape invalid")
        vocab = logits.stat().st_size//(4*len(ids))
        if not(100000 <= vocab <= 110000) or (expected_vocab is not None and vocab != expected_vocab):
            raise ValueError("AF reference vocabulary or shape mismatch")
        view=np.memmap(logits,dtype="<f4",mode="r",shape=(len(ids),vocab))
        ref=np.asarray(view[-1],dtype=np.float64).copy()
        del view
        if not np.isfinite(ref).all() or ref.size != vocab:
            raise ValueError("original AF non-finite or incomplete reference")
        return ref,vocab
    finally:
        shutil.rmtree(scratch)

def fidelity(af:np.ndarray,q:np.ndarray)->dict:
    if af.shape!=q.shape or not np.isfinite(q).all():
        raise ValueError("policy logits invalid or vocab differs from original AF")
    af=af.astype(np.float64,copy=False)
    q=q.astype(np.float64,copy=False)
    centered=(q-q.mean())-(af-af.mean())
    diff=q-af
    pa=np.exp(af-np.max(af)); pa/=pa.sum()
    pq=np.exp(q-np.max(q)); pq/=pq.sum()
    kl=float(np.dot(pa,np.log(np.maximum(pa,1e-300))-
                          np.log(np.maximum(pq,1e-300))))
    top20a=set(np.argpartition(af,-20)[-20:].tolist())
    top20q=set(np.argpartition(q,-20)[-20:].tolist())
    return {"first_token_matches_af":int(np.argmax(q))==int(np.argmax(af)),
            "top20_overlap":len(top20a&top20q)/20.0,
            "centered_logit_rmse":float(np.sqrt(np.mean(centered*centered))),
            "centered_logit_rmse_over_af_std":float(np.sqrt(np.mean(centered*centered))
                  /max(float(np.std(af)),1e-12)),
            "raw_logit_max_abs_difference":float(np.max(np.abs(diff))),
            "af_to_policy_kl_nats":max(0.,kl)}

def run(*,binary,source,root,bank,tokenizer,index,moe_base,feature,output,count):
    if count not in (2,12):raise ValueError("P6 only allows preregistered 2 or 12 prompts")
    info=validate(binary,source,root,index,feature,moe_base)
    root=info["root"]
    frozen=json.loads(pathlib.Path(feature).read_text())
    if (frozen.get("schema")!="beglin-hcp-p4-long-actual-af-risk-dataset-v1"
        or frozen.get("record_count")!=36):
        raise ValueError("P6 original 36-prompt AF evidence malformed")
    original={x["prompt_id"]:x for x in frozen["runs"]}
    corpus=prepare(bank,tokenizer)
    selected=[corpus["prompts"][3*i+i%3] for i in range(12)][:count]
    if len({x["domain"] for x in selected})!=count:
        raise ValueError("P6 missing distinct domains")
    for row in selected:
        p=original[row["id"]]
        if (p["token_sha256"]!=row["token_sha256"] or
            p["prompt_length"]!=row["tokens"] or
            not 104<=row["tokens"]<=112 or row["tokens"]+4>=128):
            raise ValueError("P6 AF and qNg64 source input hash mismatch")

    stage=info["stage"]
    live=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
    live_before=sha(live) if live.is_file() else None
    af_refs={}; vocab=None
    for i,row in enumerate(selected):
        af,vocab_new=af_reference(binary,moe_base,row["token_ids"],
                    root/("af_%02d"%i),expected_vocab=vocab)
        vocab=vocab_new
        af_refs[row["id"]]=af
        print("P6_ACTUAL_AF_REFERENCE",json.dumps({"progress":i+1,
              "positions":row["tokens"],"vocab":vocab,
              "prompt_sha_p4_matched":True,"raw_logits_exported":False}),flush=True)

    orig_bin=ps.PERSISTENT_BINARY
    orig_sha=ps.EXPECTED_PERSISTENT_BINARY_SHA
    ps.PERSISTENT_BINARY=pathlib.Path(binary).resolve(strict=True)
    ps.EXPECTED_PERSISTENT_BINARY_SHA=info["binary_sha"]
    ps.REQUEST_TIMEOUT_SECONDS=max(int(ps.REQUEST_TIMEOUT_SECONDS),180)
    worker_root=root/("smoke_worker" if count==2 else "measure_worker")
    outlog=root/GPU_CAPTURE
    records=[];start=time.monotonic()
    try:
        worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=worker_root)
        original_env=worker._env
        def p6_env():
            env=original_env()
            env["QWEN_MOE_CB_SLOTS"]="1"
            env["QWEN_MOE_CB_REQS"]="1"
            env["BEGLIN_HCP_P6_CAPTURE_ROOT"]=str(root)
            env["BEGLIN_HCP_P6_LOGITS_OUTPUT"]=str(outlog)
            return env
        worker._env=p6_env
        try:
            health=worker.start()
            if not health.get("alive") or health["runtime_policy_hash"]!=pc.policy_hash(A):
                raise ValueError("P6 scratch worker did not start at A")
            pid=int(health["pid"])
            if health["rss_bytes"]>RSS_LIMIT:
                raise ValueError("P6 scratch worker start RSS exceeds 14GiB")
            for i,row in enumerate(selected):
                variants={}
                order=("A","B") if i%2==0 else ("B","A")
                for kind in order:
                    if outlog.exists():
                        raise ValueError("P6 stale logits present before admission")
                    pol=A if kind=="A" else B
                    native=worker.submit_with_precision_policy(
                        [(row["token_ids"],4)],target_policy=pol,
                        admission_id="p6-logit-%02d-%s"%(i,kind))
                    if native.get("finite_logits") is not True or native.get("inference_passes",1)!=1:
                        raise ValueError("P6 native finite/one-pass invariant failed")
                    seq=native.get("responses")
                    if not isinstance(seq,list) or len(seq)!=1 or not seq[0] or len(seq[0])>4:
                        raise ValueError("P6 missing real generated token sequence")
                    ack=worker.health()
                    if (not ack.get("alive") or ack["runtime_policy_hash"]!=pc.policy_hash(pol)
                        or int(ack["pid"])!=pid or int(ack["rss_bytes"])>RSS_LIMIT):
                        raise ValueError("P6 precise policy ACK/PID/RSS invariant failed")
                    epoch=native.get("precision_epoch") or {}
                    if epoch.get("transitioned"):
                        if (len(epoch.get("changed_targets",[]))!=2 or
                            worker.ack.get("txn_id")!=epoch.get("txn_id") or
                            worker.ack.get("status")!="REBIND_SET_APPLIED"):
                            raise ValueError("P6 native two-target REBIND_SET was not applied")
                    if not outlog.is_file() or outlog.is_symlink():
                        raise ValueError("P6 GPU last-pre-fill logit capture missing")
                    raw=np.fromfile(outlog,dtype="<f4")
                    outlog.unlink()
                    if len(raw)!=vocab:
                        raise ValueError("P6 GPU logit capture vocabulary mismatch")
                    if int(np.argmax(raw))!=int(seq[0][0]):
                        raise ValueError("captured GPU first token argmax != native response")
                    met=fidelity(af_refs[row["id"]],raw)
                    variants[kind]={"fidelity":met,"first_token":int(seq[0][0]),
                        "engine_wall_ms":float(native["engine_wall_ms"]),
                        "roundtrip_ms":float(native["roundtrip_ms"]),
                        "transition_ms":float((epoch.get("transition_cost") or {}).get("transition_wall_ms",0))}
                a,b=variants["A"],variants["B"]
                records.append({"prompt_id":row["id"],"domain":row["domain"],
                    "split":row["split"],"input_tokens":row["tokens"],
                    "original_AF_prompt_SHA256_matched":True,
                    "order":list(order),"A":a["fidelity"],"B":b["fidelity"],
                    "A_B_first_token_matches":a["first_token"]==b["first_token"]})
                print("P6_ACTUAL_LOGIT_COMPARE_PROGRESS",json.dumps({
                    "pairs":len(records),
                    "A_AF_match":sum(x["A"]["first_token_matches_af"] for x in records),
                    "B_AF_match":sum(x["B"]["first_token_matches_af"] for x in records),
                    "elapsed_s":round(time.monotonic()-start,1)},sort_keys=True),flush=True)
        finally:
            worker.stop(force=True)
    finally:
        ps.PERSISTENT_BINARY=orig_bin
        ps.EXPECTED_PERSISTENT_BINARY_SHA=orig_sha
        if outlog.is_file():outlog.unlink()
    if (sha(stage)!=info["old_stage_sha"]
        or (sha(live) if live.is_file() else None)!=live_before):
        raise ValueError("P6 old stage/live production route modified")
    if len(records)!=count:raise ValueError("P6 result incomplete")
    summary={}
    for name in ("A","B"):
        summary[name]={
            "af_first_token_agreement":sum(r[name]["first_token_matches_af"] for r in records)/count,
            "top20_overlap_p50":float(np.median([r[name]["top20_overlap"] for r in records])),
            "centered_rmse_p50":float(np.median([r[name]["centered_logit_rmse"] for r in records])),
            "centered_rmse_relative_p50":float(np.median([r[name]["centered_logit_rmse_over_af_std"] for r in records])),
            "af_to_policy_kl_nats_p50":float(np.median([r[name]["af_to_policy_kl_nats"] for r in records])),
            "max_abs_error_max":max(r[name]["raw_logit_max_abs_difference"] for r in records)}
    report={"schema":SCHEMA,"status":"ACTUAL_ORIGINAL_AF_VS_QNG64_LAST_LOGIT_RESEARCH_ONLY",
        "prompt_count":count,"input_tokens_min":min(x["input_tokens"] for x in records),
        "input_tokens_max":max(x["input_tokens"] for x in records),"vocab_size":vocab,
        "af_reference":"same-model original quantized AF, not BF16 teacher",
        "model_accuracy_vs_ground_truth_measured":False,
        "first_generated_token_logits_only":True,
        "full_4token_teacher_forced_logit_equivalence_measured":False,
        "source_P4_SHA256":P4_FEATURE_SHA,"checkpoint_index_sha256":INDEX_SHA,
        "original_worker_sha256":info["old_stage_sha"],
        "p6_scratch_binary_sha256":info["binary_sha"],
        "p6_scratch_c_sha256":info["c_sha"],"gpu_KV128_source_guards":True,
        "native_epoch_ack_not_mocked":True,
        "timing_contaminated_by_logit_capture":True,
        "speed_improvement_claim_allowed":False,
        "raw_logits_exported":False,"raw_generated_token_ids_exported":False,
        "raw_prompt_ids_exported":False,
        "summary":summary,"records":records,
        "production_touched":False,"automatic_live_promotion":False}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("P6_ACTUAL_AF_QNG64_LOGIT_RESULT",json.dumps({
        "prompts":count,"summary":summary,"production_touched":False,
        "speed_improvement_proven":False},sort_keys=True),flush=True)
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("binary","source","root","bank","tokenizer","index","moe-base","feature","output"):
        ap.add_argument("--"+name,required=True,type=pathlib.Path)
    ap.add_argument("--count",type=int,default=2)
    a=ap.parse_args()
    run(binary=a.binary,source=a.source,root=a.root,bank=a.bank,
        tokenizer=a.tokenizer,index=a.index,moe_base=a.moe_base,feature=a.feature,
        output=a.output,count=a.count)
