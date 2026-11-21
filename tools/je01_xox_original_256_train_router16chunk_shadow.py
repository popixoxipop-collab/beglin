#!/usr/bin/env python3
"""JE01 REAL 256 unlabelled, frozen WikiText calibration original AF routing.

Runs the already SHA-proven, isolated 384-position XOX original-model Metal
router tracer over EXACTLY 256 original DeepSeek calibration prompts. B1 and
chunk size 16, original pretrained model weights READ ONLY. No heldout/ood
sources, correct labels, Bayesian tier decisions or serving changes.
Selecting the pilot 8 (layer,expert) slots by 256 pre-label real routed counts
is allowed only after every chunk succeeded and full provenance is sealed.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import time
import os
import sys

ROOT=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original384_20261117")
EXE=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original_router16_20261118_v1/build/je01_real_original_router_trace384")
EXE_SHA="37eccd3870e4b95be493ae18773497b8c6ab1ba54fff37d317477c0c7cfc78ad"
HELPER=Path("/Users/xox/mcp-sandbox/tailnet-commander/JE01_XOX_REAL_ROUTER16_SOURCE_20261118.py")
HELPER_GIT_BLOB="70c0e287e670f4a5726f9b6486e683492c0939cc"
RUN=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original_router256_20261118_v1")
MANIFEST=ROOT/"manifests_xox/calibration.manifest"
EXPECTED_MANIFEST_SHA="6674cefd09124f45599ba40a33a32ebf513968aa91f562f201f4e10e9f34cb3d"
INPUTS=ROOT/"inputs/je01_inputs"
SOURCE_INPUT_SHA="2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34"
SOURCE_INDEX_SHA="0aef144a8d549aa9e33c3a9fd9e899e0301e826bd563ceb92ad541cff1bbf768"
SCHEMA="beglin-je01-original-AF-256-unlabeled-router26-full64-source/1"
MAGIC=b"JE01_REAL_ORIGINAL_ROUTER_L26_T6_V1\n"
LOG_MAGIC=b"JE01_NATIVE_FULLLOGITS_F32_V1\n"
CHUNKS=16
N=16
NROUTE=26
NE=64
TOPK=6

def shaobj(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
            allow_nan=False).encode()).hexdigest()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def emit(event,**fields):
    print(json.dumps({"event":event,**fields},sort_keys=True),flush=True)

def local_helper():
    raw=HELPER.read_bytes()
    git=hashlib.sha1(b"blob "+str(len(raw)).encode()+bytes([0])+raw).hexdigest()
    if git!=HELPER_GIT_BLOB:raise ValueError("JE01_SOURCE_HELPER_GITBLOB_DRIFT")
    spec=importlib.util.spec_from_file_location("je01_prior_real_router_helper",HELPER)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def verify_all_inputs():
    if sha(EXE)!=EXE_SHA:raise ValueError("JE01_NATIVE_SOURCE_INSTRUMENTED_BINARY_DRIFT")
    if sha(MANIFEST)!=EXPECTED_MANIFEST_SHA:raise ValueError("JE01_FROZEN_256_NATIVE_MANIFEST_SHA_DRIFT")
    index=json.loads((INPUTS/"JE01_NATIVE_384_INPUTS_RECEIPT.json").read_text())
    sig=index.pop("receipt_sha256",None)
    if sig!=SOURCE_INPUT_SHA or shaobj(index)!=sig:raise ValueError("JE01_ORIGINAL_512_SOURCE_RECEIPT_DRIFT")
    if index.get("total_input_records")!=512:raise ValueError("JE01_ORIGINAL_SOURCE_512_COUNT_CHANGED")
    manifest_index=json.loads((ROOT/"manifests_xox/receipt.json").read_text())
    if manifest_index["receipt_sha256"]!=SOURCE_INDEX_SHA:raise ValueError("JE01_XOX_SOURCE_INPUT_INDEX_CHANGED")
    if manifest_index["manifest_sha256"]["calibration"]!=EXPECTED_MANIFEST_SHA:
        raise ValueError("JE01_MANIFEST_INDEX_CHANGED")
    expected=index["cohorts"]["calibration"]["file_receipts"]
    lines=MANIFEST.read_text().splitlines()
    if len(lines)!=256 or len(expected)!=256:raise ValueError("JE01_NOT_256_REAL_CAL_INPUTS")
    inputs=[]
    seen=set()
    for i,(entry,line) in enumerate(zip(expected,lines)):
        p_str,budget=line.rsplit(" ",1)
        p=Path(p_str)
        if budget!="1" or p.is_symlink() or not p.is_file() or p.parent.resolve()!=(INPUTS/"calibration").resolve():
            raise ValueError("JE01_NATIVE_MANIFEST_PATH_OR_BUDGET_DRIFT")
        if p.name!=entry["input_id"]+".bin" or entry["input_id"] in seen:
            raise ValueError("JE01_CAL_INPUT_ORDER_NOT_FROM_512_SOURCE")
        seen.add(entry["input_id"])
        raw=p.read_bytes()
        if len(raw)!=entry["native_int32_bytes"] or hashlib.sha256(raw).hexdigest()!=entry["native_int32_sha256"]:
            raise ValueError("JE01_CAL_256_NATIVE_INPUT_SHA_CHANGED:"+str(i))
        ids=struct.unpack("<"+str(len(raw)//4)+"i",raw)
        if not 2<=len(ids)<384 or ids[0]!=100000 or len(ids)!=entry["token_count_with_bos"]:
            raise ValueError("JE01_CAL_256_ORIGINAL_TOKENIZER_DRIFT")
        inputs.append({"index":i,"source_id":entry["input_id"],"token_len":len(ids),
                       "input_file_sha256":entry["native_int32_sha256"],"path":str(p)})
    return inputs

def audit_chunk(raw_route,raw_logits,inputs):
    route=raw_route.read_bytes()
    logits=raw_logits.read_bytes()
    if not route.startswith(MAGIC) or not logits.startswith(LOG_MAGIC):
        raise ValueError("JE01_RAW_ROUTER_AND_LOGIT_MAGIC_DRIFT")
    if len(route)!=len(MAGIC)+N*(12+26*64*4+26*6*4):
        raise ValueError("JE01_NATIVE_ROUTER_CHUNK_BYTES_DRIFT")
    if len(logits)!=len(LOG_MAGIC)+N*(16+102400*4):
        raise ValueError("JE01_NATIVE_LOGIT_CHUNK_BYTES_DRIFT")
    p=len(MAGIC);q=len(LOG_MAGIC)
    freq=[[0]*64 for _ in range(26)]
    sample_records=[]
    min_margin=1.0;max_sumerr=0
    for ix,source in enumerate(inputs):
        rid,pos,num=struct.unpack("<3I",route[p:p+12]);p+=12
        if rid!=ix or pos!=source["token_len"]-1 or num!=26:
            raise ValueError("JE01_ROUTER_CHUNK_PROMPT_LENGTH_OR_ORDER_DRIFT")
        prob=struct.unpack("<"+str(26*64)+"f",route[p:p+26*64*4]);p+=26*64*4
        top=struct.unpack("<"+str(26*6)+"i",route[p:p+26*6*4]);p+=26*6*4
        r,lpos,vocab,length=struct.unpack("<4I",logits[q:q+16]);q+=16
        if r!=ix or lpos!=pos or vocab!=102400 or length!=source["token_len"]:
            raise ValueError("JE01_LOGIT_CHUNK_PROMPT_LENGTH_OR_ORDER_DRIFT")
        lg=struct.unpack("<102400f",logits[q:q+102400*4]);q+=102400*4
        if any(not math.isfinite(x) for x in lg):raise ValueError("JE01_NONFINITE_NATIVE_PRETRAINED_LOGITS")
        arg=max(range(102400),key=lg.__getitem__)
        sel_rows=[]
        for layer in range(26):
            scores=prob[layer*64:(layer+1)*64]
            ids=top[layer*6:(layer+1)*6]
            if (len(set(ids))!=6 or any(j<0 or j>=64 for j in ids) or
                any(not math.isfinite(x) or x<0 for x in scores)):
                raise ValueError("JE01_REAL_ROUTER_CONTENT_DRIFT")
            sumerr=abs(math.fsum(scores)-1)
            if sumerr>1e-4:raise ValueError("JE01_ORIGINAL_SOFTMAX_SUM_BAD")
            max_sumerr=max(max_sumerr,sumerr)
            gap=min(scores[j] for j in ids)-max(scores[j] for j in range(64) if j not in ids)
            if gap<0:raise ValueError("JE01_NONGREEDY_ORIGINAL_ROUTE")
            min_margin=min(min_margin,gap)
            for j in ids:freq[layer][j]+=1
            sel_rows.append(list(ids))
        sample_records.append({"source_id":source["source_id"],"native_input_sha256":source["input_file_sha256"],
                               "positions":source["token_len"],"next_token_argmax":arg,
                               "per_26_layer_real_top6":sel_rows})
    if p!=len(route) or q!=len(logits):raise ValueError("JE01_NATIVE_ROUTER_TRUNCATED_OR_EXTRA")
    return {"real_routed_token_layer_events":N*26,
            "all_real_expert_decisions":N*26*6,
            "native_logits_count":N*102400,
            "min_topk_boundary_margin":min_margin,
            "max_router_prob_sum_error":max_sumerr,
            "frequency_matrix":freq,
            "sample_records":sample_records,
            "raw_original_router_sha256":sha(raw_route),
            "raw_original_logits_sha256":sha(raw_logits)}

def run_chunk(binary,chunk_index,inputs,baseline,helper):
    name="chunk"+str(chunk_index).zfill(2)
    dest=RUN/name
    if dest.exists():raise ValueError("JE01_CHUNK_DIRECTORY_ALREADY_EXISTS")
    dest.mkdir(mode=0o700)
    manifest=dest/"manifest.txt"
    manifest.write_text("".join(f'{x["path"]} 1\n' for x in inputs))
    route=dest/"real64_route.bin";logits=dest/"full102400_logits.bin"
    out=dest/"stdout.txt";err=dest/"stderr.txt"
    env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
      "HOME":"/Users/xox","TMPDIR":str(dest),"DYLD_LIBRARY_PATH":str(helper.MLXLIB),
      "QWEN_MOE_BASE":str(helper.WEIGHTS),"QWEN_MOE_GPU_CBATCH_ONLINE":"1",
      "QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"16",
      "QWEN_MOE_CB_PREFILL_BUDGET":"1","QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
      "QWEN_MOE_GPU_SORT_THRESHOLD":"64","QWEN_MOE_CBATCH_BUDGET":"0",
      "JE01_REAL_ROUTER_TRACE_OUTPUT":str(route),"JE01_REAL_ROUTER_LOGITS_OUTPUT":str(logits)}
    killed=None;lowest=helper.memfree();start=time.monotonic()
    with out.open("xb") as stdout,err.open("xb") as stderr:
        p=subprocess.Popen([str(binary)],cwd=dest,env=env,stdout=stdout,stderr=stderr)
        while p.poll() is None:
            if time.monotonic()-start>150:
                killed="CHUNK_TIMEOUT";p.terminate();break
            try:
                m=helper.memfree();lowest=min(m,lowest)
                if m<35:killed="LOW_SYSTEM_MEMORY";p.terminate();break
                if helper.health()!=baseline:killed="PRODUCTION_IDENTITY_DRIFT";p.terminate();break
                ps=subprocess.run(["/bin/ps","-o","rss=","-p",str(p.pid)],capture_output=True,text=True,timeout=4)
                if ps.stdout.strip().isdigit() and int(ps.stdout.strip())>16*1024*1024:
                    killed="SHADOW_RSS_GT16G";p.terminate();break
            except BaseException as e:killed="LIVE_MONITOR_ERROR_"+type(e).__name__;p.terminate();break
            time.sleep(1)
        if killed and p.poll() is None:
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:p.kill()
        rc=p.wait()
    stderr=err.read_text(errors="replace")
    if (rc!=0 or killed or "FATAL:" in stderr or
      "RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16" not in stderr):
        raise RuntimeError("JE01_NATIVE_CHUNK_REAL_FORWARD_FAIL:"+name+":"+str(rc)+":"+str(killed)+":"+stderr[-900:])
    if helper.health()!=baseline:raise ValueError("JE01_PRODUCTION_CHANGED_POST_CHUNK")
    audit=audit_chunk(route,logits,inputs)
    doc={"schema":"beglin-je01-real-original-256-calibration-chunk/1",
         "status":"PASS_REAL_ORIGINAL_PRETRAINED_ROUTER_AND_LOGITS",
         "chunk_index":chunk_index,"real_records":N,
         "calibration_only":True,"gold_labels_used":False,
         "source_512_original_input_receipt_sha256":SOURCE_INPUT_SHA,
         "source_256_original_manifest_sha256":EXPECTED_MANIFEST_SHA,
         "run_time_seconds":round(time.monotonic()-start,3),"min_mem_free_percent":lowest,
         "runtime_xox_production_workers_unchanged":True,
         "native_binary_sha256":EXE_SHA,
         "native_router_and_logits_file_bytes":[route.stat().st_size,logits.stat().st_size],
         "original_native_expert_nll_delta":"NOT_MEASURED",**audit}
    doc["receipt_sha256"]=shaobj(doc)
    with (dest/"JE01_CHUNK_RECEIPT.json").open("x") as f:
        f.write(json.dumps(doc,sort_keys=True,indent=2)+"\n")
    emit("CHUNK_SUCCEEDED",chunk=chunk_index,processed=(chunk_index+1)*16,
         receipt_sha256=doc["receipt_sha256"],min_free_pct=lowest,
         seconds=doc["run_time_seconds"],min_router_boundary=doc["min_topk_boundary_margin"])
    return doc

def main():
    helper=local_helper()
    before=helper.health()
    if helper.memfree()<53:raise ValueError("JE01_SHADOW_BEGIN_FREE_MEMORY_LT53")
    inputs=verify_all_inputs()
    if RUN.exists() or RUN.is_symlink():raise ValueError("JE01_OUTPUT_ROOT_ALREADY_EXISTS")
    RUN.mkdir(mode=0o700)
    emit("REAL_ORIGINAL_256_CAL_INPUTS_VALIDATED",records=len(inputs),
         manifest_sha256=EXPECTED_MANIFEST_SHA,binary_sha256=EXE_SHA)
    freq=[[0]*64 for _ in range(26)]
    receipts=[]
    for chunk in range(CHUNKS):
        part=inputs[chunk*16:(chunk+1)*16]
        if len(part)!=16:raise ValueError("JE01_MISSING_CAL_INPUTS")
        r=run_chunk(EXE,chunk,part,before,helper)
        receipts.append({"chunk":chunk,"receipt_sha256":r["receipt_sha256"],
          "native_router_sha256":r["raw_original_router_sha256"],
          "native_logits_sha256":r["raw_original_logits_sha256"],
          "seconds":r["run_time_seconds"],"min_mem_free_percent":r["min_mem_free_percent"],
          "route_events":r["real_routed_token_layer_events"]})
        for layer in range(26):
            for expert in range(64):freq[layer][expert]+=r["frequency_matrix"][layer][expert]
    if helper.health()!=before:raise ValueError("JE01_ORIGINAL_PRODUCTION_CHANGED")
    if sum(map(sum,freq))!=256*26*6:raise ValueError("JE01_REAL_ROUTE_COUNTS_MISMATCH")
    ranks=sorted([{"layer":l+1,"expert":e,"real_original_calibration_routed_calls":freq[l][e]}
          for l in range(26) for e in range(64)],
          key=lambda d:(-d["real_original_calibration_routed_calls"],d["layer"],d["expert"]))
    if not all(x["real_original_calibration_routed_calls"]>=32 for x in ranks[:8]):
        # This is an important prereg eligibility screen. Unrouted experts
        # cannot be silently made eligible; report honest infeasibility.
        min_32_met=False
    else:min_32_met=True
    doc={"schema":SCHEMA,"status":"PASS_256_REAL_ORIGINAL_AF_ROUTER_LAST_PROMPT_ONLY",
       "source_original_native_512_receipt_sha256":SOURCE_INPUT_SHA,
       "source_calibration256_manifest_sha256":EXPECTED_MANIFEST_SHA,
       "native_model_binary_sha256":EXE_SHA,"original_model_actual_inference":True,
       "unlabeled_calibration_only":True,"calibration_prompt_count":256,
       "routed_model_layers":26,"experts_per_layer":64,"topk":6,
       "observed_route_events":256*26,"observed_routed_expert_ids":256*26*6,
       "real_64way_source_scores":256*26*64,"max_routed_counts_over_256_prompts_per_pair":max(v["real_original_calibration_routed_calls"] for v in ranks),
       "exactly_eight_pilot_slots_frozen_by_unlabeled_calibration_last_token_routing":ranks[:8],
       "all_8_pilot_slots_have_at_least_32_routed_observations":min_32_met,
       "real256_frequency_matrix_sha256":shaobj(freq),
       "real256_frequency_matrix":freq,
       "all_chunks":receipts,
       "original_256_full_gold_NLL":"NOT_MEASURED",
       "real_bayesian_expert_precision_tier_delta_NLL":"NOT_MEASURED",
       "source_unquantized_BF16_quality":"NOT_MEASURED",
       "production_before":before,"production_after":helper.health(),
       "production_weights_mutated":False,"release_modified":False,
       "production_promotion_allowed":False}
    doc["receipt_sha256"]=shaobj(doc)
    with (RUN/"JE01_ACTUAL_ORIGINAL_256_ROUTER_CAL_FINAL.json").open("x") as f:
        f.write(json.dumps(doc,sort_keys=True,indent=2)+"\n")
    emit("REAL_256_CAL_ROUTER_FINAL",status=doc["status"],
         source_full_receipt_sha256=doc["receipt_sha256"],
         original_router_events=doc["observed_route_events"],
         selected_top8=doc["exactly_eight_pilot_slots_frozen_by_unlabeled_calibration_last_token_routing"],
         all_8_eligible_32=doc["all_8_pilot_slots_have_at_least_32_routed_observations"],
         min_free_pct=min(x["min_mem_free_percent"] for x in receipts),
         original_full_gold_NLL="NOT_MEASURED",promotion=False)

if __name__=="__main__":
    try:main()
    except BaseException as e:
        emit("FAIL_CLOSED_OR_INTERRUPTED",kind=type(e).__name__,error=str(e)[:1500],
             promotion=False)
        sys.exit(2)
