#!/usr/bin/env python3
"""JE01 EOE real original27 native FFN h2 trace on 16 TRAIN last tokens.

Captures actual normalized 2048-d FFN input only for 8 frozen selected
Experts routed in top6; no invented Fisher, Hessian or gradients. Requires
gold-NLL output SHA to match the previously certified uninstrumented original
AF4 B1 16x656 training gold exactly, or entire trace rejected. All resources
are isolated D50 research only; heldout and production paths untouched.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,struct,subprocess,time
import numpy as np
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/je01_eoe_original27_h2_trace16_20261122_v1")
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work/je01_eoe_realaf_nll_pilot16_20261122_v1")
CHECKPOINT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/native_moe_base_source_copy")
BINARY_SHA="ddcbbce7e8bddad691dcfa12a8345ac6df7afb266330c35c322fde656a8336ef"
AF_SHA="9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
ORIGINAL16_OUTPUT_SHA="b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c"
BASE_RECEIPT="7dd5da25e500007a2a35fd724d82a7c8c6966727458763160f6d4ee7b3ac54ec"
OLD_STAGE="a814cdfd84f544132c4d9d8aa9e6e3c4845252e49a467d61c623e93931d3106e"
FROZEN={(12,13),(13,22),(13,28),(14,63),(20,17),(21,61),(24,26),(24,60)}
MAGIC=b"JE01_ACTUAL_REAL_AF4_TRAIN16_ROUTED_H2_V1\n"
GOLD=b"JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1\n"
GOLDMETA=b"JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n"
RECORD=16+4+2048*4
SCHEMA="beglin-je01-real-original27-routed-FFN-h2-16train-after-fullgold-parity/1"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for v in iter(lambda:f.read(1<<20),b""):h.update(v)
    return h.hexdigest()
def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def signed(path,expected):
    if path.is_symlink() or not path.is_file():raise ValueError("JE01_SIGNED_EVIDENCE_MISSING")
    d=json.loads(path.read_text())
    sig=d.pop("receipt_sha256",None)
    if sig!=expected or canonical(d)!=sig:raise ValueError("JE01_SOURCE_SIGNED_DIGEST_CHANGED:"+path.name)
    return d
def memfree():
    output=subprocess.check_output(["/usr/bin/memory_pressure","-Q"],timeout=8,text=True)
    m=re.search(r"System-wide memory free percentage:\s*(\d+)%",output)
    if not m:raise ValueError("JE01_MEMORY_SENSOR_MISSING")
    return int(m.group(1))
def preflight():
    if ROOT.is_symlink() or not ROOT.is_dir() or BASE.is_symlink() or not BASE.is_dir():
        raise ValueError("JE01_ISOLATED_D50_ROOT_REQUIRED")
    bin=ROOT/"bin"/"je01_eoe_original27_routed_h2_trace16_fullgold"
    build=ROOT/"bin"/"JE01_REAL_ORIGINAL27_ROUTED_H2_BUILD_RECEIPT.json"
    if bin.is_symlink() or not bin.is_file() or sha(bin)!=BINARY_SHA or \
       build.is_symlink() or not build.is_file():
        raise ValueError("JE01_NATIVE_METAL_TRACE_BINARY_SHA_DRIFT")
    b=json.loads(build.read_text())
    if b["status"]!="JE01_ACTUAL_ORIGINAL27_ROUTED_H2_NATIVE_TRACE_COMPILED" or \
       b["binary_sha256"]!=BINARY_SHA or b["production_modified"] or \
       b["original_source_patch"]["status"]!="JE01_ISOLATED_ORIGINAL27_LASTPROMPT_ROUTED_H2_TRACE_SOURCE_STAGED" or \
       b["gold_NLL_parity"]!="NOT_TESTED":
        raise ValueError("JE01_NATIVE_H2_TRUE_SOURCE_BINARY_RECEIPT_WRONG")
    baseline=signed(BASE/"JE01_EOE_PILOT16_REAL_NLL_RESULT.json",BASE_RECEIPT)
    if baseline["status"]!="PASS_EOE_REAL_ORIGINAL_AF384_FULLGOLD_TRAIN16_NLL_PARITY" or \
       baseline["full_gold_tokens"]!=656 or baseline["real_cases"]!=16 or \
       baseline["original_AF_source_sha256"]!=AF_SHA or \
       baseline["native_output_sha256"]!=ORIGINAL16_OUTPUT_SHA:
        raise ValueError("JE01_ORIGINAL_REAL656_REFERENCE_EVIDENCE_MISMATCH")
    old=signed(BASE/"JE01_EOE_STAGED_PILOT16_PROVENANCE.json",OLD_STAGE)
    input_candidates=[p for p in (BASE/"inputs",BASE/"inputs"/"je01_real_pilot16")
       if (p/"JE01_EOE_ORIGINAL_PILOT16_INPUT_RECEIPT.json").is_file()]
    if len(input_candidates)!=1:raise ValueError("JE01_ORIGINAL_TRAIN16_INPUT_SOURCE_AMBIGUOUS")
    inputs=input_candidates[0]
    goldmeta=inputs/"chunk00"/"gold_meta.bin"
    manifest=BASE/"manifest.eoe.txt"
    if any(x.is_symlink() or not x.is_file() for x in (goldmeta,manifest)) or \
       sha(goldmeta)!=old["gold_meta_sha256"] or sha(manifest)!=old["manifest_current_host_sha256"]:
        raise ValueError("JE01_GOLD16_ORIGINAL_INPUT_SHA_DRIFT")
    af=CHECKPOINT/"weights_moe"/"deepseek_moe_af.bin"
    if af.is_symlink() or af.stat().st_size!=9814343680 or sha(af)!=AF_SHA:
        raise ValueError("JE01_ORIGINAL_AF4_SOURCE_CHECKPOINT_CHANGED")
    import mlx,mlx.core as mx
    if mx.__version__!="0.32.1":raise ValueError("JE01_PINNED_MLX_NATIVE_ABI_CHANGED")
    lib=pathlib.Path(list(mlx.__path__)[0])/"lib"
    if not (lib/"libmlx.dylib").is_file():raise ValueError("JE01_MLX_NATIVE_DYLIB_MISSING")
    if memfree()<55:raise ValueError("JE01_MEMORY_BELOW55_BEFORE_NATIVE_RUN")
    return bin,manifest,goldmeta,lib,baseline

def parse_trace(path,meta):
    src=meta.read_bytes()
    if not src.startswith(GOLDMETA):raise ValueError("JE01_ORIGINAL_TRAIN16_META_MAGIC_DRIFT")
    pos=len(GOLDMETA);prompt_ends=[]
    for i in range(16):
        start,count=struct.unpack_from("<II",src,pos);pos+=8+1536
        if not 2<=start<384 or not 1<=count<=384-start:raise ValueError("JE01_GOLD_SOURCE_PROMPT_BOUNDS_DRIFT")
        prompt_ends.append(start-1)
    if pos!=len(src):raise ValueError("JE01_GOLD_META_TRAILING_DATA")
    trace=path.read_bytes()
    if not trace.startswith(MAGIC):raise ValueError("JE01_TRUE_ROUTED_ACT_H2_FORMAT_HEADER_MISSING")
    raw=trace[len(MAGIC):]
    if not raw or len(raw)%RECORD:
        raise ValueError("JE01_NATIVE_H2_BINARY_RECORD_TRUNCATED")
    result=[];seen=set()
    for offset in range(0,len(raw),RECORD):
        case,layer,expert,last=struct.unpack_from("<4I",raw,offset)
        probability=struct.unpack_from("<f",raw,offset+16)[0]
        actual=np.frombuffer(raw,dtype="<f4",count=2048,offset=offset+20)
        if case>=16 or (layer,expert) not in FROZEN or last!=prompt_ends[case] or \
           (case,layer,expert) in seen or not 0<probability<=1 or \
           not np.isfinite(actual).all():
            raise ValueError("JE01_ACTUAL_ORIGINAL_H2_ROUTER_OR_PROMPT_MAPPING_WRONG")
        seen.add((case,layer,expert))
        energy=float(np.mean(actual.astype(np.float64)**2))
        result.append({"case":case,"layer":layer,"expert":expert,
            "source_prompt_last_pos":last,
            "actual_router_weight":float(probability),
            "actual_real_h2_l2_norm":float(np.sqrt(energy*2048)),
            "actual_real_h2_rms":float(np.sqrt(energy)),
            "actual_real_h2_max_abs":float(np.max(np.abs(actual))),
            "actual_h2_vector_sha256":hashlib.sha256(actual.tobytes()).hexdigest()})
    return result,len(trace)

def run():
    bin,manifest,goldmeta,lib,baseline=preflight()
    output=ROOT/"run"
    if output.exists() or output.is_symlink():raise ValueError("JE01_NATIVE_H2_RUN_ALREADY_EXISTS")
    output.mkdir(mode=0o700)
    temp=output/"tmp";temp.mkdir(mode=0o700)
    gold_out=output/"gold_native.bin"
    h2_out=output/"native_original_h2.bin"
    err=output/"stderr.txt";std=output/"stdout.txt"
    env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
       "HOME":"/Users/eoe","TMPDIR":str(temp),"DYLD_LIBRARY_PATH":str(lib),
       "QWEN_MOE_BASE":str(CHECKPOINT),"QWEN_MOE_GPU_CBATCH_ONLINE":"1",
       "QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"16",
       "QWEN_MOE_CB_PREFILL_BUDGET":"1","QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
       "QWEN_MOE_GPU_SORT_THRESHOLD":"64","QWEN_MOE_GPU_CBATCH_BUDGET":"0",
       "JE01_ORIGINAL_GOLD_META":str(goldmeta),"JE01_ORIGINAL_GOLD_OUTPUT":str(gold_out),
       "JE01_ROUTED_LAST_H2_OUT":str(h2_out)}
    low=memfree();start=time.monotonic();stop=None
    print(json.dumps({"event":"JE01_ORIGINAL27_REAL_ROUTED_H2_TRAIN16_START",
        "source_baseline_NLL":baseline["measured_native_EOE_mean_gold_NLL"],
        "expected_gold_output_SHA256":ORIGINAL16_OUTPUT_SHA,
        "binary_SHA256":BINARY_SHA,"heldout_labels_accessed":False},sort_keys=True),flush=True)
    with std.open("xb") as stdout,err.open("xb") as stderr:
        p=subprocess.Popen([str(bin)],cwd=str(output),env=env,stdout=stdout,stderr=stderr)
        while p.poll() is None:
            try:
                current=memfree();low=min(low,current)
                if current<35:stop="JE01_REAL_H2_MEMORY_FREE_BELOW35"
                if time.monotonic()-start>450:stop="JE01_REAL_H2_TIMEOUT450"
                status=subprocess.run(["/bin/ps","-o","rss=","-p",str(p.pid)],
                    capture_output=True,text=True,timeout=6).stdout.strip()
                if status.isdigit() and int(status)>16*1024*1024:
                    stop="JE01_REAL_H2_MODEL_RSS_GT16GB"
            except Exception as e:stop="JE01_H2_MEMORY_WATCHDOG_"+type(e).__name__
            if stop:p.terminate();break
            time.sleep(.8)
        if stop and p.poll() is None:
            try:p.wait(timeout=6)
            except subprocess.TimeoutExpired:p.kill()
        exitcode=p.wait()
    stderr=err.read_text(errors="replace")
    required="RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16"
    if exitcode!=0 or stop or "FATAL:" in stderr or required not in stderr or \
       "JE01_ORIGINAL_ACTUAL_ROUTED_H2_TRAIN16_RECORDS=" not in stderr:
        raise ValueError("JE01_REAL_MODEL_H2_CAPTURE_FAILED:"+str(stop)+"/"+str(exitcode)+":"+stderr[-1500:])
    if not gold_out.is_file() or sha(gold_out)!=ORIGINAL16_OUTPUT_SHA:
        raise ValueError("JE01_H2_INSTRUMENTED_MODEL_GOLD_OUTPUT_NOT_BITIDENTICAL_TO_REAL_ORIGINAL")
    records,nbytes=parse_trace(h2_out,goldmeta)
    if len(records)>128:raise ValueError("JE01_REAL16_SELECTED_ROUTE_COUNT_INVALID")
    groups=[]
    for l,e in sorted(FROZEN):
        matches=[x for x in records if (x["layer"],x["expert"])==(l,e)]
        groups.append({"layer":l,"expert":e,"routed_prompt_count":len(matches),
                       "empirical_actual_h2_rms_mean":float(np.mean([x["actual_real_h2_rms"] for x in matches])) if matches else None,
                       "mean_original_router_weight":float(np.mean([x["actual_router_weight"] for x in matches])) if matches else None})
    result={"schema":SCHEMA,"status":"JE01_ACTUAL_ORIGINAL27_B1_LASTPROMPT_ROUTED_H2_TRAIN16_TRACE_AND_GOLD_PARITY_PASS",
        "original_true_27layer_native_binary_sha256":BINARY_SHA,
        "original_AF4_checkpoint_sha256":AF_SHA,
        "actual_gold16_output_sha256":ORIGINAL16_OUTPUT_SHA,
        "full_teacherforced_gold_tokens":656,
        "actual_orig_gold_mean_NLL":baseline["measured_native_EOE_mean_gold_NLL"],
        "actual_routed_hidden_records":len(records),
        "actual_binary_trace_size_bytes":nbytes,
        "actual_binary_trace_SHA256":sha(h2_out),
        "actual_training_prompt_cases":16,
        "selected_eight_expert_slot_summary":groups,
        "per_record_training_prompt_h2_summary":records,
        "min_memory_free_percent":low,
        "elapsed_wall_seconds":round(time.monotonic()-start,3),
        "scientific_scope":"TRAIN16_TECHNICAL_H2_PROBE_NOT_HELDOUT_NOT_POSTERIOR_ADOPTION",
        "actual_weight_Fisher_Hessian":"NOT_MEASURED",
        "actual_activation_diagonal_covariance":"NOT_YET_CALCULATED",
        "source_true_256_training_activation_vectors":"NOT_MEASURED",
        "source_heldout_or_ood_labels_used":False,
        "production_weights_modified":False}
    result["receipt_sha256"]=canonical(result)
    with (output/"JE01_TRUE_ORIGINAL_ROUTED_H2_TRAIN16_RECEIPT.json").open("x") as f:
        f.write(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"receipt_sha256":result["receipt_sha256"],
        "actual_hidden_records":len(records),"trace_sha256":result["actual_binary_trace_SHA256"],
        "gold_bitwise_baseline_parity":True,"groups":groups},sort_keys=True),flush=True)
if __name__=="__main__":
    try:run()
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error_type":type(e).__name__,
            "reason":str(e)[:450]},sort_keys=True),flush=True)
        raise SystemExit(2)
