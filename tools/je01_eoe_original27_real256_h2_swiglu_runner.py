#!/usr/bin/env python3
"""JE01 real original27 CAL256 2048d h2 plus 1408d SwiGLU capture.

Only 256 immutable original WikiText TRAIN native prompt files from the
previously SHA-sealed 512-input source, on the original 27-layer pretrained AF4
Metal binary with scratch-only hidden trace. Exactly 16 independent B1/16case
jobs; no gold labels, no heldout or OOD, no weight/serving changes. Requires
parity of first 16 real h2 vectors versus prior original16 full-gold source
before running subsequent 240 calibration prompts.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,struct,subprocess,time
import numpy as np
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/je01_real_original256_activation_20261122_v1")
WORK=pathlib.Path("/Volumes/D50/TailnetCommander/work")
SRC=ROOT/"portable512"/"je01_inputs"
MNF=ROOT/"manifests"/"calibration.manifest"
EXE=ROOT/"bin"/"je01_eoe_real_original27_cal256_h2_swiglu"
BUILD=ROOT/"bin"/"JE01_REAL256_H2_SW_NATIVE_BUILD_RECEIPT.json"
NATIVE_SHA="77fd53111cf8316c9d9186f21ace65bccaf3c6fb1aba54c43c698076d5dc914d"
CHECKPOINT=WORK/"hcp-p18-external-multimetric-20261015/native_moe_base_source_copy"
AF_SHA="9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
SOURCE512_SHA="2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34"
ROUTER256_SHA="62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb"
PRIOR_H2_SHA="238642649f68d111df3b60f60083a47d3a6074ada881ccb36f813dd606f9a5b4"
PRIOR_H2_RECORD_MAGIC=b"JE01_ACTUAL_REAL_AF4_TRAIN16_ROUTED_H2_V1\n"
MAGIC=b"JE01_REAL_ORIGINAL_AF4_CAL256_ACTUAL_H2_SWIGLU_V1\n"
RECORD=16+4+(2048+1408)*4
OLD_RECORD=16+4+2048*4
FROZEN={(12,13),(13,22),(13,28),(14,63),(20,17),(21,61),(24,26),(24,60)}
REAL16=WORK/"je01_eoe_original27_h2_trace16_20261122_v2"/"run"/"native_original_h2.bin"
SCHEMA="beglin-je01-original27-256-train-routed-H2-and-SwiGLU-real-vectors/1"

def sha_file(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def memoryfree():
    t=subprocess.check_output(["/usr/bin/memory_pressure","-Q"],timeout=8,text=True)
    m=re.search(r"System-wide memory free percentage:\s*(\d+)%",t)
    if not m:raise ValueError("JE01_CAL256_ACTUAL_MEMORY_SENSOR_INVALID")
    return int(m.group(1))
def sealed(path,expected):
    if not path.is_file() or path.is_symlink():raise ValueError("JE01_FROZEN_SOURCE_MISSING:"+path.name)
    d=json.loads(path.read_text());sig=d.pop("receipt_sha256",None)
    if sig!=expected or canonical(d)!=sig:
        raise ValueError("JE01_ACTUAL_SOURCE_SHA_RECEIPT_INVALID:"+path.name)
    return d
def parse_block(raw,magic,size,global_first):
    if not raw.startswith(magic) or len(raw)<len(magic) or (len(raw)-len(magic))%size:
        raise ValueError("JE01_REAL_MODEL_ACTIVATION_RECORD_FORMAT_DRIFT")
    rows=[]
    for offset in range(len(magic),len(raw),size):
        case,l,e,pos=struct.unpack_from("<4I",raw,offset)
        prob=struct.unpack_from("<f",raw,offset+16)[0]
        h2=np.frombuffer(raw,dtype="<f4",count=2048,offset=offset+20)
        sw=(np.frombuffer(raw,dtype="<f4",count=1408,offset=offset+8212) if size==RECORD else None)
        if case>=16 or (l,e) not in FROZEN or not 0<prob<=1 or \
           not np.isfinite(h2).all() or (sw is not None and not np.isfinite(sw).all()):
            raise ValueError("JE01_TRUE_ORIGINAL_H2_SW_ROUTING_VALUES_INVALID")
        rows.append({"global_case":global_first+case,"layer":l,"expert":e,"pos":pos,
                     "prob":float(prob),"h2":h2,"sw":sw,
                     "h2_digest":hashlib.sha256(h2.tobytes()).hexdigest(),
                     "sw_digest":hashlib.sha256(sw.tobytes()).hexdigest() if sw is not None else None})
    if len(rows)!=len({(v["global_case"],v["layer"],v["expert"]) for v in rows}):
        raise ValueError("JE01_REAL_CAL256_H2_ONE_ROUTE_PER_SELECTED_EXPERT_ONLY")
    return rows
def preflight():
    if ROOT.is_symlink() or not ROOT.is_dir() or MNF.is_symlink() or not MNF.is_file() or \
       EXE.is_symlink() or not EXE.is_file() or sha_file(EXE)!=NATIVE_SHA:
        raise ValueError("JE01_REAL256_ORIGINAL_NATIVE_SOURCE_NOT_READY")
    b=json.loads(BUILD.read_text())
    if b["status"]!="JE01_TRUE_ORIGINAL27_CAL256_H2_AND_SWIGLU_APPLE_METAL_COMPILED" or \
       b["native_binary_sha256"]!=NATIVE_SHA or \
       b["source_patch"]["status"]!="JE01_ORIGINAL27_ACTUAL_CAL256_TRAIN_ONLY_H2_SWIGLU_SCRATCH_PATCHED" or \
       b["operating_model_modified"]:
        raise ValueError("JE01_REAL256_MLX_SCRATCH_BUILD_RECEIPT_DRIFT")
    original=sealed(SRC/"JE01_NATIVE_384_INPUTS_RECEIPT.json",SOURCE512_SHA)
    moved=ROOT/"manifests"/"JE01_RELOCATION_PASS.json"
    d=json.loads(moved.read_text());sig=d.pop("receipt_sha256",None)
    if sig!=canonical(d) or d["status"]!="RELOCATED_512_REAL_NATIVE_FILES_MANIFESTS_PASS" or \
       d["source_receipt_sha256"]!=SOURCE512_SHA or d["files"]!=512 or \
       d["original_data_modified"] or d["production_promotion_allowed"]:
        raise ValueError("JE01_REAL512_ORIGINAL_NATIVE_RELOCATION_SHA_CHANGED")
    source_router=sealed(ROOT/"source"/"JE01_ROUTER256.json",ROUTER256_SHA)
    if source_router["calibration_prompt_count"]!=256 or \
       source_router["observed_route_events"]!=6656 or \
       source_router["observed_routed_expert_ids"]!=39936 or \
       source_router["unlabeled_calibration_only"] is not True:
        raise ValueError("JE01_ACTUAL_ORIGINAL_ROUTER256_SOURCE_DRIFT")
    counts={(x["layer"],x["expert"]):x["real_original_calibration_routed_calls"]
            for x in source_router["exactly_eight_pilot_slots_frozen_by_unlabeled_calibration_last_token_routing"]}
    if set(counts)!=FROZEN or sum(counts.values())!=682:
        raise ValueError("JE01_ROUTER256_EXACT_FROZEN_LASTTOKEN_COUNT_DRIFT")
    entries=original["cohorts"]["calibration"]["file_receipts"]
    lines=MNF.read_text().splitlines()
    if len(lines)!=256 or len(entries)!=256:
        raise ValueError("JE01_ORIGINAL256_TRAIN_MANIFEST_COUNT_DRIFT")
    prompts=[]
    for i,(line,rec) in enumerate(zip(lines,entries)):
        path,one=line.rsplit(" ",1)
        f=pathlib.Path(path)
        if one!="1" or f.is_symlink() or not f.is_file() or \
           f.parent!=SRC/"calibration" or f.name!=rec["input_id"]+".bin":
            raise ValueError("JE01_CAL256_UNMODIFIED_INPUT_MANIFEST_ID_DRIFT")
        raw=f.read_bytes()
        if sha_file(f)!=rec["native_int32_sha256"] or len(raw)!=rec["native_int32_bytes"]:
            raise ValueError("JE01_SOURCE_FROZEN_TRAIN256_RAW_TOKEN_SHA_DRIFT")
        tokens=struct.unpack("<"+"i"*(len(raw)//4),raw)
        if tokens[0]!=100000 or len(tokens)!=rec["token_count_with_bos"] or not 2<=len(tokens)<384 or \
           any(x<0 or x>=102400 for x in tokens):
            raise ValueError("JE01_TRUE_ORIGINAL_CAL256_NATIVE_TOKEN_IDS_CHANGED")
        prompts.append((str(f),len(tokens)-1))
    file=CHECKPOINT/"weights_moe"/"deepseek_moe_af.bin"
    if not file.is_file() or file.is_symlink() or file.stat().st_size!=9814343680 or \
       sha_file(file)!=AF_SHA:raise ValueError("JE01_REAL_ORIGINAL_PRETRAINED_9GB_AF4_SHA_DRIFT")
    if REAL16.is_symlink() or not REAL16.is_file() or sha_file(REAL16)!=PRIOR_H2_SHA:
        raise ValueError("JE01_TRUE_PRIOR_GOLD16_H2_REFERENCE_SHA_CHANGED")
    import mlx,mlx.core as mx
    if mx.__version__!="0.32.1":raise ValueError("JE01_REAL_NATIVE_MLX_ABI_DRIFT")
    lib=pathlib.Path(list(mlx.__path__)[0])/"lib"
    if not (lib/"libmlx.dylib").is_file():raise ValueError("JE01_REAL256_NATIVE_MLX_DLL_MISSING")
    if memoryfree()<55:raise ValueError("JE01_REAL256_MEM_FREE_BELOW55")
    return prompts,counts,lib
def execute():
    prompts,expected,lib=preflight()
    stage=ROOT/"run256"
    if stage.is_symlink() or stage.exists():
        raise ValueError("JE01_TRUE_REAL256_OUTPUT_ROOT_ALREADY_EXISTS")
    stage.mkdir(mode=0o700)
    prior=parse_block(REAL16.read_bytes(),PRIOR_H2_RECORD_MAGIC,OLD_RECORD,0)
    ref={(x["global_case"],x["layer"],x["expert"]):x for x in prior}
    if len(ref)!=45:raise ValueError("JE01_REAL_PREVIOUS_H2_TRAIN16_RECORD_COUNT_NOT45")
    all_counts={key:0 for key in FROZEN}
    chunk_evidence=[]; max_first_h2=0; max_first_prob=0; max_first_position=0
    for block in range(16):
        root=stage/("chunk"+str(block).zfill(2))
        root.mkdir(mode=0o700)
        section=prompts[16*block:16*(block+1)]
        manifest=root/"train16.manifest"
        manifest.write_text("".join(f"{path} 1\n" for path,_ in section))
        out=root/"actual_h2_swiglu.bin"
        stdout=root/"stdout.txt";stderr=root/"stderr.txt"
        env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
             "HOME":"/Users/eoe","DYLD_LIBRARY_PATH":str(lib),
             "TMPDIR":str(root),
             "QWEN_MOE_BASE":str(CHECKPOINT),
             "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
             "QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"16",
             "QWEN_MOE_CB_PREFILL_BUDGET":"1",
             "QWEN_MOE_GPU_SORT_THRESHOLD":"64",
             "QWEN_MOE_GPU_CBATCH_BUDGET":"0",
             "QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
             "JE01_ROUTED_LAST_H2_OUT":str(out)}
        if memoryfree()<55:raise ValueError("JE01_REAL256_SUBJOB_MEMORY_BELOW55")
        started=time.monotonic();low=memoryfree();stop=None
        with stdout.open("xb") as o,stderr.open("xb") as e:
            proc=subprocess.Popen([str(EXE)],cwd=str(root),env=env,stdout=o,stderr=e)
            while proc.poll() is None:
                try:
                    free=memoryfree();low=min(low,free)
                    if free<35:stop="JE01_REAL256_MEMORY_FREE_BELOW35"
                    if time.monotonic()-started>180:stop="JE01_REAL256_CHUNK_TIMEOUT180"
                    rss=subprocess.run(["/bin/ps","-o","rss=","-p",str(proc.pid)],
                         capture_output=True,text=True,timeout=5).stdout.strip()
                    if rss.isdigit() and int(rss)>16*1024*1024:
                        stop="JE01_REAL256_NATIVE_RSS_OVER16G"
                except Exception as ex:stop="JE01_REAL256_WATCHDOG_"+type(ex).__name__
                if stop:proc.terminate();break
                time.sleep(1)
            if stop and proc.poll() is None:
                try:proc.wait(timeout=6)
                except subprocess.TimeoutExpired:proc.kill()
            ret=proc.wait()
        err=stderr.read_text(errors="replace")
        if ret or stop or "FATAL:" in err or \
           "RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16" not in err or \
           "JE01_ORIGINAL_ACTUAL_ROUTED_H2_TRAIN16_RECORDS=" not in err:
            raise RuntimeError("JE01_CAL256_TRUE_ORIGINAL_NATIVE_FORWARD_FAILURE:"+
                               str(block)+":"+str(stop)+":"+str(ret)+":"+err[-900:])
        raw=out.read_bytes()
        local=parse_block(raw,MAGIC,RECORD,block*16)
        if len(local)<1:raise ValueError("JE01_REAL256_ORIGINAL_H2_EMPTY_CHUNK")
        for r in local:
            key=r["layer"],r["expert"]
            all_counts[key]+=1
            idx=r["global_case"]%16
            if r["pos"]!=section[idx][1]:
                raise ValueError("JE01_TRUE_ORIGINAL_LAST_PROMPT_POSITION_CHANGED")
        if block==0:
            current={(x["global_case"],x["layer"],x["expert"]):x for x in local}
            if set(current)!=set(ref):
                raise ValueError("JE01_NATIVE_CAL256_FIRST16_REAL_ROUTER_NOT_IDENTICAL_TO_OLD16")
            for key,expected_h2 in ref.items():
                observed=current[key]
                max_first_h2=max(max_first_h2,float(np.max(np.abs(observed["h2"]-expected_h2["h2"]))))
                max_first_prob=max(max_first_prob,abs(observed["prob"]-expected_h2["prob"]))
                max_first_position=max(max_first_position,abs(observed["pos"]-expected_h2["pos"]))
            if max_first_h2>2e-5 or max_first_prob>2e-6 or max_first_position:
                raise ValueError("JE01_REAL256_FIRST16_H2_PRIOR_GOLD16_PARITY_FAILED")
        entry={"chunk":block,"source_case_first":block*16,"cases":16,
               "real_routed_selected_records":len(local),
               "real_binary_sha256":hashlib.sha256(raw).hexdigest(),
               "binary_file_bytes":len(raw),"original_int32_manifest_sha256":sha_file(manifest),
               "min_memory_free_percent":low,
               "native_wall_seconds":time.monotonic()-started,
               "excluded_heldout_and_ood_labels":True}
        entry["receipt_sha256"]=canonical(entry)
        (root/"JE01_CAL256_ACT_H2_SW_CHUNK_RECEIPT.json").write_text(
               json.dumps(entry,sort_keys=True,indent=2,allow_nan=False)+"\n")
        chunk_evidence.append(entry)
        print(json.dumps({"chunk":block,"routed_records":len(local),"cumulative_records":sum(all_counts.values()),
             "gold16_h2_parity_maxabs":max_first_h2 if block==0 else None,
             "training_only":True},sort_keys=True),flush=True)
    if all_counts!=expected or sum(all_counts.values())!=682:
        raise ValueError("JE01_REAL256_ACTUAL_LASTTOKEN_EXPERT_COUNTS_DISAGREE_ORIGINAL_ROUTER256")
    result={"schema":SCHEMA,"status":"JE01_TRUE_ORIGINAL27_256TRAIN_682ROUTED_H2_SWIGLU_MODEL_FORWARD_PASS",
       "source_original_512_int32_receipt_sha256":SOURCE512_SHA,
       "source_original_router256_receipt_sha256":ROUTER256_SHA,
       "original_27layer_MLX_native_binary_sha256":NATIVE_SHA,
       "original_AF4_full_checkpoint_sha256":AF_SHA,
       "selected_eight_frozen_experts":[list(k) for k in sorted(FROZEN)],
       "per_expert_original_router256_expected_count":{f"{l}:{e}":c for (l,e),c in sorted(expected.items())},
       "per_expert_actual_routed_last_prompt_count":{f"{l}:{e}":c for (l,e),c in sorted(all_counts.items())},
       "total_unique_original_training_prompts":256,"true_model_inference_chunks":16,
       "actual_original_routed_hidden_and_swiglu_records":sum(all_counts.values()),
       "native_original_FFN_input_H2_dimension":2048,"native_original_FFN_down_SwiGLU_dimension":1408,
       "original_true_gold16_reference_hidden_trace_sha256":PRIOR_H2_SHA,
       "real16_control_max_abs_h2":max_first_h2,
       "real16_control_max_abs_router_weight":max_first_prob,
       "gold_token_loss_gradient":"NOT_MEASURED","true_Fisher":"NOT_MEASURED",
       "true_Hessian":"NOT_MEASURED","gold_full256_NLL":"NOT_MEASURED_THIS_RUN",
       "independent_holdout_OOD_gold_labels_accessed":False,
       "production_modified":False,"per_chunk":chunk_evidence}
    result["receipt_sha256"]=canonical(result)
    (stage/"JE01_ORIGINAL27_CAL256_TRUE_H2_SWIGLU_RECEIPT.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"receipt":result["receipt_sha256"],
          "source256":256,"observations":682,"h2_dim":2048,"down_sw_dim":1408,
          "first16_parity_maxabs":max_first_h2},sort_keys=True),flush=True)
if __name__=="__main__":
    try:execute()
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","type":type(e).__name__,
            "reason":str(e)[:750]},sort_keys=True),flush=True)
        raise SystemExit(2)
