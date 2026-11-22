#!/usr/bin/env python3
"""JE01 original27 frozen CAL240 rolewise 12AF4/6Q4/6Q5 exact-byte TRAIN16 sanity gold NLL OFF/ON.

Strict source SHA, original EOE real 656-token baseline, 35% memory watchdog.
Only isolated D50 scratch 27-layer model; no production, no independent labels.
Run --mode off first; require EXACT original output SHA parity before rolewise ON. Pilot16 is exploratory; not a generalization claim.
"""
import argparse,hashlib,json,math,os,pathlib,re,struct,subprocess,time
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work/je01_eoe_realaf_nll_pilot16_20261122_v1")
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/je01_cal240_rolewise_native_stage_20261122_v1")
CHECKPOINT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/native_moe_base_source_copy")
BASELINE_RECEIPT="7dd5da25e500007a2a35fd724d82a7c8c6966727458763160f6d4ee7b3ac54ec"
CANDIDATE_SHA="db028b252d489861349259ec7cd08c87086e617d9f331b1de4a5f660bed79f8f"
CANDIDATE_FILE_SHA="ada43c5e233aa4154d37de5988aa22662ed16496363e84183a784c4c588a63ac"
CANDIDATE_SRC_STAGE="JE01_CAL240_ORIGINAL27_NATIVE_ROLEWISE_RESEARCH_STAGE_PASS"
AF_SHA="9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
INPUT_SHA="4678ae08006174ed1f472273fdca8b6667dce5ad930951211dbe4cc09e130993"
OLD_MODEL_SHA="2b9a665e17b4cbfed8112a5d3271e56365495a80049cd1f19af25098a8f46be7"
GOLD_MAGIC=b"JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1\n"
META_MAGIC=b"JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def canon(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def signed(p,expected):
    if p.is_symlink() or not p.is_file():raise ValueError("JE01_AUDIT_MISSING:"+p.name)
    d=json.loads(p.read_text())
    x=d.pop("receipt_sha256",None)
    if x!=expected or canon(d)!=expected:
        raise ValueError("JE01_REAL_SOURCE_RECEIPT_INVALID:"+p.name)
    return d
def memfree():
    d=subprocess.check_output(["/usr/bin/memory_pressure","-Q"],text=True,timeout=6)
    m=re.search(r"System-wide memory free percentage:\s*(\d+)%",d)
    if not m:raise ValueError("JE01_MEMORY_SENSOR_MISSING")
    return int(m.group(1))
def original_656(raw,meta):
    if not raw.startswith(GOLD_MAGIC):raise ValueError("JE01_PREVIOUS_EOE_REAL656_GOLD_HEADER_CHANGED")
    if not meta.startswith(META_MAGIC):raise ValueError("JE01_GOLD_META_HEADER_CHANGED")
    offs=len(META_MAGIC);records=[];lens=[];starts=[]
    for i in range(16):
        start,count=struct.unpack_from("<II",meta,offs);offs+=8
        gold=list(struct.unpack_from("<384i",meta,offs));offs+=1536
        if count<=0 or start<2 or start+count>384:
            raise ValueError("JE01_GOLD_CASE_BOUNDS_CHANGED")
        lens.append(gold[:count]);starts.append(start)
    if offs!=len(meta) or sum(map(len,lens))!=656:
        raise ValueError("JE01_ORIGINAL_GOLD655_OR_METADATA_LEN_CHANGED")
    pos=len(GOLD_MAGIC)
    for i in range(16):
        row=[]
        for t,actual_gold in enumerate(lens[i]):
            if pos+32>len(raw):raise ValueError("JE01_OUTPUT_GOLD_TRUNCATED")
            rr,index,gold,position=struct.unpack_from("<4I",raw,pos)
            nll,top1,logit=struct.unpack_from("<dIf",raw,pos+16)
            pos+=32
            if rr!=i or index!=t or gold!=actual_gold or position!=starts[i]+t-1 or \
               not math.isfinite(nll) or nll<0 or not math.isfinite(logit):
                raise ValueError("JE01_NATIVE_GOLD_ID_OR_LOGIT_INVALID")
            row.append(nll)
        records.append(row)
    if pos!=len(raw):raise ValueError("JE01_NATIVE_RECORD_TRAILING_BYTES")
    return records,lens,starts

def preflight(mode):
    if not ROOT.is_dir() or ROOT.is_symlink() or not BASE.is_dir() or BASE.is_symlink():
        raise ValueError("JE01_RESEARCH_D50_ROOT_INVALID")
    baseline=signed(BASE/"JE01_EOE_PILOT16_REAL_NLL_RESULT.json",BASELINE_RECEIPT)
    if baseline["status"]!="PASS_EOE_REAL_ORIGINAL_AF384_FULLGOLD_TRAIN16_NLL_PARITY" or \
       baseline["full_gold_tokens"]!=656 or baseline["real_cases"]!=16 or \
       baseline["original_AF_source_sha256"]!=AF_SHA or \
       baseline["native_eoe_arm64_binary_sha256"]!=OLD_MODEL_SHA or \
       baseline["NIG_candidate_precision_comparison"]!="NOT_MEASURED":
        raise ValueError("JE01_PREVIOUS_REAL_ORIGINAL27_AF_PARITY_INVALID")
    stage_file=ROOT/"JE01_CAL240_ROLEWISE_STAGE_RECEIPT.json"
    if not stage_file.is_file() or stage_file.is_symlink():
        raise ValueError("JE01_RESEARCH_ROLEWISE_STAGE_RECEIPT_MISSING")
    stage=json.loads(stage_file.read_text())
    claim=stage.pop("receipt_sha256",None)
    if claim!=canon(stage) or stage.get("status")!=CANDIDATE_SRC_STAGE or \
       stage.get("binary_sha256")!=CANDIDATE_SHA or \
       stage.get("bundle_sha256")!=CANDIDATE_FILE_SHA or \
       stage.get("physical_bytes")!=43253760 or \
       stage.get("role_tiers")!={"AF4":12,"Q4":6,"Q5":6} or \
       stage.get("changed_expert_overlays")!=7 or \
       stage.get("original_e64_passthrough_expert")!=[14,63] or \
       stage.get("original_AF4_production_modified") is not False or \
       stage.get("gold_NLL")!="NOT_RUN" or \
       stage.get("real_native_GPU_decode_parity")!="NOT_RUN" or \
       stage.get("github_actions_run_id")!=44995805607:
        raise ValueError("JE01_CAL240_ROLEWISE_STAGE_SOURCE_CONTRACT_INVALID")
    exe=pathlib.Path(stage["binary_path"])
    bundle=pathlib.Path(stage["bundle_path"])
    tsv=pathlib.Path(stage["tsv_path"])
    if exe.parent!=ROOT/"bin" or exe.name!="je01_eoe_original384_cal240_rolewise_fullgold" or \
       not exe.is_file() or exe.is_symlink() or sha(exe)!=CANDIDATE_SHA:
        raise ValueError("JE01_NATIVE_27L_BINARY_SHA_CHANGED")
    if bundle!=pathlib.Path("/Volumes/D50/TailnetCommander/work/je01_real_cal240_frozen_bundle_v3_20261122/JE01_CAL240_FROZEN_SSE_12AF4_6Q4_6Q5_EXACT43MB.bin") or \
       bundle.is_symlink() or not bundle.is_file() or bundle.stat().st_size!=43253760 or \
       sha(bundle)!=CANDIDATE_FILE_SHA or \
       tsv!=bundle.parent/"JE01_CAL240_FROZEN_SSE_24ROLE_SOURCE.tsv" or \
       tsv.is_symlink() or not tsv.is_file() or sha(tsv)!=stage["tsv_sha256"]:
        raise ValueError("JE01_CAL240_REAL_PHYSICAL_ROLE_BUNDLE_SHA_CHANGED")
    receipt=json.loads((bundle.parent/"JE01_CAL240_FROZEN_SSE_PHYSICAL_SOURCE_RECEIPT.json").read_text())
    seal=receipt.pop("receipt_sha256",None)
    if seal!="77d587e923283dc81d7c354ecc379298712f8a76868cb518208eecbe5a6df8b4" or \
       seal!=canon(receipt) or \
       receipt["physical_file_sha256"]!=CANDIDATE_FILE_SHA or \
       receipt["role_manifest_sha256"]!=stage["tsv_sha256"] or \
       receipt["first16_train_pilot_excluded"] is not True or \
       receipt["heldout_OOD_gold_accessed"] is not False or \
       receipt["production_modified"] is not False or \
       receipt["true_full_model_gold_NLL"]!="NOT_MEASURED":
        raise ValueError("JE01_FROZEN_REAL_CAL240_SOURCE_RECEIPT_INVALID")
    orig=CHECKPOINT/"weights_moe"/"deepseek_moe_af.bin"
    if orig.is_symlink() or not orig.is_file() or orig.stat().st_size!=9814343680 or sha(orig)!=AF_SHA:
        raise ValueError("JE01_ORIGINAL_AF_MODEL_SOURCE_NOT_EQUAL")
    oldstage=signed(BASE/"JE01_EOE_STAGED_PILOT16_PROVENANCE.json",
        "a814cdfd84f544132c4d9d8aa9e6e3c4845252e49a467d61c623e93931d3106e")
    if oldstage["original16_train_input_sha256"]!=INPUT_SHA:
        raise ValueError("JE01_SOURCE_NATIVE_TRAIN16_CHANGED")
    inputs=[p for p in (BASE/"inputs",BASE/"inputs"/"je01_real_pilot16")
            if (p/"JE01_EOE_ORIGINAL_PILOT16_INPUT_RECEIPT.json").is_file()]
    if len(inputs)!=1:raise ValueError("JE01_TRAIN16_SOURCE_INPUT_AMBIGUOUS")
    inp=inputs[0]
    goldmeta=inp/"chunk00"/"gold_meta.bin"
    manifest=BASE/"manifest.eoe.txt"
    if sha(goldmeta)!=oldstage["gold_meta_sha256"] or sha(manifest)!=oldstage["manifest_current_host_sha256"]:
        raise ValueError("JE01_GOLD_16_SOURCE_MANIFEST_CHANGED")
    reference=BASE/"gold_original_nll.bin"
    if sha(reference)!=baseline["native_output_sha256"]:
        raise ValueError("JE01_ORIGINAL_16_NATIVE_EOE_OUTPUT_SHA_CHANGED")
    ref,_,_=original_656(reference.read_bytes(),goldmeta.read_bytes())
    for i,row in enumerate(ref):
        if abs(math.fsum(row)-baseline["per_case"][i]["actual_EOE_native_gold_sum"])>1e-8:
            raise ValueError("JE01_SOURCE_BASELINE_CASE_NLL_RECORDS_CHANGED")
    import mlx,mlx.core as mx
    if mx.__version__!="0.32.1":raise ValueError("JE01_MLX_ABI_NO_LONGER_0_32_1")
    lib=pathlib.Path(list(mlx.__path__)[0])/"lib"
    if not (lib/"libmlx.dylib").is_file():raise ValueError("JE01_MLX_DYLIB_UNAVAILABLE")
    if memfree()<55:raise ValueError("JE01_AVAILABLE_MEMORY_BELOW55_PRELAUNCH")
    if mode=="rolewise":
        prev=ROOT/"off"/"JE01_CAL240_ROLEWISE_16_GOLD_NLL.json"
        if not prev.is_file() or prev.is_symlink():
            raise ValueError("JE01_NO_OVERLAY_MODE_NOT_EXECUTED")
        v=json.loads(prev.read_text())
        claim=v.pop("receipt_sha256",None)
        if claim!=canon(v) or v["status"]!="PASS_ORIGINAL27_CAL240_ROLEWISE_BINARY_OFF_GOLD_PARITY":
            raise ValueError("JE01_NO_OVERLAY_ORIGINAL_NLL_NOT_PASSED")
        if v.get("real_native_output_sha256")!=baseline["native_output_sha256"]:
            raise ValueError("JE01_NO_OVERLAY_ORIGINAL_GOLD_OUTPUT_SHA_NOT_EXACT")
    return exe,bundle,tsv,manifest,goldmeta,lib,ref

def run(mode):
    exe,bundle,tsv,manifest,goldmeta,lib,ref=preflight(mode)
    folder=ROOT/mode
    if folder.exists() or folder.is_symlink():raise ValueError("JE01_ALREADY_EXECUTED_MODE")
    folder.mkdir(mode=0o700)
    tmp=folder/"tmp";tmp.mkdir(mode=0o700)
    target=folder/"gold_native.bin"
    stdout=folder/"stdout.txt";stderr=folder/"stderr.txt"
    env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
         "HOME":"/Users/eoe","TMPDIR":str(tmp),
         "DYLD_LIBRARY_PATH":str(lib),
         "QWEN_MOE_BASE":str(CHECKPOINT),"QWEN_MOE_GPU_CBATCH_ONLINE":"1",
         "QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"16",
         "QWEN_MOE_CB_PREFILL_BUDGET":"1",
         "QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
         "QWEN_MOE_GPU_SORT_THRESHOLD":"64","QWEN_MOE_GPU_CBATCH_BUDGET":"0",
         "JE01_ORIGINAL_GOLD_META":str(goldmeta),
         "JE01_ORIGINAL_GOLD_OUTPUT":str(target)}
    if mode=="rolewise":
        # Legacy scratch research API name Q35; pinned SHA below is ACTUAL Q4/Q5.
        env["JE01_REAL_Q35_BUNDLE"]=str(bundle)
        env["JE01_REAL_Q35_TSV"]=str(tsv)
    started=time.monotonic();low=memfree();stop=None
    print(json.dumps({"event":"JE01_REAL_ORIGINAL27_CAL240_ROLEWISE_16GOLD_START",
                      "mode":mode,"cases":16,"gold_tokens":656,
                      "native_binary_sha256":CANDIDATE_SHA,
                      "source_real_affine_AF4_mixedq45_sha256":CANDIDATE_FILE_SHA,
                      "original_AF_weight_sha256":AF_SHA,
                      "initial_memory_free_percent":low}),flush=True)
    with stdout.open("xb") as out,stderr.open("xb") as err:
        p=subprocess.Popen([str(exe)],cwd=str(folder),env=env,
                          stdout=out,stderr=err)
        while p.poll() is None:
            try:
                free=memfree();low=min(low,free)
                if free<35:stop="JE01_MEMORY_FREE_BELOW35"
                if time.monotonic()-started>420:stop="JE01_REAL_ORIGINAL27_TIMEOUT420"
                proc=subprocess.run(["/bin/ps","-o","rss=","-p",str(p.pid)],
                          capture_output=True,text=True,timeout=5).stdout.strip()
                if proc.isdigit() and int(proc)>16*1024*1024:
                    stop="JE01_REAL_MODEL_RSS_GT16G"
            except Exception as e:stop="JE01_MEMORY_WATCHDOG_"+type(e).__name__
            if stop:p.terminate();break
            time.sleep(0.8)
        if stop and p.poll() is None:
            try:p.wait(timeout=6)
            except subprocess.TimeoutExpired:p.kill()
        code=p.wait()
    duration=time.monotonic()-started
    error=stderr.read_text(errors="replace")
    marker="JE01_CAL240_FROZEN_ROLEWISE_SEVEN_EXPERTS_REGISTERED"
    required="RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16"
    if code!=0 or stop or "FATAL:" in error or required not in error or \
       ((marker in error)!=(mode=="rolewise")):
        raise RuntimeError("JE01_REAL_ORIGINAL27_CAL240_ROLEWISE_NATIVE_FAILURE:"
               +mode+"/"+str(stop)+"/"+str(code)+"/stderr:"+error[-1500:])
    original=goldmeta.read_bytes()
    measured,_,_=original_656(target.read_bytes(),original)
    actual=[math.fsum(row) for row in measured]
    controls=[math.fsum(row) for row in ref]
    deltas=[(x-y) for x,y in zip(actual,controls)]
    value=math.fsum(actual)/656
    old=math.fsum(controls)/656
    if mode=="off" and (max(map(abs,deltas))>1e-10 or abs(value-old)>1e-10 or sha(target)!=sha(BASE/"gold_original_nll.bin")):
        raise ValueError("JE01_CANDIDATE_BINARY_OFF_ORIGINAL27_PARITY_FAILED")
    report={"schema":"beglin-je01-original27-real-CAL240-frozen-7expert-rolewise-fullgold16/1",
       "status":("PASS_ORIGINAL27_CAL240_ROLEWISE_BINARY_OFF_GOLD_PARITY"
                 if mode=="off" else "ACTUAL_REAL_ORIGINAL27_CAL240_ROLEWISE_PILOT16_MEASURED"),
       "mode":mode,"actual_model_complete_27layers":True,
       "native_7_experts_rebound_only":True,"original_1_Expert_E64_native_path_preserved":True,
       "scientific_scope":"CAL240_FROZEN_SELECTION_ALREADY_OBSERVED_TRAIN16_IS_EXPLORATORY_PARITY_ONLY_NO_HOLDOUT",
       "actual_preserved_original_affine_AF4_roles":12,"actual_q4_roles":6,"actual_q5_roles":6,
       "actual_original_AF_source_sha256":AF_SHA,
       "actual_compiled_CAL240_rolewise_native_model_sha256":CANDIDATE_SHA,
       "original_real_CAL240_rolewise_BF16_source_sha256":CANDIDATE_FILE_SHA,
       "gold_train_cases":16,"gold_tokens":656,
       "actual_native_mean_gold_NLL":value,
       "same_EOE_original_native_mean_gold_NLL":old,
       "paired_mean_NLL_candidate_minus_original":value-old,
       "paired_max_abs_percase_gold_NLL_sum_delta":max(map(abs,deltas)),
       "per_case":[{"row":i,"tokens":len(ref[i]),
                    "original_NLL_sum":controls[i],
                    "measured_mode_NLL_sum":actual[i],
                    "candidate_minus_original_NLL_sum":deltas[i]} for i in range(16)],
       "per_gold_target_actual_NLL":[[float(v) for v in row]for row in measured],
       "real_native_output_sha256":sha(target),
       "minimum_memory_free_percent":low,
       "actual_wall_seconds":round(duration,3),
       "true_source_heldout_ood_labels_accessed":False,
       "meets_preregistered_adoption_gate":"NOT_EVALUATED",
       "full256_calibration_NLL":"NOT_MEASURED_THIS_RUN",
       "real_OOD_and_same_source_128":"NOT_MEASURED",
       "production_checkpoint_modified":False}
    report["receipt_sha256"]=canon(report)
    with (folder/"JE01_CAL240_ROLEWISE_16_GOLD_NLL.json").open("x") as f:
        f.write(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("per_case","per_gold_target_actual_NLL")},
                      sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["off","rolewise"],required=True)
    a=p.parse_args()
    try:run(a.mode)
    except BaseException as e:
        print(json.dumps({"status":"JE01_FULL_MODEL_REAL_AFFINE_MIX4_FAIL_CLOSED",
                          "mode":a.mode,"type":type(e).__name__,
                          "reason":str(e)[:1000]},sort_keys=True),flush=True)
        raise SystemExit(2)
