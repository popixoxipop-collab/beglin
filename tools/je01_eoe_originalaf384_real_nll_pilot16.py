#!/usr/bin/env python3
"""JE01 REAL original 27-layer pretrained AF model 384-position TRAIN16 NLL.

Executes only as Fleet-approved resource_submit job on EOE D50 with prior
real native ARM64 MLX0.32.1 and 656 gold tokens SHA staged in NEW scratch.
Pins original five AF source files, executable and original input receipts;
kills on memory free <35%, RSS>16GiB, or hard wall deadline.
Does NOT compare or promote NIG/q3q5. Never accesses heldout/OOD labels.
"""
import hashlib,json,math,os,pathlib,re,struct,subprocess,sys,time
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/je01_eoe_realaf_nll_pilot16_20261122_v1")
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015/native_moe_base_source_copy")
GOLD_REF=json.loads(r'''[{"source_id":"1238f877c10f039ed6e9d4fca0f5dfa1c515cc70a2f069765e9426b14e0474c1","tokens":23,"first_gold_nll":7.1120428959585205,"nll_sum":49.16774326183457},{"source_id":"db037f9d1c936fa3e13ae8667bf4aa74f6fe17b952686277edd5950ade7a3c9d","tokens":50,"first_gold_nll":4.039534692098496,"nll_sum":123.234992588518},{"source_id":"61fc48ad81ab3c00a2307c5f0b92c0de4e59201b540fe4127aec10215c2278f3","tokens":35,"first_gold_nll":4.8272450708785435,"nll_sum":60.13843405971973},{"source_id":"b817e623fb45033a3383adf2689e06054f0e62bacdeff14c4fba01049e3c8281","tokens":31,"first_gold_nll":0.004194551516448541,"nll_sum":19.71960302574572},{"source_id":"bd267b8bef31882bbc39d44b3bb87f87c47633c8528db5180805fe1cbd91c175","tokens":39,"first_gold_nll":1.1658431908199312,"nll_sum":117.97686035892602},{"source_id":"be10ace1de627d1fdab75013523e91a09672fc2877d6d09a21bfe68b47fc427e","tokens":50,"first_gold_nll":3.033621097068824,"nll_sum":142.07529803051588},{"source_id":"9d9671cd0433525a28dc697e5b7e43ba114301ad281a410e497f5b4f5b4784aa","tokens":42,"first_gold_nll":3.5131328103581083,"nll_sum":90.98023812664215},{"source_id":"b1395890313d1db56d5baf55ac2a61017bc39ed932a47d4c44b29f8c1b187709","tokens":34,"first_gold_nll":6.588435158175212,"nll_sum":94.62393182300669},{"source_id":"21485cd7a42a7b0180b995aad00a7598f45138c497663311aba9197d5d0d6ece","tokens":44,"first_gold_nll":0.11989905268655221,"nll_sum":78.60769912414074},{"source_id":"f651c4275e584ff21f47ad2a702d88a349ae1bfb62a22df6539562c66603395b","tokens":64,"first_gold_nll":1.558261415577121,"nll_sum":151.2537156835404},{"source_id":"ebe4268484169ccb5a2dbaea9bf49ceaa1c07f246cf0c3f1e1cd07f90f467274","tokens":28,"first_gold_nll":5.917331420467427,"nll_sum":79.5989917779307},{"source_id":"a0a1ca143d329e60afdaf18bb5d93dccc3d6759008e33195dbd1c6f53a165526","tokens":54,"first_gold_nll":7.1191251047781705,"nll_sum":116.2275376776223},{"source_id":"86575fad179792e10eda968cbddcf5645377ff90508debeb0381e3133410d009","tokens":66,"first_gold_nll":4.796408203901365,"nll_sum":180.95090642692654},{"source_id":"56626b7dd8366e88dc85e4b14e384487c14a1029bf1bc9d0f3c00d311bf1291e","tokens":24,"first_gold_nll":0.8466158053825765,"nll_sum":54.10806250154663},{"source_id":"938d483fad002fe5dde65024b1e8fac903f9b79d3a209e49603065b6344780c8","tokens":15,"first_gold_nll":0.00878762376329334,"nll_sum":30.006509076826617},{"source_id":"7f8f6013fb5386cbba9c11b47238482dc29b185a3d23cdd6ae65cb1538f5def0","tokens":57,"first_gold_nll":3.4032788231491082,"nll_sum":114.7500812009826}]''')
EXPECTED_MEAN=2.291799702354307
BIN_SHA="2b9a665e17b4cbfed8112a5d3271e56365495a80049cd1f19af25098a8f46be7"
STAGE_SHA="a814cdfd84f544132c4d9d8aa9e6e3c4845252e49a467d61c623e93931d3106e"
INPUT_SHA="4678ae08006174ed1f472273fdca8b6667dce5ad930951211dbe4cc09e130993"
FILE_SHA={"deepseek_moe_af.bin":"9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4",
"deepseek_moe_f32.bin":"1cc5386c8b8773ee802e059be22f0b7b317dacfc3c9bbf0b724b4e411b596033",
"layout_af.txt":"ba3898f3bf366b1aec8471d9896885f101bf64b78c816828ea82bd7d58851c3e",
"layout_f32.txt":"2ecc5fcb2b4d26302fdd26e9f7d9167811b431a5649d467fa5e5912dbb5a5e4c",
"arch_config_moe.txt":"43f205631553478b204f761d5894aaeec67100ed276aca762a24c62e63a91c50"}
META=b"JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n"
MAGIC=b"JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1\n"

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda:f.read(1<<20),b""):h.update(piece)
    return h.hexdigest()
def canonical(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def memory():
    x=subprocess.check_output(["/usr/bin/memory_pressure","-Q"],text=True,timeout=6)
    m=re.search(r"System-wide memory free percentage:\s*(\d+)%",x)
    if not m:raise RuntimeError("JE01_REAL_MEMORY_SENSOR_MISSING")
    return int(m.group(1))
def ensure():
    if not ROOT.is_dir() or ROOT.is_symlink() or not BASE.is_dir():
        raise ValueError("JE01_MODEL_SCRATCH_DIRECTORY_UNAVAILABLE")
    source=ROOT/"JE01_EOE_STAGED_PILOT16_PROVENANCE.json"
    d=json.loads(source.read_text())
    sig=d.pop("receipt_sha256",None)
    if sig!=STAGE_SHA or canonical(d)!=STAGE_SHA or \
       d["original16_train_input_sha256"]!=INPUT_SHA or \
       d["original_native_binary_sha256"]!=BIN_SHA:
        raise ValueError("JE01_STAGED_SOURCE_FULL_SHA_CHANGED")
    exe=ROOT/"bin"/"je01_eoe_original384_fullgold"
    if not exe.is_file() or exe.is_symlink() or sha(exe)!=BIN_SHA:
        raise ValueError("JE01_NATIVE_ARM64_MODEL_EXE_SHA_NOT_PINNED")
    for name,expected in FILE_SHA.items():
        f=BASE/"weights_moe"/name
        if not f.is_file() or f.is_symlink() or sha(f)!=expected:
            raise ValueError("JE01_SOURCE_REAL_PRETRAINED_AF_SHA_DRIFT:"+name)
    dirs=[p for p in (ROOT/"inputs",ROOT/"inputs"/"je01_real_pilot16")
          if (p/"JE01_EOE_ORIGINAL_PILOT16_INPUT_RECEIPT.json").is_file()]
    if len(dirs)!=1:raise ValueError("JE01_REAL16_INPUT_DIR_AMBIGUOUS")
    iroot=dirs[0]
    input_receipt=json.loads((iroot/"JE01_EOE_ORIGINAL_PILOT16_INPUT_RECEIPT.json").read_text())
    isig=input_receipt.pop("receipt_sha256",None)
    if isig!=INPUT_SHA or canonical(input_receipt)!=INPUT_SHA:
        raise ValueError("JE01_PILOT16_ORIGINAL_INPUT_SHA_CHANGED")
    manifest=ROOT/"manifest.eoe.txt"
    if sha(manifest)!=d["manifest_current_host_sha256"]:
        raise ValueError("JE01_ORIGINAL_MANIFEST_SHA_CHANGED")
    lines=manifest.read_text().splitlines()
    if len(lines)!=16:raise ValueError("JE01_ORIGINAL_TRAIN16_MANIFEST_ROW_COUNT")
    for item,line in zip(input_receipt["pilot_evidence"],lines):
        filepart,count=line.rsplit(" ",1)
        f=pathlib.Path(filepart)
        if count!="1" or f!=iroot/"inputs"/(item["input_id"]+".bin") or \
           not f.is_file() or f.is_symlink() or sha(f)!=item["raw_teacherforce_sha256"]:
            raise ValueError("JE01_ORIGINAL_TF_INPUT_ROW_SHA_CHANGED")
    meta=iroot/"chunk00"/"gold_meta.bin"
    raw=meta.read_bytes()
    if not raw.startswith(META) or sha(meta)!=d["gold_meta_sha256"]:
        raise ValueError("JE01_REAL_ORIGINAL_GOLD_META_SHA_DRIFT")
    off=len(META);targets=[];starts=[]
    for j in range(16):
        start,n=struct.unpack_from("<II",raw,off);off+=8
        gold=list(struct.unpack_from("<384i",raw,off));off+=384*4
        if n!=GOLD_REF[j]["tokens"] or start<2 or \
           not all(0<=k<102400 for k in gold[:n]):
            raise ValueError("JE01_ORIGINAL_GOLD_SOURCE_ALIGN_"+str(j))
        targets.append(gold[:n]);starts.append(start)
    if off!=len(raw):raise ValueError("JE01_REAL_ORIGINAL_GOLD_METADATA_TAIL")
    import mlx,mlx.core as mx
    if mx.__version__!="0.32.1":raise ValueError("JE01_RUNTIME_MLX_ABI_VERSION_DRIFT")
    lib=pathlib.Path(list(mlx.__path__)[0])/"lib"
    if not (lib/"libmlx.dylib").is_file():raise ValueError("JE01_RUNTIME_MLX_LIB_MISSING")
    if memory()<55:raise ValueError("JE01_INITIAL_MEMORY_FREE_BELOW55")
    return exe,lib,manifest,meta,starts,targets

def run():
    exe,lib,manifest,meta,starts,targets=ensure()
    raw_out=ROOT/"gold_original_nll.bin"
    std=ROOT/"run_stdout.txt"
    err=ROOT/"run_stderr.txt"
    receipt=ROOT/"JE01_EOE_PILOT16_REAL_NLL_RESULT.json"
    tmp=ROOT/"tmp"
    if any(p.exists() for p in (raw_out,std,err,receipt,tmp)):
        raise ValueError("JE01_REAL_MODEL_RUN_PREVIOUS_OUTPUT_PRESENT")
    tmp.mkdir(mode=0o700)
    env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
         "HOME":"/Users/eoe","TMPDIR":str(tmp),"DYLD_LIBRARY_PATH":str(lib),
         "QWEN_MOE_BASE":str(BASE),"QWEN_MOE_GPU_CBATCH_ONLINE":"1",
         "QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"16",
         "QWEN_MOE_CB_PREFILL_BUDGET":"1",
         "QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
         "QWEN_MOE_GPU_SORT_THRESHOLD":"64","QWEN_MOE_GPU_CBATCH_BUDGET":"0",
         "JE01_ORIGINAL_GOLD_META":str(meta),
         "JE01_ORIGINAL_GOLD_OUTPUT":str(raw_out)}
    print(json.dumps({"event":"JE01_REAL_EOE_ORIGINAL_AF384_START",
          "original_AF_sha256":FILE_SHA["deepseek_moe_af.bin"],
          "compiled_exe_sha256":BIN_SHA,"gold_tokens":656,
          "original_pilot16_XOX_mean_NLL":EXPECTED_MEAN,
          "memory_free_initial_pct":memory()}),flush=True)
    began=time.monotonic()
    lowest=memory();stop=None
    with std.open("xb") as stdout,err.open("xb") as stderr:
        p=subprocess.Popen([str(exe)],cwd=str(ROOT),env=env,
                           stdout=stdout,stderr=stderr)
        while p.poll() is None:
            try:
                free=memory();lowest=min(lowest,free)
                if free<35:stop="JE01_MEMORY_FREE_BELOW35"
                if time.monotonic()-began>420:stop="JE01_REAL_MODEL_EXEC_OVER420S"
                raw=subprocess.run(["/bin/ps","-o","rss=","-p",str(p.pid)],
                                   capture_output=True,text=True,timeout=5).stdout.strip()
                if raw.isdigit() and int(raw)>16*1024*1024:
                    stop="JE01_ORIGINAL_MODEL_RSS_OVER16G"
            except Exception as ex:
                stop="JE01_WATCHDOG_EXCEPTION:"+type(ex).__name__
            if stop:
                p.terminate();break
            time.sleep(.8)
        if stop and p.poll() is None:
            try:p.wait(timeout=6)
            except subprocess.TimeoutExpired:p.kill()
        rc=p.wait()
    duration=time.monotonic()-began
    stderr=err.read_text(errors="replace")
    if rc!=0 or stop or "FATAL:" in stderr or \
       "RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16" not in stderr:
        raise RuntimeError("JE01_ORIGINAL_PRETRAINED_NATIVE_FORWARD_FAIL:"
                +str(stop)+"/EXIT"+str(rc)+"/STDERR:"+stderr[-1000:])
    b=raw_out.read_bytes()
    if not b.startswith(MAGIC):raise ValueError("JE01_ORIGINAL_FULLGOLD_BINARY_MAGIC_MISMATCH")
    offset=len(MAGIC);records=[];tokens=0;total=0.0
    for i,expect in enumerate(GOLD_REF):
        losses=[]
        for t,gold in enumerate(targets[i]):
            row=b[offset:offset+32]
            if len(row)!=32:raise ValueError("JE01_NATIVE_GOLD_OUTPUT_TRUNCATED")
            req,index,actual,pos=struct.unpack_from("<4I",row)
            nll,best,logit=struct.unpack_from("<dIf",row,16)
            offset+=32
            if req!=i or index!=t or actual!=gold or \
               pos!=starts[i]+t-1 or not math.isfinite(nll) or \
               nll<-1e-7 or not math.isfinite(logit):
                raise ValueError("JE01_ORIGINAL_GOLD_TOKEN_NLL_ID_MISMATCH")
            losses.append(nll)
        diff=sum(losses)-expect["nll_sum"]
        if abs(diff)>0.005 or abs(losses[0]-expect["first_gold_nll"])>1e-4:
            raise ValueError("JE01_REAL_EOE_VS_XOX_ORIGINAL_GOLD_NLL_PARITY_FAIL:"
                             +str(i)+":"+str(diff))
        total+=sum(losses);tokens+=len(losses)
        records.append({"source_id":expect["source_id"],"row":i,
             "gold_tokens":len(losses),"actual_EOE_native_gold_sum":sum(losses),
             "original_XOX_native_gold_sum":expect["nll_sum"],
             "delta":diff})
    if offset!=len(b) or tokens!=656:
        raise ValueError("JE01_NATIVE_GOLD_RECORD_COUNT_OR_TAIL_DRIFT")
    actual=total/tokens
    if abs(actual-EXPECTED_MEAN)>0.005:
        raise ValueError("JE01_NATIVE_EOE_GOLD_PILOT_MEAN_NLL_DRIFT")
    result={"schema":"beglin-je01-eoe-original-AF384-native-train16-fullgold/1",
      "status":"PASS_EOE_REAL_ORIGINAL_AF384_FULLGOLD_TRAIN16_NLL_PARITY",
      "original_AF_source_sha256":FILE_SHA["deepseek_moe_af.bin"],
      "native_eoe_arm64_binary_sha256":BIN_SHA,
      "real_cases":16,"full_gold_tokens":tokens,
      "measured_native_EOE_mean_gold_NLL":actual,
      "prior_XOX_original_mean_gold_NLL":EXPECTED_MEAN,
      "max_abs_per_case_gold_sum_diff":max(abs(x["delta"]) for x in records),
      "per_case":records,"native_output_sha256":sha(raw_out),
      "minimum_memory_free_percent":lowest,
      "elapsed_seconds":round(duration,3),
      "original_model_full_27layer_inference":True,
      "training_only":True,"heldout_or_ood_gold_used":False,
      "NIG_candidate_precision_comparison":"NOT_MEASURED",
      "production_checkpoint_changed":False}
    result["receipt_sha256"]=canonical(result)
    with receipt.open("x") as f:
        f.write(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="per_case"},sort_keys=True),flush=True)

if __name__=="__main__":
    try:run()
    except BaseException as ex:
        print(json.dumps({"status":"JE01_REAL_AF384_FULLGOLD_FAIL_CLOSED",
                          "error_type":type(ex).__name__,
                          "reason":str(ex)[:900]},sort_keys=True),flush=True)
        raise SystemExit(2)
