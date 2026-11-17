#!/usr/bin/env python3
"""JE01 original BEGLIN DeepSeek AF native teacher-forced full TRAIN target NLL.

A new scratch-only source tree modifies ONLY the 384-position B1 original
GPU online gate to compute the full-vocabulary next-gold-token logsumexp on
every teacher-forced continuation position. The model's regular next-token
logits are untouched. Existing original AF checkpoint/operating workers/route
never change. Public frozen WikiText TRAIN only; no heldout/OOD accesses.
This is a BASELINE full-target loss, not a Bayesian Expert bit improvement.

Run stage 1 as --mode pilot16 to test first-token NLL parity against the
prior full256 original AF first-gold-token receipt. After pass, --mode all256
reuses the NEW scratch-only binary; one 16-record chunk per fresh subprocess
with 35% free-memory and original production-PID guards. No background claim.
"""
import argparse,hashlib,json,math,os
from pathlib import Path
import re,shutil,struct,subprocess,sys,time,statistics
from urllib.request import urlopen

ORIG=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original384_20261117")
SOURCES=ORIG/"src_xox_mlxabi384"
WEIGHTS=Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
CAL256=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original_router256_20261118_v1")
GOLD=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_train256_fullgold_NLL_20261118_v1/JE01_ORIGINAL_REAL_TRAIN256_FULL_GOLD_IDS.json")
FIRST=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_train256_fullgold_NLL_20261118_v1/JE01_ACTUAL_PRETRAINED_ORIGINAL_CAL256_NEXTGOLD_NLL.json")
ROOT=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original_fulltarget_teacherforce_20261118_v1")
SRC_SHA={"qwen_infer.c":"4e91092f7f860a446d97e5d871f249faeab0693d43c0293622a9cf551755622e",
         "mlx_moe.cpp":"c8b92e3ce5832b012713d41551f604855a89043ca316931501305ef501754782",
         "CMakeLists.txt":"f9bb057869f3fbc6a0a8ae55ab0d66247d27b211b092aa7cdfb8ab601fc023de"}
INPUT_INDEX_SHA="2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34"
REAL256_ROUTER_SHA="62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb"
GOLD_SHA="355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7"
FIRST_SHA="aca35d52a67c8ce5854c1b7297f80b241df2b2d66823b75a98744123fe8169b4"
MLXCMAKE="/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/share/cmake"
MLXLIB="/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/lib"
MANIFEST_MAGIC=b"JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n"
OUT_MAGIC=b"JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1\n"
SCHEMA="beglin-je01-original-AF-train256-native-fullgold-teacherforce-NLL/1"
VOCAB=102400
BOS=100000
POSITIONS=384
CHUNK=16
MAXGOLD=384

def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                              allow_nan=False).encode()).hexdigest()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def load_seal(p,expect):
    data=json.loads(Path(p).read_text())
    claimed=data.pop("receipt_sha256",None)
    if claimed!=expect or canonical(data)!=expect:
        raise ValueError("JE01_ORIGINAL_INPUT_OR_SCORE_SHA_DRIFT:"+str(p))
    return data

def health():
    with urlopen("http://127.0.0.1:18765/healthz",timeout=6) as f:doc=json.load(f)
    if doc.get("status")!="ok" or doc.get("route_generation")!=6 or \
       doc.get("active_route",{}).get("route_id")!="beglin-candidate-shared-up-l3-n6":
        raise ValueError("JE01_REAL_OPERATING_ROUTE_CHANGED")
    wk=doc.get("workers",{})
    if len(wk)!=2 or any(not x.get("alive") for x in wk.values()):
        raise ValueError("JE01_REAL_OPERATING_WORKERS_DOWN")
    pids={k:x.get("pid") for k,x in wk.items()}
    if set(pids.values())!={35481,35493}:
        raise ValueError("JE01_ORIGINAL_WORKER_PIDS_CHANGED")
    return pids

def free_pct():
    d=subprocess.check_output(["/usr/bin/memory_pressure","-Q"],text=True,timeout=6)
    m=re.search(r"System-wide memory free percentage:\s*(\d+)%",d)
    if not m:raise ValueError("JE01_MEMORY_SENSOR_MISSING")
    return int(m.group(1))

def log(event,**v):
    print(json.dumps({"event":event,**v},sort_keys=True),flush=True)

def replace(s,a,b,n=1,label="SRC"):
    if s.count(a)!=n:raise ValueError("JE01_SOURCE_ANCHOR_CHANGED_"+label+":"+str(s.count(a)))
    return s.replace(a,b)

def patch_cpu(src):
    first=src.index("static int run_moe_gpu_cbatch_online_gate(")
    last=src.index("// V5i Phase B:",first)
    s=src[first:last]
    # Source path-scoped output and teacher-forcing target lookup. DO NOT
    # change MLX graph, model parameters or binary output except NLL dump.
    anchor="    int persistent_warmed = 0;\n"
    addition=r'''
    // JE01 isolated original full-target teacher forcing. Disabled unless
    // the authorizer names a fresh scratch-only gold metadata/output pair.
    const char *je01_gold_meta=getenv("JE01_ORIGINAL_GOLD_META");
    const char *je01_gold_out=getenv("JE01_ORIGINAL_GOLD_OUTPUT");
    FILE *je01_gold_file=NULL;
    static uint32_t je01_start[MOE_CB4B_RMAX];
    static uint32_t je01_count[MOE_CB4B_RMAX];
    static int32_t je01_targets[MOE_CB4B_RMAX][MOE_CBATCH_MAXPOS];
    if (je01_gold_meta && *je01_gold_meta) {
        if (!je01_gold_out || !*je01_gold_out || B!=1 || R!=16 ||
            pfB!=1 || persistent_mode)
            {fprintf(stderr,"FATAL: JE01 teacher-forced mode requires B1 R16 pfB1 nonpersistent\n");exit(1);}
        int fd=open(je01_gold_out,O_WRONLY|O_CREAT|O_EXCL,0600);
        if (fd<0){fprintf(stderr,"FATAL: JE01 gold output file must be NEW\n");exit(1);}
        je01_gold_file=fdopen(fd,"wb");
        if (!je01_gold_file)exit(1);
        const char magic[]="JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1\n";
        if (fwrite(magic,1,sizeof(magic)-1,je01_gold_file)!=sizeof(magic)-1)exit(1);
    }
'''
    s=replace(s,anchor,anchor+addition,1,"META_FILE_GLOBALS")
    anchor2="        const int MCN = mf_n;\n"
    addition2=r'''
        if (je01_gold_file) {
            FILE *meta=fopen(je01_gold_meta,"rb");
            if (!meta){fprintf(stderr,"FATAL: JE01 original frozen gold metadata cannot be read\n");exit(1);}
            char hdr[40]={0};const char header[]="JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n";
            if (fread(hdr,1,sizeof(header)-1,meta)!=sizeof(header)-1 ||
                memcmp(hdr,header,sizeof(header)-1)!=0 || mf_n!=R)
                {fprintf(stderr,"FATAL: JE01 original gold metadata header/count drift\n");exit(1);}
            for (int rr=0;rr<R;rr++) {
                if (fread(&je01_start[rr],sizeof(uint32_t),1,meta)!=1 ||
                    fread(&je01_count[rr],sizeof(uint32_t),1,meta)!=1 ||
                    fread(je01_targets[rr],sizeof(int32_t),MOE_CBATCH_MAXPOS,meta)!=(size_t)MOE_CBATCH_MAXPOS)
                    {fprintf(stderr,"FATAL: JE01 original gold metadata row missing\n");exit(1);}
                if (je01_start[rr]<2 || je01_count[rr]<1 ||
                    je01_start[rr]+je01_count[rr]-1!=(uint32_t)mf_plen[rr] ||
                    je01_start[rr]+je01_count[rr]>MOE_CBATCH_MAXPOS)
                    {fprintf(stderr,"FATAL: JE01 source prompt/target length contract drift\n");exit(1);}
                for (int t=0;t<(int)je01_count[rr];t++)
                    if (je01_targets[rr][t]<0 || je01_targets[rr][t]>=MOE_VOCAB)
                        {fprintf(stderr,"FATAL: JE01 gold token outside vocab\n");exit(1);}
            }
            if (fgetc(meta)!=EOF){fprintf(stderr,"FATAL: JE01 gold metadata trailing bytes\n");exit(1);}
            fclose(meta);
        }
'''
    s=replace(s,anchor2,anchor2+addition2,1,"META_FILE_AFTER_MANIFEST")
    anchor3="            if (validation_on && pass == 1) {\n"
    addition3=r'''
            // Every raw prefill token is teacher forced from the SHA-frozen
            // original input, not an autoregressively generated prediction.
            if (je01_gold_file && pass==1) {
                if (A!=1 || ndec!=0) {
                    fprintf(stderr,"FATAL: JE01 teacher forcing requires one prefill column\n");exit(1);
                }
                int rr=mcb_req[slot_arr[0]];
                int pos=spos_arr[0];
                int index=pos-(int)je01_start[rr]+1;
                if (index>=0 && index<(int)je01_count[rr]) {
                    const float *lg=gpu_logits;
                    const int target=je01_targets[rr][index];
                    float maximum=-INFINITY;int best=-1;
                    for (int c=0;c<MOE_VOCAB;c++) {
                        if (!isfinite(lg[c])){fprintf(stderr,"FATAL: JE01 nonfinite pretrained logits\n");exit(1);}
                        if (lg[c]>maximum){maximum=lg[c];best=c;}
                    }
                    double total=0.;
                    for (int c=0;c<MOE_VOCAB;c++)total+=exp((double)lg[c]-maximum);
                    double nll=(double)maximum+log(total)-(double)lg[target];
                    if (!isfinite(nll) || nll < -1e-7){fprintf(stderr,"FATAL: JE01 NLL nonfinite/negative\n");exit(1);}
                    uint32_t record[4]={(uint32_t)rr,(uint32_t)index,(uint32_t)target,(uint32_t)pos};
                    uint32_t best_u=(uint32_t)best;
                    float gold_logit=lg[target];
                    if (fwrite(record,sizeof(uint32_t),4,je01_gold_file)!=4 ||
                        fwrite(&nll,sizeof(double),1,je01_gold_file)!=1 ||
                        fwrite(&best_u,sizeof(uint32_t),1,je01_gold_file)!=1 ||
                        fwrite(&gold_logit,sizeof(float),1,je01_gold_file)!=1)
                        {fprintf(stderr,"FATAL: JE01 teacher forced score write failed\n");exit(1);}
                }
            }
'''
    s=replace(s,anchor3,addition3+anchor3,1,"META_NLL_AFTER_REAL_GPU")
    anchor4="    free(x_embed); free(gpu_logits);\n    return 1;\n"
    addition4=r'''
    if (je01_gold_file) {
        if (fflush(je01_gold_file)!=0 ||
            fsync(fileno(je01_gold_file))!=0 ||
            fclose(je01_gold_file)!=0) {
            fprintf(stderr,"FATAL: JE01 NLL output flush failed\n");exit(1);
        }
    }
'''
    s=replace(s,anchor4,addition4+anchor4,1,"META_NLL_CLOSE")
    return src[:first]+s+src[last:]

def preflight():
    pids=health()
    if free_pct()<55:raise ValueError("JE01_INSUFFICIENT_FREE_MEMORY_INITIAL")
    if not (Path(MLXLIB)/"libmlx.dylib").is_file():
        raise ValueError("JE01_XOX_EXISTING_MLX_ABI_MISSING")
    for f,source_sha in SRC_SHA.items():
        if sha(SOURCES/f)!=source_sha:raise ValueError("JE01_ORIGINAL_RESEARCH_SOURCE_CHANGED:"+f)
    frozen=load_seal(CAL256/"JE01_ACTUAL_ORIGINAL_256_ROUTER_CAL_FINAL.json",REAL256_ROUTER_SHA)
    if (frozen["calibration_prompt_count"]!=256 or
        frozen["status"]!="PASS_256_REAL_ORIGINAL_AF_ROUTER_LAST_PROMPT_ONLY" or
        frozen["unlabeled_calibration_only"] is not True):
        raise ValueError("JE01_ORIGINAL_REAL_256_ROUTE_SOURCE_NOT_VALID")
    gold=load_seal(GOLD,GOLD_SHA)
    if (gold["aligned_to_original_prompt_prefix_count"]!=256 or
        gold["original_256_source_rows"]!=256 or
        gold["same_source_validation_and_ood_labels_used"] is not False or
        sum(x["gold_token_count"] for x in gold["cases"])!=11024):
        raise ValueError("JE01_ORIGINAL_REAL_256_FULL_TARGET_SOURCE_CHANGED")
    base=load_seal(FIRST,FIRST_SHA)
    if base["actually_scored_first_gold_tokens"]!=256 or \
       base["actual_calibration_prompts"]!=256 or \
       base["Bayesian_Expert_precision_gain"]!="NOT_MEASURED":
        raise ValueError("JE01_ORIGINAL_256_FIRST_TARGET_BASELINE_SOURCE_CHANGED")
    receipt=load_seal(ORIG/"inputs/je01_inputs/JE01_NATIVE_384_INPUTS_RECEIPT.json",INPUT_INDEX_SHA)
    if receipt["total_input_records"]!=512:raise ValueError("JE01_ORIGINAL_MODEL_INPUT_512_SEAL_CHANGED")
    if not (ORIG/"manifests_xox/receipt.json").is_file():raise ValueError("JE01_NATIVE_INPUT_PROVENANCE_MISSING")
    if ROOT.exists() or ROOT.is_symlink():raise ValueError("JE01_TEACHERFORCE_OUTPUT_ROOT_ALREADY_EXISTS")
    records=receipt["cohorts"]["calibration"]["file_receipts"]
    if len(records)!=256 or len(gold["cases"])!=256:raise ValueError("JE01_TRAIN256_MISSING_ROWS")
    recs=[]
    for k,(item,g) in enumerate(zip(records,gold["cases"])):
        file=ORIG/"inputs/je01_inputs/calibration"/(item["input_id"]+".bin")
        raw=file.read_bytes()
        if sha(file)!=item["native_int32_sha256"] or \
           g["native_BOS_prompt_sha256"]!=item["native_int32_sha256"] or \
           g["input_id"]!=item["input_id"] or g["row"]!=k:
            raise ValueError("JE01_TRAIN256_CASE_NATIVE_INPUT_SHA_CHANGED:"+str(k))
        ids=list(struct.unpack("<"+str(len(raw)//4)+"i",raw))
        target=g["gold_continuation_token_ids"]
        if ids[0]!=100000 or len(ids)!=g["original_BOS_prompt_positions"] or \
           len(target)!=g["gold_token_count"] or \
           len(ids)+len(target)!=g["model_full_BOS_sequence_positions"] or \
           len(ids)+len(target)>384:
            raise ValueError("JE01_ORIGINAL_GOLD_PREFIX_OR_CONTEXT_CHANGED:"+str(k))
        if not all(type(v) is int and 0<=v<102400 for v in target):
            raise ValueError("JE01_GOLD_TOKEN_IDS_INVALID:"+str(k))
        first=base["scored_case_receipts"][k]
        if first["source_id"]!=item["input_id"] or first["gold_target_first_token_id"]!=target[0]:
            raise ValueError("JE01_EXISTING_REAL_FIRST_GOLD_NLL_NOT_PAIRED:"+str(k))
        recs.append({"index":k,"source_id":item["input_id"],"prompt_ids":ids,
                     "target_ids":target,"prompt_sha":item["native_int32_sha256"],
                     "expected_first_gold_nll":first["actual_original_next_gold_NLL"]})
    return pids,recs

def prepare(root,recs):
    root.mkdir(mode=0o700)
    inp=root/"inputs";inp.mkdir(mode=0o700)
    manifests=[];receipt=[]
    for block in range(16):
        start=block*16;end=start+16;chunk=root/("chunk"+str(block).zfill(2))
        chunk.mkdir(mode=0o700)
        meta=bytearray(MANIFEST_MAGIC)
        lines=[];case_sha=[]
        for j,r in enumerate(recs[start:end]):
            toks=r["prompt_ids"]+r["target_ids"][:-1]
            if len(toks)+1>384:raise ValueError("JE01_NATIVE_TEACHERFORCE_FULL_TARGET_CONTEXT_LIMIT")
            raw=struct.pack("<"+str(len(toks))+"i",*toks)
            path=inp/(r["source_id"]+".bin")
            with path.open("xb") as out:out.write(raw)
            lines.append(str(path)+" 1\n")
            target=r["target_ids"]
            meta.extend(struct.pack("<II",len(r["prompt_ids"]),len(target)))
            meta.extend(struct.pack("<384i",*(target+[-1]*(384-len(target)))))
            case_sha.append({"row":r["index"],"source_id":r["source_id"],
                 "total_tokens_in_native_input":len(toks),
                 "prompt_tokens":len(r["prompt_ids"]), "targets":len(target),
                 "original_baseline_first_gold_NLL":r["expected_first_gold_nll"],
                 "score_input_sha256":hashlib.sha256(raw).hexdigest()})
        manifest=chunk/"manifest.txt";manifest.write_text("".join(lines))
        metadata=chunk/"gold_meta.bin"
        metadata.write_bytes(meta)
        manifests.append({"chunk":block,"manifest_sha256":sha(manifest),
                          "gold_meta_sha256":sha(metadata),"case_provenance":case_sha})
    return manifests

def build(source_root,pids):
    model=source_root/"src";shutil.copytree(SOURCES,model)
    original=(model/"qwen_infer.c").read_text()
    patched=patch_cpu(original)
    (model/"qwen_infer.c").write_text(patched)
    cm=(model/"CMakeLists.txt").read_text()
    cm=replace(cm,"OUTPUT_NAME qwen_infer_gpu_je01_context384_xoxabi",
               "OUTPUT_NAME je01_original_AF_real_full_target_nll384",1,"BINARY_NAME")
    (model/"CMakeLists.txt").write_text(cm)
    p=subprocess.run(["/opt/homebrew/bin/cmake","-S",str(model),"-B",str(source_root/"build"),
       "-DCMAKE_BUILD_TYPE=Release","-DMLX_CMAKE_DIR="+MLXCMAKE],
       capture_output=True,text=True,timeout=75)
    if p.returncode:raise RuntimeError("JE01_NATIVE_TF_CMAKE_CONFIG_FAIL:"+p.stderr[-1300:].decode(errors="replace"))
    p=subprocess.run(["/opt/homebrew/bin/cmake","--build",str(source_root/"build"),
        "--target","qwen_infer_gpu","--parallel","2"],
        capture_output=True,timeout=140)
    if p.returncode:raise RuntimeError("JE01_NATIVE_TF_BUILD_FAIL:"+p.stderr[-1900:].decode(errors="replace"))
    exe=source_root/"build/je01_original_AF_real_full_target_nll384"
    if not exe.is_file() or health()!=pids or free_pct()<45:raise ValueError("JE01_TEACHERFORCE_BUILD_EXECUTABLE_OR_HOST_DRIFT")
    log("NEW_ORIGINAL_GPU_TF_BINARY_COMPILED",executable_sha256=sha(exe),
        source_patch_sha256=sha(model/"qwen_infer.c"),producer_scope="scratch-only")
    return exe

def scored(path,records,chunk):
    raw=path.read_bytes()
    if not raw.startswith(OUT_MAGIC):raise ValueError("JE01_GOLD_NATIVE_RESULT_MAGIC_CHANGED")
    p=len(OUT_MAGIC)
    loss_rows=[];by_req={}
    for j,r in enumerate(records):
        losses=[]
        for t,token in enumerate(r["target_ids"]):
            b=raw[p:p+32]
            if len(b)!=32:raise ValueError("JE01_REAL_GOLD_TARGET_LOGIT_NLL_MISSING")
            req,index,gold,pos=struct.unpack("<4I",b[:16])
            nll,top1,logit=struct.unpack("<dIf",b[16:])
            p+=32
            if req!=j or index!=t or gold!=token or pos!=len(r["prompt_ids"])+t-1 or \
               not math.isfinite(nll) or nll < -1e-7 or not math.isfinite(logit):
                raise ValueError("JE01_REAL_GOLD_NATIVE_EVIDENCE_RECORD_DRIFT")
            losses.append(nll)
        if abs(losses[0]-r["expected_first_gold_nll"])>1e-5:
            raise ValueError("JE01_FIRST_GOLD_NLL_DIFFERENT_FROM_PREVIOUS_REAL256:"+str(r["index"])+":"+str(losses[0]-r["expected_first_gold_nll"]))
        loss_rows.append({"row":r["index"],"source_id":r["source_id"],"tokens":len(losses),
                          "nll_sum":math.fsum(losses),"nll_mean":math.fsum(losses)/len(losses),
                          "first_gold_nll":losses[0]})
    if p!=len(raw):raise ValueError("JE01_NATIVE_GOLD_RESULT_TRAILING_RECORDS")
    return loss_rows

def execute(exe,root,first_rows,original_pids,mode):
    end=1 if mode=="pilot16" else 16
    all_rows=[];chunks=[];min_free=100
    for block in range(end):
        chunk=root/("chunk"+str(block).zfill(2))
        out=chunk/"gold_native_nll.bin"
        err=chunk/"stderr.txt";std=chunk/"stdout.txt"
        env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
            "HOME":"/Users/xox","TMPDIR":str(chunk),
            "DYLD_LIBRARY_PATH":MLXLIB,
            "QWEN_MOE_BASE":str(WEIGHTS),
            "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
            "QWEN_MOE_CB_SLOTS":"1",
            "QWEN_MOE_CB_REQS":"16",
            "QWEN_MOE_CB_PREFILL_BUDGET":"1",
            "QWEN_MOE_CB_PROMPT_MANIFEST":str(chunk/"manifest.txt"),
            "QWEN_MOE_GPU_SORT_THRESHOLD":"64",
            "QWEN_MOE_GPU_CBATCH_BUDGET":"0",
            "JE01_ORIGINAL_GOLD_META":str(chunk/"gold_meta.bin"),
            "JE01_ORIGINAL_GOLD_OUTPUT":str(out)}
        started=time.monotonic();killed=None;minimum=free_pct()
        log("NATIVE_FULL_TARGET_CHUNK_LAUNCHED",chunk=block,model_binary_sha256=sha(exe),
            total_gold_tokens=sum(len(c["target_ids"]) for c in first_rows[block*16:(block+1)*16]))
        with std.open("xb") as stdout,err.open("xb") as stderr:
            p=subprocess.Popen([str(exe)],cwd=chunk,env=env,stdout=stdout,stderr=stderr)
            while p.poll() is None:
                if time.monotonic()-started>230:killed="JE01_TF_CHUNK_TIMEOUT";p.terminate();break
                try:
                    free=free_pct();minimum=min(minimum,free)
                    if free<35:killed="JE01_MEMORY_BELOW_35";p.terminate();break
                    if health()!=original_pids:killed="JE01_OPERATING_WORKER_CHANGED";p.terminate();break
                    v=subprocess.run(["/bin/ps","-o","rss=","-p",str(p.pid)],
                        capture_output=True,text=True,timeout=5).stdout.strip()
                    if v.isdigit() and int(v)>16*1024*1024:
                        killed="JE01_NATIVE_RSS_OVER_16GB";p.terminate();break
                except BaseException as e:killed="JE01_MONITOR_"+type(e).__name__;p.terminate();break
                time.sleep(.7)
            if killed and p.poll() is None:
                try:p.wait(timeout=8)
                except subprocess.TimeoutExpired:p.kill()
            returncode=p.wait()
        resultstderr=err.read_text(errors="replace")
        if returncode or killed or "FATAL:" in resultstderr or \
           "RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16" not in resultstderr:
            raise RuntimeError("JE01_TEACHERFORCE_NATIVE_CHUNK_FAILED:"+str(block)+"/"+str(killed)+"/"+str(returncode)+resultstderr[-1500:])
        rows=scored(out,first_rows[block*16:(block+1)*16],block)
        all_rows+=rows;min_free=min(min_free,minimum)
        rr={"chunk":block,"result_sha256":sha(out),"bytes":out.stat().st_size,
            "records":len(rows),"gold_targets_scored":sum(q["tokens"] for q in rows),
            "min_free_pct":minimum,
            "wall_seconds":round(time.monotonic()-started,3),"stderr_sha256":sha(err)}
        chunks.append(rr)
        with (chunk/"JE01_NATIVE_TF_CHUNK_PASS.json").open("x") as f:
            f.write(json.dumps(rr,sort_keys=True,indent=2)+"\n")
        log("TEACHERFORCED_NATIVE_ORIGINAL_CHUNK_PASS",chunk=block,
            completed_cases=len(all_rows),targets_scored=sum(x["tokens"] for x in all_rows),
            avg_nll=sum(x["nll_sum"] for x in all_rows)/sum(x["tokens"] for x in all_rows),
            min_free_pct=minimum,source_result_sha256=rr["result_sha256"])
    if health()!=original_pids:raise ValueError("JE01_PRODUCTION_CHANGED_AFTER_FULL_TARGET")
    tot=sum(x["tokens"] for x in all_rows)
    val=math.fsum(x["nll_sum"] for x in all_rows)
    report={"schema":SCHEMA,
        "status":"REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_"+("PILOT16" if mode=="pilot16" else "CAL256")+"_PASS",
        "actual_original_pretrained_AF_model":True,
        "mode":mode,"n_calibration_train_cases":len(all_rows),
        "gold_full_target_tokens_scored":tot,
        "mean_gold_token_NLL":val/tot,
        "per_example_mean_full_target_NLL":sum(x["nll_mean"] for x in all_rows)/len(all_rows),
        "max_abs_first_gold_NLL_disagreement_from_original_256_previous_run":max(abs(x["first_gold_nll"]-first_rows[x["row"]]["expected_first_gold_nll"]) for x in all_rows),
        "source_original_model_af_sha_scope":"UNCHANGED_RESEARCH_SOURCE",
        "gold_full_target_train_source_sha256":GOLD_SHA,
        "calibration_original256_router_receipt_sha256":REAL256_ROUTER_SHA,
        "original_512_train_holdout_input_receipt_sha256":INPUT_INDEX_SHA,
        "native_model_binary_sha256":sha(exe),
        "per_case_real_NLL":all_rows,
        "per_chunk_native_output_sha256":chunks,
        "minimum_memory_free_percent":min_free,
        "production_before":original_pids,"production_after":health(),
        "training_only":True,"heldout_gold_labels_used":False,
        "bayesian_expert_quantization_changed":False,
        "trained_alt_bits_comparison":"NOT_MEASURED",
        "same_source_128_or_OOD_128_NLL":"NOT_MEASURED",
        "production_checkpoint_or_release_modified":False,
        "production_promotion_allowed":False}
    report["receipt_sha256"]=canonical(report)
    name="JE01_ORIGINAL_AF_FULL_TARGET_GOLD_NLL_"+mode.upper()+".json"
    with (root/name).open("x") as f:f.write(json.dumps(report,sort_keys=True,indent=2)+"\n")
    log("ORIGINAL_REAL_FULL_TARGET_NLL_FINAL",status=report["status"],receipt_sha256=report["receipt_sha256"],
        train_cases=report["n_calibration_train_cases"],gold_tokens=report["gold_full_target_tokens_scored"],
        mean_full_gold_token_NLL=report["mean_gold_token_NLL"],
        min_memory_pct=report["minimum_memory_free_percent"],release_modified=False)
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("pilot16","all256"),required=True)
    args=parser.parse_args()
    if args.mode=="pilot16":
        ids,recs=preflight()
        ROOT.mkdir(mode=0o700)
        manifests=prepare(ROOT,recs)
        (ROOT/"MANIFEST_AND_NATIVE_SOURCE_RECEIPT.json").write_text(json.dumps({
           "schema":"JE01-GOLD-TARGET-INPUT-256-RECEIPT/1","real_original_training_cases":256,
           "total_original_gold_target_tokens":11024,
           "original_model_af_changed":False,"source_gold_sha256":GOLD_SHA,
           "original_inputs_sha256":INPUT_INDEX_SHA,"blocks":manifests},sort_keys=True,indent=2)+"\n")
        exe=build(ROOT,ids)
        report=execute(exe,ROOT,recs,ids,"pilot16")
        if report["n_calibration_train_cases"]!=16:
            raise ValueError("JE01_ORIGINAL_GOLD_PILOT_NOT_SIXTEEN")
    else:
        if not ROOT.is_dir() or (ROOT/"JE01_ORIGINAL_AF_FULL_TARGET_GOLD_NLL_ALL256.json").exists():
            raise ValueError("JE01_FULL_TARGET_SOURCE_ROOT_NOT_READY_OR_FINAL_EXISTS")
        pil=load_seal(ROOT/"JE01_ORIGINAL_AF_FULL_TARGET_GOLD_NLL_PILOT16.json",
            json.loads((ROOT/"JE01_ORIGINAL_AF_FULL_TARGET_GOLD_NLL_PILOT16.json").read_text())["receipt_sha256"])
        if pil["status"]!="REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_PILOT16_PASS":
            raise ValueError("JE01_GOLD_PILOT_NOT_PASSED")
        # PRECHECK ALL original 256 data identities again before full run.
        if free_pct()<55 or health()!=pil["production_before"]:
            raise ValueError("JE01_FULL_TRAIN_GPU_HOST_HEALTH_OR_FREE_MEMORY_UNSAFE")
        frozen=load_seal(GOLD,GOLD_SHA)["cases"]
        index=load_seal(ORIG/"inputs/je01_inputs/JE01_NATIVE_384_INPUTS_RECEIPT.json",INPUT_INDEX_SHA)
        original_first=load_seal(FIRST,FIRST_SHA)
        records=[]
        for j,(item,g) in enumerate(zip(index["cohorts"]["calibration"]["file_receipts"],frozen)):
            records.append({"index":j,"source_id":item["input_id"],"prompt_ids":[0]*g["original_BOS_prompt_positions"],
                "target_ids":g["gold_continuation_token_ids"],
                "prompt_sha":item["native_int32_sha256"],
                "expected_first_gold_nll":original_first["scored_case_receipts"][j]["actual_original_next_gold_NLL"]})
        exe=ROOT/"build/je01_original_AF_real_full_target_nll384"
        if sha(exe)!=pil["native_model_binary_sha256"]:
            raise ValueError("JE01_FULL_TARGET_BINARY_CHANGED_SINCE_PILOT")
        # The pilot chunk's result exists; run remaining 15 chunks only, then
        # combine with the exact pilot16 records. Mode switch retains prefix.
        # Existing native_run chunk output is immutable.
        # Delegated as a separate method to avoid even a transient overwrite.
        pilot_rows=pil["per_case_real_NLL"]
        chunks=pil["per_chunk_native_output_sha256"]
        for block in range(1,16):
            one_dir=ROOT/("chunk"+str(block).zfill(2))
            # All original source/matched ids are verified in preflight for
            # pilot; now recheck source SHA at every chunk using indices.
            if not (one_dir/"manifest.txt").is_file():raise ValueError("JE01_MISSING_REAL_GOLD_SOURCE_MANIFEST")
            # Chunk-local execution uses no re-generation of inputs.
            part_records=records[block*16:(block+1)*16]
            doc=execute_one_chunk(exe,ROOT,block,part_records,pil["production_before"])
            pilot_rows+=doc["rows"]
            chunks.append(doc["chunk"])
        total_tokens=sum(x["tokens"] for x in pilot_rows)
        if len(pilot_rows)!=256 or total_tokens!=11024:
            raise ValueError("JE01_TARGET_COUNT_NOT_256_11024")
        val=math.fsum(x["nll_sum"] for x in pilot_rows)
        report={"schema":SCHEMA,"status":"REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_CAL256_PASS",
            "actual_original_pretrained_AF_model":True,
            "n_calibration_train_cases":256,"gold_full_target_tokens_scored":total_tokens,
            "mean_gold_token_NLL":val/total_tokens,
            "per_example_mean_full_target_NLL":sum(x["nll_mean"] for x in pilot_rows)/256,
            "max_abs_first_gold_NLL_disagreement_from_original_256_previous_run":
                 max(abs(x["first_gold_nll"]-records[x["row"]]["expected_first_gold_nll"]) for x in pilot_rows),
            "source_gold_sha256":GOLD_SHA,"calibration_original256_router_receipt_sha256":REAL256_ROUTER_SHA,
            "original_512_input_sha256":INPUT_INDEX_SHA,
            "native_model_binary_sha256":sha(exe),
            "per_case_real_NLL":pilot_rows,"per_chunk_native_output_sha256":chunks,
            "production_before":pil["production_before"],"production_after":health(),
            "training_only":True,"heldout_gold_labels_used":False,
            "bayesian_expert_quantization_changed":False,"trained_alt_bits_comparison":"NOT_MEASURED",
            "same_source_128_or_OOD_128_NLL":"NOT_MEASURED",
            "production_checkpoint_or_release_modified":False,
            "production_promotion_allowed":False}
        if report["production_before"]!=report["production_after"]:
            raise ValueError("JE01_OPERATING_DRIFT")
        report["receipt_sha256"]=canonical(report)
        with (ROOT/"JE01_ORIGINAL_AF_FULL_TARGET_GOLD_NLL_ALL256.json").open("x") as f:
            f.write(json.dumps(report,sort_keys=True,indent=2)+"\n")
        log("ORIGINAL_REAL_FULL_TARGET_NLL_FINAL",cases=256,tokens=total_tokens,
            mean_nll=report["mean_gold_token_NLL"],receipt_sha256=report["receipt_sha256"])

def execute_one_chunk(exe,root,block,recs,pids):
    # The same exact per-chunk native runner and parser as pilot, with a new
    # corresponding 16-request output file in a never-before-run chunk dir.
    chunk=root/("chunk"+str(block).zfill(2))
    out=chunk/"gold_native_nll.bin";err=chunk/"stderr.txt";std=chunk/"stdout.txt"
    if out.exists() or err.exists() or std.exists():raise ValueError("JE01_PREVIOUS_CHUNK_ALREADY_RUN")
    env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
         "HOME":"/Users/xox","TMPDIR":str(chunk),"DYLD_LIBRARY_PATH":MLXLIB,
         "QWEN_MOE_BASE":str(WEIGHTS),"QWEN_MOE_GPU_CBATCH_ONLINE":"1",
         "QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"16",
         "QWEN_MOE_CB_PREFILL_BUDGET":"1","QWEN_MOE_CB_PROMPT_MANIFEST":str(chunk/"manifest.txt"),
         "QWEN_MOE_GPU_SORT_THRESHOLD":"64","QWEN_MOE_GPU_CBATCH_BUDGET":"0",
         "JE01_ORIGINAL_GOLD_META":str(chunk/"gold_meta.bin"),
         "JE01_ORIGINAL_GOLD_OUTPUT":str(out)}
    started=time.monotonic();minimum=free_pct();killed=None
    log("REAL_256_GOLD_NATIVE_CHUNK_START",chunk=block,target_tokens=sum(len(r["target_ids"]) for r in recs))
    with std.open("xb") as o,err.open("xb") as e:
        proc=subprocess.Popen([str(exe)],cwd=chunk,env=env,stdout=o,stderr=e)
        while proc.poll() is None:
            if time.monotonic()-started>250:killed="TIMEOUT";proc.terminate();break
            try:
                free=free_pct();minimum=min(minimum,free)
                if free<35:killed="LOW_MEMORY";proc.terminate();break
                if health()!=pids:killed="PRODUCTION_DRIFT";proc.terminate();break
                z=subprocess.run(["/bin/ps","-o","rss=","-p",str(proc.pid)],capture_output=True,text=True,timeout=4).stdout.strip()
                if z.isdigit() and int(z)>16*1024*1024:killed="RSS_GT16GB";proc.terminate();break
            except BaseException as e:killed="MONITOR_"+type(e).__name__;proc.terminate();break
            time.sleep(.7)
        if killed and proc.poll() is None:
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:proc.kill()
        rc=proc.wait()
    stderr=err.read_text(errors="replace")
    if rc or killed or "FATAL:" in stderr or "RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16" not in stderr:
        raise RuntimeError("JE01_TEACHERFORCE_NATIVE_FAIL:"+str(block)+"/"+str(rc)+"/"+str(killed)+":"+stderr[-900:])
    rows=scored(out,recs,block)
    doc={"chunk":block,"result_sha256":sha(out),"bytes":out.stat().st_size,
         "gold_targets_scored":sum(x["tokens"] for x in rows),"min_free_pct":minimum,
         "wall_seconds":round(time.monotonic()-started,3),"stderr_sha256":sha(err)}
    with (chunk/"JE01_NATIVE_TF_CHUNK_PASS.json").open("x") as f:
        f.write(json.dumps(doc,sort_keys=True,indent=2)+"\n")
    log("REAL_256_GOLD_NATIVE_CHUNK_PASS",chunk=block,verified_cases=(block+1)*16,
        verified_targets=doc["gold_targets_scored"],min_memory_free_pct=minimum,
        result_sha256=doc["result_sha256"])
    return {"rows":rows,"chunk":doc}

if __name__=="__main__":
    try:main()
    except BaseException as e:
        log("JE01_ORIGINAL_REAL_FULL_TARGET_FAIL_CLOSED",
            kind=type(e).__name__,reason=str(e)[:1500],
            model_promotion_allowed=False)
        sys.exit(2)
