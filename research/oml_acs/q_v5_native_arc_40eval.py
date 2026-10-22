#!/usr/bin/env python3
"""Real native BEGLIN 28-layer Qv5 six-arm four-choice validation harness.

One BF16-sourced Qwen projection policy per full native process. For each of
40 ARC-Challenge validation questions score the correct answer token NLL AND
accuracy against all A/B/C/D options using a single unperturbed-choice prefix
pass. Qv5 selected mask must be source-only SHA frozen before any eval runs.

Only sandbox C source copies are extended; unchanged original native ppl mode,
official HF source, git main and live serving remain untouched.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,subprocess,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
ORIGIN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
QV4=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv4-masked-nll-20261022")
LAB=ROOT/"lab_local/qv5_fisher_20261022"
RUN=LAB/"native_40val_6mode"
DEV=LAB/"development_val_frozen.json"
PRE=ROOT/"configs/research/BEGLIN_QV5_FISHER_GRAD_PRELOCK_20261022.json"
MILP=ROOT/"docs/research/BEGLIN_QV5_FISHER_HIGHS_SOURCE_MILP_20261022.json"
POLICY=ROOT/"configs/research/BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRE_SHA="a886418affc9f665d64ce4ca9eb0b5ea2aef4ae97bc59bf58ea1affb53bf19ef"
DEV_SHA="a00985cc73f80518f448c2b8a1295cf894283723f87b5af53321d486ce0187ae"
MODELS=("f32","q4","role_random","v2_diag","v3_cov","v5_fisher")
PROJ={
 "f32":(196,0,0),"q4":(0,0,196),"role_random":(0,28,168),
 "v2_diag":(0,28,168),"v3_cov":(0,28,168),"v5_fisher":(0,28,168)}
QV4_BIN_SOURCE={"base":QV4/"lab_local/q_v4_masked_answer_20261022/bin/qv4_base.c",
 "mixed":QV4/"lab_local/q_v4_masked_answer_20261022/bin/qv4_mixed.c"}
OLD_MASKS={
 "role_random":(ORIGIN/"lab_local/qng64_n8_repacked_full_20261022/policy_random.txt",
      "72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5"),
 "v2_diag":(ROOT/"configs/research/BEGLIN_QV2_ASYNC_DAG_N8_POLICY_20261022.txt",
      "e044675ea5b47a5aca0ca75980a55070b932d1a4dc330815bbb097acbf75df7b"),
 "v3_cov":(ROOT/"configs/research/BEGLIN_Q_V3_FROZEN_28N8_POLICY_20261022.txt",
      "82ae93e06afa7380175afa57535c1f0d1f7fd4320b8d28f981cdc8ee0cf24ef3")
}
PATCH='    } else if (!strcmp(mode,"masked_nll")) {'
BRANCH=r'''    } else if (!strcmp(mode,"masked_choice_qv5")) {
        const char *qv5=getenv("BEGLIN_QV5_FROZEN_ARC_CHALLENGE");
        if (!qv5 || strcmp(qv5,"1")) {fprintf(stderr,"FATAL: Qv5 native choice gate absent\n");return 1;}
        int nw=argc>2?atoi(argv[2]):-1;
        if (nw<1 || nw>128) {fprintf(stderr,"FATAL: bad Qv5 validation count\n");return 1;}
        int ids[g_cfg.maxseq]; int won=0; double total=0.0;
        for(int wi=0;wi<nw;wi++){
            snprintf(path,sizeof(path),"%s/eval/ppl_work/w%d.i32",base,wi);
            int n=load_ids(path,ids,g_cfg.maxseq);
            if(n<2 || n>208){fprintf(stderr,"FATAL: invalid Qv5 prefix tokens\n");return 1;}
            snprintf(path,sizeof(path),"%s/eval/ppl_work/w%d.choices",base,wi);
            FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"FATAL: Qv5 choice tokens absent\n");return 1;}
            int v[5]={-1,-1,-1,-1,-1}; char junk=0;
            int count=fscanf(f,"%d %d %d %d %d %c",&v[0],&v[1],&v[2],&v[3],&v[4],&junk);
            fclose(f);
            if(count!=5 || v[4]<0 || v[4]>3 || v[0]<0 || v[1]<0 || v[2]<0 || v[3]<0 ||
               v[0]>=g_cfg.vocab || v[1]>=g_cfg.vocab || v[2]>=g_cfg.vocab || v[3]>=g_cfg.vocab ||
               v[0]==v[1] || v[0]==v[2] || v[0]==v[3] || v[1]==v[2] || v[1]==v[3] || v[2]==v[3]) {
                fprintf(stderr,"FATAL: Qv5 invalid frozen A/B/C/D ids or answer\n");return 1;
            }
            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);
            final_logits(xtmp,logits);
            float mx;vDSP_maxv(logits,1,&mx,g_cfg.vocab);double s=0;
            for(int j=0;j<g_cfg.vocab;j++)s+=exp((double)(logits[j]-mx));
            int answer=v[4], best=0;
            for(int j=1;j<4;j++)if(logits[v[j]]>logits[v[best]])best=j;
            double nll=-(double)logits[v[answer]]+mx+log(s);
            if(!isfinite(nll) || nll<0){fprintf(stderr,"FATAL: nonfinite Qv5 answer NLL\n");return 1;}
            total+=nll;won+=(best==answer);
            fprintf(stderr,"[qv5-choice] window %d NLL %.17g expected %d predicted %d prefix %d\n",wi,nll,answer,best,n);
        }
        fprintf(stdout,"qv5_choice NLL %.17g items %d correct %d mean %.17g ppl %.17g\n",
                total,nw,won,total/nw,exp(total/nw));
    } else if (!strcmp(mode,"masked_nll")) {'''

def sha(b):return hashlib.sha256(b).hexdigest()
def secure(path:pathlib.Path):
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("missing/unprotected frozen private source "+str(path))
    return path.read_bytes()
def put(path:pathlib.Path,b:bytes):
    with path.open("xb") as f:
        f.write(b);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def sources():
    if sha(PRE.read_bytes())!=PRE_SHA or sha(DEV.read_bytes())!=DEV_SHA:
        raise RuntimeError("original Qv5/ARC preregistration change")
    if not MILP.is_file() or not POLICY.is_file():
        raise RuntimeError("source-only Fisher policy not yet frozen")
    policy=json.loads(MILP.read_text())
    if (policy["status"]!="FROZEN_SOURCE_ONLY_EXACT_MILP_NOT_YET_ARC_VALIDATION_SCORED" or
        sha(POLICY.read_bytes())!=policy["mask_sha256"] or
        not policy["development_ARC_40_labels_unopened"]):
        raise RuntimeError("post-eval policy refit forbidden")
    val=json.loads(secure(DEV))
    if val["question_count"]!=40 or len(val["questions"])!=40:
        raise RuntimeError("not original 40 ARC-Challenge validation questions")
    if not (SNAP/"model.safetensors").is_file():
        raise RuntimeError("source BF16 checkpoint missing")
    old=OLD_MASKS.copy()
    old["v5_fisher"]=(POLICY,policy["mask_sha256"])
    for mode,(path,h) in old.items():
        if sha(path.read_bytes())!=h:raise RuntimeError("original n8 role mask changed "+mode)
    return val,old,policy

def patch(old:str):
    if old.count(PATCH)!=1 or "masked_choice_qv5" in old:
        raise RuntimeError("native research source ambiguity")
    new=old.replace(PATCH,BRANCH,1)
    if new.split(PATCH,1)[1]!=old.split(PATCH,1)[1]:
        raise RuntimeError("prior Qv4 masked_NLL mode modified")
    return new

def prepare():
    val,masks,policy=sources()
    if RUN.exists():raise FileExistsError("do not replace original Qv5 native comparison")
    os.umask(0o077)
    RUN.mkdir(parents=True,mode=0o700)
    locked=[]
    for i,row in enumerate(val["questions"]):
        ids=row["prefix_ids"]
        if len(ids)<2 or len(ids)>208:raise RuntimeError("prefix length outside frozen policy")
        correct=row["correct"];suffix=row["suffix_ids"]
        if set(suffix)!=set("ABCD"):raise RuntimeError("no A-D choice encoding")
        import struct
        data=struct.pack("<"+"i"*len(ids),*ids)
        choice=(" ".join(str(suffix[x]) for x in "ABCD")+" "+"ABCD".index(correct).__str__()+"\n").encode()
        locked.append({"id_sha256":sha(row["id"].encode()),"prefix_sha256":sha(data),
           "choices_SHA256":sha(choice),"correct_SHA256":sha(correct.encode()),
           "prompt_length":len(ids)})
        for mode in MODELS:
            dir=RUN/mode
            (dir/"eval/ppl_work").mkdir(parents=True,exist_ok=True,mode=0o700)
            (dir/"results").mkdir(exist_ok=True,mode=0o700)
            put(dir/f"eval/ppl_work/w{i}.i32",data)
            put(dir/f"eval/ppl_work/w{i}.choices",choice)
            if i==0:put(dir/"prompt_ids.i32",data)
    record={
      "schema":"BEGLIN-Qv5-ARC-validation-40-six-native-mode-pre-execution-lock/1",
      "status":"INPUTS_AND_SOURCE_FISHER_POLICY_FROZEN_BEFORE_NATIVE_RESULTS",
      "source_DEV_file_SHA256":DEV_SHA,
      "source_MILP_document_SHA256":sha(MILP.read_bytes()),
      "source_fisher_policy_SHA256":policy["mask_sha256"],
      "BF16_original_rev":SNAP.name,"masks":{k:v[1] for k,v in masks.items()},
      "conditions":list(MODELS),"mode_cases":40,
      "validation":locked,"not_source_calibration":True,
      "no_prior_QV4_8_questions_reused":True,"no_operational_mutation":True
    }
    put(RUN/"pre_native_sixmode_manifest.json",(json.dumps(record,sort_keys=True,indent=2)+"\n").encode())
    print("QV5_REAL_NATIVE_PREPARED",json.dumps({
        "manifest_sha256":sha(secure(RUN/"pre_native_sixmode_manifest.json")),
        "modes":len(MODELS),"questions":len(locked),
        "policy_mask_sha256":policy["mask_sha256"]},sort_keys=True),flush=True)

def build():
    if not RUN.is_dir() or not (RUN/"pre_native_sixmode_manifest.json").is_file():
        raise RuntimeError("native evaluation input lock required before C compile")
    folder=RUN/"bin"
    if folder.exists():raise FileExistsError("native qv5 bin already built")
    os.umask(0o077);folder.mkdir(mode=0o700)
    for kind in ("base","mixed"):
        oldfile=QV4_BIN_SOURCE[kind]
        receipt=json.loads(secure(QV4/"lab_local/q_v4_masked_answer_20261022/bin"/(kind+"_build_receipt.json")))
        old=secure(oldfile)
        if sha(old)!=receipt["new_source_sha256"]:
            raise RuntimeError("original Qv4 independently audited C source changed")
        target=folder/f"qv5_{kind}.c"
        put(target,patch(old.decode()).encode())
        obj=folder/f"qv5_{kind}.o"
        bin=folder/f"beglin_qv5_{kind}"
        subprocess.run(["clang","-O3","-w","-I",str(ORIGIN),"-c",str(target),"-o",str(obj)],check=True,timeout=120)
        objects=sorted((ORIGIN/".build").glob("*.o"))
        if kind=="base":
            deps=[p for p in objects if p.name!="qwen_infer.o"]
            if len(deps)!=21:raise RuntimeError("native BF16/Q4 original dependency set changed")
        else:
            base=ORIGIN/"lab_local/qng64_n8_repacked_full_20261022"
            deps=[base/"qng64_original_aliased.o",base/"n8_repacked_simd.o",base/"n8_repacked_dispatch.o"]
            deps += [p for p in objects if p.name not in ("qwen_infer.o","qng64_dense.o")]
            if len(deps)!=23:raise RuntimeError("native mixed NEON dependencies changed")
        subprocess.run(["clang","-O3",str(obj)]+list(map(str,deps))+[
            "-o",str(bin),"-framework","Accelerate","-lpthread","-lm"],check=True,timeout=120)
        bin.chmod(0o700)
        put(folder/f"{kind}_build_receipt.json",(json.dumps({
           "schema":"BEGLIN-Qv5-native-ARC-fourchoice-BF16-Q4-n8-binary/1",
           "orig_Qv4_C_SHA256":sha(old),"new_C_SHA256":sha(target.read_bytes()),
           "actual_engine_binary_SHA256":sha(bin.read_bytes()),
           "original_ppl_unchanged":True,"Qv4_masked_nll_unchanged":True,
           "no_production_binary_modified":True},sort_keys=True,indent=2)+"\n").encode())
        print("QV5_NATIVE_BUILD",kind,sha(bin.read_bytes()),flush=True)

REGEX_ROW=re.compile(r"\[qv5-choice\] window (\d+) NLL ([\w.+\-eE]+) expected (\d+) predicted (\d+) prefix (\d+)")
REGEX_SUM=re.compile(r"qv5_choice NLL ([\w.+\-eE]+) items (\d+) correct (\d+) mean ([\w.+\-eE]+) ppl ([\w.+\-eE]+)")
REGEX_ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def parse(log,mode,manifest):
    counts=REGEX_ROLE.findall(log)
    if len(counts)!=1 or tuple(map(int,counts[0]))!=PROJ[mode]:
        raise RuntimeError("native BEGLIN wrong actual projection precision "+mode)
    items=REGEX_ROW.findall(log)
    if len(items)!=40:raise RuntimeError("not 40 source-frozen ARC native answer likelihoods")
    losses=[];correct=0
    for i,(idx,nll,a,b,n) in enumerate(items):
        num=float(nll)
        if int(idx)!=i or not math.isfinite(num) or num<0 or \
           int(n)!=manifest["validation"][i]["prompt_length"] or \
           not 0<=int(a)<4 or not 0<=int(b)<4:
            raise RuntimeError("native ARC attention/choice boundary failed "+mode)
        losses.append(num);correct+=int(a==b)
    result=REGEX_SUM.findall(log)
    if len(result)!=1:raise RuntimeError("not unique native 40-sample aggregate")
    tot,n,acc,mean,ppl=map(float,result[0])
    if n!=40 or int(acc)!=correct or abs(tot-sum(losses))>1e-7 or \
       abs(mean-tot/40)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
        raise RuntimeError("native BEGLIN ARC-NLL transcript arithmetic mismatch")
    return {"items":40,"correct":correct,"accuracy":correct/40.,
       "correct_answer_mean_nll":mean,"correct_answer_ppl":ppl,
       "per_question_correct_answer_nll":losses,
       "per_question_native_right":[int(a==b) for _,_,a,b,_ in items],
       "native_role_counts":list(PROJ[mode])}

def evaluate(mode):
    val,masks,policy=sources()
    if mode not in MODELS:raise ValueError("invalid mode")
    frozen=json.loads(secure(RUN/"pre_native_sixmode_manifest.json"))
    if frozen["status"]!="INPUTS_AND_SOURCE_FISHER_POLICY_FROZEN_BEFORE_NATIVE_RESULTS" or \
       frozen["source_fisher_policy_SHA256"]!=policy["mask_sha256"] or \
       frozen["source_DEV_file_SHA256"]!=DEV_SHA:
        raise RuntimeError("native evaluated after source policy tampering")
    dir=RUN/mode;kind="base" if mode in ("f32","q4") else "mixed"
    exe=RUN/"bin"/f"beglin_qv5_{kind}"
    buildrec=json.loads(secure(RUN/"bin"/f"{kind}_build_receipt.json"))
    if sha(exe.read_bytes())!=buildrec["actual_engine_binary_SHA256"]:
        raise RuntimeError("actual native model binary changed")
    logfile=dir/"native_40choice_log.txt"
    if logfile.exists():raise FileExistsError("cannot rerun original native quality arm")
    for i,m in enumerate(frozen["validation"]):
        if sha(secure(dir/f"eval/ppl_work/w{i}.i32"))!=m["prefix_sha256"] or \
           sha(secure(dir/f"eval/ppl_work/w{i}.choices"))!=m["choices_SHA256"]:
            raise RuntimeError("answer token or prompt SHA changed after freeze")
    env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV4_","BEGLIN_QV5_"))}
    env.update({"VECLIB_MAXIMUM_THREADS":"1",
      "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
      "QWEN_HF_CONFIG":str(SNAP/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(dir.resolve()),"QWEN_PROMPT":str((dir/"prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":"f32" if mode=="f32" else "q4" if mode=="q4" else "mixed",
      "BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0"})
    if mode in masks:
        env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(masks[mode][0])
    tic=time.monotonic()
    proc=subprocess.run([str(exe),"masked_choice_qv5","40"],cwd=str(ORIGIN),env=env,
        text=True,capture_output=True,timeout=1800)
    log=proc.stdout+"\n"+proc.stderr
    os.umask(0o077);put(logfile,log.encode())
    if proc.returncode:raise RuntimeError("native Qwen 28layer 40ARC fail "+mode+" exit "+str(proc.returncode)+" "+proc.stderr[-500:])
    score=parse(log,mode,frozen)
    receipt={
      "schema":"BEGLIN-Qv5-native-28layer-40-ARC-fourchoice-arm/1",
      "native_precision_policy":mode,"native_binary_SHA256":sha(exe.read_bytes()),
      "native_program_log_SHA256":sha(log.encode()),
      "dev_frozen_manifest_SHA256":sha(secure(RUN/"pre_native_sixmode_manifest.json")),
      "original_BF16_snapshot":SNAP.name,
      "source_only_Fisher_policy_mask_SHA256":policy["mask_sha256"],
      "scores":score,"elapsed_wall_seconds":round(time.monotonic()-tic,2),
      "model_layers_executed":28,"source_checkpoint_modified":False,
      "user_deployment_modified":False,"sample_scope":"ARC-Challenge validation DEV_ONLY"}
    path=dir/"native_40choice_receipt.json"
    put(path,(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
    print("QV5_REAL_NATIVE_40_ARC",mode,json.dumps({"acc":score["accuracy"],
      "correct_nll":score["correct_answer_mean_nll"],"receipt_SHA":sha(secure(path)),
      "seconds":round(time.monotonic()-tic,2)},sort_keys=True),flush=True)

def run():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("prepare","build","eval"))
    p.add_argument("--mode",choices=MODELS)
    args=p.parse_args()
    if args.stage=="prepare":prepare()
    if args.stage=="build":build()
    if args.stage=="eval":
        if not args.mode:p.error("eval needs --mode")
        evaluate(args.mode)
if __name__=="__main__":run()
