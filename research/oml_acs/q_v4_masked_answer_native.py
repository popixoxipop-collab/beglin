#!/usr/bin/env python3
"""Research-only assistant-suffix NLL for native Qwen BEGLIN precision policies.

Uses 8 preregistered *development* two-choice prompts and single-token A/B
suffixes, scoring only the answer position. The original native C "ppl"
mode and all operational source/binaries remain UNMODIFIED.
Source-only v2/v3 masks and historical outputs are NEVER refit here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ORIGIN = Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
LAB = ROOT / "lab_local/q_v4_masked_answer_20261022"
PROTO = ROOT / "configs/research/BEGLIN_Q_V4_MASKED_ANSWER_DEV_PRELOCK_20261022.json"
SNAP = Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRELOCK_SHA = "984adbb205bc14e3ff76c1654a3143529fdd07a8ba017e445462cb3707c3677c"
SOURCE_SHA = {
    "base": "c03099aed130a5fd9222f0ba4d0fdf00d3e9bd1b1c203bf145ccb8290a0ad131",
    "mixed": "d8e149092c73d858dfb7a066f697f2b95e0668ffa747b9d66ddc4ee3617f447a",
}
MASKS = {
    "role_random": (ORIGIN/"lab_local/qng64_n8_repacked_full_20261022/policy_random.txt",
        "72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5"),
    "v2_diag": (ROOT/"configs/research/BEGLIN_QV2_ASYNC_DAG_N8_POLICY_20261022.txt",
        "e044675ea5b47a5aca0ca75980a55070b932d1a4dc330815bbb097acbf75df7b"),
    "v3_cov": (ROOT/"configs/research/BEGLIN_Q_V3_FROZEN_28N8_POLICY_20261022.txt",
        "82ae93e06afa7380175afa57535c1f0d1f7fd4320b8d28f981cdc8ee0cf24ef3"),
}
MODES = ("f32", "q4", "role_random", "v2_diag", "v3_cov")
OLD_SOURCES = {
    "base": ORIGIN/"lab_local/q_full_model_20261021_v2/qwen_infer_q_lab.c",
    "mixed": ORIGIN/"lab_local/q_kernel_aware_native_20261022/qwen_kernel_aware_copy.c",
}
EXPECTED_COUNTS = {"f32": (196,0,0), "q4": (0,0,196),
                   "role_random": (0,28,168), "v2_diag": (0,28,168),
                   "v3_cov": (0,28,168)}
PATCH_ANCHOR = '    } else if (!strcmp(mode,"ppl")) {'
MASKED_BRANCH = r'''    } else if (!strcmp(mode,"masked_nll")) {
        // Qv4 research-only branch: original "ppl" remains untouched below.
        // Read an exact prefix token count per locked prompt. Only tokens
        // AFTER that prefix contribute to loss; no prompt loss dilution.
        const char *gate=getenv("BEGLIN_QV4_DEV_MASKED_NLL");
        if (!gate || strcmp(gate,"1")) {
            fprintf(stderr,"FATAL: Qv4 masked NLL requires opt-in gate\n");
            return 1;
        }
        int nw=argc>2?atoi(argv[2]):0;
        if (nw<1 || nw>64) {fprintf(stderr,"FATAL: bad Qv4 windows\n");return 1;}
        int ids[g_cfg.maxseq]; long scored=0; double total=0.0;
        for (int wi=0;wi<nw;wi++) {
            snprintf(path,sizeof path,"%s/eval/ppl_work/w%d.i32",base,wi);
            int n=load_ids(path,ids,g_cfg.maxseq);
            snprintf(path,sizeof path,"%s/eval/ppl_work/w%d.prefix",base,wi);
            FILE *pf=fopen(path,"rb");
            if (!pf) {fprintf(stderr,"FATAL: missing Qv4 prefix metadata\n");return 1;}
            int k=-1; char trailing=0;
            int fields=fscanf(pf,"%d %c",&k,&trailing); fclose(pf);
            if (fields!=1 || n<2 || k<2 || k>=n) {
                fprintf(stderr,"FATAL: invalid Qv4 answer-only prefix boundary\n");
                return 1;
            }
            double wnll=0.0; int wcount=0;
            for (int p=0;p<n;p++) {
                forward_token(ids[p],p,xtmp,0);
                if (p>=k-1 && p<n-1) {
                    final_logits(xtmp,logits);
                    float mx;
                    vDSP_maxv(logits,1,&mx,g_cfg.vocab);
                    double se=0.0;
                    for(int v=0;v<g_cfg.vocab;v++) se+=exp((double)(logits[v]-mx));
                    double one=-((double)logits[ids[p+1]]-mx-log(se));
                    if (!isfinite(one)) {fprintf(stderr,"FATAL: Qv4 NaN NLL\n");return 1;}
                    wnll+=one; wcount++;
                }
            }
            fprintf(stderr,"[qv4-masked] window %d prefix %d nll %.17g tokens %d\n",wi,k,wnll,wcount);
            scored+=wcount; total+=wnll;
        }
        if (scored==0) {fprintf(stderr,"FATAL: zero answer tokens\n");return 1;}
        printf("qv4_masked_nll %.17g tokens %ld mean %.17g ppl %.17g\n",
               total,scored,total/scored,exp(total/scored));
    } else if (!strcmp(mode,"ppl")) {'''

def sha(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()

def checked_file(p: Path, expected_sha: str) -> bytes:
    if not p.is_file() or p.is_symlink():
        raise RuntimeError("missing or symlinked fixed source: " + str(p))
    b = p.read_bytes()
    if sha(b) != expected_sha:
        raise RuntimeError("SHA mismatch for frozen source: " + str(p))
    return b

def private_write(p: Path, b: bytes) -> None:
    with p.open("xb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    p.chmod(0o600)

def guarded_patch(s: str) -> str:
    if s.count(PATCH_ANCHOR) != 1:
        raise RuntimeError("ambiguous native ppl source, refusing patch")
    if '"masked_nll"' in s:
        raise RuntimeError("masked NLL already installed")
    result = s.replace(PATCH_ANCHOR, MASKED_BRANCH, 1)
    # byte identical original legacy "ppl" tail
    if result.split(PATCH_ANCHOR, 1)[1] != s.split(PATCH_ANCHOR, 1)[1]:
        raise RuntimeError("original ppl mode changed")
    return result

def config() -> dict:
    c = json.loads(checked_file(PROTO, PRELOCK_SHA))
    if (c["status"] != "FROZEN_BEFORE_NATIVE_SCORES"
            or c["classification"] != "EXPLORATORY_DEVELOPMENT_NOT_FINAL_HOLDOUT"
            or c["baseline_modes"] != list(MODES)
            or c["production_promotion"] is not False
            or len(c["examples"]) != 8):
        raise RuntimeError("Qv4 frozen evaluation contract invalid")
    examples = c["examples"]
    if len({e["question"] for e in examples}) != 8 or \
       sum(e["correct"] == "A" for e in examples) != 4 or \
       sum(e["correct"] == "B" for e in examples) != 4:
        raise RuntimeError("development examples duplicate/unbalanced")
    for k,pair in MASKS.items():
        data = checked_file(pair[0], pair[1]).decode()
        roles = [line.split() for line in data.splitlines()]
        if len(roles)!=28 or len({tuple(x) for x in roles})!=28 or any(
                len(x)!=2 or x[0] not in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
                or not x[1].isdigit() or not 0<=int(x[1])<28 for x in roles):
            raise RuntimeError("source-frozen " + k + " policy invalid")
    return c

def prepare() -> None:
    c = config()
    if LAB.exists():
        raise FileExistsError("Qv4 mask lab already exists; never overwrite results")
    if not (SNAP/"tokenizer.json").is_file():
        raise RuntimeError("original Qwen tokenizer missing")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    windows = []
    for i,e in enumerate(c["examples"]):
        prefix = "<|im_start|>user\n" + e["question"] + "<|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
        if not 5<=len(prefix_ids)<=240:
            raise RuntimeError("unexpected Qwen prompt length")
        for answer in ("A","B"):
            ids = tokenizer.encode(prefix+answer, add_special_tokens=False).ids
            if ids[:len(prefix_ids)] != prefix_ids or len(ids)!=len(prefix_ids)+1:
                raise RuntimeError("Qwen tokenizer does not preserve single-token label prefix")
            from array import array
            import struct
            blob = struct.pack("<"+"i"*len(ids),*ids)
            windows.append({"idx":len(windows),"case":e["id"],"answer":answer,
                 "is_correct":answer==e["correct"],"prefix":len(prefix_ids),
                 "token_count":len(ids),"ids_sha256":sha(blob),"blob":blob})
    os.umask(0o077)
    LAB.mkdir(mode=0o700,parents=True)
    for name in MODES:
        wd=LAB/name
        (wd/"eval/ppl_work").mkdir(parents=True,mode=0o700)
        (wd/"results").mkdir(mode=0o700)
        for w in windows:
            dest=wd/f"eval/ppl_work/w{w['idx']}.i32"
            private_write(dest,w["blob"])
            private_write(dest.with_suffix(".prefix"),(str(w["prefix"])+"\n").encode())
        private_write(wd/"prompt_ids.i32",windows[0]["blob"])
    manifest={
       "schema":"BEGLIN-Qv4-frozen-development-masked-answer-inputs/1",
       "state":"INPUTS_FROZEN_BEFORE_ANY_NATIVE_QV4_INFERENCE",
       "preregistration_sha256":PRELOCK_SHA,
       "tokenizer_sha256":sha((SNAP/"tokenizer.json").read_bytes()),
       "source_original_BF16_revision":SNAP.name,
       "mask_sha256":{k:v[1] for k,v in MASKS.items()},
       "windows":[{k:v for k,v in w.items() if k!="blob"} for w in windows],
       "all_policies_same_ids":True,
       "legacy_24_and_source_12_not_used":True,
       "result_use":"EXPLORATORY_DEV_ONLY_NO_PROMOTION"}
    private_write(LAB/"inputs_frozen.json",(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode())
    print("QV4_INPUTS_FROZEN",sha((LAB/"inputs_frozen.json").read_bytes()),len(windows),flush=True)

def build() -> None:
    config()
    if not LAB.is_dir() or not (LAB/"inputs_frozen.json").is_file():
        raise RuntimeError("freeze Qv4 inputs before building")
    target=LAB/"bin"
    if target.exists():
        raise FileExistsError("research native build exists")
    target.mkdir(mode=0o700)
    for kind in ("base","mixed"):
        old = checked_file(OLD_SOURCES[kind], SOURCE_SHA[kind]).decode()
        new = guarded_patch(old)
        source=target/f"qv4_{kind}.c"
        private_write(source,new.encode())
        obj=target/f"qv4_{kind}.o"
        exe=target/f"beglin_qv4_{kind}"
        subprocess.run(["clang","-O3","-w","-I",str(ORIGIN),"-c",str(source),"-o",str(obj)],
                       check=True,timeout=120)
        objects=sorted((ORIGIN/".build").glob("*.o"))
        if kind=="base":
            links=[p for p in objects if p.name!="qwen_infer.o"]
            if len(links)!=21:raise RuntimeError("base compiled object closure drift")
        else:
            links=[
              ORIGIN/"lab_local/qng64_n8_repacked_full_20261022/qng64_original_aliased.o",
              ORIGIN/"lab_local/qng64_n8_repacked_full_20261022/n8_repacked_simd.o",
              ORIGIN/"lab_local/qng64_n8_repacked_full_20261022/n8_repacked_dispatch.o",
            ]+[p for p in objects if p.name not in ("qwen_infer.o","qng64_dense.o")]
            if len(links)!=23:raise RuntimeError("mixed compiled object closure drift")
        subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in links]+
                       ["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],
                       check=True,timeout=120)
        exe.chmod(0o700)
        private_write(target/f"{kind}_build_receipt.json",(
            json.dumps({"old_source_sha256":SOURCE_SHA[kind],
              "new_source_sha256":sha(source.read_bytes()),"exe_sha256":sha(exe.read_bytes()),
              "legacy_ppl_unchanged":True,"production_touched":False},sort_keys=True,indent=2)+"\n").encode())
        print("QV4_NATIVE_BUILD",kind,sha(exe.read_bytes()),flush=True)

def frozen_windows() -> tuple[dict,list]:
    c=config()
    p=LAB/"inputs_frozen.json"
    if not p.is_file() or (p.stat().st_mode & 0o777)!=0o600:
        raise RuntimeError("missing protected frozen input receipt")
    frozen=json.loads(p.read_text())
    if frozen["preregistration_sha256"]!=PRELOCK_SHA or frozen["state"]!="INPUTS_FROZEN_BEFORE_ANY_NATIVE_QV4_INFERENCE":
        raise RuntimeError("Qv4 input receipt not frozen")
    if len(frozen["windows"])!=16 or frozen["mask_sha256"]!={k:v[1] for k,v in MASKS.items()}:
        raise RuntimeError("not exactly the frozen Qv4 task cohort")
    for mode in MODES:
        for w in frozen["windows"]:
            i=w["idx"]; wd=LAB/mode/"eval/ppl_work"
            p=wd/f"w{i}.i32"; meta=wd/f"w{i}.prefix"
            if (not p.is_file() or p.is_symlink() or sha(p.read_bytes())!=w["ids_sha256"]
                or p.stat().st_mode&0o777!=0o600 or not meta.is_file()
                or meta.stat().st_mode&0o777!=0o600 or meta.read_text()!=str(w["prefix"])+"\n"):
                raise RuntimeError("Qv4 native input token/answer boundary drift")
    return frozen,frozen["windows"]

RESULT_ROW=re.compile(r"\[qv4-masked\] window (\d+) prefix (\d+) nll ([\w.+\-eE]+) tokens (\d+)")
FINAL_ROW=re.compile(r"qv4_masked_nll ([\w.+\-eE]+) tokens (\d+) mean ([\w.+\-eE]+) ppl ([\w.+\-eE]+)")
def parse_native_log(log:str,windows:list) -> dict:
    rows=RESULT_ROW.findall(log)
    final=FINAL_ROW.findall(log)
    if len(rows)!=len(windows) or len(final)!=1:
        raise RuntimeError("missing actual masked native per-window loss")
    vals=[]
    for i,(index,prefix,loss,tokens) in enumerate(rows):
        w=windows[i]; value=float(loss)
        if int(index)!=i or int(prefix)!=w["prefix"] or int(tokens)!=1 or not math.isfinite(value) or value<0:
            raise RuntimeError("wrong answer-only token span, order or nonfinite loss")
        vals.append(value)
    total,toks,mean,ppl=map(float,final[0])
    if int(toks)!=len(windows) or abs(total-sum(vals))>1e-8 or abs(mean-total/toks)>1e-8:
        raise RuntimeError("native masked loss aggregate inconsistent")
    if abs(math.log(ppl)-mean)>1e-8:
        raise RuntimeError("masked PPL != exponential answer-only NLL")
    good=[]; margins=[]; hits=0
    for i in range(0,len(windows),2):
        a,b=windows[i:i+2]
        if (a["case"]!=b["case"] or (a["answer"],b["answer"])!=("A","B")
            or a["is_correct"]==b["is_correct"]):
            raise RuntimeError("pairwise label order invalid")
        correct=vals[i] if a["is_correct"] else vals[i+1]
        wrong=vals[i+1] if a["is_correct"] else vals[i]
        good.append(correct); margins.append(wrong-correct)
        hits+=int(correct < wrong)
    return {"masked_suffix_tokens":int(toks),"choice_accuracy":hits/len(good),
        "correct_answer_mean_nll":sum(good)/len(good),
        "correct_answer_ppl":math.exp(sum(good)/len(good)),
        "correct_minus_wrong_nll_mean":-sum(margins)/len(margins),
        "mean_correct_logprob_margin":sum(margins)/len(margins),
        "per_case_margin":margins}

def eval_policy(name:str)->None:
    if name not in MODES:raise ValueError("unknown policy")
    frozen,windows=frozen_windows()
    kind="base" if name in ("f32","q4") else "mixed"
    exe=LAB/"bin"/f"beglin_qv4_{kind}"
    receipt=LAB/"bin"/f"{kind}_build_receipt.json"
    if not exe.is_file() or not receipt.is_file() or \
       sha(exe.read_bytes())!=json.loads(receipt.read_text())["exe_sha256"]:
        raise RuntimeError("source-pinned Qv4 C binary unavailable")
    dest=LAB/name/"native_masked_nll_log.txt"
    if dest.exists():raise FileExistsError("this frozen native mode already ran: "+name)
    os.umask(0o077)
    env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_QV","BEGLIN_Q_LAB_"))}
    env.update({"QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
      "QWEN_HF_CONFIG":str(SNAP/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str((LAB/name).resolve()),"QWEN_PROMPT":str((LAB/name/"prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":("f32" if name=="f32" else "q4" if name=="q4" else "mixed"),
      "BEGLIN_QV4_DEV_MASKED_NLL":"1","BEGLIN_QV2_NATIVE_ASYNC":"0",
      "VECLIB_MAXIMUM_THREADS":"1"})
    if name in MASKS:
        env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(MASKS[name][0].resolve())
    t0=time.monotonic()
    run=subprocess.run([str(exe),"masked_nll",str(len(windows))],env=env,
        cwd=str(ORIGIN),capture_output=True,text=True,timeout=1200)
    log=run.stdout+"\n"+run.stderr
    private_write(dest,log.encode())
    if run.returncode!=0:raise RuntimeError("native Qwen masked eval failed: "+name+" return="+str(run.returncode))
    counts=re.search(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)",log)
    if not counts or tuple(map(int,counts.groups()))!=EXPECTED_COUNTS[name]:
        raise RuntimeError("full Qwen actual mode projection counts differ from source policy")
    score=parse_native_log(log,windows)
    output={"schema":"BEGLIN-Qv4-native-answer-only-choice-development/1",
        "mode":name,"fixed_input_receipt_sha256":sha((LAB/"inputs_frozen.json").read_bytes()),
        "native_binary_sha256":sha(exe.read_bytes()),"original_bf16_revision":SNAP.name,
        "log_sha256":sha(log.encode()),"projected_roles":list(EXPECTED_COUNTS[name]),
        "model_layers":28,"elapsed_wall_s":round(time.monotonic()-t0,2),
        "scores":score,"classification":"EXPLORATORY_DEVELOPMENT_ONLY",
        "no_original_model_or_live_modification":True}
    private_write(LAB/name/"native_masked_answer_receipt.json",(
        json.dumps(output,sort_keys=True,indent=2)+"\n").encode())
    print("QV4_REAL_NATIVE",name,json.dumps({"accuracy":score["choice_accuracy"],
        "nll":score["correct_answer_mean_nll"],
        "mean_margin":score["mean_correct_logprob_margin"],
        "receipt_sha256":sha((LAB/name/"native_masked_answer_receipt.json").read_bytes())},
        sort_keys=True),flush=True)

def audit() -> None:
    frozen,windows=frozen_windows()
    report={}
    for name in MODES:
        receipt=LAB/name/"native_masked_answer_receipt.json"
        log=LAB/name/"native_masked_nll_log.txt"
        if not receipt.is_file() or not log.is_file() or \
           (receipt.stat().st_mode & 0o777)!=0o600 or (log.stat().st_mode & 0o777)!=0o600:
            raise RuntimeError("missing native original or insecure output: "+name)
        original=json.loads(receipt.read_text())
        native_log=log.read_text()
        if original["log_sha256"]!=sha(native_log.encode()):
            raise RuntimeError("native process log tampered: "+name)
        x=parse_native_log(native_log,windows)
        for k in ("choice_accuracy","correct_answer_mean_nll",
                  "mean_correct_logprob_margin","correct_answer_ppl"):
            if abs(x[k]-original["scores"][k])>1e-10:
                raise RuntimeError("native score was not independently reproducible")
        report[name]={"accuracy":x["choice_accuracy"],
            "correct_answer_mean_nll":x["correct_answer_mean_nll"],
            "mean_margin":x["mean_correct_logprob_margin"]}
    summary={"schema":"BEGLIN-Qv4-masked-answer-independent-development-audit/1",
      "preregistered_input_sha256":PRELOCK_SHA,"cases":8,"pairs_per_policy":8,
      "candidate_models":list(MODES),"results":report,
      "v3_beats_random_on_correct_only_NLL":report["v3_cov"]["correct_answer_mean_nll"] < report["role_random"]["correct_answer_mean_nll"],
      "v3_beats_Q4_on_correct_only_NLL":report["v3_cov"]["correct_answer_mean_nll"] < report["q4"]["correct_answer_mean_nll"],
      "no_IID_OOD_quality_claim":True,"no_production_promotion":True,
      "no_old_V6_holdout_24":True}
    dest=LAB/"qv4_independent_masked_answer_audit.json"
    private_write(dest,(json.dumps(summary,sort_keys=True,indent=2)+"\n").encode())
    print("QV4_INDEPENDENT_AUDIT_PASS",json.dumps(summary,sort_keys=True),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("check","prepare","build","eval","audit"))
    parser.add_argument("--policy",choices=MODES)
    args=parser.parse_args()
    if args.action=="check":
        config()
        for kind,p in OLD_SOURCES.items():
            guarded_patch(checked_file(p,SOURCE_SHA[kind]).decode())
        print("QV4_ORIGINAL_SOURCE_PATCH_SITES_PASS")
    elif args.action=="prepare":prepare()
    elif args.action=="build":build()
    elif args.action=="eval":
        if not args.policy:parser.error("eval requires --policy")
        eval_policy(args.policy)
    else:audit()
if __name__=="__main__":
    main()
