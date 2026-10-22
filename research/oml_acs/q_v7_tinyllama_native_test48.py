#!/usr/bin/env python3
"""Frozen TinyLlama real BEGLIN full-model Q4 vs source-only Fisher-22n8 test48.

The same official BF16 TinyLlama checkpoint, tokenizer and OpenBookQA test
48 exact prompt/answer IDs under original BEGLIN native CPU C Q4G64 and
repacked QNg64-n8. The source-only Fisher 22n8 mask is SHA locked before
native test is opened. Outcomes are development evidence, not publication
quality superiority or a live runtime switch.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,re,struct,subprocess,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
MODEL=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv7-tinyllama-model-20261022")
LAB=ROOT/"lab_local/qv7_tinyllama_20261022"
TEST=LAB/"eval_openbookqa_test48_tinyllama_frozen.json"
TRAIN=LAB/"source_arc64_tinyllama_frozen.json"
POLICY=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_154_COST_FISHER_POLICY_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV7_TINYLLAMA_22N8_FISHER_FROZEN_POLICY_20261022.txt"
RUN=LAB/"native_openbookqa_test48_two_arms"
ORIG=BASE/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base"
MIX=LAB/"native_mixed_22/beglin_tinyllama_22layer_mixed"
BUILD=LAB/"native_mixed_22/build_receipt.json"
SHAS={"base":"408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c",
  "mixed":"f0dc6789b23892719b0d132c8c173f7d06eefe2e793dcb29bc88821e783e28a3"}
TEST_SHA="8807002e79afadcb2d150b32dffdefa204afb0d70678375a608b7900f18d0ab0"
TRAIN_SHA="59c1d22197ba83ac3769049c01de4ae98414c678066ef6faa48fa2a0859bbe41"
TYPES={"q4":(0,0,154),"fisher_22n8":(0,22,132)}
NAME=("q4","fisher_22n8")
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUMMARY=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
ROLES=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("unsealed original TinyLlama private eval input/outcome "+str(p))
    return p.read_bytes()
def seal(p,b):
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def source():
    if sha(read(TEST))!=TEST_SHA or sha(read(TRAIN))!=TRAIN_SHA:
        raise RuntimeError("original Tiny Llama frozen source/eval cohort drift")
    pol=json.loads(POLICY.read_text())
    if pol["status"]!="SOURCE_ONLY_EXACT_LLAMA_22N8_MASK_FROZEN_BEFORE_CROSSCORPUS_NATIVE_QUALITY" or \
       not pol["unseen_48_test_accuracy_not_opened_by_optimizer"] or sha(MASK.read_bytes())!=pol["mask_sha256"]:
        raise RuntimeError("TinyLlama 22n8 source-only MILP not SHA frozen")
    if sha(ORIG.read_bytes())!=SHAS["base"] or sha(MIX.read_bytes())!=SHAS["mixed"] or \
       json.loads(read(BUILD))["native_binary_SHA256"]!=SHAS["mixed"]:
        raise RuntimeError("BEGLIN C complete mixed and Q4 executables have drifted")
    test=json.loads(read(TEST))["test48"]
    if len(test)!=48:raise RuntimeError("not 48 previously frozen OpenBookQA test cases")
    return pol,test
def prepare():
    pol,test=source()
    if RUN.exists():raise FileExistsError("TinyLlama native test48 already prepared; no retest/tuning")
    os.umask(0o077)
    RUN.mkdir(mode=0o700,parents=True)
    cases=[]
    for idx,q in enumerate(test):
        tok=q["prefix_ids"];correct=q["correct"];choices=q["suffix_ids"]
        if not 2<=len(tok)<=208 or set(choices)!=set("ABCD") or correct not in choices:raise RuntimeError("test token shape")
        ids=struct.pack("<"+"i"*len(tok),*tok)
        letters=(" ".join(str(choices[l]) for l in "ABCD")+" "+str("ABCD".index(correct))+"\n").encode()
        cases.append({"i":idx,"ids_sha":sha(ids),"choices_sha":sha(letters),"length":len(tok),"case_SHA":sha(q["id"].encode())})
        for mode in NAME:
            dst=RUN/mode
            (dst/"eval/ppl_work").mkdir(parents=True,exist_ok=True,mode=0o700)
            (dst/"results").mkdir(mode=0o700,exist_ok=True)
            seal(dst/f"eval/ppl_work/w{idx}.i32",ids)
            seal(dst/f"eval/ppl_work/w{idx}.choices",letters)
            if idx==0:seal(dst/"prompt_ids.i32",ids)
    manifest={"schema":"BEGLIN-Qv7-TinyLlama-pinned-2native-mode-test48-before-eval/1",
       "status":"TINYLLAMA_SOURCE_ONLY_MASK_AND_CROSSCORPUS_48_INPUTS_FROZEN",
       "source64_SHA256":TRAIN_SHA,"held48_SHA256":TEST_SHA,
       "source_only_Fisher_policy_public_SHA256":sha(POLICY.read_bytes()),
       "n8_mask_SHA256":pol["mask_sha256"],
       "executable_sha256":SHAS,"eval_count":48,
       "tokens":cases,"modes":NAME,
       "held_first_prompt_used_prior_for_Q4_implementation_smoke":True,
       "one_test_prompt_contamination_explicitly_disclosed":True,
       "native_eval_is_exploratory_not_final_acceptance":True,
       "no_main_or_live_runtime_mutation":True}
    b=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    seal(RUN/"frozen_test48_native_manifest.json",b)
    print("QV7_TINYLLAMA_48_FISHER_AND_Q4_NATIVE_INPUTS_FROZEN",
       json.dumps({"manifest_sha":sha(b),"policy_mask_sha":pol["mask_sha256"],"cases":48},sort_keys=True),flush=True)
def parse(log:str,mode:str,manifest:dict):
    counts=ROLES.findall(log)
    if len(counts)!=1 or tuple(map(int,counts[0]))!=TYPES[mode]:
        raise RuntimeError("TinyLlama actual Q4/n8 source projection tier unexpected")
    matches=ROW.findall(log)
    if len(matches)!=48:raise RuntimeError("not 48 full TinyLlama one-letter outcome rows")
    values=[];hits=[]
    for i,(idx,val,y,yhat,tokens) in enumerate(matches):
        nll=float(val)
        if int(idx)!=i or int(tokens)!=manifest["tokens"][i]["length"] or \
           not math.isfinite(nll) or nll<0 or not 0<=int(y)<4 or not 0<=int(yhat)<4:
            raise RuntimeError("TinyLlama per-test answer-only NLL/argmax incorrect")
        values.append(nll);hits.append(int(y==yhat))
    v=SUMMARY.findall(log)
    if len(v)!=1:raise RuntimeError("TinyLlama actual C summary missing")
    total,n,right,mean,ppl=map(float,v[0])
    if n!=48 or int(right)!=sum(hits) or abs(total-sum(values))>1e-7 or \
       abs(mean-sum(values)/48)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
        raise RuntimeError("native test48 mean and percase logs disagree")
    return {"correct":int(right),"N":48,"accuracy":sum(hits)/48.,
        "correct_answer_NLL":mean,"percase_answer_NLL":values,"percase_right":hits}
def eval_mode(mode:str):
    pol,test=source()
    if mode not in NAME:raise ValueError("unpermitted native trial")
    manifest=json.loads(read(RUN/"frozen_test48_native_manifest.json"))
    if manifest["status"]!="TINYLLAMA_SOURCE_ONLY_MASK_AND_CROSSCORPUS_48_INPUTS_FROZEN" or \
       manifest["n8_mask_SHA256"]!=pol["mask_sha256"] or manifest["held48_SHA256"]!=TEST_SHA:
        raise RuntimeError("loss-bearing TinyLlama evaluation lacks original prelock")
    dst=RUN/mode
    logpath=dst/"native_tinyllama_48choice_log.txt"
    if logpath.exists():raise FileExistsError("cannot overwrite original native family outcome")
    for item in manifest["tokens"]:
        i=item["i"]
        if sha(read(dst/f"eval/ppl_work/w{i}.i32"))!=item["ids_sha"] or \
           sha(read(dst/f"eval/ppl_work/w{i}.choices"))!=item["choices_sha"]:
            raise RuntimeError("source frozen TinyLlama 48 validation prompt changed")
    exe=ORIG if mode=="q4" else MIX
    env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV5_","BEGLIN_QV6_"))}
    env.update({"QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
      "QWEN_HF_CONFIG":str(MODEL/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(dst.resolve()),"QWEN_PROMPT":str((dst/"prompt_ids.i32").resolve()),
      "VECLIB_MAXIMUM_THREADS":"1","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
      "BEGLIN_QV2_NATIVE_ASYNC":"0","BEGLIN_Q_LAB_PRECISION":"q4" if mode=="q4" else "mixed"})
    if mode!="q4":env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(MASK.resolve())
    start=time.monotonic()
    proc=subprocess.run([str(exe),"masked_choice_qv5","48"],cwd=str(ROOT),env=env,
        capture_output=True,text=True,timeout=1200)
    raw=(proc.stdout+"\n"+proc.stderr).encode()
    os.umask(0o077);seal(logpath,raw)
    if proc.returncode!=0:raise RuntimeError("native TinyLlama failed "+mode+" rc="+str(proc.returncode)+" "+proc.stderr[-900:])
    metrics=parse(raw.decode(),mode,manifest)
    receipt={"schema":"BEGLIN-Qv7-real-native-Llama-family-48-crosscorpus-task-quality/1",
       "status":"TINYLLAMA_REAL_22LAYERS_NATIVE_C_EXECUTED_CROSSDATASET48",
       "mode":mode,"model_revision":pol["model_revision"],
       "binary_sha256":sha(exe.read_bytes()),"original_model_type":"llama",
       "policy_mask_sha256":None if mode=="q4" else pol["mask_sha256"],
       "frozen_test_manifest_SHA256":sha(read(RUN/"frozen_test48_native_manifest.json")),
       "native_log_sha256":sha(raw),"actual_projection_counts":TYPES[mode],
       "model_layers":22,"metrics":metrics,
       "wall_clock_seconds_not_unconfounded_serving_latency":round(time.monotonic()-start,2),
       "evaluation_classification":"DEVELOPMENT_WITH_ONE_SMOCK_CONSUMED_TEST",
       "no_live_release_or_main_changes":True}
    b=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode()
    seal(dst/"native_fullmodel_quality_receipt.json",b)
    print("QV7_TINYLLAMA_ACTUAL_NATIVE_CROSS_FAMILY_48",mode,json.dumps({
       "correct":metrics["correct"],"accuracy":metrics["accuracy"],
       "NLL":metrics["correct_answer_NLL"],"receipt_SHA":sha(b)},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("prepare","eval"))
    p.add_argument("--mode",choices=NAME)
    a=p.parse_args()
    if a.stage=="prepare":prepare()
    elif a.mode:eval_mode(a.mode)
    else:p.error("eval requires --mode")
