#!/usr/bin/env python3
"""Cross-dataset BEGLIN native Q4/Qv5-24/Qv6-64/Qv6-128 OpenBookQA comparison.

The exact *previously audited* Qv5 28-layer Apple CPU C binaries are reused,
no model or kernel source modification. Each of 48 OpenBookQA validation
questions uses one identical official Qwen prompt and four frozen A/B/C/D
tokens in every policy. The source-only 64/128 masks must be frozen before any
model evaluation, with original Fisher source scores and prelocked dataset SHA.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,struct,subprocess,time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
BEG=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
LAB=ROOT/"lab_local/qv6_fisher_scaling_20261022"
FROZEN=LAB/"openbookqa_val48_frozen.json"
SRC=LAB/"source_arc_128_frozen.json"
MILP=ROOT/"docs/research/BEGLIN_QV6_NESTED_64_128_SOURCE_POLICY_20261022.json"
RUN=LAB/"native_openbookqa_48_four_arms"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
SRC_SHA="78d267bef54240cfb32b292c5d1885b6716fadd9bbd78ec7b3b1bf97822a1ab2"
HELD_SHA="c09831bf86c6eedebed433dc2a0398a0992afb19fddc9f86750c2d137d71d1b0"
BINS={"base":(OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base",
      "408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"),
      "mixed":(OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed",
      "f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221")}
MODES=("q4","v5_fisher_24","v6_fisher_64","v6_fisher_128")
MASKS={
 "v5_fisher_24":(ROOT/"configs/research/BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt",
       "88fdc38e6f55a16b0bdd4f7385e70f6de053db577a26d334a3ef3cb831113533"),
 "v6_fisher_64":(ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",None),
 "v6_fisher_128":(ROOT/"configs/research/BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt",None)}
COUNT={"q4":(0,0,196),"v5_fisher_24":(0,28,168),"v6_fisher_64":(0,28,168),"v6_fisher_128":(0,28,168)}
ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
PER=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)

def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def secure(p:pathlib.Path):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("unsealed Qv6 private input or model output "+str(p))
    return p.read_bytes()
def put(p:pathlib.Path,b:bytes):
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def verified():
    if sha(secure(SRC))!=SRC_SHA or sha(secure(FROZEN))!=HELD_SHA:
        raise RuntimeError("source128 and independent cross-corpus 48 corpus drift")
    plan=json.loads(MILP.read_text())
    if plan["status"]!="FROZEN_64_128_POLICIES_BEFORE_NEW_OPENBOOKQA_VALIDATION" or \
       plan["cross_corpus_48_frozen_private_SHA256"]!=HELD_SHA or not plan["evaluation_48_OpenBookQA_logits_not_opened"]:
        raise RuntimeError("attempt to tune precision bits from held data")
    for p,(binary,dig) in BINS.items():
        if not binary.is_file() or sha(binary.read_bytes())!=dig:
            raise RuntimeError("original audited native C binary unavailable/drift "+p)
    for mode,(mask,dig) in MASKS.items():
        original=dig if dig else plan["selected_policies"][mode.rsplit("_",1)[1]]["mask_sha256"]
        if sha(mask.read_bytes())!=original:raise RuntimeError("source-only Fisher mask SHA changed "+mode)
        rows=[tuple(line.split()) for line in mask.read_text().splitlines()]
        if len(rows)!=28 or len(set(rows))!=28:
            raise RuntimeError("not 28 original source tensor upgrades")
    held=json.loads(secure(FROZEN))
    if held["sample_count"]!=48 or len(held["unseen_questions"])!=48:
        raise RuntimeError("not prelocked 48 cross-corpus cases")
    return held,plan

def prepare():
    held,policy=verified()
    if RUN.exists():raise FileExistsError("native OpenBookQA 48 already prepared; no reselection")
    os.umask(0o077)
    RUN.mkdir(parents=True,mode=0o700)
    locked=[]
    for i,case in enumerate(held["unseen_questions"]):
        prompt=case["prefix_ids"];answer=case["correct"];choices=case["suffix_ids"]
        if not 2<=len(prompt)<=208 or set(choices)!={"A","B","C","D"} or answer not in choices:
            raise RuntimeError("invalid frozen OpenBookQA four letter tokenizer")
        data=struct.pack("<"+"i"*len(prompt),*prompt)
        suffix=(" ".join(str(choices[x]) for x in "ABCD")+" "+str("ABCD".index(answer))+"\n").encode()
        locked.append({"index":i,"prompt_sha256":sha(data),"choice_sha256":sha(suffix),
          "prompt_length":len(prompt),"item_id_hash":sha(case["id"].encode())})
        for name in MODES:
            dest=RUN/name
            (dest/"eval/ppl_work").mkdir(mode=0o700,parents=True,exist_ok=True)
            (dest/"results").mkdir(mode=0o700,exist_ok=True)
            put(dest/f"eval/ppl_work/w{i}.i32",data)
            put(dest/f"eval/ppl_work/w{i}.choices",suffix)
            if i==0:put(dest/"prompt_ids.i32",data)
    out={"schema":"BEGLIN-Qv6-frozen-exact-48-OpenBookQA-native-eval/1",
      "status":"FROZEN_FOUR_ARMS_BEFORE_FIRST_CROSSCORPUS_NATIVE_FORWARD",
      "source_train_128_private_SHA256":SRC_SHA,
      "cross_validation_48_private_SHA256":HELD_SHA,
      "source_only_two_policies_document_SHA256":sha(MILP.read_bytes()),
      "all_48_prompt_choice_hashes":locked,
      "native_Qv5_verbatim_source_binary_SHA256":{k:v[1] for k,v in BINS.items()},
      "all_mask_SHA256":{k:sha(v[0].read_bytes()) for k,v in MASKS.items()},
      "modes":MODES,"dataset_revision":held["dataset_sha256"],"scored_item_count":48,
      "no_model_outputs_before_policy_freeze":True,
      "not_training_or_calibration_corpus":True,"no_live_modification":True}
    p=(json.dumps(out,sort_keys=True,indent=2)+"\n").encode()
    put(RUN/"frozen_natural_science_48_manifest.json",p)
    print("QV6_OPENBOOKQA48_NATIVE_FROZEN",json.dumps({
       "manifest_SHA":sha(p),"dataset":held["dataset_sha256"],"cases":48,
       "runs":len(MODES),"native_bin_SHA":{k:v[1] for k,v in BINS.items()}},sort_keys=True),flush=True)

def parse_log(log,mode,frozen):
    counts=ROLE.findall(log)
    if len(counts)!=1 or tuple(map(int,counts[0]))!=COUNT[mode]:
        raise RuntimeError("BEGLIN original C executed wrong q4/n8 projection count "+mode)
    rows=PER.findall(log)
    if len(rows)!=48:raise RuntimeError("not 48 original native answer token losses "+mode)
    expected=[];got=[];losses=[]
    for i,(id,nll,y,yp,n) in enumerate(rows):
        v=float(nll)
        if (int(id)!=i or not math.isfinite(v) or v<0 or
             int(n)!=frozen["all_48_prompt_choice_hashes"][i]["prompt_length"] or
             not 0<=int(y)<4 or not 0<=int(yp)<4):
            raise RuntimeError("original 48 C source scores invalid")
        losses.append(v);expected.append(int(y));got.append(int(yp))
    totals=SUM.findall(log)
    if len(totals)!=1:raise RuntimeError("missing unique native 48 mean score")
    sumv,nwin,right,mean,ppl=map(float,totals[0])
    if (nwin!=48 or right!=sum(y==yp for y,yp in zip(expected,got)) or
        abs(sumv-sum(losses))>1e-7 or abs(mean-sum(losses)/48)>1e-7 or
        abs(math.log(ppl)-mean)>1e-7):
        raise RuntimeError("native source score mean or count differs")
    return {"cases":48,"correct":int(right),"accuracy":right/48,
      "answer_only_correct_letter_NLL":mean,
      "per_case_correct_letter_NLL":losses,
      "per_case_expected_option":expected,
      "per_case_native_argmax":got,
      "per_case_right":[int(y==yp) for y,yp in zip(expected,got)]}

def evaluate(mode):
    held,policy=verified()
    if mode not in MODES:raise ValueError("unpermitted source policy")
    manifest=json.loads(secure(RUN/"frozen_natural_science_48_manifest.json"))
    if manifest["status"]!="FROZEN_FOUR_ARMS_BEFORE_FIRST_CROSSCORPUS_NATIVE_FORWARD" or \
       manifest["source_only_two_policies_document_SHA256"]!=sha(MILP.read_bytes()) or \
       manifest["cross_validation_48_private_SHA256"]!=HELD_SHA:
        raise RuntimeError("cannot run model on refitted or unbound original source")
    dest=RUN/mode
    logpath=dest/"openbookqa48_native_log.txt"
    if logpath.exists():raise FileExistsError("native 48 policy output already exists; refuse overwrite")
    for i,x in enumerate(manifest["all_48_prompt_choice_hashes"]):
        if sha(secure(dest/f"eval/ppl_work/w{i}.i32"))!=x["prompt_sha256"] or \
           sha(secure(dest/f"eval/ppl_work/w{i}.choices"))!=x["choice_sha256"]:
            raise RuntimeError("candidate prefix or options changed after freeze")
    which="base" if mode=="q4" else "mixed"
    exe=BINS[which][0]
    env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV5_","BEGLIN_QV6_","BEGLIN_QV4_"))}
    env.update({"VECLIB_MAXIMUM_THREADS":"1",
      "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
      "QWEN_HF_CONFIG":str(SNAP/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(dest.resolve()),
      "QWEN_PROMPT":str((dest/"prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":"q4" if mode=="q4" else "mixed",
      "BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0"})
    if mode!="q4":env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(MASKS[mode][0].resolve())
    t=time.monotonic()
    result=subprocess.run([str(exe),"masked_choice_qv5","48"],cwd=str(BEG),
       env=env,text=True,capture_output=True,timeout=1800)
    log=(result.stdout+"\n"+result.stderr).encode()
    os.umask(0o077);put(logpath,log)
    if result.returncode!=0:raise RuntimeError("real 48 OpenBookQA CPU native model failed "+mode+str(result.returncode)+result.stderr[-500:])
    metrics=parse_log(log.decode(),mode,manifest)
    receipt={"schema":"BEGLIN-Qv6-native-OpenBookQA48-policy-comparison/1",
      "status":"ACTUAL_NATIVE_C_QWEN28_FULL_FOURCHOICE_48_COMPLETE",
      "mode":mode,"origin_qv5_28layer_native_binary_SHA":sha(exe.read_bytes()),
      "frozen_input_manifest_SHA256":sha(secure(RUN/"frozen_natural_science_48_manifest.json")),
      "actual_native_log_SHA256":sha(log),"same_source_model_BF16_cache_rev":SNAP.name,
      "N8_policy_mask_sha256":None if mode=="q4" else sha(MASKS[mode][0].read_bytes()),
      "metrics":metrics,"elapsed_wall_s":round(time.monotonic()-t,2),
      "no_cost_benchmark_claim":True,"research_only":True,"production_touched":False}
    b=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
    put(dest/"openbookqa48_native_receipt.json",b)
    print("QV6_OPENBOOKQA48_NATIVE_C",mode,json.dumps({
      "accuracy":metrics["accuracy"],"correct":metrics["correct"],
      "NLL":metrics["answer_only_correct_letter_NLL"],
      "native_receipt_SHA":sha(b),"wall_s":receipt["elapsed_wall_s"]},sort_keys=True),flush=True)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","eval"))
    parser.add_argument("--mode",choices=MODES)
    args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.mode:evaluate(args.mode)
    else:parser.error("eval requires --mode")
