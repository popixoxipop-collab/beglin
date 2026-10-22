#!/usr/bin/env python3
"""Five-arm true 28-layer BEGLIN CPU-native Qwen MMLU 48 evaluation.

All arms share pinned original BF16 checkpoint and original Qv5 audited Q4 or
n8-repacked C binary, exact source-only masks and identical 48 MMLU 4-choice
Qwen ChatML tokens. Qv9 mask is frozen BEFORE invoking any validation logits.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,re,struct,subprocess,time

ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
LAB=ROOT/"lab_local/qv9_ggn_20261023"
TEST=LAB/"MMLU_psych_CS_48_eval_frozen.json"
PRE=ROOT/"configs/research/BEGLIN_QV9_GGN_SOURCE_PRELOCK_20261023.json"
POL=ROOT/"docs/research/BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
MASK=ROOT/"configs/research/BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt"
RUN=LAB/"MMLU48_5arm_original_CPU_native"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRE_SHA="73dc2829da79637a97d68f903f72fed753eace2b154caec111c1fe60a7f4f983"
VAL_SHA="2203bf54dfbc20e07488a234a012011efd78b098876e981de8ebed12cf8045c9"
MODES=("q4","qv6_64","qv6_128","qv8_joint96","qv9_ggn24")
BINS={
 "base":(OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base",
    "408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"),
 "mixed":(OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed",
    "f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221")}
MASKS={
 "qv6_64":(ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",
            "74fb949e45d13f63a26d412effc39377492ea71bda20ae3690f56fddf6303e08"),
 "qv6_128":(ROOT/"configs/research/BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt",
            "8b9e2404f9a38d7b57784b7d663efa19661553ccaa7dc79bfcbf6e6180fc7997"),
 "qv8_joint96":(ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt",
                 "9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922"),
 "qv9_ggn24":(MASK,None)}
LOG_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
LOG_TOTAL=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
LOG_ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("unsealed Qv9 original validation data/results "+str(p))
    return p.read_bytes()
def write(p,data):
    with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def verify():
    if sha(PRE.read_bytes())!=PRE_SHA or sha(sealed(TEST))!=VAL_SHA:
        raise RuntimeError("Qv9 new MMLU dataset prelock or original 48 questions changed")
    if not POL.exists():raise RuntimeError("GGN output curvature mask not frozen")
    p=json.loads(POL.read_text())
    if (p["status"]!="GGN_SOURCE_ONLY_FROZEN_BEFORE_DIAG_OR_MMLU" or
        p["MMLU_eval_48_logits_unopened"] is not True or
        not p["source_only_choice_GGN_not_full_vocab"] or sha(MASK.read_bytes())!=p["mask_sha256"]):
        raise RuntimeError("new GGN mask not frozen before validation outputs")
    val=json.loads(sealed(TEST))
    if len(val["items"])!=48 or not val["new_eval_quality_labels_not_used_by_optimizer"]:
        raise RuntimeError("MMLU validation cohort not 48")
    for label,(exe,dig) in BINS.items():
        if not exe.is_file() or sha(exe.read_bytes())!=dig:raise RuntimeError("actual C binary changed "+label)
    for mode,(mask,h) in MASKS.items():
        expect=h if h else p["mask_sha256"]
        if not mask.is_file() or sha(mask.read_bytes())!=expect:raise RuntimeError("policy mask drift "+mode)
        selected=[tuple(line.split()) for line in mask.read_text().splitlines()]
        if len(selected)!=28 or len(set(selected))!=28:
            raise RuntimeError("frozen native 28 tensors not unique "+mode)
    return val,p
def prepare():
    val,policy=verify()
    if RUN.exists():raise FileExistsError("cannot reprepare held MMLU 5-arm after inference")
    os.umask(0o077)
    RUN.mkdir(mode=0o700,parents=True)
    locked=[]
    for index,q in enumerate(val["items"]):
        ids=q["prefix_ids"]; correct=q["correct"];letter=q["suffix_ids"]
        if not 18<=len(ids)<=208 or set(letter)!=set("ABCD"):
            raise RuntimeError("original Qwen single-letter token boundary invalid")
        data=struct.pack("<"+"i"*len(ids),*ids)
        labels=(" ".join(str(letter[x]) for x in "ABCD")+" "+str("ABCD".index(correct))+"\n").encode()
        locked.append({"index":index,"id_sha256":sha(q["id"].encode()),
           "prompt_SHA256":sha(data),"choices_SHA256":sha(labels),"length":len(ids),
           "subject":q["subject"]})
        for mode in MODES:
            loc=RUN/mode
            (loc/"eval/ppl_work").mkdir(mode=0o700,parents=True,exist_ok=True)
            (loc/"results").mkdir(mode=0o700,exist_ok=True)
            write(loc/f"eval/ppl_work/w{index}.i32",data)
            write(loc/f"eval/ppl_work/w{index}.choices",labels)
            if index==0:write(loc/"prompt_ids.i32",data)
    manifest={
       "schema":"BEGLIN-Qv9-MMLU48-5arm-precisions-original-Qwen-native-input-lock/1",
       "status":"FROZEN_SOURCE_ONLY_GGN_MASKS_AND_48_MMLU_INPUTS_BEFORE_C_FORWARD",
       "prelock_SHA256":PRE_SHA,"original_TEST48_SHA256":VAL_SHA,
       "original_GGN_source_policy_doc_SHA256":sha(POL.read_bytes()),
       "source_only_GGN_mask_SHA256":policy["mask_sha256"],
       "all_policy_masks_SHA256":{m:(h or policy["mask_sha256"]) for m,(_,h) in MASKS.items()},
       "native_binary_SHAs":{k:d for k,(_,d) in BINS.items()},
       "modes":list(MODES),"inputs":locked,
       "MMLU_subjects":val["subjects"],"source_only_train24_no_MMLU_model_outputs":True,
       "main_release_or_checkpoint_modified":False}
    data=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    write(RUN/"frozen_5x48_manifest.json",data)
    print("QV9_MMLU48_5ARM_ACTUAL_NATIVE_INPUTS_FROZEN",json.dumps({
        "manifest_sha":sha(data),"mask_sha":policy["mask_sha256"],"cases":len(locked)},sort_keys=True),flush=True)
def evaluate(mode):
    val,policy=verify()
    if mode not in MODES:raise ValueError("unapproved native condition")
    lock=json.loads(sealed(RUN/"frozen_5x48_manifest.json"))
    if lock["status"]!="FROZEN_SOURCE_ONLY_GGN_MASKS_AND_48_MMLU_INPUTS_BEFORE_C_FORWARD" or \
       lock["original_GGN_source_policy_doc_SHA256"]!=sha(POL.read_bytes()) or \
       lock["original_TEST48_SHA256"]!=VAL_SHA:
        raise RuntimeError("Qv9 GGN bit policy or MMLU validation input changed")
    dest=RUN/mode
    logfile=dest/"real_mmlu48_native_C_log.txt"
    if logfile.exists():raise FileExistsError("cannot overwrite real source 48 native evaluation")
    for i,meta in enumerate(lock["inputs"]):
        if sha(sealed(dest/f"eval/ppl_work/w{i}.i32"))!=meta["prompt_SHA256"] or \
           sha(sealed(dest/f"eval/ppl_work/w{i}.choices"))!=meta["choices_SHA256"]:
            raise RuntimeError("original frozen MMLU tokens tampered")
    tag="base" if mode=="q4" else "mixed"
    executable=BINS[tag][0]
    env={k:v for k,v in os.environ.items() if not k.startswith((
      "QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV2_","BEGLIN_QV4_","BEGLIN_QV5_","BEGLIN_QV6_","BEGLIN_QV7_","BEGLIN_QV8_","BEGLIN_QV9_"))}
    env.update({"VECLIB_MAXIMUM_THREADS":"1","QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
      "QWEN_HF_CONFIG":str(SNAP/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(dest.resolve()),"QWEN_PROMPT":str((dest/"prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":"q4" if mode=="q4" else "mixed",
      "BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0"})
    if mode!="q4":env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(MASKS[mode][0].resolve())
    t=time.monotonic()
    ret=subprocess.run([str(executable),"masked_choice_qv5","48"],env=env,cwd=str(QWEN),
       capture_output=True,text=True,timeout=1800)
    log=(ret.stdout+"\n"+ret.stderr).encode()
    os.umask(0o077);write(logfile,log)
    if ret.returncode!=0:
        raise RuntimeError("original full 28-layer Qwen C native MMLU failed "+str(ret.returncode)+" "+ret.stderr[-600:])
    roles=LOG_ROLE.findall(log.decode())
    if len(roles)!=1 or tuple(map(int,roles[0]))!=((0,0,196) if mode=="q4" else (0,28,168)):
        raise RuntimeError("actual mixed native Q4/n8 mask count not correct")
    cases=LOG_ROW.findall(log.decode())
    if len(cases)!=48:raise RuntimeError("not original 48 MMLU next-token full vocab losses")
    scores=[];right=[]
    for i,(idx,nll,y,pred,n) in enumerate(cases):
        loss=float(nll)
        if int(idx)!=i or not math.isfinite(loss) or loss<0 or \
           int(n)!=lock["inputs"][i]["length"] or not 0<=int(y)<=3 or not 0<=int(pred)<=3:
            raise RuntimeError("incorrect native MMLU token boundary")
        scores.append(loss);right.append(int(y==pred))
    totals=LOG_TOTAL.findall(log.decode())
    if len(totals)!=1:raise RuntimeError("native MMLU source aggregate missing")
    total,n,nright,mean,ppl=map(float,totals[0])
    if n!=48 or nright!=sum(right) or abs(total-sum(scores))>1e-7 or abs(mean-total/48)>1e-7:
        raise RuntimeError("native MMLU answer NLL/accuracy arithmetic mismatch")
    metric={
       "correct":int(nright),"total":48,"accuracy":nright/48,
       "mean_correct_fullvocab_next_token_NLL":mean,
       "per_question_right":right,"per_question_answer_NLL":scores}
    record={
       "schema":"BEGLIN-Qv9-MMLU48-original-C-Qwen28-Q4n8-fullmodel-one-arm/1",
       "status":"REAL_APPLE_BEGLIN_C_QWEN28_MMLU48_ONE_ARM_PASS",
       "mode":mode,"model_rev":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
       "original_binary_SHA256":sha(executable.read_bytes()),"native_log_SHA256":sha(log),
       "source_mask_SHA256":None if mode=="q4" else sha(MASKS[mode][0].read_bytes()),
       "frozen_input_manifest_SHA256":sha(sealed(RUN/"frozen_5x48_manifest.json")),
       "metrics":metric,
       "elapsed_wall_seconds":round(time.monotonic()-t,2),
       "not_sustained_decode_speed_evidence":True,"no_operational_promotion":True}
    data=(json.dumps(record,sort_keys=True,indent=2)+"\n").encode()
    write(dest/"original_mmlu48_native_receipt.json",data)
    print("QV9_REAL_MMLU48_NATIVE_C",mode,json.dumps({
       "correct":metric["correct"],"NLL":mean,"native_receipt_sha":sha(data),
       "elapsed_s":record["elapsed_wall_seconds"]},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("prepare","eval"))
    p.add_argument("--mode",choices=MODES)
    args=p.parse_args()
    if args.stage=="prepare":prepare()
    elif args.mode:evaluate(args.mode)
    else:p.error("mode required")
